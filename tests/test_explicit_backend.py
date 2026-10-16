import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.explicit_material import advance,response,geometry,hex_to_tet,oriented
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.adaptive_tet import propose,quality,boundary
from packing_fem.periodic_contact import Contact
from packing_fem import sintered

ROOT=Path(__file__).resolve().parents[1]


def config():return json.loads((ROOT/'configs/explicit-prototype.json').read_text())


def test_finite_plastic_material_objectivity_and_isochoric_flow():
    c=config();b=np.eye(3)[None];ep=np.zeros(1);F=np.diag([1.1,.94,.96])[None]
    b1,ep1=advance(b,ep,F,c);assert ep1[0]>0
    assert np.linalg.det(b1)[0]==pytest.approx(np.linalg.det(F)[0]**2,rel=1e-12)
    theta=.83;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
    br,epr=advance(b,ep,R[None]@F,c)
    sigma=response(b1,ep1,c)[0];rotated=response(br,epr,c)[0]
    np.testing.assert_allclose(rotated,R[None]@sigma@R.T,atol=1e-13)
    np.testing.assert_allclose(epr,ep1)
    # A rigid rotation after plastic loading adds no plastic strain or energy.
    b2,ep2=advance(b1,ep1,R[None],c)
    np.testing.assert_allclose(ep2,ep1,atol=1e-14)
    np.testing.assert_allclose(response(b2,ep2,c)[1],response(b1,ep1,c)[1],rtol=1e-12)


def test_hencky_elastic_hydrostatic_limit_and_internal_force():
    c=config();c['yield_mpa']=1e6;b=np.eye(3)[None]*1.03**2
    sigma,_,J=response(b,np.zeros(1),c);K=c['young_mpa']/(3*(1-2*c['poisson']))
    np.testing.assert_allclose(sigma[0],np.eye(3)*K*np.log(1.03**3)/1.03**3,atol=1e-14)
    assert J[0]==pytest.approx(1.03**3)
    x=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float);t=np.arange(4)[None]
    from packing_fem.explicit_material import internal_force
    force,*_=internal_force(x*1.03,t,b,np.zeros(1),c)
    np.testing.assert_allclose(force.sum(axis=0),0,atol=1e-15)
    np.testing.assert_allclose(np.cross(x*1.03,force).sum(axis=0),0,atol=1e-15)


def test_periodic_affine_patch_and_virtual_work():
    x,h,_=sintered.mesh(np.ones((2,2,2),bool),1.);p=Periodic(x,2.)
    rng=np.random.default_rng(42);q=rng.normal(size=p.ndof)*.01
    current=p.positions(q);F=deformation(q[-6:])
    for axis in range(3):
        a=np.where(x[:,axis]==0)[0];b=np.where(x[:,axis]==2)[0]
        other=[i for i in range(3) if i!=axis]
        for i in a:
            j=b[np.all(x[b][:,other]==x[i,other],axis=1)][0]
            np.testing.assert_allclose(current[j]-current[i],2*F[:,axis],atol=1e-14)
    force=rng.normal(size=x.shape);dq=rng.normal(size=p.ndof)
    assert p.force(force)@dq==pytest.approx(np.sum(force*p.velocities(dq)),rel=1e-14)
    p.mass(np.ones(len(x)))
    assert np.linalg.eigvalsh(p.Mff.toarray()).min()>0


def test_local_remesh_improves_quality_preserves_boundary_and_plastic_state():
    x=np.array([[1,0,0],[-.5,np.sqrt(3)/2,0],[-.5,-np.sqrt(3)/2,0],[0,0,.15],[0,0,-.15]])
    t=oriented(x,np.array([[0,1,2,3],[0,2,1,4]]));V0=geometry(x,t)[0]
    F=np.diag([1.05,.95,1.]);b0,ep0=advance(np.eye(3)[None],np.zeros(1),F[None],config())
    b=np.repeat(b0,2,axis=0);ep=np.repeat(ep0,2);V0=V0/np.linalg.det(F)
    proposal=propose(x,t,b,ep,V0,config(),threshold=1.,energy_tolerance=1e-10)
    assert proposal is not None
    t2,b2,ep2,V02,event=proposal
    assert len(t2)==3 and event['quality_after']>event['quality_before']
    assert V02.sum()==pytest.approx(V0.sum(),rel=1e-14)
    assert set(map(tuple,np.sort(boundary(t),axis=1)))==set(map(tuple,np.sort(boundary(t2),axis=1)))
    np.testing.assert_allclose(b2,np.repeat(b0,3,axis=0),atol=1e-13)
    np.testing.assert_allclose(ep2,ep[0],atol=1e-14)


def contact_fixture(periodic=False):
    # Two tetrahedra, opposed parallel free faces. Their remote back vertices
    # make outward normals unambiguous. Periodic image contact uses x faces.
    if not periodic:
        x=np.array([[.2,.2,.4],[.8,.2,.4],[.2,.8,.4],[.2,.2,.1],
                    [.2,.2,.6],[.2,.8,.6],[.8,.2,.6],[.2,.2,.9]])
        current=x.copy();current[4:,2]-=.22
    else:
        x=np.array([[.1,.2,.2],[.1,.8,.2],[.1,.2,.8],[.4,.2,.2],
                    [.9,.2,.2],[.9,.2,.8],[.9,.8,.2],[.6,.2,.2]])
        current=x.copy();current[:4,0]-=.22
    t=oriented(x,np.array([[0,1,2,3],[4,5,6,7]]));p=Periodic(x,1.)
    return x,current,t,p


@pytest.mark.parametrize('periodic',[False,True])
def test_periodic_contact_action_reaction_coulomb_and_image_work(periodic):
    X,x,t,p=contact_fixture(periodic);contact=Contact(X,t,p,100.,.3,.1)
    vel=np.zeros_like(x);vel[4:,1]=1.
    force,extra,hist,report=contact.evaluate(x,vel,np.eye(3),np.zeros((3,3)),.01)
    assert report['contacts']>0 and report['penetration_mm']==pytest.approx(.02)
    np.testing.assert_allclose(force.sum(axis=0),0,atol=1e-14)
    assert report['max_coulomb_ratio']<=.3+1e-12
    assert report['friction_increment']>=0
    assert np.linalg.norm(extra)>0 if periodic else np.linalg.norm(extra)==0
    contact.history=hist
    force2,_,_,report2=contact.evaluate(x,vel,np.eye(3),np.zeros((3,3)),.01)
    assert report2['friction_increment']>0
    # The second pass reaches the Coulomb cap, opposing relative sliding.
    assert (force2[4:].sum(axis=0)@vel[4])<=1e-12


def test_nonadjacent_same_body_contact_not_filtered_by_component_label():
    X,x,t,_=contact_fixture(False)
    bridge=np.array([[1.3,1.2,.1],[1.6,1.4,.3],[1.4,1.6,.5],[1.7,1.3,.7],[1.5,1.7,1.]])
    X=np.r_[X,bridge];x=np.r_[x,bridge]
    chain=np.array([[0,1,3,8],[1,3,8,9],[3,8,9,10],[8,9,10,11],
                    [9,10,11,12],[10,11,12,7],[11,12,7,4],[12,7,4,5]])
    t=oriented(X,np.r_[t,chain]);p=Periodic(X,3.)
    # All elements are connected through shared FACES, not just body labels.
    visited={0}
    while True:
        new=visited|{j for j,row in enumerate(t) if any(len(set(row)&set(t[i]))>=3 for i in visited)}
        if new==visited:break
        visited=new
    assert len(visited)==len(t)
    contact=Contact(X,t,p,100.,.3,.1)
    force,_,_,report=contact.evaluate(x,np.zeros_like(x),np.eye(3)*3,np.zeros((3,3)),0.)
    assert report['contacts']>0
    assert np.linalg.norm(force[:8])>0


def test_nonuniform_plastic_history_remesh_rejected_when_energy_would_be_lost():
    x=np.array([[1,0,0],[-.5,np.sqrt(3)/2,0],[-.5,-np.sqrt(3)/2,0],[0,0,.15],[0,0,-.15]])
    t=oriented(x,np.array([[0,1,2,3],[0,2,1,4]]));V0=geometry(x,t)[0]
    b=np.repeat(np.eye(3)[None],2,axis=0);ep=np.array([0.,.2]);saved=ep.copy()
    assert propose(x,t,b,ep,V0,config(),1.,1e-6) is None
    np.testing.assert_array_equal(ep,saved)


def test_contact_periodic_image_force_is_derivative_of_penalty_energy():
    X,x,t,p=contact_fixture(True);contact=Contact(X,t,p,100.,0.,.1)
    force,extra,_,_=contact.evaluate(x,np.zeros_like(x),np.eye(3),np.zeros((3,3)),0.)
    energies=[];delta=1e-7
    for shift in [-delta,delta]:
        H=np.eye(3);H[0,0]+=shift
        energies.append(contact.evaluate(x,np.zeros_like(x),H,np.zeros((3,3)),0.)[-1]['normal_energy'])
    assert -(energies[1]-energies[0])/(2*delta)==pytest.approx(extra[0],rel=1e-6)


def test_friction_spring_is_objective_under_rigid_rotation():
    X,x,t,p=contact_fixture(False);contact=Contact(X,t,p,100.,.3,.1)
    velocity=np.zeros_like(x);velocity[4:,0]=.1
    force,_,hist,_=contact.evaluate(x,velocity,np.eye(3),np.zeros((3,3)),.01);contact.history=hist
    angle=.7;R=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
    rotated,_,_,_=contact.evaluate(x@R.T,np.zeros_like(x),R,np.zeros((3,3)),0.)
    np.testing.assert_allclose(rotated,force@R.T,atol=1e-13)


def test_elastic_unloading_preserves_finite_plastic_memory():
    c=config();G=c['young_mpa']/(2*(1+c['poisson']));strain=.05
    F=np.diag(np.exp([strain,-strain/2,-strain/2]))[None]
    b,ep=advance(np.eye(3)[None],np.zeros(1),F,c)
    expected=(3*G*strain-c['yield_mpa'])/(3*G+c['hardening_mpa'])
    assert ep[0]==pytest.approx(expected,rel=1e-13)
    elastic=strain-expected
    unload=np.diag(np.exp([-elastic,elastic/2,elastic/2]))[None]
    bu,epu=advance(b,ep,unload,c)
    np.testing.assert_allclose(response(bu,epu,c)[0],0,atol=1e-13)
    assert epu[0]==pytest.approx(expected,rel=1e-13)
    assert (unload@F)[0,0,0]==pytest.approx(np.exp(expected))


def test_explicit_homogeneous_cell_against_free_lateral_hencky_solution(tmp_path):
    from packing_fem.explicit_solver import solve
    c=config();c.update(length_mm=1.,yield_mpa=1e6,compression_fraction=.001,contact_penalty_mpa_mm=1.,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.)
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.)
    result=solve(tmp_path/'native_explicit',X,hex_to_tet(X,h),c)
    assert all(result['checks'].values()),result
    last=np.load(tmp_path/'native_explicit/last-state.npz');F=deformation(last['generalized_position'][-6:])
    assert F[0,0]==pytest.approx(F[2,2]**(-c['poisson']),rel=1e-6)
    history=np.genfromtxt(tmp_path/'native_explicit/history.csv',delimiter=',',names=True)
    nominal=-c['young_mpa']*np.log(F[2,2])/F[2,2]
    assert history[-1]['nominal_compression_MPa']==pytest.approx(nominal,rel=1e-5)


def test_periodic_cut_face_only_removed_when_opposite_material_face_exists():
    X,h,_=sintered.mesh(np.ones((2,2,2),bool),1.);t=hex_to_tet(X,h)
    assert len(Contact(X,t,Periodic(X,2.),100.,.3,.1).faces)==0
    mask=np.zeros((3,3,3),bool);mask[0,0,0]=True
    X,h,_=sintered.mesh(mask,1.);t=hex_to_tet(X,h)
    # The three faces on coordinate zero border pores in the next image.
    assert len(Contact(X,t,Periodic(X,3.),100.,.3,.1).faces)==12
    mask[2,0,0]=True;X,h,_=sintered.mesh(mask,1.);t=hex_to_tet(X,h)
    assert len(Contact(X,t,Periodic(X,3.),100.,.3,.1).faces)==20


def test_legacy_audits_cannot_read_the_new_explicit_material_history(tmp_path):
    from packing_fem.energy import audit
    from packing_fem import plastic
    np.savez(tmp_path/'initial-mesh.npz',points=np.zeros((4,3)))
    with pytest.raises(ValueError,match='explicit'):audit(tmp_path,tmp_path/'must_not_exist')
    assert not (tmp_path/'must_not_exist').exists()
    with pytest.raises(ValueError,match='explicit'):plastic.analyze(tmp_path,True,0)


def test_closed_boundary_volume_independent_of_tetra_interior_and_translation():
    from packing_fem.adaptive_tet import closed_surface_volume
    mask=np.ones((3,3,3),bool);mask[1,1,1]=False
    X,h,_=sintered.mesh(mask,1.);t=hex_to_tet(X,h)
    F=np.array([[1.1,.1,.03],[.1,.95,0],[.03,0,.8]])
    current=X@F.T+np.array([10.,-3.,2.])
    assert closed_surface_volume(current,t)==pytest.approx(26*np.linalg.det(F),rel=1e-13)


def test_explicit_unloading_releases_load_and_leaves_permanent_compression(tmp_path):
    from packing_fem.explicit_solver import solve
    c=config();c.update(length_mm=1.,compression_fraction=.03,contact_penalty_mpa_mm=1.,ramp_s=.001,hold_s=.0005,unload_s=.001,recovery_hold_s=.0005,damping_per_s=20000.)
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.)
    result=solve(tmp_path/'unload',X,hex_to_tet(X,h),c)
    assert all(result['checks'].values()),result
    assert result['residual_compression_strain']==pytest.approx(.02465,rel=.01)
    history=np.genfromtxt(tmp_path/'unload/history.csv',names=True,delimiter=',')
    assert abs(history[-1]['nominal_compression_MPa'])<1e-7


def test_reference_contact_area_keeps_penalty_potential_conservative_on_stretch():
    X,x,t,p=contact_fixture(False);contact=Contact(X,t,p,100.,0.,.1)
    old=contact.evaluate(x,np.zeros_like(x),np.eye(3),np.zeros((3,3)),0.)[-1]
    stretched=x.copy();stretched[:,:2]*=1.05
    new=contact.evaluate(stretched,np.zeros_like(x),np.eye(3),np.zeros((3,3)),0.)[-1]
    assert new['contacts']==old['contacts']>0
    assert new['normal_energy']==pytest.approx(old['normal_energy'],rel=1e-12)


def test_same_particle_can_contact_its_remote_periodic_image():
    X=np.array([[.2,.4,.4],[.6,.3,.3],[.6,.6,.3],[.6,.3,.6]])
    t=oriented(X,np.arange(4)[None]);p=Periodic(X,1.)
    contact=Contact(X,t,p,100.,.3,.1)
    current=X.copy();current[1:,0]=1.22
    force,extra,_,report=contact.evaluate(current,np.zeros_like(X),np.eye(3),np.zeros((3,3)),0.)
    assert report['contacts']>0
    assert report['penetration_mm']==pytest.approx(.02)
    assert np.linalg.norm(extra)>0
    np.testing.assert_allclose(force.sum(axis=0),0,atol=1e-14)
