import numpy as np
import pytest
from scipy.spatial.distance import pdist
from packing_fem.rigid_inclusion import RigidPeriodic,RigidPeriodicIPC,rotation_derivatives,cylinder_surface,rigid_volume
from packing_fem.explicit_material import oriented
from packing_fem.fixed_point import affine_predictor
from packing_fem.friction_audit import friction_report
import ipctk as ipc

C=dict(rigid_inclusion=True,rigid_diameter_mm=2.,rigid_height_mm=2.,rigid_segments=16,length_mm=10.,friction=.3,ipc_dhat_mm=.15,ipc_stiffness_mpa=10.,ipc_slip_regularization_mm=1e-4)
X=np.array([[6.08,5.1,5.1],[7.,4.2,4.],[7.,5.8,4.],[7.,5.,6.]])
T=oriented(X,np.array([[0,1,2,3]]))


def test_rigid_finite_rotation_and_hessian_match_derivatives():
    a=np.array([.2,-.3,.4]);R,d,dd=rotation_derivatives(a);np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-14)
    assert np.linalg.det(R)==pytest.approx(1.)
    h=1e-5
    for j in range(3):
        v=np.eye(3)[j]*h;rp,dp,_=rotation_derivatives(a+v);rm,dm,_=rotation_derivatives(a-v)
        np.testing.assert_allclose((rp-rm)/(2*h),d[j],atol=1e-10)
        np.testing.assert_allclose((dp-dm)/(2*h),dd[:,j],atol=1e-10)


def test_macro_compression_preserves_rigid_shape_and_periodic_images():
    p=RigidPeriodic(X,10.,C);contact=RigidPeriodicIPC(X,T,p,C);q=np.zeros(p.ndof);q[p.rigid_dofs]=[.1,-.2,.3,.2,-.3,.4]
    original=q.copy();moved=affine_predictor(p,q,-.35)
    np.testing.assert_array_equal(moved[p.rigid_dofs[3:]],original[p.rigid_dofs[3:]])
    x=contact.positions(moved)[contact.body_vertices:];local,faces=cylinder_surface(C);N=len(local)
    np.testing.assert_allclose(pdist(x[:N]),pdist(local),atol=1e-13)
    np.testing.assert_allclose(x[-N:]-x[:N],np.broadcast_to([20.,20.,13.],(N,3)),atol=1e-13)
    signed=abs(np.einsum('ij,ij->i',local[faces[:,0]],np.cross(local[faces[:,1]],local[faces[:,2]])).sum()/6)
    assert signed==pytest.approx(rigid_volume(C),rel=1e-14)


@pytest.mark.parametrize('mu',[0.,.3])
def test_nonlinear_contact_jacobian_includes_rigid_geometric_curvature(mu):
    c=dict(C,friction=mu);p=RigidPeriodic(X,10.,c);contact=RigidPeriodicIPC(X,T,p,c,radius=0)
    old=np.zeros(p.ndof);q=old.copy();q[p.rigid_dofs]=[.01,-.01,.005,.03,.02,-.04]
    _,g,J,count=contact.evaluate_coupled(q,old,True);assert count>0
    force,torque=contact.rigid_resultant(q);assert contact.rigid_contact_count()>0
    _,(R,dR,_)=p.pose(q);B=np.array([np.array([W[2,1],W[0,2],W[1,0]]) for W in dR@R.T]).T
    np.testing.assert_allclose(g[p.rigid_dofs[:3]],-force,atol=1e-12)
    np.testing.assert_allclose(g[p.rigid_dofs[3:]],-B.T@torque/p.rigid_radius,atol=1e-12)
    d=np.random.default_rng(133).normal(size=p.ndof);h=1e-7
    fd=(contact.evaluate_coupled(q+h*d,old)[1]-contact.evaluate_coupled(q-h*d,old)[1])/(2*h)
    np.testing.assert_allclose(J@d,fd,rtol=2e-5,atol=2e-7)
    if mu==0:
        eplus=contact.evaluate_coupled(q+h*d,old)[0];eminus=contact.evaluate_coupled(q-h*d,old)[0]
        np.testing.assert_allclose((eplus-eminus)/(2*h),g@d,rtol=1e-6)
    else:
        assert friction_report(contact,q,old,mu)['passed']
        assert np.linalg.norm(contact.displacement(q,old).ravel()-contact.R@(q-old))>1e-5


def test_curved_rotation_ccd_prevents_crossing_along_actual_path():
    p=RigidPeriodic(X,10.,C);contact=RigidPeriodicIPC(X,T,p,C,radius=0);q=np.zeros(p.ndof);d=q.copy();d[p.rigid_dofs]=[.15,0,0,0,.6,.2]
    bound=contact.ccd(q,d);assert 0<bound<=1
    for alpha in np.linspace(0,.9*bound,12):assert not ipc.has_intersections(contact.mesh,contact.positions(q+alpha*d))


def test_macro_loading_predictor_keeps_gap_around_a_nonstraining_inclusion():
    Xcap=np.array([[5.1,5.1,6.05],[4.2,4.,7.],[5.8,4.,7.],[5.,6.,7.]])
    t=oriented(Xcap,np.array([[0,1,2,3]]));p=RigidPeriodic(Xcap,10.,C);contact=RigidPeriodicIPC(Xcap,t,p,C,radius=0)
    old=np.zeros(p.ndof);predicted=affine_predictor(p,old,-.2)
    assert contact.loading_path_clear(old,predicted)
    assert not ipc.has_intersections(contact.mesh,contact.positions(predicted))
    local,_=cylinder_surface(C);rigid=contact.positions(predicted)[contact.body_vertices:]
    np.testing.assert_allclose(pdist(rigid),pdist(local),atol=1e-12)


def test_rigid_center_section_area_matches_faceted_cylinder_volume():
    from tools.export_rigid_compaction import rigid_section
    local,faces=cylinder_surface(C);polygon=rigid_section(local+5,faces,2,5,(0,1))
    area=.5*abs(polygon[:,0]@np.roll(polygon[:,1],-1)-polygon[:,1]@np.roll(polygon[:,0],-1))
    assert area==pytest.approx(rigid_volume(C)/C['rigid_height_mm'],abs=1e-9)


def test_curved_swept_box_cache_preserves_actual_contact_force_and_jacobian():
    p=RigidPeriodic(X,10.,C);cached=RigidPeriodicIPC(X,T,p,dict(C,ipc_cache_swept_candidates=True),radius=0);fresh=RigidPeriodicIPC(X,T,p,C,radius=0)
    old=np.zeros(p.ndof);direction=old.copy();direction[p.rigid_dofs]=[.03,.01,0,.08,.15,.05];bound=cached.ccd(old,direction)
    for fraction in [0.,.3,.7]:
        q=old+fraction*bound*direction;a=cached.evaluate_coupled(q,old,True);b=fresh.evaluate_coupled(q,old,True)
        assert cached.last_collision_cache_hit and a[3]==b[3]
        np.testing.assert_allclose(a[1],b[1],rtol=1e-10,atol=1e-12)
        np.testing.assert_allclose(a[2].toarray(),b[2].toarray(),rtol=1e-9,atol=1e-10)
    outside=old.copy();outside[p.rigid_dofs[0]]=-.4
    a=cached.evaluate_coupled(outside,old);b=fresh.evaluate_coupled(outside,old)
    assert not cached.last_collision_cache_hit and a[3]==b[3]
    np.testing.assert_allclose(a[1],b[1],atol=1e-12)


def test_enlarged_tiles_preserve_friction_and_rigid_resultant_across_periodic_seam(tmp_path):
    import json
    from tools.audit_rigid_periodic_contact import audit
    c=dict(C,rigid_center_mm=[.05,5.,5.]);x=X.copy();x[:,0]=15.05-x[:,0]
    t=oriented(x,np.array([[0,1,2,3]]));p=RigidPeriodic(x,10.,c)
    old=np.zeros(p.ndof);q=old.copy();q[p.rigid_dofs]=[.01,-.01,.005,.03,.02,-.04]
    (tmp_path/'config.json').write_text(json.dumps(c))
    for step,unknown in enumerate([old,q]):np.savez(tmp_path/f'state-{step:04d}.npz',X=x,t=t,V0=np.ones(1),parent=np.zeros(1,int),q=unknown,x=p.positions(unknown))
    out=tmp_path/'audit.json';assert audit(tmp_path,1,out)
    report=json.loads(out.read_text());assert report['tiles'][0]['rigid_contacts']>0
    assert report['generalized_force_max_difference_N']<1e-10


def test_rigid_block_smoothing_reduces_full_contact_residual_without_body_or_history_change():
    from packing_fem.quasistatic_coupled import rigid_block_relaxation
    c=dict(C,implicit_regularization=1e-7,implicit_force_tolerance_N=1e-5,implicit_peak_rms_fraction=1.,implicit_rigid_block_iterations=12)
    p=RigidPeriodic(X,10.,c);contact=RigidPeriodicIPC(X,T,p,c,radius=0);old=np.zeros(p.ndof);q=old.copy()
    _,g,K,_=contact.evaluate_coupled(q,old,True);initial=float(abs(g[:p.nmicro]).max())
    def measure(g):return .5*float(abs(g).max())**2
    def derivative(g):
        active=abs(g)>=abs(g).max()*(1-1e-12)
        return np.where(active,g/active.sum(),0.)
    events=[];candidate=rigid_block_relaxation(p,contact,q,old,g,K[:p.nmicro,:p.nmicro],np.zeros(p.ndof),measure,derivative,c,events.append)
    assert candidate is not None and events
    np.testing.assert_array_equal(p.positions(candidate),p.positions(q));np.testing.assert_array_equal(candidate[-6:],q[-6:]);np.testing.assert_array_equal(old,0)
    _,after,_,_=contact.evaluate_coupled(candidate,old)
    assert abs(after[:p.nmicro]).max()<.5*initial
    assert not ipc.has_intersections(contact.mesh,contact.positions(candidate))


def test_periodic_audit_does_not_confuse_zero_candidates_with_active_contact():
    from tools.audit_rigid_periodic_contact import stencil_contribution
    p=RigidPeriodic(X,10.,C);contact=RigidPeriodicIPC(X,T,p,C,radius=0);q=np.zeros(p.ndof)
    far=ipc.VertexVertexNormalCollision(0,contact.body_vertices+contact.rigid_local_count-1)
    assert stencil_contribution(contact,q,q,'vv_collisions',far)['exactly_zero']
    assert stencil_contribution(contact,q,q,'vv_collisions',far)['inert']
    normal=contact.collisions(contact.positions(q));active=[]
    for kind in ['vv_collisions','ev_collisions','ee_collisions','fv_collisions']:
        active.extend(stencil_contribution(contact,q,q,kind,item) for item in getattr(normal,kind))
    assert active and any(not r['inert'] and r['normal_generalized_gradient_max']>1e-6 for r in active)


@pytest.mark.parametrize('weights,mu',[( [1.5,-.5],0.),([.3,.7],.3)])
def test_periodic_allocation_audit_checks_summed_signed_forces_energy_and_work(weights,mu):
    from tools.audit_rigid_periodic_contact import stencil_group_contribution
    points=X.copy();points[0]=[6.02,5.,6.02];c=dict(C,friction=mu)
    p=RigidPeriodic(points,10.,c);contact=RigidPeriodicIPC(points,oriented(points,np.array([[0,1,2,3]])),p,c,radius=0)
    old=np.zeros(p.ndof);q=old.copy();q[p.rigid_dofs[1]]=.001
    first=ipc.VertexVertexNormalCollision(0,contact.body_vertices+16)
    split=[]
    for weight in weights:
        entry=ipc.VertexVertexNormalCollision(0,contact.body_vertices+16);entry.weight=weight;split.append(('vv_collisions',entry))
    a,ga,fa=stencil_group_contribution(contact,q,old,[('vv_collisions',first)])
    b,gb,fb=stencil_group_contribution(contact,q,old,split)
    assert not a['inert'] and a['normal_generalized_gradient_max']>1e-6
    np.testing.assert_allclose(ga,gb,rtol=1e-12,atol=1e-12);np.testing.assert_allclose(fa,fb,rtol=1e-12,atol=1e-12)
    for key in ['weight_sum','normal_energy_Nmm','friction_energy_Nmm','physical_friction_work_Nmm']:assert a[key]==pytest.approx(b[key],rel=1e-12,abs=1e-12)
