import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.contact_tracking import adjacent_patch,material_rotation,closest_segments,swept_edge_event
from packing_fem.tracked_contact import TrackedContact
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import oriented,hex_to_tet
from packing_fem import sintered


def test_patch_matching_lifted_periodic_and_distant_rejection():
    X=np.array([[0,0,0],[0,1,0],[.5,.5,0],[1,0,0],[1,1,0],[.5,.5,1],[.1,.1,.01],[.1,.2,.01],[.2,.1,.01]],float)
    f=np.array([[0,1,2],[3,5,4],[6,7,8]])
    assert adjacent_patch(X,f,1.,0,[0,0,0],1,[-1,0,0])
    assert not adjacent_patch(X,f,1.,0,[0,0,0],1,[0,0,0])
    assert not adjacent_patch(X,f,1.,0,[0,0,0],2,[0,0,0])


def test_material_transport_includes_spin_and_stretch():
    tri=np.array([[0,0,0],[1,0,0],[0,1,0]],float);n=np.array([0.,0.,1.]);a=.7
    R=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
    stretch=np.diag([1.1,.9,1.]);new=tri@stretch@R.T+[3.,1.,2.]
    np.testing.assert_allclose(material_rotation(tri,new,n,R@n),R,atol=1e-13)


def test_segments_and_swept_crossing_without_endpoint_contact():
    old=np.array([[-1,0,0],[1,0,0],[0,-1,1],[0,1,1]],float);new=old.copy();new[2:,2]=-1
    assert closest_segments(*old)['distance']==pytest.approx(1.)
    assert closest_segments(*new)['distance']==pytest.approx(1.)
    event=swept_edge_event(old,new);assert event['time_fraction']==pytest.approx(.5)
    far=new.copy();far[2:,0]=3
    assert swept_edge_event(old+np.array([0,0,3]),old+np.array([0,0,3])) is None
    parallel=np.array([[0,0,0],[1,0,0],[0,1,0],[1,1,0]],float)
    assert closest_segments(*parallel)['distance']==pytest.approx(1.)
    moved=parallel.copy();moved[2:,1]=0
    assert swept_edge_event(parallel,moved) is not None


def fixture():
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.);X[:,2]-=.6
    t=hex_to_tet(X,h);extra=np.array([[.4,.43,.6],[.3,.33,.8],[.5,.33,.8],[.4,.53,.8]])
    ids=np.arange(len(X),len(X)+4);X=np.r_[X,extra];t=oriented(X,np.r_[t,ids[None]])
    p=Periodic(X,3.);contact=TrackedContact(X,t,p,100.,.3,.1)
    x=X.copy();x[-4:,2]-=.21
    return X,x,t,p,contact


def test_tracked_contact_energy_force_and_automatic_seam_history():
    X,x,t,p,c=fixture();zero=np.zeros_like(x);H=np.eye(3)*3
    a=c.evaluate(x,zero,H,H*0,0.);assert a[-1]['contacts']>0
    c.history=a[2]
    transferred=0;energy=[]
    for u in np.linspace(.4,.47,71):
        xx=x.copy();xx[-4:,0]+=u-.4
        result=c.evaluate(xx,zero,H,H*0,.001);c.history=result[2]
        transferred+=result[-1]['transferred_pairs'];energy.append(result[-1]['tangent_energy'])
        np.testing.assert_allclose(result[0].sum(axis=0),0,atol=1e-14)
        assert result[-1]['max_coulomb_ratio']<=.3+1e-12
    assert transferred>0 and max(energy)>0
    assert energy[-1]>0 # no reset at the diagonal
    # Differential normal potential, with friction disabled and no history.
    c.mu=0.;c.history={};base=c.evaluate(x,zero,H,H*0,0.)
    d=1e-7;xp=x.copy();xm=x.copy();xp[-4,2]+=d;xm[-4,2]-=d
    grad=(c.evaluate(xp,zero,H,H*0,0.)[-1]['normal_energy']-c.evaluate(xm,zero,H,H*0,0.)[-1]['normal_energy'])/(2*d)
    assert -grad==pytest.approx(base[0][-4,2],rel=1e-6)


def test_opt_in_solver_connection_and_energy_column(tmp_path):
    from packing_fem.explicit_solver import solve
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(contact_backend='tracked_guarded',length_mm=1.,yield_mpa=1e6,compression_fraction=.001,contact_penalty_mpa_mm=1.,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.)
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.)
    r=solve(tmp_path/'run',X,hex_to_tet(X,h),c)
    assert all(r['checks'].values()),r
    assert r['contact_backend']=='tracked_guarded'
    history=np.genfromtxt(tmp_path/'run/history.csv',names=True,delimiter=',')
    assert 'contact_algorithmic_Nmm' in history.dtype.names


def test_coplanar_swept_boxes_can_be_certified_separate():
    old=np.array([[1.,2.,2.45],[2.,3.,2.45],[2.,1.,2.45],[3.,2.,2.45]])
    new=old.copy();new[[1,3],2]+=.0001
    # Overlapping boxes alone must not block distant parallel edges.
    assert swept_edge_event(old,new) is None


@pytest.mark.parametrize('periodic',[False,True])
def test_tracked_contact_periodic_action_reaction_and_image_work(periodic):
    from test_explicit_backend import contact_fixture
    X,x,t,p=contact_fixture(periodic);c=TrackedContact(X,t,p,100.,0.,.1)
    zero=np.zeros_like(x);H=np.eye(3);force,extra,_,report=c.evaluate(x,zero,H,H*0,0.)
    assert report['contacts']>0 and report['penetration_mm']==pytest.approx(.02)
    np.testing.assert_allclose(force.sum(axis=0),0,atol=1e-13)
    if periodic:
        d=1e-7;plus=H.copy();minus=H.copy();plus[0,0]+=d;minus[0,0]-=d
        grad=(c.evaluate(x,zero,plus,H*0,0.)[-1]['normal_energy']-c.evaluate(x,zero,minus,H*0,0.)[-1]['normal_energy'])/(2*d)
        assert -grad==pytest.approx(extra[0],rel=1e-6)


def test_tracked_same_body_remote_image_is_not_a_local_neighbour():
    X=np.array([[.2,.4,.4],[.6,.3,.3],[.6,.6,.3],[.6,.3,.6]])
    t=oriented(X,np.arange(4)[None]);c=TrackedContact(X,t,Periodic(X,1.),100.,.3,.1)
    x=X.copy();x[1:,0]=1.22
    r=c.evaluate(x,np.zeros_like(x),np.eye(3),np.zeros((3,3)),0.)[-1]
    assert r['contacts']>0 and r['penetration_mm']==pytest.approx(.02)


def test_separation_releases_history_without_calling_it_friction():
    X,x,t,p,c=fixture();H=np.eye(3)*3;zero=np.zeros_like(x)
    c.history=c.evaluate(x,zero,H,H*0,0.)[2]
    xx=x.copy();xx[-4:,0]+=.001
    r=c.evaluate(xx,zero,H,H*0,.001);c.history=r[2]
    energy=r[-1]['tangent_energy'];assert energy>0
    separated=c.evaluate(X,zero,H,H*0,.001)
    assert separated[-1]['released_history_energy']==pytest.approx(energy)
    assert separated[-1]['friction_increment']==0 and not separated[2]


def test_nonadjacent_history_transition_is_refused_without_mutation():
    X,x,t,p,c=fixture();H=np.eye(3)*3;zero=np.zeros_like(x)
    r=c.evaluate(x,zero,H,H*0,0.);key,state=next(iter(r[2].items()))
    far=next(i for i in range(len(c.faces)) if not adjacent_patch(X,c.faces,3.,i,[0,0,0],key[1],key[2:]))
    bogus=(key[0],far,0,0,0);c.history={bogus:state};saved=state['energy']
    with pytest.raises(ValueError,match='Ambiguous contact patch'):c.evaluate(x,zero,H,H*0,.001)
    assert c.history[bogus]['energy']==saved


def test_verlet_friction_work_does_not_use_backward_euler_artificial_loss():
    from packing_fem.contact_probe import friction_return
    n=np.array([0.,0.,1.]);s=np.zeros(3)
    r=friction_return(s,n,n,[.001,0,0],1.,100.,.3)
    assert r['algorithmic_loss']>0 # BE identity, not Verlet mechanical work.
    assert r['trapezoidal_correction']==pytest.approx(0.,abs=1e-15)
    for du,fn in (([.006,0,0],1.),([-.002,0,0],.5)):
        r=friction_return(r['spring'],n,n,du,fn,100.,.3)
        assert r['trapezoidal_work']==pytest.approx(r['elastic_change']+r['friction_loss']+r['trapezoidal_correction'],abs=1e-15)
