from copy import deepcopy
import numpy as np
import pytest
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.explicit_periodic import Periodic
from packing_fem.triangle_quadrature import RULES
from packing_fem.barrier_contact import barrier
from tools.verify_plane_contact import fixture


def setup(slip=0.,refined=True):
    X,t,x,H,upper=fixture(1,.015,0.);p=Periodic(X,1.)
    old=EmbeddedContact(X,t,p,20.,.3,.1,thickness=.02)
    rules={}
    if refined:
        for i,face in enumerate(old.faces):
            if not np.all(np.isclose(X[face,2],.6)):continue
            tri=X[face];A=np.linalg.norm(np.cross(tri[1]-tri[0],tri[2]-tri[0]))/2
            B,w=RULES[8];rules[str(i)]=dict(bary=B.tolist(),weights=(A*w).tolist())
    contact=EmbeddedContact(X,t,p,20.,.3,.1,thickness=.02,quadrature_rules=rules)
    return X,t,x,H,upper,contact


@pytest.mark.parametrize('refined',[False,True])
def test_embedded_plane_area_force_and_virtual_work(refined):
    X,t,x,H,upper,c=setup(refined=refined)
    f,extra,hist,r=c.evaluate(x,x*0,H,H*0,0.)
    E,F,_=barrier(.015,20.,.02)
    assert r['normal_energy']==pytest.approx(E,rel=1e-8)
    assert f[upper,2].sum()==pytest.approx(F,rel=1e-8)
    np.testing.assert_allclose(f.sum(0),0,atol=1e-12)
    assert len(c.edge_stiffness_diagonal())==len(X)
    assert c.integration_area0.sum()==pytest.approx(c.node_area0.sum())


@pytest.mark.parametrize('slip',[.0001,.01])
def test_embedded_friction_preserves_committed_and_applied_states(slip):
    from packing_fem.area_contact import SurfaceResolutionError
    rules={}
    for attempt in range(12):
        X,t,_,_,upper=fixture(1,.015,0.)
        c=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_rules=rules)
        try:
            for gap in np.linspace(.025,.015,9):
                _,_,x,H,_=fixture(1,float(gap),0.)
                result=c.evaluate(x,x*0,H,H*0,0.);c.history=result[2];c.accept_geometry(x,H)
            original=x.copy();initial_cell=H.copy()
            for displacement in np.linspace(0,slip,9)[1:]:
                x=original.copy();x[upper,0]+=displacement;H=initial_cell.copy();H[0,2]+=displacement
                result=c.evaluate(x,x*0,H,H*0,.001/8);c.history=result[2];c.accept_geometry(x,H)
            break
        except SurfaceResolutionError as error:
            feature=getattr(error,'feature',None)
            assert feature and feature.get('requires_quadrature_refinement'),str(error)
            rules.update(feature['rules'])
    else:raise AssertionError('Prescribed friction replay budget exhausted')
    saved=deepcopy(c.history);replay=c.replay_force_state(x,x*0,H)
    np.testing.assert_allclose(result[0],replay[0],atol=1e-12)
    assert result[-1]['max_coulomb_ratio']<=.3+1e-12
    normal=barrier(.015,20.,.02)[1];spring=min(slip,.3*normal/10.)
    assert result[-1]['tangent_energy']==pytest.approx(5*spring**2,rel=2e-3)
    for key in saved:np.testing.assert_array_equal(c.history[key]['plastic'],saved[key]['plastic'])


def test_invalid_partial_face_quadrature_is_rejected():
    X,t,x,H,upper,c=setup(refined=False)
    with pytest.raises(ValueError,match='quadrature'):
        EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_rules={'0':dict(bary=[[1/3]*3],weights=[1.])})


def test_nonmatching_periodic_mesh_refines_only_by_inactive_replay():
    from packing_fem.area_contact import SurfaceResolutionError
    rules={};requests=0
    for attempt in range(12):
        X,t,_,_,upper=fixture(1,.015,.25)
        c=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02,quadrature_rules=rules)
        try:
            for gap in np.linspace(.025,.015,9):
                _,_,x,H,_=fixture(1,float(gap),.25)
                value=c.evaluate(x,x*0,H,H*0,0.)
                if gap==.025:assert not value[2]
                c.history=value[2];c.accept_geometry(x,H)
            break
        except SurfaceResolutionError as error:
            feature=getattr(error,'feature',None)
            assert feature and feature.get('requires_quadrature_refinement'),str(error)
            rules.update(feature['rules']);requests+=1
    else:raise AssertionError('Replay budget exhausted')
    assert requests>0
    E,F,_=barrier(.015,20.,.02)
    assert value[-1]['normal_energy']==pytest.approx(E,rel=1e-3)
    assert value[0][upper,2].sum()==pytest.approx(F,rel=1e-3)


def test_active_contact_history_cannot_be_reset_by_remeshing():
    from packing_fem.embedded_contact import reinitialize_inactive
    X,t,x,H,upper,old=setup(refined=False)
    response=old.evaluate(x,x*0,H,H*0,0.);old.history=response[2];old.accept_geometry(x,H)
    saved=deepcopy(old.history)
    new=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    with pytest.raises(ValueError,match='active-history reset prohibited'):
        reinitialize_inactive(old,new,x,x*0,H,False)
    assert old.history.keys()==saved.keys()
    for key in saved:np.testing.assert_array_equal(saved[key]['plastic'],old.history[key]['plastic'])


def test_inactive_checkpoint_initialization_is_not_an_artificial_motion_step():
    from packing_fem.embedded_contact import initialize_inactive_geometry
    X,t,_,_,upper=fixture(1,.025,.25)
    c=EmbeddedContact(X,t,Periodic(X,1.),20.,.3,.1,thickness=.02)
    c.evaluate(X,X*0,np.eye(3),np.zeros((3,3)),0.);c.accept_geometry(X,np.eye(3))
    _,_,x,H,_=fixture(1,.025,.25)
    with pytest.raises(ValueError,match='contact-free'):
        initialize_inactive_geometry(c,x,H,False)
    initialize_inactive_geometry(c,x,H,True)
    response=c.evaluate(x,x*0,H,H*0,0.)
    assert not response[2]
    assert response[-1]['normal_energy']==0.
    c.accept_geometry(x,H)
    _,_,y,G,_=fixture(1,.0249,.25)
    assert c.evaluate(y,y*0,G,G*0,.00001)[-1]['normal_energy']==0.


def test_lazy_candidates_use_the_verlet_pose_even_after_cache_restore():
    X,t,x,H,upper,c=setup(refined=False)
    guard=c.inner.guard;guard.neighbor_builds=1;guard.trial_shifts=[(0,0,0)]
    guard.neighbor_geometry=(x.copy(),H.copy())
    c._candidates([0]);assert set(c._face_candidates)=={0}
    c._x=x+.001;c._H=H
    c._candidates([1]);assert set(c._face_candidates)=={0,1}
    np.testing.assert_array_equal(c._face_search[0],x[c.faces])
    # Restoring an equal build counter with a different saved pose must
    # invalidate the derived tree, not combine two geometries in one cache.
    moved=x.copy();moved[upper,0]+=.002
    guard.neighbor_geometry=(moved,H.copy())
    c._candidates([2]);assert set(c._face_candidates)=={2}
    np.testing.assert_array_equal(c._face_search[0],moved[c.faces])
