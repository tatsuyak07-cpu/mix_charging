"""Small, reproducible contact integration checks; optional IPC dependency."""
import numpy as np
import pytest
ipc=pytest.importorskip('ipctk')
from packing_fem.explicit_material import oriented
from packing_fem.explicit_periodic import Periodic
from packing_fem.ipc_periodic import PeriodicIPC

C=dict(friction=.3,ipc_dhat_mm=.02,ipc_stiffness_mpa=10.,ipc_slip_regularization_mm=1e-4)
X=np.array([[1.,1,1],[2,1,1],[1,2,1],[1,1,2],[2.03,.2,.2],[2.03,1.8,.2],[2.03,.2,2.2],[3.03,1,1]])
T=oriented(X,np.array([[0,1,2,3],[4,5,6,7]]))


def moved(p,distance):
    q=np.zeros(p.ndof)
    for node in range(4,8):q[3*(p.classes[node]-1)]=distance
    return q


def test_contact_potential_gradient_and_periodic_image_ownership():
    p=Periodic(X,10.);q=moved(p,-.02);values=[]
    for radius in [1,2]:
        contact=PeriodicIPC(X,T,p,C,radius);contact.set_lag(q,q)
        energy,g,_,count=contact.evaluate(q)
        assert count>0
        values.append((energy,g,count))
    np.testing.assert_allclose(values[0][0],values[1][0],rtol=1e-12)
    np.testing.assert_allclose(values[0][1],values[1][1],rtol=1e-11,atol=1e-12)
    assert values[0][2]==values[1][2]
    d=np.random.default_rng(14).normal(size=p.ndof);d[-6:]=0;h=1e-7
    finite=(contact.evaluate(q+h*d)[0]-contact.evaluate(q-h*d)[0])/(2*h)
    np.testing.assert_allclose(finite,values[1][1]@d,rtol=1e-6,atol=1e-7)


def test_continuous_collision_detection_blocks_crossing():
    p=Periodic(X,10.);contact=PeriodicIPC(X,T,p,C)
    q=np.zeros(p.ndof)
    assert contact.ccd(q,moved(p,-.02))==pytest.approx(1.)
    alpha=contact.ccd(q,moved(p,-.05))
    assert 0<alpha<.7


def test_unprojected_contact_tangent_matches_force_derivative():
    p=Periodic(X,10.);q=moved(p,-.02)
    contact=PeriodicIPC(X,T,p,dict(C,friction=0.,ipc_project_tangent=False));contact.set_lag(q,q)
    _,_,K,count=contact.evaluate(q,True)
    assert count>0
    d=np.random.default_rng(18).normal(size=p.ndof);d[-6:]=0;h=1e-7
    gp=contact.evaluate(q+h*d)[1];gm=contact.evaluate(q-h*d)[1]
    np.testing.assert_allclose(K@d,(gp-gm)/(2*h),rtol=1e-5,atol=1e-6)


def test_large_proposal_uses_bounded_ccd_sweep_and_remains_collision_free():
    p=Periodic(X,10.);contact=PeriodicIPC(X,T,p,dict(C,ipc_ccd_maximum_displacement_mm=.1))
    q=np.zeros(p.ndof);dq=moved(p,-1e6);alpha=contact.ccd(q,dq)
    assert 0<alpha<=1e-7
    assert contact.last_ccd_horizon==pytest.approx(1e-7)
    assert not ipc.has_intersections(contact.mesh,contact.positions(q+.9*alpha*dq))


def test_coupled_friction_force_jacobian_includes_normal_force_variation():
    p=Periodic(X,10.);q=moved(p,-.02);old=moved(p,-.01)
    for node in range(4,8):old[3*(p.classes[node]-1)+1]=-.005
    contact=PeriodicIPC(X,T,p,C)
    value,g,J,count=contact.evaluate_coupled(q,old,True)
    contact.set_lag(q,old);ve,vg,_,cn=contact.evaluate(q)
    assert count==cn and count>0
    np.testing.assert_allclose(value,ve,rtol=1e-12)
    np.testing.assert_allclose(g,vg,atol=1e-12)
    d=np.random.default_rng(181).normal(size=p.ndof);d[-6:]=0;h=1e-7
    fd=(contact.evaluate_coupled(q+h*d,old)[1]-contact.evaluate_coupled(q-h*d,old)[1])/(2*h)
    np.testing.assert_allclose(J@d,fd,rtol=1e-5,atol=1e-6)
    assert np.max(abs((J-J.T).data))>1e-3


def test_swept_candidates_preserve_contact_and_reject_queries_outside_sweep():
    p=Periodic(X,10.);q=moved(p,-.005);dq=moved(p,-.02)
    cached=PeriodicIPC(X,T,p,dict(C,ipc_cache_swept_candidates=True))
    fresh=PeriodicIPC(X,T,p,C)
    np.testing.assert_allclose(cached.ccd(q,dq),fresh.ccd(q,dq),rtol=1e-12)
    for fraction in [0.,.25,.75,1.]:
        query=q+fraction*dq
        a=cached.evaluate_coupled(query,q);b=fresh.evaluate_coupled(query,q)
        assert cached.last_collision_cache_hit
        assert a[3]==b[3]
        np.testing.assert_allclose(a[0],b[0],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(a[1],b[1],rtol=1e-11,atol=1e-11)
    cached.evaluate_coupled(q+1.01*dq,q)
    assert not cached.last_collision_cache_hit
    outside=q+.5*dq;outside[1]+=.0001
    cached.evaluate_coupled(outside,q)
    assert not cached.last_collision_cache_hit


def test_swept_candidates_ccd_blocks_a_crossing_trajectory():
    p=Periodic(X,10.);q=np.zeros(p.ndof);dq=moved(p,-.05)
    cached=PeriodicIPC(X,T,p,dict(C,ipc_cache_swept_candidates=True))
    fresh=PeriodicIPC(X,T,p,C)
    a=cached.ccd(q,dq);b=fresh.ccd(q,dq)
    assert 0<a<1
    np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-12)
