import numpy as np
import pytest
from packing_fem.edge_contact import start_pair,response,potential


def fixture():
    old=np.array([[-1.,0,0],[1.,0,0],[0,-1,.01],[0,1,.01]])
    new=old.copy();new[2:,2]=-.01
    return old,new,start_pair(old,new,.5)


def test_interior_edge_normal_potential_gradient_balance_and_moment():
    old,x,state=fixture();r=response(x,state,10.,5.,0.)
    assert r['normal_force_N']==pytest.approx(.1)
    assert r['normal_energy']==pytest.approx(.0005)
    np.testing.assert_allclose(r['force'].sum(axis=0),0,atol=1e-14)
    np.testing.assert_allclose(np.cross(x,r['force']).sum(axis=0),0,atol=1e-14)
    d=1e-7
    for i in range(4):
        for axis in range(3):
            shift=np.zeros_like(x);shift[i,axis]=d
            grad=(potential(x+shift,state,10.,5.,r['state']['plastic'])-potential(x-shift,state,10.,5.,r['state']['plastic']))/(2*d)
            assert -grad==pytest.approx(r['force'][i,axis],abs=1e-9)


def test_edge_friction_frame_gradient_objectivity_and_periodic_work():
    old,x,state=fixture();r=response(x,state,10.,5.,.3)
    x[2:,0]+=.003;x[2:,1]+=.004;r=response(x,r['state'],10.,5.,.3)
    assert r['tangent_energy']>0 and r['coulomb_ratio']<=.3+1e-14
    np.testing.assert_allclose(r['force'].sum(axis=0),0,atol=1e-14)
    # Includes the rotating-frame terms missing from simple traction splitting.
    np.testing.assert_allclose(np.cross(x,r['force']).sum(axis=0),0,atol=1e-14)
    a=.8;R=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
    rr=response(x@R.T+[4,2,1],state,10.,5.,.3)
    np.testing.assert_allclose(rr['force'],r['force']@R.T,atol=1e-14)
    assert rr['potential']==pytest.approx(r['potential'])
    d=1e-7
    for i in range(4):
        for axis in range(3):
            delta=np.zeros_like(x);delta[i,axis]=d
            grad=(potential(x+delta,state,10.,5.,r['state']['plastic'])-potential(x-delta,state,10.,5.,r['state']['plastic']))/(2*d)
            assert -grad==pytest.approx(r['force'][i,axis],abs=1e-9)
    for axis in range(3):
        plus=x.copy();minus=x.copy();plus[2:,axis]+=d;minus[2:,axis]-=d
        grad=(potential(plus,state,10.,5.,r['state']['plastic'])-potential(minus,state,10.,5.,r['state']['plastic']))/(2*d)
        assert -grad==pytest.approx(r['force'][2:,axis].sum(),abs=1e-9)


def test_friction_return_dissipation_and_separation():
    _,x,state=fixture();total=0.
    for shift in np.linspace(0,.05,51):
        points=x.copy();points[2:,0]+=shift
        r=response(points,state,10.,5.,.3);state=r['state'];total+=r['friction_increment']
        assert r['friction_increment']>=-1e-14 and r['coulomb_ratio']<=.3+1e-14
    assert total>0
    points=x.copy();points[2:,2]=.001
    assert response(points,state,10.,5.,.3) is None


def test_unsupported_endpoint_and_parallel_transitions_refused():
    old,x,state=fixture();x[2:,0]+=1.1
    with pytest.raises(ValueError,match='endpoint'):response(x,state,10.,5.,.3)
    x[3]=x[2]+[1,0,0]
    with pytest.raises(ValueError,match='Parallel'):response(x,state,10.,5.,.3)
