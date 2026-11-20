"""Necessary consistency checks for the new incremental solver, not validation."""
import numpy as np
from packing_fem.explicit_material import geometry
from packing_fem.implicit_material import evaluate

C=dict(young_mpa=10.,poisson=.33,yield_mpa=.05,hardening_mpa=.2)
X=np.array([[0.,0,0],[1.,0,0],[0,1.,0],[0,0,1.]])
T=np.array([[0,1,2,3]])


def test_plastic_incremental_potential_has_internal_force_gradient():
    V,_,D=geometry(X,T);B=np.eye(3)[None];ep=np.zeros(1)
    x=X@np.array([[.8,.1,0],[0,1.1,.03],[0,0,1.05]]).T
    direction=np.random.default_rng(231).normal(size=x.shape)
    _,g,_,s=evaluate(x,T,B,ep,np.linalg.inv(D),V,C)
    h=1e-6
    fd=(evaluate(x+h*direction,T,B,ep,np.linalg.inv(D),V,C)[0]-evaluate(x-h*direction,T,B,ep,np.linalg.inv(D),V,C)[0])/(2*h)
    assert s['ep'][0]>0
    np.testing.assert_allclose(fd,np.sum(g*direction),rtol=1e-7,atol=1e-9)
    np.testing.assert_allclose(g.sum(axis=0),0,atol=1e-12)


def test_elastic_tangent_matches_force_derivative_and_translation_nullspace():
    V,_,D=geometry(X,T);B=np.eye(3)[None];ep=np.zeros(1)
    x=X@np.diag([.999,1.0002,1.0001]);d=np.random.default_rng(4).normal(size=x.shape)
    _,g,K,_=evaluate(x,T,B,ep,np.linalg.inv(D),V,C,True)
    h=1e-7
    gp=evaluate(x+h*d,T,B,ep,np.linalg.inv(D),V,C)[1]
    gm=evaluate(x-h*d,T,B,ep,np.linalg.inv(D),V,C)[1]
    # PSD projection can remove compressive geometric modes, O(stress).
    np.testing.assert_allclose(K@d.ravel(),((gp-gm)/(2*h)).ravel(),rtol=.003,atol=.005)
    np.testing.assert_allclose(K@np.tile([1.,2.,3.],4),0,atol=1e-12)


def test_unprojected_plastic_consistent_tangent_matches_force_derivative():
    V,_,D=geometry(X,T);B=np.eye(3)[None];ep=np.zeros(1)
    c=dict(C,implicit_project_material_tangent=False)
    x=X@np.array([[.8,.1,0],[0,1.1,.03],[0,0,1.05]]).T
    direction=np.random.default_rng(17).normal(size=x.shape)
    _,_,K,state=evaluate(x,T,B,ep,np.linalg.inv(D),V,c,True)
    assert state['ep'][0]>0
    h=1e-7
    gp=evaluate(x+h*direction,T,B,ep,np.linalg.inv(D),V,c)[1]
    gm=evaluate(x-h*direction,T,B,ep,np.linalg.inv(D),V,c)[1]
    np.testing.assert_allclose(K@direction.ravel(),((gp-gm)/(2*h)).ravel(),rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(K@np.tile([1.,2.,3.],4),0,atol=1e-12)


def test_history_subcells_preserve_potential_without_averaging_plastic_strain():
    V,_,D=geometry(X,T)
    B=np.array([np.diag([1.02,.99,1/1.0098]),np.eye(3)])
    ep=np.array([.7,.1]);weights=np.array([.3,.7])*V[0];parent=np.array([0,0])
    x=X@np.diag([.98,1.01,1.]);d=np.random.default_rng(17).normal(size=x.shape)
    value,g,K,state=evaluate(x,T,B,ep,np.linalg.inv(D),weights,C,True,parent)
    h=1e-6
    plus=evaluate(x+h*d,T,B,ep,np.linalg.inv(D),weights,C,parent=parent)[0]
    minus=evaluate(x-h*d,T,B,ep,np.linalg.inv(D),weights,C,parent=parent)[0]
    np.testing.assert_allclose((plus-minus)/(2*h),np.sum(g*d),rtol=1e-7,atol=1e-9)
    assert state['ep'].shape==(2,)
    assert K.shape==(12,12)


def test_mixed_rotated_subcell_history_tangent_and_frame_covariance():
    rng=np.random.default_rng(428)
    # Disconnected, differently distorted tets expose cross-element scatter
    # errors; shuffled subcells exercise the history-parent accumulation.
    n=7
    transforms=np.eye(3)[None]+.12*rng.normal(size=(n,3,3))
    old=np.concatenate([X@A.T+[3*i,0,0] for i,A in enumerate(transforms)])
    t=np.arange(4*n).reshape(n,4)
    V,_,D=geometry(old,t)
    parent=np.repeat(np.arange(n),[1,3,2,4,1,2,3]);rng.shuffle(parent)
    weights=V[parent]/np.bincount(parent)[parent]
    maps=np.eye(3)[None]+.08*rng.normal(size=(len(parent),3,3))
    b=maps@maps.swapaxes(1,2);ep=rng.uniform(0,.7,len(parent))
    old_b=b.copy();old_ep=ep.copy()
    current=old.copy()
    for i,A in enumerate(np.eye(3)[None]+.05*rng.normal(size=(n,3,3))):
        ids=t[i];current[ids]=(old[ids]-old[ids[0]])@A.T+old[ids[0]]
    c=dict(C,implicit_project_material_tangent=False)
    inverse=np.linalg.inv(D);direction=rng.normal(size=current.shape)
    value,g,K,state=evaluate(current,t,b,ep,inverse,weights,c,True,parent)
    assert np.any(state['ep']>ep)
    h=1e-7
    gp=evaluate(current+h*direction,t,b,ep,inverse,weights,c,parent=parent)[1]
    gm=evaluate(current-h*direction,t,b,ep,inverse,weights,c,parent=parent)[1]
    np.testing.assert_allclose(K@direction.ravel(),((gp-gm)/(2*h)).ravel(),rtol=2e-6,atol=2e-7)
    rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    if np.linalg.det(rotation)<0:rotation[:,0]*=-1
    vr,gr,Kr,sr=evaluate(current@rotation.T,t,rotation@b@rotation.T,ep,inverse@rotation.T,weights,c,True,parent)
    np.testing.assert_allclose(vr,value,rtol=2e-13,atol=1e-14)
    np.testing.assert_allclose(gr,g@rotation.T,rtol=2e-12,atol=2e-13)
    expected=(K@direction.ravel()).reshape(-1,3)@rotation.T
    np.testing.assert_allclose((Kr@(direction@rotation.T).ravel()).reshape(-1,3),expected,rtol=2e-11,atol=2e-12)
    np.testing.assert_array_equal(b,old_b)
    np.testing.assert_array_equal(ep,old_ep)


def test_repeated_principal_stretches_have_finite_tangent():
    V,_,D=geometry(X,T);b=np.eye(3)[None];ep=np.zeros(1)
    c=dict(C,implicit_project_material_tangent=False)
    for scale in [1.,.97]:
        _,_,K,_=evaluate(scale*X,T,b,ep,np.linalg.inv(D),V,c,True)
        direction=np.random.default_rng(611).normal(size=X.shape);h=1e-7
        gp=evaluate(scale*X+h*direction,T,b,ep,np.linalg.inv(D),V,c)[1]
        gm=evaluate(scale*X-h*direction,T,b,ep,np.linalg.inv(D),V,c)[1]
        assert np.all(np.isfinite(K.data))
        np.testing.assert_allclose(K@direction.ravel(),((gp-gm)/(2*h)).ravel(),rtol=2e-7,atol=2e-7)


def test_history_partition_across_assembly_chunks_preserves_full_operator():
    V,_,D=geometry(X,T);inverse=np.linalg.inv(D)
    b=np.array([[[1.03,.01,0],[.01,.98,0],[0,0,1.01]]]);ep=np.array([.17])
    current=X@np.array([[.91,.04,.02],[0,1.02,0],[.03,0,.98]]).T
    c=dict(C,implicit_project_material_tangent=False)
    value,g,K,state=evaluate(current,T,b,ep,inverse,V,c,True)
    count=8195;rng=np.random.default_rng(941);weights=rng.uniform(.5,2,count);weights*=V[0]/weights.sum()
    v2,g2,K2,s2=evaluate(current,T,np.repeat(b,count,axis=0),np.repeat(ep,count),inverse,weights,c,True,np.zeros(count,dtype=int))
    np.testing.assert_allclose(v2,value,rtol=1e-12)
    np.testing.assert_allclose(g2,g,rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(K2.toarray(),K.toarray(),rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(s2['ep'],state['ep'][0],rtol=1e-13)
