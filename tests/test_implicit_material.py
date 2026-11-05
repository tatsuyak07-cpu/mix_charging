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
