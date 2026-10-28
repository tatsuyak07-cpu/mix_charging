import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.macro_loading import (BASIS,first_piola,generalized_load,
    recover_cauchy,uniaxial_load,diagnostic_cauchy)


@pytest.fixture
def F():
    return np.array([[1.08,.03,.12],[.03,1.04,-.08],[.12,-.08,.72]])


def test_spatial_power_and_zero_rigid_rotation_work(F):
    sigma=np.array([[.2,.03,.01],[.03,-.1,.04],[.01,.04,-.3]])
    dF=np.array([[.02,.04,.01],[.04,-.01,.03],[.01,.03,-.05]])
    P=first_piola(F,sigma);V0=17.
    assert V0*np.sum(P*dF)==pytest.approx(V0*np.linalg.det(F)*np.sum(sigma*(dF@np.linalg.inv(F))),abs=1e-14)
    dq=np.array([dF[i,j] for i,j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))])
    assert generalized_load(F,sigma,V0)@dq==pytest.approx(V0*np.sum(P*dF),abs=1e-14)
    omega=np.array([[0,.3,-.2],[-.3,0,.4],[.2,-.4,0]])
    assert np.sum(P*(omega@F))==pytest.approx(0.,abs=1e-14)


def test_rotating_cell_and_loading_axis_rotates_piola(F):
    theta=.73;R=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
    sigma=np.diag([0.,0.,-.1])
    np.testing.assert_allclose(first_piola(R@F,R@sigma@R.T),R@first_piola(F,sigma),atol=1e-15)


def test_stress_reconstruction_and_nominal_shear(F):
    sigma=np.diag([0.,0.,-.1]);load=uniaxial_load(F,.1,8.)
    np.testing.assert_allclose(recover_cauchy(F,load,8.),sigma,atol=1e-15)
    old=np.array([0.,0.,-.8,0.,0.,0.])
    assert np.linalg.norm(recover_cauchy(F,old,8.)[:2,2])>.005


def test_hydrostatic_load_is_exact_volume_gradient(F):
    p=.31;V0=7.;eps=1e-6
    numerical=np.array([-p*V0*(np.linalg.det(F+eps*B)-np.linalg.det(F-eps*B))/(2*eps) for B in BASIS])
    np.testing.assert_allclose(generalized_load(F,-p*np.eye(3),V0),numerical,atol=2e-10)


def test_follower_path_work_converges_quadratically(F):
    from scipy.integrate import quad
    dF=F-np.eye(3)
    exact=quad(lambda s:np.sum(first_piola(np.eye(3)+s*dF,np.diag([0.,0.,-.1]))*dF),0,1,epsabs=1e-13)[0]
    errors=[]
    for n in (8,16,32):
        power=[np.sum(first_piola(np.eye(3)+s*dF,np.diag([0.,0.,-.1*s*s]))*dF) for s in np.linspace(0,1,n+1)]
        # Independent high-order quadrature also includes the pressure ramp.
        reference=quad(lambda s:np.sum(first_piola(np.eye(3)+s*dF,np.diag([0.,0.,-.1*s*s]))*dF),0,1,epsabs=1e-13)[0]
        errors.append(abs(np.trapezoid(power,dx=1/n)-reference))
    assert exact>0
    assert errors[0]/errors[1]>3.9 and errors[1]/errors[2]>3.9


def test_invalid_gauge_does_not_break_archiving():
    assert diagnostic_cauchy(np.diag([-1.,-1.,1.]),np.zeros(6),1.) is None
    with pytest.raises(ValueError):first_piola(np.diag([1.,1.,-1.]),np.eye(3))


def test_cauchy_pressure_solver_and_unloading(tmp_path):
    from packing_fem.sintered import mesh
    from packing_fem.explicit_material import hex_to_tet
    from packing_fem.explicit_solver import solve
    X,h,_=mesh(np.ones((1,1,1),bool),1.);t=hex_to_tet(X,h)
    c=json.loads(Path('configs/explicit-area-cauchy-pressure.json').read_text())
    c.update(length_mm=1.,pressure_mpa=.02,ramp_s=.0002,hold_s=.0002,
             unload_s=.0002,recovery_hold_s=.0002,max_steps=50000,output_every=10,remesh_every=0)
    r=solve(tmp_path/'run',X,t,c)
    assert r['failure'] is None and r['checks']['requested_pressure_reached']
    assert r['checks']['unload_completed'] and r['pressure_measure']=='cauchy'
    np.testing.assert_allclose(r['final_external_work_equivalent_cauchy_stress_MPa'],0.,atol=1e-15)
    history=np.genfromtxt(tmp_path/'run/history.csv',delimiter=',',names=True)
    assert history['command_pressure_MPa'].max()==pytest.approx(.02)
    assert np.all(np.isfinite(history['external_work_Nmm']))
