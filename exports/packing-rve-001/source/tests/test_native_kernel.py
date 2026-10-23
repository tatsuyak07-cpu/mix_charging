import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.native_kernel import NativeKernel
from packing_fem import explicit_material as material
from packing_fem.adaptive_tet import quality
from packing_fem.explicit_timestep import triangle_frequency_bound
from packing_fem.explicit_solver import lumped_mass
from packing_fem.sintered import mesh


@pytest.fixture(scope='module')
def kernel():
    import os
    path=os.environ.get('PACKING_FEM_NATIVE_LIBRARY')
    if not path:pytest.skip('Optional native backend not built; set PACKING_FEM_NATIVE_LIBRARY')
    return NativeKernel(path)


def case():
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=material.hex_to_tet(X,h);rng=np.random.default_rng(93);x=X@np.array([[.8,.05,0],[0,.95,.08],[.01,0,.9]]).T+.003*rng.normal(size=X.shape)
    F=material.geometry(x,t)[2]@np.linalg.inv(material.geometry(X,t)[2]);b=np.repeat(np.eye(3)[None],len(t),axis=0);ep=np.linspace(0,.1,len(t));c=dict(young_mpa=10.,poisson=.33,yield_mpa=.05,hardening_mpa=.2,density_tonne_mm3=1e-9)
    return X,t,x,F,b,ep,c


def test_native_geometry_and_quality_match_numpy(kernel):
    X,t,x,*_=case()
    for actual,expected in zip(kernel.geometry(x,t),material.geometry(x,t)):np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-14)
    np.testing.assert_allclose(kernel.quality(x,t),quality(x,t),rtol=1e-13,atol=1e-14)


@pytest.mark.parametrize('yield_stress',[.05,1e6])
def test_native_plastic_increment_and_force_energy_match_numpy(kernel,yield_stress):
    X,t,x,F,b,ep,c=case();c['yield_mpa']=yield_stress
    bn,pn=kernel.advance(b,ep,F,c);br,pr=material.advance(b,ep,F,c)
    np.testing.assert_allclose(bn,br,rtol=2e-13,atol=2e-14);np.testing.assert_allclose(pn,pr,rtol=1e-12,atol=1e-14)
    for actual,expected in zip(kernel.internal_force(x,t,bn,pn,c),material.internal_force(x,t,br,pr,c)):np.testing.assert_allclose(actual,expected,rtol=2e-10,atol=2e-12)
    V=material.geometry(X,t)[0];mass=lumped_mass(t,V,1e-9,len(X));stress=material.response(br,pr,c)[0]
    assert kernel.bound(x,t,mass,c,stress,br)==pytest.approx(triangle_frequency_bound(x,t,mass,c,stress,br),rel=1e-12)


def test_native_repeated_reversed_load_and_objective_material_update(kernel):
    _,t,_,_,b,ep,c=case();br=b.copy();pr=ep.copy();theta=.31;R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    for k in range(50):
        A=np.diag([1.001,1.0003,.9987]) if k<25 else np.diag([.999, .9997,1.0013]);F=np.repeat((R@A@R.T)[None],len(t),axis=0)
        b,ep=kernel.advance(b,ep,F,c);br,pr=material.advance(br,pr,F,c)
    np.testing.assert_allclose(b,br,rtol=1e-11,atol=1e-12);np.testing.assert_allclose(ep,pr,rtol=1e-11,atol=1e-12)


def test_native_rejects_invalid_states(kernel):
    X,t,x,F,b,ep,c=case();b[0,0,0]=-1
    with pytest.raises(ValueError):kernel.advance(b,ep,F,c)
    with pytest.raises(ValueError):kernel.geometry(X,t[:,[0,2,1,3]])
    with pytest.raises(ValueError):kernel.geometry(X,t+len(X))


def test_native_pressure_integrator_matches_numpy_and_analytic_limit(kernel,tmp_path):
    from packing_fem.explicit_solver import solve
    from scipy.optimize import brentq
    X,h,_=mesh(np.ones((2,2,2),bool),.5);t=material.hex_to_tet(X,h);c=json.loads(Path('configs/explicit-prototype.json').read_text());c.update(length_mm=1.,contact_backend='area_guarded',loading_mode='cauchy_pressure',pressure_mpa=.01,yield_mpa=1e6,ramp_s=.0006,hold_s=.0003,damping_per_s=20000.,timestep_method='triangle_mass_bound',active_contact_timestep=True,remesh_every=0)
    a=solve(tmp_path/'numpy',X,t,c);b=solve(tmp_path/'native',X,t,dict(c,element_backend='native',native_library=str(kernel.path)))
    assert all(a['checks'].values()) and all(b['checks'].values())
    np.testing.assert_allclose(b['final_axial_compression'],a['final_axial_compression'],atol=1e-12)
    expected=brentq(lambda lam:-c['young_mpa']*np.log(lam)/lam-.01,.99,1.)
    # Cauchy pressure with free lateral stresses: axial Hencky stress is
    # E ln(lam)/J, and J=lam**(1-2*nu).
    expected=brentq(lambda lam:-c['young_mpa']*np.log(lam)/lam**(1-2*c['poisson'])-.01,.99,1.)
    assert 1-b['final_axial_compression']==pytest.approx(expected,abs=1e-7)
