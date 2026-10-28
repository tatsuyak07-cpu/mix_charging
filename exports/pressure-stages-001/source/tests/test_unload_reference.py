"""Independent elastic formula and nonvanishing cycle-energy diagnostics."""
import numpy as np
import pytest
from tools.verify_unload_quasistatic import elastic_reference,cycle_metrics
from packing_fem.explicit_material import response


def test_lambert_reference_has_exact_cauchy_stress_energy_and_linear_limit():
    E,nu=10.,.33;p=np.array([0.,1e-8,.02]);r=elastic_reference(p,E,nu,1.)
    stretch=np.c_[r['lateral_stretch'],r['lateral_stretch'],1-r['axial_compression']]
    b=np.array([np.diag(v*v) for v in stretch])
    sigma,energy,_=response(b,np.zeros(3),dict(young_mpa=E,poisson=nu,yield_mpa=.05,hardening_mpa=.2))
    target=np.zeros_like(sigma);target[:,2,2]=-p
    np.testing.assert_allclose(sigma,target,atol=1e-14,rtol=1e-12)
    np.testing.assert_allclose(energy,r['energy_Nmm'],atol=1e-18,rtol=1e-12)
    assert r['axial_compression'][1]==pytest.approx(p[1]/E,rel=1e-8)


def test_cycle_energy_metric_includes_zero_pressure_and_distinguishes_fast_motion():
    a=np.zeros((5,11));a[:,0]=[0,1,2,3,4];a[:,3]=[0,1,1,0,0];a[:,5]=[0,1,1,1e-12,0];a[:,4]=[0,1e-5,0,1e-10,1e-12]
    r=cycle_metrics(a,1,1,1,1.)
    assert r['peak_kinetic_over_reference']==1e-5
    assert r['unloading']['peak_kinetic_Nmm']==1e-10
    # At zero load, a global denominator still sees an independently large KE.
    a[-1,4]=.1
    assert cycle_metrics(a,1,1,1,1.)['peak_kinetic_over_reference']==.1
    assert cycle_metrics(a,1,1,1,1.)['recovery']['peak_kinetic_over_reference']==.1


def test_cycle_energy_metric_rejects_unavailable_reference_and_nonfinite_rows():
    a=np.zeros((2,11))
    with pytest.raises(ValueError):cycle_metrics(a,1,1,1,0.)
    a[0,4]=np.nan
    with pytest.raises(ValueError):cycle_metrics(a,1,1,1,1.)
