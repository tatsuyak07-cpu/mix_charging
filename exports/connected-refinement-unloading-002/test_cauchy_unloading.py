"""An elastic periodic cube must recover after true-pressure unloading."""
import json
from pathlib import Path
import numpy as np
from packing_fem import sintered
from packing_fem.explicit_material import hex_to_tet
from packing_fem.explicit_periodic import deformation
from packing_fem.explicit_solver import solve


def test_true_pressure_elastic_cycle_returns_to_identity(tmp_path):
    c=json.loads((Path(__file__).resolve().parents[1]/'configs/explicit-area-cauchy-pressure.json').read_text())
    c.update(length_mm=1.,pressure_mpa=.02,ramp_s=.008,hold_s=.002,unload_s=.008,recovery_hold_s=.002,output_every=200,max_steps=100000)
    X,h,_=sintered.mesh(np.ones((1,1,1),bool),1.)
    r=solve(tmp_path/'cycle',X,hex_to_tet(X,h),c)
    assert r['failure'] is None
    assert r['checks']['unload_completed']
    audit=np.loadtxt(tmp_path/'cycle/accepted-step-audit.csv.gz',delimiter=',',skiprows=1)
    assert np.max(audit[:,3])>=.02*(1-1e-10)
    assert audit[-1,3]==0.
    with np.load(tmp_path/'cycle/last-state.npz') as z:
        np.testing.assert_allclose(deformation(z['generalized_position'][-6:]),np.eye(3),atol=1e-7,rtol=0.)
        np.testing.assert_allclose(z['peeq'],0.,atol=1e-12,rtol=0.)
    np.testing.assert_allclose(r['final_work_equivalent_cauchy_stress_MPa'],0.,atol=1e-6,rtol=0.)
    assert np.max(audit[:,4])/r['energy_scale_Nmm']<.01
    # Solver acceptance is deliberately not asserted: KE / remaining elastic
    # energy can become large near zero load even when absolute KE is tiny.
