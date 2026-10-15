"""Actual material benchmarks for the nonlinear periodic writer and unload path."""
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from packing_fem import plastic,sintered

ROOT=Path(__file__).resolve().parents[1]
SOLVER='/workspace/environment/calculix/usr/bin/ccx'


@pytest.mark.skipif(shutil.which(SOLVER) is None,reason='CalculiX unavailable')
@pytest.mark.parametrize('yield_mpa,compression',[(.05,.03),(10.,.0001)])
def test_periodic_material_loading_and_true_zero_load_unloading(tmp_path,yield_mpa,compression):
    c=json.loads((ROOT/'configs/plastic-smoke.json').read_text())
    c.update(length_mm=2.,yield_mpa=yield_mpa,compression_fraction=compression)
    p,e,idx=sintered.mesh(np.ones((2,2,2),bool),1.)
    estimate={'grid_cells_per_axis':2,'porosity_error':0,'periodic_components':1,
              'winding_rank':3,'upper_bound_dofs':3*len(p)+6,'initial_porosity':0.}
    np.savez_compressed(tmp_path/'sintered-geometry.npz',points=p,elements=e)
    plastic.write_model(tmp_path,c,p,e,idx,estimate)
    r=plastic.run(tmp_path,SOLVER,30)
    assert r['checks']['solver_completed'] and r['checks']['end_of_unload_reached']
    assert r['checks']['positive_jacobians'] and r['checks']['periodic_displacements']
    assert r['checks']['physical_porosity']
    assert np.max(np.abs(r['final_mean_cauchy_stress_mpa']))<1e-7
    from packing_fem.energy import audit
    energy=audit(tmp_path,tmp_path/'energy-audit')
    assert energy['checks']['finite_solver_energy']
    assert energy['checks']['macro_micro_work']
    assert energy['checks']['compression_work_balance']
    assert energy['checks']['cycle_work_balance']
    assert energy['checks']['plastic_accumulation']
    if yield_mpa==.05:
        assert r['peak_equivalent_plastic_strain']>.02
        assert .02<r['residual_compression_strain']<.03
        # Logarithmic axial compression is split into elastic + plastic components.
        # Bilinear uniaxial law: sigma = yield + H*plastic strain.
        expected=(c['young_mpa']*(-np.log(1-compression))-yield_mpa)/(c['young_mpa']+c['hardening_mpa'])
        assert r['peak_equivalent_plastic_strain']==pytest.approx(expected,rel=.03)
        assert r['peak_compression_mpa']==pytest.approx(yield_mpa+c['hardening_mpa']*expected,rel=.01)
    else:
        assert r['peak_equivalent_plastic_strain']<1e-10
        assert abs(r['residual_compression_strain'])<1e-8
        assert r['peak_compression_mpa']==pytest.approx(c['young_mpa']*compression,rel=.001)


def test_no_silent_claim_of_self_contact_or_calibrated_clay():
    c=json.loads((ROOT/'configs/plastic-smoke.json').read_text())
    c['contact_model']='self_contact'
    with pytest.raises(ValueError,match='contact'):plastic.validate(c)
    c['contact_model']='not_implemented';c['material_calibration']='clay'
    with pytest.raises(ValueError,match='uncalibrated'):plastic.validate(c)


def test_linear_cli_refuses_nonlinear_config_and_saved_input(tmp_path):
    from packing_fem.cli import main
    out=tmp_path/'must-not-be-created'
    assert main(['prepare','--config',str(ROOT/'configs/plastic-smoke.json'),'--output',str(out)])==2
    assert not out.exists()
    (tmp_path/'plastic-metadata.json').write_text('{}')
    assert main(['solve',str(tmp_path)])==2
    assert not (tmp_path/'sintered.log').exists()
