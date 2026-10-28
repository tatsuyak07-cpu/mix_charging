"""Geometry-preserving neck refinement and independent beam verification."""
from pathlib import Path
import shutil
import numpy as np
import pytest
from packing_fem.neck_validation import neck_mesh,beam_benchmark,write_neck
from packing_fem import plastic
import json

ROOT=Path(__file__).resolve().parents[1]


def test_refinement_keeps_exact_stepped_geometry_volume_and_neck_area():
    for level in (1,2,4):
        p,e,g=neck_mesh(level)
        assert g['initial_solid_volume_mm3']==pytest.approx(238.,abs=1e-10)
        assert g['quarter_neck_area_mm2']==.75
        assert p[:,2].min()==3 and p[:,2].max()==21
        assert len(np.unique(p,axis=0))==len(p)
        assert np.all(p[e].max(axis=1)>p[e].min(axis=1))


@pytest.mark.skipif(shutil.which('/workspace/environment/calculix/usr/bin/ccx') is None,reason='CalculiX unavailable')
def test_incompatible_mode_beam_matches_independent_reference(tmp_path):
    result=beam_benchmark(tmp_path/'beam','C3D8I')
    assert result['solver_completed']
    assert result['within_5_percent_beam_reference']
    assert result['tip_displacement_mm']==pytest.approx(.03206384,rel=.05)


def test_incompatible_mode_is_not_silently_used_with_eight_corner_work_audit():
    c=json.loads((ROOT/'configs/plastic-smoke.json').read_text());c['element_type']='C3D8I'
    with pytest.raises(ValueError,match='C3D8'):plastic.validate(c)


def test_surface_mean_is_area_weighted_on_a_nonuniform_grid():
    from packing_fem.neck_validation import mesh_from_grid,top_surface_weights
    p,e=mesh_from_grid([np.array([0.,.25,1.]),np.array([0.,1.]),np.array([20.,21.])],np.ones((2,1,1),bool))
    weights=top_surface_weights(p,e)
    # Mean of u_z=x on the unit square is exactly 1/2, despite uneven nodes.
    assert np.dot(weights,p[:,0])==pytest.approx(.5)
    assert weights.sum()==pytest.approx(1)
    assert not np.isclose(p[p[:,2]==21,0].mean(),.5)


def test_component_refuses_conflicting_geometry_settings(tmp_path):
    c=json.loads((ROOT/'configs/plastic-smoke.json').read_text());c['diameter_mm']=11
    with pytest.raises(ValueError,match='Fixed component geometry'):
        write_neck(tmp_path/'must-not-exist',c,1,'C3D8I')
    assert not (tmp_path/'must-not-exist').exists()
