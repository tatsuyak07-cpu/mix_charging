"""Observed porosity target must not hide a failed volume audit."""
from tools.monitor_pressure_ramp import stop_reason


def test_both_volume_definitions_must_reach_target():
    assert stop_reason({'absolute_difference':.00004,'geometry_porosity':.09998,
                        'material_porosity':.10002,'target_final_porosity':.1},1e-4) is None


def test_observed_target_still_requires_independent_audit():
    reason=stop_reason({'absolute_difference':.00001,'geometry_porosity':.09998,
                        'material_porosity':.09999,'target_final_porosity':.1},1e-4)
    assert reason=='Observed target porosity; independent audit still required'


def test_failed_volume_audit_overrides_nominal_target():
    reason=stop_reason({'absolute_difference':.001,'geometry_porosity':.099,
                        'material_porosity':.098,'target_final_porosity':.1},1e-4)
    assert reason=='Existing volume_definition_difference audit limit exceeded; target not validated'


def test_negative_void_volume_is_not_a_reached_target():
    reason=stop_reason({'absolute_difference':0.,'geometry_porosity':-.001,
                        'material_porosity':-.001,'target_final_porosity':.1},1e-4)
    assert reason=='Nonphysical porosity; target not validated'
