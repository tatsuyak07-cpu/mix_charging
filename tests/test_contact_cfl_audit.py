"""Force-only audits retain physical checks when timestep identity is inapplicable."""
import json
from pathlib import Path
import numpy as np
from tools.audit_contact_capture import audit

BASE=Path('exports/contact-reaction-audit-001/simulation/run')

def test_force_only_retains_equilibrium_and_force_failure_checks(tmp_path):
    # Use the archived accepted plastic/contact state, not fabricated native checks.
    import shutil
    run=tmp_path/'run';run.mkdir()
    for name in ('config.json','summary.json','last-state.npz','history.csv','contact-audit.json'):
        shutil.copy2(BASE/name,run/name)
    good=audit(run,tmp_path/'good',force_only=True)
    assert good['all_checks_passed']
    assert good['baseline_invariance_requested'] is False
    assert good['full_step_baseline_bitwise_identical'] is None
    data=json.loads((run/'contact-audit.json').read_text())
    data['solver_internal_generalized_force'][0]+=.01
    (run/'contact-audit.json').write_text(json.dumps(data))
    bad=audit(run,tmp_path/'bad',force_only=True)
    assert not bad['all_checks_passed']
    assert not bad['checks']['full_force_reassembly']
    assert bad['checks']['contact_force_finite_difference']


def test_identical_run_cannot_pass_as_cfl_refinement():
    from tools.compare_contact_cfl import compare
    result,*_=compare(BASE.parent,BASE.parent)
    assert all(v == 0 for v in result['metrics'].values())
    assert not result['checks']['only_cfl_changed']
    assert not result['checks']['cfl_halved']
    assert not result['all_checks_passed']
    json.dumps(result)  # All exported check booleans must be serializable.
