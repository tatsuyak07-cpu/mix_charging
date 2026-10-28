import json
from pathlib import Path
import numpy as np
import pytest
from packing_fem.surface_remesh import closest_edge_candidates
from packing_fem.explicit_periodic import Periodic
from tools.verify_closest_retry import run


def test_saved_cell_selects_higher_quality_target_without_changing_inputs():
    root=Path('exports/packing-rve-replay-001/validation/corrected-run/calculation')
    c=json.loads((root/'config.json').read_text());s=json.loads((root/'summary.json').read_text())
    with np.load(root/'last-state.npz') as z:
        X=z['reference_points'];x=z['points'];before=x.copy()
        rows=closest_edge_candidates(X,x,z['tets'],z['elastic_b'],z['peeq'],z['reference_volume'],Periodic(X,c['length_mm']).velocities(z['generalized_velocity']),s['failure_feature'],c['length_mm'])
    assert rows[0]['side']==1 and rows[0]['edge']==[3793,3795]
    assert rows[0]['minimum_quality']>.08 and rows[1]['minimum_quality']<.032
    np.testing.assert_array_equal(before,x)


def test_real_guard_refines_and_completes_manufactured_path(tmp_path):
    r=run(tmp_path/'retry');assert r['passed'],r['solver']['failure']
    events=r['solver']['remesh_events'];assert events[0]['candidate_selection']
    files=list((tmp_path/'retry/calculation').glob('surface-remesh-attempt-*.npz'))
    with np.load(files[0]) as p:assert 'weights' in p.files


def test_first_audit_rejection_tries_other_edge(tmp_path,monkeypatch):
    import packing_fem.surface_remesh as surface
    original=surface.audit;calls=[]
    def reject_once(*a,**kw):
        report=original(*a,**kw);calls.append(report)
        if len(calls)==1:report={**report,'accepted':False,'checks':{**report['checks'],'injected_rejection':False}}
        return report
    monkeypatch.setattr(surface,'audit',reject_once)
    r=run(tmp_path/'fallback')
    events=r['solver']['remesh_events'];assert not events[0]['accepted']
    assert events[1]['accepted'] and events[0]['candidate_selection']['candidate']['side']!=events[1]['candidate_selection']['candidate']['side']
    assert r['solver']['checks']['path_completed']


def test_all_rejections_preserve_accepted_state_like_zero_event_cap(tmp_path,monkeypatch):
    baseline=run(tmp_path/'cap',max_events=0)
    import packing_fem.surface_remesh as surface
    original=surface.audit
    def reject(*a,**kw):return {**original(*a,**kw),'accepted':False}
    monkeypatch.setattr(surface,'audit',reject)
    r=run(tmp_path/'rejected')
    assert not r['solver']['checks']['path_completed'] and len(r['solver']['remesh_events'])==2
    with np.load(tmp_path/'cap/calculation/last-state.npz') as a,np.load(tmp_path/'rejected/calculation/last-state.npz') as b:
        for key in a.files:np.testing.assert_array_equal(a[key],b[key])


def test_unknown_strategy_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='strategy'):run(tmp_path/'bad',strategy='unknown')
