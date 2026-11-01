from pathlib import Path
import json
import numpy as np
import pytest
from packing_fem.embedded_contact import EmbeddedContact
from packing_fem.embedded_remesh import refine_parent_rules,transfer
from packing_fem.explicit_periodic import Periodic
from packing_fem.explicit_material import geometry
from packing_fem.surface_remesh import bisect
from tools.verify_plane_contact import fixture


from tools.verify_parent_hybrid import case as setup


def test_parent_audit_keeps_weights_even_when_child_areas_do_not_match():
    old,new,p,H,before,mapping,parent=setup()
    # Probe exposes the fine-DOF audit without authorizing a transfer.
    new._parent_transfer_probe=True
    r=new.evaluate(p['x'],p['x']*0,H,H*0,0.)
    assert all(c['passed'] for c in new.parent_contexts.values()),[(k,v['report']) for k,v in new.parent_contexts.items()]
    assert any(abs(sum(new.rules[str(f)]['weights'])-g['area']/2)>1e-3 for g in new.parent_groups.values() if len(g['faces'])==2 for f in g['faces'])
    new._parent_transfer_probe=False
    after,report=transfer(old,new,p,H,mapping,parent,before,1e-8)
    assert report['periodic_virtual_work_relative_error']<1e-8
    assert len(old.history)==len(new.history)


def test_centroid_rule_passes_total_energy_but_fails_new_node_reaction():
    from packing_fem.area_contact import SurfaceResolutionError
    old,new,p,H,before,mapping,parent=setup(centroid=True)
    new._parent_transfer_probe=True;new.evaluate(p['x'],p['x']*0,H,H*0,0.)
    failed=[c for c in new.parent_contexts.values() if not c['passed']]
    assert failed and max(c['report']['energy_error'] for c in failed)<1e-10
    assert max(c['report']['force_error'] for c in failed)>.1
    new._parent_transfer_probe=False
    with pytest.raises(SurfaceResolutionError,match='Parent contact audit'):new.evaluate(p['x'],p['x']*0,H,H*0,0.)


@pytest.mark.parametrize('slip',[0.,.0001,.01])
def test_hybrid_bounded_reweighting_conserves_history_work_and_energies(slip):
    from copy import deepcopy
    from packing_fem.embedded_remesh import transfer_parent
    old,new,p,H,before,mapping,parent=setup(slip=slip,quadrature_order=8)
    saved=deepcopy(old.history);weights=old.inner.integration_area0.copy()
    after,report=transfer_parent(old,new,p,H,mapping,parent,before,1e-8,redistribute=True)
    assert report['parent_audit_passed']
    assert all(c['passed'] for c in new.parent_contexts.values())
    assert report['redistribution_used']
    assert report['redistribution']['new_samples']==0
    assert report['redistribution_work_error']<1e-8
    for name in ('normal_energy','tangent_energy'):assert after[-1][name]==pytest.approx(before[-1][name],rel=1e-8,abs=1e-18)
    np.testing.assert_array_equal(old.inner.integration_area0,weights)
    nodes={int(n):int(new.face_points[mapping[face,k][0]][0][mapping[face,k][1]]) for face,(point_nodes,_,_) in old.face_points.items() for k,n in enumerate(point_nodes)}
    for key,state in saved.items():
        migrated=new.history[('area',nodes[key[1]])]
        for name in ('spring','plastic','director','coordinate'):np.testing.assert_array_equal(migrated[name],state[name])
        ratio=new.inner.integration_area0[nodes[key[1]]]/weights[key[1]]
        assert migrated['energy']==pytest.approx(state['energy']*ratio,rel=1e-12,abs=1e-30)
    for key in saved:
        for name in ('spring','plastic','director','bary','anchor_ids'):np.testing.assert_array_equal(old.history[key][name],saved[key][name])


def test_insufficient_existing_support_rejects_without_changing_old_history():
    from copy import deepcopy
    from packing_fem.embedded_remesh import transfer_parent
    old,new,p,H,before,mapping,parent=setup(quadrature_order=4)
    saved=deepcopy(old.history);w=old.inner.integration_area0.copy()
    with pytest.raises(ValueError,match='infeasible'):transfer_parent(old,new,p,H,mapping,parent,before,1e-8,redistribute=True)
    np.testing.assert_array_equal(old.inner.integration_area0,w)
    assert not new._parent_transfer_probe
    for key in saved:np.testing.assert_array_equal(old.history[key]['plastic'],saved[key]['plastic'])


def test_parent_mode_does_not_allow_blind_replay_or_repeated_unknown_lineage():
    from packing_fem.area_contact import SurfaceResolutionError
    old,new,p,H,before,mapping,parent=setup()
    new._requests={'1':dict(bary=[[1/3]*3],weights=[.5])}
    with pytest.raises(SurfaceResolutionError,match='lineage-aware') as error:new._raise_refinement()
    assert 'rules' not in error.value.feature
    assert error.value.feature['proposed_rules']==new._requests
    with pytest.raises(ValueError,match='multi-event'):refine_parent_rules(new,new.faces,p)


def test_parent_constructor_rejects_loss_of_historical_metadata_and_total_area():
    from copy import deepcopy
    old,new,p,H,before,mapping,parent=setup()
    rules=deepcopy(new.rules);group=next(g for g in new.parent_groups.values() if len(g['faces'])==2)
    rules[str(group['faces'][0])]['parent_atlas']=[dict(source=[[0,0,0]]*3)]
    with pytest.raises(ValueError,match='loses historical'):EmbeddedContact(new.X,p['t'],Periodic(new.X,1.),20.,.3,.1,quadrature_rules=rules,audit_mode='parent',error_budget='normalized')
    rules=deepcopy(new.rules)
    for face in group['faces']:rules[str(face)]['weights']=(2*np.asarray(rules[str(face)]['weights'])).tolist()
    with pytest.raises(ValueError,match='area mismatch'):EmbeddedContact(new.X,p['t'],Periodic(new.X,1.),20.,.3,.1,quadrature_rules=rules,audit_mode='parent',error_budget='normalized')
