from copy import deepcopy
import pickle
import numpy as np
import pytest
from packing_fem.checkpoint import contact_state
from packing_fem.conservative_quadrature import project
from packing_fem.embedded_remesh import transfer_parent
from packing_fem.runtime_quadrature import repair
from tools.verify_parent_hybrid import case


def prepared(slip=.0001):
    old,new,p,H,before,mapping,parent=case(slip=slip,quadrature_order=8)
    response,_=transfer_parent(old,new,p,H,mapping,parent,before,1e-8,redistribute=True)
    new.accept_geometry(p['x'],H)
    return new,p,H,response


@pytest.mark.parametrize('slip',[.0001,.01])
def test_lookahead_preserves_accepted_force_history_and_old_object(slip):
    c,p,H,before=prepared(slip);snapshot=pickle.dumps(contact_state(c));rules=deepcopy(c.rules)
    y=p['x'].copy();G=H.copy();y[p['X'][:,2]>.5,0]+=.0001;G[0,2]+=.0001
    candidate,response,report=repair(c,p['t'],p['x'],p['x']*0,H,y,y*0,G,G*0,.0001,before)
    assert pickle.dumps(contact_state(c))==snapshot
    assert c.rules==rules and not c._parent_transfer_probe
    assert report['accepted_work_error']<1e-8
    assert max(report['accepted_energy_errors'].values())<1e-8
    assert report['new_samples']==0 and not report['history_interpolation']
    assert all(ctx['passed'] for ctx in candidate.parent_contexts.values())
    assert response[-1]['max_coulomb_ratio']<=c.mu+1e-10
    assert candidate.history.keys()==c.history.keys()
    for key,state in c.history.items():
        for field in ('spring','plastic','director','coordinate','bary','anchor_ids'):
            np.testing.assert_array_equal(candidate.history[key][field],state[field])
        ratio=candidate.inner.integration_area0[key[1]]/c.inner.integration_area0[key[1]]
        assert candidate.history[key]['energy']==pytest.approx(state['energy']*ratio,rel=1e-12,abs=1e-30)


def test_rejected_projection_leaves_all_old_contact_state_unchanged(monkeypatch):
    import packing_fem.runtime_quadrature as runtime
    c,p,H,before=prepared();snapshot=pickle.dumps(contact_state(c));rules=deepcopy(c.rules)
    def infeasible(*args,**kwargs):raise ValueError('Deliberately infeasible conservation constraints')
    monkeypatch.setattr(runtime,'redistribute',infeasible)
    with pytest.raises(ValueError,match='infeasible'):
        repair(c,p['t'],p['x'],p['x']*0,H,p['x'],p['x']*0,H,H*0,.0001,before)
    assert pickle.dumps(contact_state(c))==snapshot and c.rules==rules


def test_fixed_original_weight_bounds_prevent_repeated_projection_ratcheting():
    initial=np.array([.25,.75]);current=np.array([.5,.5]);C=np.ones((1,2))
    ctx=dict(matrix=np.array([[1.,0.]]),exact=np.array([.7]),scales=np.ones(1))
    proposed,_=project(current,C,[ctx],1e-7)
    assert proposed[0]>.69
    with pytest.raises(ValueError,match='infeasible'):
        project(current,C,[ctx],1e-7,reference_weights=initial)
