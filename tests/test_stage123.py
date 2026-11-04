from copy import deepcopy
import pickle
import numpy as np
import pytest
from tools.enrich_precontact_rules import enriched
from packing_fem.triangle_quadrature import gauss_rule
from packing_fem.runtime_quadrature import repair
from packing_fem.checkpoint import contact_state
from tests.test_runtime_quadrature import prepared
from tools.fit_replay_rules import fit
from tools.verify_plane_contact import fixture
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_periodic import Periodic
from packing_fem.parent_quadrature import source_on_child
from tools.run_parent_replay import require_free_prefix


def test_positive_enrichment_keeps_material_points_moments_and_all_past_poses():
    B,w=gauss_rule(4);rules={'0':dict(bary=B.tolist(),weights=(2*w).tolist(),atlas=[dict(tag='retained')])};saved=deepcopy(rules)
    for level in (0,1,2):
        new,report=enriched(rules,level,dict(contact_free_prefix=True,contact=dict(history={})))
        np.testing.assert_array_equal(new['0']['bary'][:len(B)],B)
        assert new['0']['atlas']==saved['0']['atlas'] and report['new_samples']>0
        assert np.all(np.asarray(new['0']['weights'])>0) and report['faces'][0]['moment_error']<1e-10
    assert rules==saved


@pytest.mark.parametrize('free,history',[(False,{}),(True,{'unknown':{}})])
def test_active_history_cannot_be_created_for_added_samples(free,history):
    with pytest.raises(ValueError,match='contact-free'):enriched({},1,dict(contact_free_prefix=free,contact=dict(history=history)))


def test_forecast_discards_future_friction_and_preserves_current_potential():
    c,p,H,before=prepared(.0001);snapshot=pickle.dumps(contact_state(c));y=p['x'].copy();G=H.copy();y[p['X'][:,2]>.5,0]+=.0001;G[0,2]+=.0001
    candidate,response,report=repair(c,p['t'],p['x'],p['x']*0,H,y,y*0,G,G*0,.0001,before,prediction_only=True)
    assert report['prediction_only'] and pickle.dumps(contact_state(c))==snapshot
    f=c.periodic.force(before[0],before[1]);again=candidate.replay_force_state(p['x'],p['x']*0,H)
    np.testing.assert_allclose(c.periodic.force(response[0],response[1]),f,rtol=1e-10,atol=1e-15)
    np.testing.assert_allclose(c.periodic.force(again[0],again[1]),f,rtol=1e-10,atol=1e-15)
    for key,state in c.history.items():
        for name in ('spring','plastic','coordinate','director'):np.testing.assert_array_equal(candidate.history[key][name],state[name])
    np.testing.assert_array_equal(candidate.inner.guard.accepted_geometry[0],candidate.P@p['x'])


def test_positive_replay_fit_resolves_planned_child_basis_without_old_history():
    X,t,*_=fixture(1,.025,0.);base=AreaContact(X,t,Periodic(X,1.),20.,.3,.1)
    face=next(i for i,ids in enumerate(base.faces) if np.all(np.isclose(X[ids,2],.4)))
    ids=base.faces[face];source=X[ids];B,w=gauss_rule(4);area=np.linalg.norm(np.cross(source[1]-source[0],source[2]-source[0]))/2
    pose=dict(source=source.tolist(),targets=[(source+[0,0,.015]).tolist()],mapping=[[0,1,2]],groups=[[0]],tie_rule='clarke')
    rules={str(face):dict(bary=B.tolist(),weights=(area*w).tolist(),atlas=[pose])};saved=deepcopy(rules)
    state=dict(contact_free_prefix=True,contact=dict(history={}),X=X,t=t)
    config=dict(length_mm=1.,contact_penalty_mpa_mm=20.,friction=.3,contact_skin_mm=.1,surface_remesh_edge=ids[:2].tolist())
    new,report=fit(rules,state,config,{str(face):pose},0)
    assert report['requires_coupled_replay'] and not report['history_transfer_from_stop']
    assert report['faces'][0]['child_source_basis'] and report['faces'][0]['response_error']<.0005
    assert report['faces'][0]['moment_error']<1e-10 and rules==saved
    assert np.all(np.asarray(new[str(face)]['weights'])>0)
    assert new[str(face)]['atlas']==saved[str(face)]['atlas']


def test_curved_retained_geometry_keeps_midpoint_and_material_lineage():
    I=np.eye(3);mid=(I[0]+I[1])/2;cells=np.array([[I[0],mid,I[2]],[mid,I[1],I[2]]]);source=np.array([[0.,0,0],[1.,0,0],[0.,1,0]]);bent=np.array([.5,0,.2])
    child=[np.array([source[0],bent,source[2]]),np.array([bent,source[1],source[2]])]
    pose=dict(source=source.tolist(),source_cells=cells.tolist(),child_sources=[c.tolist() for c in child])
    np.testing.assert_allclose(source_on_child(cells[0][[2,0,1]],pose),child[0][[2,0,1]],atol=1e-15)
    with pytest.raises(ValueError,match='cover'):source_on_child(I,pose)
    del pose['source_cells']
    with pytest.raises(ValueError,match='lineage'):source_on_child(cells[0],pose)


@pytest.mark.parametrize('free,history',[(False,{}),(True,{'active':{}})])
def test_automatic_replay_never_starts_with_active_or_unknown_prefix(free,history):
    with pytest.raises(ValueError,match='before ALL'):require_free_prefix(dict(contact_free_prefix=free,contact=dict(history=history)))
