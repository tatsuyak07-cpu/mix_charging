from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
from packing_fem.embedded_remesh import refine_rules
from packing_fem.triangle_quadrature import RULES


def fixture():
    X=np.array([[0.,0,.015],[1,0,.015],[0,1,.015]])
    Y=np.r_[X,[[.5,0,.015]]];faces=np.array([[0,3,2],[3,1,2]])
    C=[np.array([[1,0,0],[.5,.5,0],[0,0,1]]),np.array([[.5,.5,0],[0,1,0],[0,0,1]])]
    B,w=RULES[4];weights=np.r_[w,w]*.25
    pose=dict(source=X.tolist(),targets=[(X-[0,0,.015]).tolist()],mapping=[[0,1,2]],groups=[[0]],tie_rule='clarke')
    rule=dict(bary=np.r_[B@C[0],B@C[1]].tolist(),weights=weights.tolist(),atlas=[pose],atlas_controls=[dict(deepcopy(pose),lookahead_steps=4)])
    old=SimpleNamespace(X=X,faces=np.array([[0,1,2]]),rules={'0':rule},L=1.,kn=20.,thickness=.02,quadrature_tolerance=.001)
    return old,faces,dict(X=Y),C


def test_retains_all_past_and_control_poses_in_each_child_without_new_samples():
    old,faces,p,C=fixture();saved=deepcopy(old.rules)
    rules,points,parent=refine_rules(old,faces,p)
    assert len(points)==len(old.rules['0']['weights'])
    for j,rule in rules.items():
        assert len(rule['atlas'])==len(rule['atlas_controls'])==1
        np.testing.assert_allclose(rule['atlas'][0]['source'],C[int(j)]@old.X,atol=1e-15)
        assert len(rule['refinement_pose_audits'])==2
        assert max(a['force_error'] for a in rule['refinement_pose_audits'])<1e-10
    assert old.rules==saved


def test_rejects_bad_historical_response_even_when_child_areas_match():
    old,faces,p,_=fixture();weights=old.rules['0']['weights'];count=len(weights)//2
    original=np.asarray(weights[:count]);skew=.1*original;skew[0]+=.9*original.sum();weights[:count]=skew.tolist()
    with pytest.raises(ValueError,match='retained-pose audit'):refine_rules(old,faces,p)
