import numpy as np
import pytest
from packing_fem.contact_atlas import build,density,scales
from tools.verify_onset_budget import plane_reference
from packing_fem.surface_patch import IntegrationRefinement


def poses():
    rows=[]
    for width in (.05,.055,.1):
        depth=.001;z=.02*(1-depth)
        rows.append(dict(source=[[0,0,z],[1,0,z+.02*depth/width],[0,1,z]],targets=[[[-2,-2,0],[4,-2,0],[-2,4,0]]],mapping=[[0,1,2]],groups=[[0]]))
    return rows


def test_positive_rule_matches_all_previous_poses_and_independent_decimal_integral():
    history=poses();previous=None
    for count in range(1,4):
        rule=build(history[:count],.5,10.,.02,previous_bary=previous);previous=rule['bary']
        assert np.all(rule['weights']>0) and rule['weights'].sum()==pytest.approx(.5,abs=1e-12)
        assert len(rule['weights'])<rule['candidate_points']
        for pose in history[:count]:
            exact,_=plane_reference(np.asarray(pose['source']))
            applied=rule['weights']@density(rule['bary'],pose,10.,.02)
            assert np.max(abs(applied-exact)/scales(exact))<.001
        assert rule['moment_error']<1e-11


def test_atlas_refuses_to_discard_old_poses_to_meet_budget():
    with pytest.raises(IntegrationRefinement,match='no past pose discarded'):build(poses(),.5,10.,.02,max_poses=2)


def test_workspace_budget_changes_do_not_change_an_already_feasible_rule():
    pose=poses()[:1]
    normal=build(pose,.5,10.,.02)
    larger=build(pose,.5,10.,.02,max_entries=48000000,max_points=160000)
    np.testing.assert_array_equal(normal['bary'],larger['bary'])
    np.testing.assert_array_equal(normal['weights'],larger['weights'])
    with pytest.raises(IntegrationRefinement,match='entries=.*no reference pose discarded'):
        build(pose,.5,10.,.02,max_entries=10)


def test_bounded_fallback_retains_moments_and_audits_every_response(monkeypatch):
    import packing_fem.contact_atlas as atlas
    from types import SimpleNamespace
    original=atlas.linprog;calls=[]
    def simulated_numeric_rejection(*args,**kwargs):
        calls.append(kwargs)
        if len(calls)==1:return SimpleNamespace(success=False,message='simulated singular exact system')
        return original(*args,**kwargs)
    monkeypatch.setattr(atlas,'linprog',simulated_numeric_rejection)
    rule=atlas.build(poses()[:1],.5,10.,.02)
    assert rule['fit_mode']=='minimax_responses_exact_moments'
    assert rule['moment_error']<1e-11
    assert all(max(r.values())<.001/8 for r in rule['audits'])
    assert np.all(rule['weights']>0)

    independent,_=plane_reference(np.asarray(poses()[0]['source']))
    applied=rule['weights']@density(rule['bary'],poses()[0],10.,.02)
    assert np.max(abs(applied-independent)/scales(independent))<.001


def test_minimax_residual_for_two_inconsistent_scalar_integrals():
    from packing_fem.contact_atlas import minimax_fit
    from packing_fem.triangle_quadrature import RULES
    B,_=RULES[4]
    C=np.array([np.ones(len(B)),B[:,0],B[:,1],B[:,0]**2,B[:,0]*B[:,1],B[:,1]**2])
    field=B[:,0]**3/.1
    delta=.00019
    matrix=np.r_[C,field[None],field[None]]
    target=np.r_[[1,1/3,1/3,1/6,1/12,1/6],1-delta,1+delta]
    result=minimax_fit(matrix,target)
    assert result.success
    assert result.x[-1]==pytest.approx(delta,rel=1e-6,abs=1e-10)
    assert np.min(result.x[:-1])>=-1e-12
    np.testing.assert_allclose(C@result.x[:-1],target[:6],atol=1e-10)


def test_dual_enrichment_adds_support_without_relaxing_moment_or_response_error():
    from packing_fem.contact_atlas import enrich_minimax,minimax_fit
    B=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]])
    def columns(points):
        return np.array([np.ones(len(points)),points[:,0],points[:,1],points[:,0]**2,points[:,0]*points[:,1],points[:,1]**2,points[:,0]**3/.1])
    rhs=np.array([1,1/3,1/3,1/6,1/12,1/6,1.]);matrix=columns(B)
    original=minimax_fit(matrix,rhs);assert original.success and original.x[-1]>.001
    points,expanded,result,rounds=enrich_minimax(B,matrix,rhs,columns,1e-8)
    assert result.success and rounds>0 and len(points)>len(B)
    assert result.x[-1]<1e-8
    np.testing.assert_allclose(expanded[:6]@result.x[:-1],rhs[:6],atol=1e-10)
    assert np.min(result.x[:-1])>=-1e-12
