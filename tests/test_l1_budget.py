import numpy as np
import pytest
from packing_fem.conservative_quadrature import project,response_audit


def test_concentrated_force_excess_is_not_divided_among_components():
    w=np.ones(2)/2
    c=dict(matrix=np.zeros((11,2)),exact=np.r_[0.,.09,np.zeros(9)],scales=np.ones(11),energy_budget=0.,force_budget=.1)
    proposed,report=project(w,np.eye(2),[c],1e-6)
    np.testing.assert_array_equal(proposed,w)
    assert report['baseline_admissible'] and response_audit(w,[c],1e-6)['passed']


def test_shared_face_budget_rejects_individually_admissible_excesses():
    w=np.ones(2)/2
    c=dict(matrix=np.zeros((2,2)),exact=np.array([0.,.07]),scales=np.ones(2),energy_budget=0.,force_budget=.1,budget_group='pose',global_energy_budget=0.,global_force_budget=.1)
    assert response_audit(w,[c],1e-6)['passed']
    assert not response_audit(w,[c,dict(c)],1e-6)['passed']
    with pytest.raises(ValueError,match='infeasible'):project(w,np.eye(2),[c,dict(c)],1e-6)


def test_l1_projection_moves_weights_and_independently_passes_total_budget():
    w=np.ones(3)/3
    c=dict(matrix=np.array([[0.,0.,0.],[0.,.5,1.],[0.,1.,.5]]),exact=np.array([0.,.65,.58]),scales=np.ones(3),energy_budget=0.,force_budget=.025,budget_group='pose',global_energy_budget=0.,global_force_budget=.025)
    p,r=project(w,np.ones((1,3)),[c],1e-5)
    assert r['relative_area_weight_change']>0
    assert response_audit(p,[c],1e-5)['passed']
    assert p.sum()==pytest.approx(1.,abs=1e-12)


def test_energy_and_force_budgets_are_separate_and_poses_are_independent():
    w=np.ones(2)/2
    c=dict(matrix=np.zeros((2,2)),exact=np.array([.09,.09]),scales=np.ones(2),energy_budget=.1,force_budget=.1,budget_group='accepted',global_energy_budget=.1,global_force_budget=.1)
    next_pose=dict(c,budget_group='trial')
    assert response_audit(w,[c,next_pose],1e-6)['passed']
    assert not response_audit(w,[c,dict(c)],1e-6)['passed']


def test_boundary_certified_rule_is_retained_without_artificial_safety_rejection():
    w=np.ones(2)/2;c=dict(matrix=np.zeros((2,2)),exact=np.array([0.,.125]),scales=np.ones(2),energy_budget=0.,force_budget=.124)
    p,r=project(w,np.eye(2),[c],.001)
    np.testing.assert_array_equal(p,w)
    assert r['baseline_admissible']
