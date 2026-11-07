import numpy as np
from packing_fem.fixed_point import anderson_candidate


def test_acceleration_recovers_known_coupled_affine_fixed_point():
    A=np.array([[.8,.1],[0.,-.7]]);rhs=np.array([.3,.2]);x=np.array([1.,-1.]);outputs=[];residuals=[]
    for _ in range(3):
        y=A@x+rhs;outputs.append(y);residuals.append(y-x);x=y
    candidate=anderson_candidate(outputs,residuals,np.ones(2))
    expected=np.linalg.solve(np.eye(2)-A,rhs)
    np.testing.assert_allclose(candidate,expected,atol=1e-12)
    assert np.linalg.norm(A@candidate+rhs-candidate)<1e-12


def test_single_map_evaluation_does_not_invent_an_extrapolation():
    assert anderson_candidate([np.ones(2)],[np.zeros(2)],np.ones(2)) is None
