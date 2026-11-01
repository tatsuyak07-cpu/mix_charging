import numpy as np
import pytest
from packing_fem.conservative_quadrature import project


def test_positive_reweighting_preserves_moment_and_energy_while_fitting_new_mode():
    # Current coarse force, area and energy are all the same conserved row.
    w=np.ones(4)/4;C=np.ones((3,4));M=np.array([[0.,.2,.6,1.],[1,1,1,1.]])
    ctx=dict(matrix=M,exact=np.array([.55,1.]),scales=np.ones(2))
    for factor in (1.5,2.,4.):
        proposed,report=project(w,C,[ctx],1e-7,factor=factor)
        assert np.all(proposed>0)
        np.testing.assert_allclose(C@proposed,C@w,atol=1e-10)
        np.testing.assert_allclose(M@proposed,ctx['exact'],atol=1.1e-7)
        assert report['new_samples']==0
    np.testing.assert_array_equal(w,np.ones(4)/4)


def test_infeasible_conservation_or_support_never_returns_relaxed_solution():
    w=np.ones(2)/2;C=np.array([[0.,1.],[1.,1.]])
    ctx=dict(matrix=np.array([[0.,1.]]),exact=np.array([.9]),scales=np.ones(1))
    with pytest.raises(ValueError,match='infeasible'):project(w,C,[ctx],.001)
    with pytest.raises(ValueError,match='budget'):project(w,C,[ctx],max_entries=1)


def test_tall_conservation_operator_preserves_all_original_rows():
    w=np.ones(4)/4;C=np.ones((1000,4));ctx=dict(matrix=np.array([[0.,.2,.6,1.]]),exact=np.array([.55]),scales=np.ones(1))
    proposed,report=project(w,C,[ctx],1e-7)
    np.testing.assert_allclose(C@proposed,C@w,atol=1e-12)
    assert abs(ctx['matrix'][0]@proposed-.55)<1e-7
    assert report['conservation_nullity']==3
