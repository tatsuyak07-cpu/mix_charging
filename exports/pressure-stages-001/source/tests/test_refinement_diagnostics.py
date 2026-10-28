"""Independent geometric limits for stopped-path diagnostics."""
import numpy as np
import pytest

from tools.diagnose_refinement_chain import active_interval, measure, point_gap
from tools.probe_refinement_location import geometry_split
from packing_fem.explicit_material import geometry, oriented


def crossed(offset=.015):
    return np.array([[-1., 0, 0], [1., 0, 0], [0, -1., offset], [0, 1., offset]])


def test_crossed_edges_interval_has_analytic_width_and_root():
    p = crossed()
    a, b = active_interval(p, .02, .5)
    half = np.sqrt(.02**2 - .015**2) / 2
    assert a == pytest.approx(.5-half, abs=1e-13)
    assert b == pytest.approx(.5+half, abs=1e-13)
    assert point_gap(p, .5) == pytest.approx(.015)


def test_endpoint_interval_and_inactive_edges():
    p = crossed();p[:2, 0] += 1.
    assert active_interval(p, .02, 0.)[0] == 0.
    assert active_interval(crossed(.03), .02, .5) is None
    assert measure(p, .02)['minimum_gap_mm'] == pytest.approx(.015)


@pytest.mark.parametrize('fraction', [.2, .5, .8])
def test_geometry_split_preserves_affine_map_and_volume(fraction):
    X = np.array([[1., 1, 1], [2., 1, 1], [1., 2, 1], [1., 1, 2]])
    t = oriented(X, np.array([[0, 1, 2, 3]]))
    F = np.diag([1.1, .9, .8]);x = X @ F.T + [.1, -.2, .3]
    Y, y, newt = geometry_split(X, x, t, np.array([0, 1]), fraction, 5.)
    np.testing.assert_allclose(y, Y @ F.T + [.1, -.2, .3], atol=1e-15)
    assert geometry(Y, newt)[0].sum() == pytest.approx(geometry(X, t)[0].sum())
    assert geometry(y, newt)[0].sum() == pytest.approx(geometry(x, t)[0].sum())
    np.testing.assert_allclose(Y[-1], (1-fraction)*X[0]+fraction*X[1])


def test_geometry_split_matches_translated_edges_with_reversed_node_order():
    first = np.array([[1., 1, 1], [2., 1, 1], [1., 2, 1], [1., 1, 2]])
    X = np.r_[first, (first + [5., 0, 0])[[1, 0, 2, 3]]]
    t = oriented(X, np.array([[0, 1, 2, 3], [4, 5, 6, 7]]))
    Y, y, newt = geometry_split(X, X, t, np.array([0, 1]), .2, 5.)
    assert len(Y) == len(X) + 2
    np.testing.assert_allclose(Y[-1]-Y[-2], [5., 0, 0], atol=1e-14)
    assert geometry(Y, newt)[0].sum() == pytest.approx(geometry(X, t)[0].sum())
    with pytest.raises(ValueError, match='interior'):
        geometry_split(X, X, t, np.array([0, 1]), 0., 5.)
