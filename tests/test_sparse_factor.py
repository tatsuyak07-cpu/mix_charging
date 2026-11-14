import numpy as np
import pytest
from scipy.sparse import csr_matrix
from packing_fem.sparse_factor import factorize


@pytest.mark.parametrize('backend', ['superlu', 'pardiso'])
def test_nonsymmetric_multiple_rhs_and_input_preservation(backend):
    if backend == 'pardiso':
        pytest.importorskip('pypardiso')
    matrix = csr_matrix([[4., -2., 1.], [1., 3., 0.], [-1., 2., 5.]])
    original = matrix.copy()
    expected = np.array([[1., 2.], [-3., 4.], [5., -6.]])
    rhs = matrix @ expected
    before = rhs.copy()
    factor = factorize(matrix, dict(implicit_sparse_solver=backend, implicit_sparse_threads=4))
    np.testing.assert_allclose(factor.solve(rhs), expected, atol=1e-12)
    np.testing.assert_allclose(factor.solve(rhs[:, 0]), expected[:, 0], atol=1e-12)
    np.testing.assert_array_equal(rhs, before)
    np.testing.assert_array_equal(matrix.toarray(), original.toarray())
    if backend == 'pardiso':
        factor.close()
        factor.close()
        with pytest.raises(RuntimeError):
            factor.solve(rhs)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match='Unknown sparse solver'):
        factorize(csr_matrix(np.eye(2)), dict(implicit_sparse_solver='unknown'))
