"""Optional parallel nonsymmetric direct solve; no dependency installation here."""
import numpy as np
import importlib.metadata
import sys
from scipy.sparse.linalg import splu


def solver_metadata(config):
    backend = config.get('implicit_sparse_solver', 'superlu')
    if backend not in ('superlu', 'pardiso'):
        raise ValueError(f'Unknown sparse solver: {backend}')
    packages = ['numpy', 'scipy']
    if backend == 'pardiso':
        packages += ['pypardiso', 'mkl', 'threadpoolctl']
    return dict(backend=backend, threads=config.get('implicit_sparse_threads', 4) if backend == 'pardiso' else None,
                versions={name: importlib.metadata.version(name) for name in packages})


class PardisoFactor:
    def __init__(self, matrix, threads):
        from pypardiso import PyPardisoSolver
        from threadpoolctl import threadpool_limits
        self._limits = threadpool_limits
        self.threads = threads
        self.matrix = matrix.astype(np.float64).tocsr(copy=True)
        self.matrix.sum_duplicates()
        self.matrix.sort_indices()
        self.solver = PyPardisoSolver(mtype=11)
        try:
            with self._limits(limits=threads, user_api='blas'):
                self.solver.factorize(self.matrix)
        except Exception:
            self.close()
            raise

    def solve(self, rhs):
        if self.solver is None:
            raise RuntimeError('Factor has been closed')
        rhs = np.array(rhs, dtype=np.float64, order='F', copy=True)
        with self._limits(limits=self.threads, user_api='blas'):
            result = self.solver.solve(self.matrix, rhs)
        error = np.max(abs(self.matrix @ result - rhs))
        if not np.all(np.isfinite(result)) or error > 1e-7 * max(1., np.max(abs(rhs))):
            raise ValueError(f'Parallel sparse solve residual too large: {error}')
        return result

    def close(self):
        solver = getattr(self, 'solver', None)
        if solver is not None:
            self.solver = None
            with self._limits(limits=self.threads, user_api='blas'):
                solver.free_memory(everything=True)

    def __del__(self, finalizing=sys.is_finalizing):
        # Native allocations are reclaimed by process exit. Do not import
        # SciPy from PyPardiso cleanup after Python module teardown has begun.
        if not finalizing():
            self.close()


def factorize(matrix, config=None):
    config = config or {}
    backend = config.get('implicit_sparse_solver', 'superlu')
    if backend == 'superlu':
        return splu(matrix.tocsc(), permc_spec='COLAMD')
    if backend != 'pardiso':
        raise ValueError(f'Unknown sparse solver: {backend}')
    threads = config.get('implicit_sparse_threads', 4)
    if not isinstance(threads, int) or threads < 1:
        raise ValueError('Sparse solver thread count must be a positive integer')
    return PardisoFactor(matrix, threads)
