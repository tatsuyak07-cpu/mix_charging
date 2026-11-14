"""Exact block elimination of the active periodic macro deformation unknowns."""
import numpy as np
from packing_fem.sparse_factor import factorize


def solve(K,rhs,nmicro,c=None):
    B=K[:nmicro,nmicro:].toarray();C=K[nmicro:,nmicro:].toarray()
    factor=factorize(K[:nmicro,:nmicro],c)
    response=factor.solve(np.column_stack([rhs[:nmicro],B]))
    macro=np.linalg.solve(C-B.T@response[:,1:],rhs[nmicro:]-B.T@response[:,0])
    result=np.r_[response[:,0]-response[:,1:]@macro,macro]
    error=np.max(abs(K@result-rhs))
    if not np.all(np.isfinite(result)) or error>1e-7*max(1.,np.max(abs(rhs))):
        raise ValueError(f'Block linear solve residual too large: {error}')
    return result
