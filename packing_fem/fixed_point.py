"""Small-memory Anderson proposal; callers must safeguard physical admissibility."""
import numpy as np
from .explicit_periodic import deformation


def affine_predictor(p,q,prescribed):
    previous=deformation(q[-6:]);result=q.copy();result[p.prescribed]=prescribed
    change=deformation(result[-6:])@np.linalg.inv(previous)
    result[:p.nmicro]=(q[:p.nmicro].reshape(-1,3)@change.T).ravel()
    return result


def anderson_candidate(outputs,residuals,scale):
    if len(outputs)<2:return None
    R=np.column_stack(residuals)*np.asarray(scale)[:,None]
    differences=R[:,:-1]-R[:,-1:]
    weights=np.linalg.lstsq(differences,-R[:,-1],rcond=1e-10)[0]
    weights=np.r_[weights,1-weights.sum()]
    if not np.all(np.isfinite(weights)) or abs(weights).sum()>10:return None
    return np.column_stack(outputs)@weights
