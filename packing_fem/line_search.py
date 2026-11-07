"""Bounded strong-Wolfe search with cached FE evaluations and domain rejection."""
import warnings
import numpy as np
from scipy.optimize import line_search


def bounded_wolfe(x,d,energy,gradient,values,maximum_alpha,roundoff=0.):
    slope=float(gradient@d)
    if maximum_alpha<=0 or not np.isfinite(slope) or slope>=0:return None
    pivot=int(np.argmax(abs(d)));cached_x=None;cached=None;best=None
    norm=float(np.max(abs(gradient),initial=0.))
    def evaluate(y):
        nonlocal cached_x,cached,best
        if cached_x is not None and np.array_equal(y,cached_x):return cached
        alpha=float((y[pivot]-x[pivot])/d[pivot])
        try:
            value,g,payload=values(y)
            if not np.isfinite(value) or not np.all(np.isfinite(g)):raise ValueError('Nonfinite line-search value')
            record=dict(alpha=alpha,energy=float(value),gradient=g,payload=payload)
            armijo=value<=energy+1e-4*alpha*slope
            acceptable=armijo or (value<=energy+roundoff and np.max(abs(g),initial=0.)<norm*(1-1e-4*alpha))
            record['wolfe']=bool(armijo and abs(float(g@d))<=.9*abs(slope))
            if alpha>0 and acceptable and (best is None or value<best['energy']):best=record
            cached=record
        except (ValueError,RuntimeError):cached=dict(alpha=alpha,energy=np.inf,gradient=np.zeros_like(gradient),payload=None,wolfe=False)
        cached_x=y.copy();return cached
    with warnings.catch_warnings(record=True) as messages:
        result=line_search(lambda y:evaluate(y)['energy'],lambda y:evaluate(y)['gradient'],x,d,gfk=gradient,old_fval=energy,c1=1e-4,c2=.9,amax=maximum_alpha,maxiter=20)
    if result[0] is not None:
        candidate=evaluate(x+float(result[0])*d)
        if candidate['wolfe']:best=candidate
    if best is not None:best=dict(best,warnings=[str(message.message) for message in messages])
    return best
