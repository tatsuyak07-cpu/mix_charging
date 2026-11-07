"""Element fields for plots of the J2 material's transferred quadrature history."""
import numpy as np


def element_peeq(state):
    """Current-volume mean, preserving the integral of equivalent plastic strain."""
    ep=np.asarray(state['ep']);count=len(state['t'])
    if 'parent' not in state:
        if len(ep)!=count:raise ValueError('Quadrature history requires an element map')
        return ep
    parent=np.asarray(state['parent'],dtype=int)
    weights=np.asarray(state['V0'],dtype=float).copy()
    if 'b' in state:weights*=np.sqrt(np.linalg.det(state['b']))
    if parent.shape!=ep.shape or weights.shape!=ep.shape or np.any(parent<0) or np.any(parent>=count) or np.any(weights<=0):raise ValueError('Invalid quadrature plotting weights')
    volume=np.bincount(parent,weights=weights,minlength=count)
    if np.any(volume<=0):raise ValueError('An element has no material history')
    return np.bincount(parent,weights=weights*ep,minlength=count)/volume
