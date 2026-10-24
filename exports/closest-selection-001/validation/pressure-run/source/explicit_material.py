"""Experimental finite-strain Hencky J2 solid; mm, N, MPa, tonne, s.

Yield and hardening are defined in Kirchhoff stress, not silently equated to
CalculiX's finite-strain integration. Plastic flow is isochoric. Material
history is elastic left Cauchy-Green plus accumulated equivalent plastic strain.
"""
import numpy as np


def constants(c):
    E,nu=c['young_mpa'],c['poisson']
    if not E>0 or not -.99<nu<.5:raise ValueError('Invalid elastic constants')
    if not np.isfinite(c['yield_mpa']) or not np.isfinite(c['hardening_mpa']) or c['yield_mpa']<=0 or c['hardening_mpa']<0:raise ValueError('Invalid plastic constants')
    return E/(2*(1+nu)),E/(3*(1-2*nu))


def geometry(x,t):
    a=x[t];D=(a[:,1:]-a[:,:1]).swapaxes(1,2);det=np.linalg.det(D)
    if not np.all(np.isfinite(det)) or np.any(det<=0):raise ValueError('Nonpositive tetrahedron Jacobian')
    gradients=np.empty((len(t),4,3));gradients[:,1:]=np.linalg.inv(D);gradients[:,0]=-gradients[:,1:].sum(axis=1)
    return det/6,gradients,D


def oriented(x,t):
    t=np.asarray(t,dtype=int).copy();a=x[t];negative=np.linalg.det((a[:,1:]-a[:,:1]).swapaxes(1,2))<0
    t[negative,1],t[negative,2]=t[negative,2].copy(),t[negative,1].copy()
    geometry(x,t);return t


def spectral_log(b):
    values,vectors=np.linalg.eigh(b)
    if np.any(values<=0) or not np.all(np.isfinite(values)):raise ValueError('Invalid elastic tensor')
    return .5*np.log(values),vectors


def response(b,ep,c):
    G,K=constants(c);e,Q=spectral_log(b);trace=e.sum(axis=1);dev=e-trace[:,None]/3
    tau=np.einsum('eik,ek,ejk->eij',Q,2*G*dev+K*trace[:,None],Q)
    J=np.exp(trace);sigma=tau/J[:,None,None]
    energy=G*np.sum(dev**2,axis=1)+.5*K*trace**2+c['yield_mpa']*ep+.5*c['hardening_mpa']*ep**2
    return sigma,energy,J


def advance(b,ep,increment,c):
    G,_=constants(c);trial=increment@b@increment.swapaxes(1,2);e,Q=spectral_log(trial)
    trace=e.sum(axis=1);dev=e-trace[:,None]/3;s=2*G*dev;seq=np.sqrt(1.5*np.sum(s*s,axis=1))
    dg=np.maximum(0,(seq-c['yield_mpa']-c['hardening_mpa']*ep)/(3*G+c['hardening_mpa']))
    corrected=e-dg[:,None]*1.5*s/np.maximum(seq[:,None],1e-30)
    updated=np.einsum('eik,ek,ejk->eij',Q,np.exp(2*corrected),Q)
    return updated,ep+dg


def internal_force(x,t,b,ep,c):
    volume,grad,D=geometry(x,t);sigma,energy,J=response(b,ep,c)
    local=volume[:,None,None]*np.einsum('eij,enj->eni',sigma,grad)
    force=np.zeros_like(x);np.add.at(force,t.ravel(),-local.reshape(-1,3))
    return force,volume,energy,J,D,sigma


def hex_to_tet(x,hexes):
    pattern=np.array([[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6],[0,5,1,6]])
    return oriented(x,hexes[:,pattern].reshape(-1,4))
