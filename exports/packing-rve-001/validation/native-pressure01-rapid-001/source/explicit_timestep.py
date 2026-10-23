"""Mass-weighted assembled row bound for a research explicit timestep.

The linear elastic bound is a Gershgorin upper bound. The finite-strain elastic
stretch/stress inflation is conservative by numerical choice, not a proof for
arbitrary Hencky-plastic states; independent CFL checks remain mandatory.
"""
import numpy as np
from .explicit_material import geometry,constants


def assembled_frequency_bound(x,t,mass,c,stress=None,elastic_b=None):
    V,g,_=geometry(x,t);G,K=constants(c);lam=K-2*G/3
    scale=np.ones(len(t))
    if elastic_b is not None:
        eig=np.linalg.eigvalsh(elastic_b)
        if eig.min()<=0:raise ValueError('Invalid elastic stretch in timestep estimate')
        scale=(1+np.max(np.abs(np.log(eig)),axis=1))/np.sqrt(np.prod(eig,axis=1))
    C=lam*np.einsum('eia,ejb->eiajb',g,g)
    C+=G*np.einsum('eij,ab->eiajb',np.einsum('eic,ejc->eij',g,g),np.eye(3))
    C+=G*np.einsum('eib,eja->eiajb',g,g)
    C*=scale[:,None,None,None,None]
    if stress is not None:
        geometric=2*np.linalg.norm(stress,axis=(1,2))[:,None,None]*np.einsum('eic,ejc->eij',g,g)
        C=np.abs(C)+np.abs(np.einsum('eij,ab->eiajb',geometric,np.eye(3)))
    denominator=np.sqrt(mass[t][:,:,None,None,None]*mass[t][:,None,None,:,None])
    rows=np.abs(C*V[:,None,None,None,None]/denominator).sum(axis=(3,4))
    bound=np.zeros((len(x),3));np.add.at(bound,t,rows)
    return float(bound.max())


def triangle_frequency_bound(x,t,mass,c,stress=None,elastic_b=None):
    """Triangle-inequality majorant of assembled_frequency_bound.

No constitutive or mass change. Bound the absolute row sum before assembling
3x3 blocks; may be more conservative than the original Gershgorin estimate.
    """
    V,g,_=geometry(x,t);G,K=constants(c);lam=K-2*G/3
    inflation=np.ones(len(t))
    if elastic_b is not None:
        eig=np.linalg.eigvalsh(elastic_b)
        if eig.min()<=0:raise ValueError('Invalid elastic stretch in timestep estimate')
        inflation=(1+np.max(np.abs(np.log(eig)),axis=1))/np.sqrt(np.prod(eig,axis=1))
    invmass=1/np.sqrt(mass[t]);ag=abs(g);ones=ag.sum(2)
    S1=np.sum(ones*invmass,axis=1)
    Sa=np.einsum('eja,ej->ea',ag,invmass)
    dots=abs(np.einsum('eia,eja->eij',g,g));Sd=np.einsum('eij,ej->ei',dots,invmass)
    rows=abs(lam)*ag*S1[:,None,None]+G*Sd[:,:,None]+G*ones[:,:,None]*Sa[:,None,:]
    rows*=inflation[:,None,None]
    if stress is not None:rows+=2*np.linalg.norm(stress,axis=(1,2))[:,None,None]*Sd[:,:,None]
    rows*=V[:,None,None]*invmass[:,:,None]
    assembled=np.zeros((len(x),3));np.add.at(assembled,t,rows)
    return float(assembled.max())
