"""Incremental Hencky J2 potential and spatial consistent T4 tangent.

Uses the same Kirchhoff yield stress and history as explicit_material. Plastic
work is included in that module's incremental potential, not added twice.
"""
import numpy as np
from scipy.sparse import coo_matrix
from .explicit_material import geometry, advance, response, constants


def evaluate(x,t,old_b,old_ep,old_inverse,reference_volume,c,tangent=False,parent=None):
    volume,grad,D=geometry(x,t)
    if parent is None:parent=np.arange(len(t))
    inc=(D@old_inverse)[parent]
    b,ep=advance(old_b,old_ep,inc,c)
    sigma,energy,J=response(b,ep,c)
    tau=sigma*J[:,None,None]
    weighted_tau=np.zeros((len(t),3,3));np.add.at(weighted_tau,parent,reference_volume[:,None,None]*tau)
    local=np.einsum('eij,eaj->eai',weighted_tau,grad)
    g=np.zeros_like(x);np.add.at(g,t.ravel(),local.reshape(-1,3))
    state=dict(b=b,ep=ep,volume=volume,J=J,D=D,sigma=weighted_tau/volume[:,None,None])
    value=float(reference_volume@energy)
    if not tangent:return value,g,None,state
    G,K=constants(c);H=c['hardening_mpa'];y=c['yield_mpa']
    trial=inc@old_b@inc.swapaxes(1,2)
    lam,Q=np.linalg.eigh(trial);loglam=np.log(lam)
    h=np.einsum('eik,ek,ejk->eij',Q,.5*loglam,Q)
    tr=np.trace(h,axis1=1,axis2=2);s=2*G*(h-tr[:,None,None]*np.eye(3)/3)
    seq=np.sqrt(1.5*np.sum(s*s,axis=(1,2)))
    plastic=seq>y+H*old_ep
    a=np.ones(len(parent));da=np.zeros(len(parent))
    a[plastic]=H/(3*G+H)+3*G/(3*G+H)*(y+H*old_ep[plastic])/seq[plastic]
    da[plastic]=-3*G/(3*G+H)*(y+H*old_ep[plastic])/seq[plastic]**2
    delta=lam[:,:,None]-lam[:,None,:]
    close=abs(delta)<1e-7*np.maximum(lam[:,:,None],lam[:,None,:])
    ratio=np.divide(loglam[:,:,None]-loglam[:,None,:],delta,out=np.zeros_like(delta),where=~close)
    ratio[close]=np.broadcast_to(2/(lam[:,:,None]+lam[:,None,:]),delta.shape)[close]
    blocks=np.empty((len(t),12,12))
    for node in range(4):
        for axis in range(3):
            L=np.zeros_like(trial);L[:,axis,:]=grad[parent,node,:]
            dB=L@trial+trial@L.swapaxes(1,2)
            dh=.5*Q@(ratio*(Q.swapaxes(1,2)@dB@Q))@Q.swapaxes(1,2)
            dtr=np.trace(dh,axis1=1,axis2=2)
            dq=3*G*np.sum(s*dh,axis=(1,2))/np.maximum(seq,1e-30)
            dtau=K*dtr[:,None,None]*np.eye(3)+2*G*a[:,None,None]*(dh-dtr[:,None,None]*np.eye(3)/3)+ (da*dq)[:,None,None]*s
            qp_grad=grad[parent]
            dg=reference_volume[:,None,None]*(np.einsum('eij,eaj->eai',dtau,qp_grad)-np.einsum('eij,ekj,eak->eai',tau,L,qp_grad))
            column=np.zeros((len(t),4,3));np.add.at(column,parent,dg)
            blocks[:,:,3*node+axis]=column.reshape(-1,12)
    blocks=.5*(blocks+blocks.swapaxes(1,2))
    if c.get('implicit_project_material_tangent',True):
        eig,U=np.linalg.eigh(blocks)
        blocks=(U*np.maximum(eig,0)[:,None,:])@U.swapaxes(1,2)
    ids=(3*t[:,:,None]+np.arange(3)).reshape(-1,12)
    rows=np.broadcast_to(ids[:,:,None],blocks.shape).ravel();cols=np.broadcast_to(ids[:,None,:],blocks.shape).ravel()
    stiffness=coo_matrix((blocks.ravel(),(rows,cols)),shape=(x.size,x.size)).tocsr()
    return value,g,stiffness,state
