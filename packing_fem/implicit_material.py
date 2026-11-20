"""Incremental Hencky J2 potential and spatial consistent T4 tangent.

Uses the same Kirchhoff yield stress and history as explicit_material. Plastic
work is included in that module's incremental potential, not added twice.
"""
import numpy as np
from scipy.sparse import coo_matrix
from .explicit_material import geometry, advance, response, constants


def consistent_blocks(grad,Q,lam,ratio,s,seq,a,da,tau,reference_volume,parent,G,K):
    """Assemble the same spatial tangent in principal axes, in bounded chunks.

    For L=e_k tensor grad(N_b), the logarithmic variation in principal axes is
    dh_ij=ratio_ij*(Q_ki*w_bj*lam_j+lam_i*w_bi*Q_kj)/2.
    Contracting this expression before rotating avoids twelve repeated full
    tensor variations per quadrature point. The geometric stress term uses the
    original spatial tau, and subcell weights are accumulated without averaging.
    """
    blocks=np.zeros((len(grad),12,12))
    for start in range(0,len(parent),8192):
        sl=slice(start,start+8192);ids=parent[sl];basis=Q[sl]
        gradients=grad[ids];w=gradients@basis
        # s is diagonal in this basis. Compute it from the original tensor to
        # retain exactly the return mapping's deviatoric stress convention.
        principal_s=np.einsum('eji,ejk,eki->ei',basis,s[sl],basis)
        ga=G*a[sl]
        plastic=(3*G*da[sl]/np.maximum(seq[sl],1e-30))[:,None,None]
        coefficient=(K-2*ga/3)[:,None,None]+plastic*principal_s[:,:,None]*principal_s[:,None,:]
        principal=np.einsum('eik,eai,ebk->eaibk',coefficient,w,w)
        coefficient=ga[:,None,None]*ratio[sl]*lam[sl,:,None]
        principal+=np.einsum('eik,ebi,eak->eaibk',coefficient,w,w)
        diagonal=np.einsum('ein,ean,ebn->eabi',ga[:,None,None]*ratio[sl]*lam[sl,None,:],w,w)
        for axis in range(3):principal[:,:,axis,:,axis]+=diagonal[:,:,:,axis]
        spatial=basis[:,None,None]@principal.transpose(0,1,3,2,4)@basis[:,None,None].swapaxes(-1,-2)
        spatial=spatial.transpose(0,1,3,2,4)
        stress_gradient=np.einsum('eij,ebj->ebi',tau[sl],gradients)
        spatial-=stress_gradient.swapaxes(1,2)[:,None,:,:,None]*gradients[:,:,None,None,:]
        spatial*=reference_volume[sl,None,None,None,None]
        np.add.at(blocks,ids,spatial.reshape(-1,12,12))
    return .5*(blocks+blocks.swapaxes(1,2))


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
    blocks=consistent_blocks(grad,Q,lam,ratio,s,seq,a,da,tau,reference_volume,parent,G,K)
    if c.get('implicit_project_material_tangent',True):
        eig,U=np.linalg.eigh(blocks)
        blocks=(U*np.maximum(eig,0)[:,None,:])@U.swapaxes(1,2)
    ids=(3*t[:,:,None]+np.arange(3)).reshape(-1,12)
    rows=np.broadcast_to(ids[:,:,None],blocks.shape).ravel();cols=np.broadcast_to(ids[:,None,:],blocks.shape).ravel()
    stiffness=coo_matrix((blocks.ravel(),(rows,cols)),shape=(x.size,x.size)).tocsr()
    return value,g,stiffness,state
