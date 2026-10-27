"""Independent small-strain sparse periodic stiffness reference (not compaction)."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu
from .explicit_material import geometry,constants
from .explicit_periodic import Periodic


def solve(X,t,c,pressure):
    V,g,_=geometry(X,t);G,K=constants(c);lam=K-2*G/3
    C=lam*np.einsum('eia,ejb->eiajb',g,g)
    C+=G*np.einsum('eij,ab->eiajb',np.einsum('eic,ejc->eij',g,g),np.eye(3))
    C+=G*np.einsum('eib,eja->eiajb',g,g);blocks=(C*V[:,None,None,None,None]).reshape(-1,12,12)
    dofs=(3*t[:,:,None]+np.arange(3)).reshape(-1,12);rows=np.repeat(dofs,12,axis=1).ravel();cols=np.tile(dofs,(1,12)).ravel()
    full=coo_matrix((blocks.ravel(),(rows,cols)),shape=(3*len(X),3*len(X))).tocsr();p=Periodic(X,c['length_mm']);reduced=(p.A.T@full@p.A).tocsc();n=p.nmicro
    factor=splu(reduced[:n,:n]);Km=reduced[:n,n:].toarray();relax=-factor.solve(Km);macro=reduced[n:,n:].toarray()+reduced[n:,:n]@relax
    eig=np.linalg.eigvalsh((macro+macro.T)/2)
    if eig.min()<=1e-10*eig.max():raise ValueError('RVE has an unresolved macroscopic mechanism')
    rhs=np.zeros(6);rhs[2]=-pressure*c['length_mm']**3;macroq=np.linalg.solve(macro,rhs);q=np.r_[relax@macroq,macroq];u=p.velocities(q)
    load=np.r_[np.zeros(n),rhs];residual=np.linalg.norm(reduced@q-load)/max(np.linalg.norm(load),1e-30)
    strain=(u[t].swapaxes(1,2)@g);strain=(strain+strain.swapaxes(1,2))/2
    sigma=2*G*strain+lam*np.trace(strain,axis1=1,axis2=2)[:,None,None]*np.eye(3)
    volume_average=np.einsum('e,eij->ij',V,sigma)/c['length_mm']**3
    energy=float(.5*q@(reduced@q));work=float(.5*q@load)
    report=dict(scope=__doc__,pressure_mpa=pressure,macro_stiffness_Nmm=macro.tolist(),macro_eigenvalues_Nmm=eig.tolist(),macro_strain=macroq.tolist(),volume_average_stress_mpa=volume_average.tolist(),relative_equilibrium_residual=float(residual),elastic_energy_Nmm=energy,half_load_work_Nmm=work,initial_porosity=float(1-V.sum()/c['length_mm']**3),material_strain_peak=float(np.max(abs(strain))),checks=dict(equilibrium=bool(residual<1e-10),positive_macro_stiffness=bool(eig.min()>0),symmetric_macro_stiffness=bool(np.max(abs(macro-macro.T))/np.max(abs(macro))<1e-10),elastic_work_identity=bool(abs(energy-work)<1e-10*max(abs(work),1e-30))))
    return u,report
