"""Finite-strain work audit: macro and micro first Piola work vs solver ELSE.

ELSE is accumulated internal work, not recoverable elastic energy. A work
balance is not a check of calibration, self-contact, or mesh convergence.
"""
import json
from pathlib import Path
import numpy as np
from .continuum import deformation_gradients,stress_tensors
from .sintered import hex_quadrature
from .results import read_dat,find


def audit(run,output):
    run=Path(run);output=Path(output)
    if (run/'initial-mesh.npz').exists() or (run/'model.json').exists():raise ValueError('Experimental explicit results require their own history/energy audit')
    if (run/'mode-metadata.json').exists():raise ValueError('Internal-mode runs require incompatible.analyze, not the eight-node work audit')
    output.mkdir(parents=True,exist_ok=False)
    m=json.loads((run/'plastic-metadata.json').read_text());c=m['config'];L=c['length_mm']
    with np.load(run/'sintered-geometry.npz') as g:p,e=g['points'],g['elements']
    w0=hex_quadrature(p,e);eye=np.eye(3)
    previousF=np.broadcast_to(eye,w0.shape+(3,3)).copy();previousP=np.zeros_like(previousF)
    macroF=eye.copy();macroP=np.zeros((3,3));micro_work=macro_work=0.;rows=[]
    prior_ep=np.zeros_like(w0);plastic_monotone=True;hardening_integral=0.
    for t,s in read_dat(run/'plastic.dat'):
        u=find(s,'displacements','PHYSICAL');u=u[np.argsort(u[:,0])][:,1:4]
        q=find(s,'displacements','MACRO');q=q[np.argsort(q[:,0])][:,1]/m['macro_scale']/L
        F=eye+np.array([[q[0],q[3],q[4]],[q[3],q[1],q[5]],[q[4],q[5],q[2]]])
        flocal=deformation_gradients(p,p+u,e);jlocal=np.linalg.det(flocal)
        sigma=find(s,'stresses','SOLID');sigma=sigma[np.lexsort((sigma[:,1],sigma[:,0]))]
        tensor=stress_tensors(sigma[:,2:8].reshape(-1,8,6))
        plocal=jlocal[...,None,None]*tensor@np.linalg.inv(flocal).swapaxes(-1,-2)
        pbar=(plocal*w0[...,None,None]).sum(axis=(0,1))/L**3
        micro_work+=float(np.sum(.5*(previousP+plocal)*(flocal-previousF)*w0[...,None,None]))
        macro_work+=float(L**3*np.sum(.5*(macroP+pbar)*(F-macroF)))
        ep=find(s,'equivalent plastic strain','SOLID');ep=ep[np.lexsort((ep[:,1],ep[:,0]))][:,-1].reshape(-1,8)
        plastic_monotone &= bool(np.all(ep>=prior_ep-1e-7))
        hardening_integral=float(np.sum((c['yield_mpa']*ep+.5*c['hardening_mpa']*ep**2)*w0))
        try:internal=float(find(s,'total internal energy','SOLID').ravel()[-1])
        except ValueError:internal=float('nan')
        rows.append([t,macro_work,micro_work,internal,hardening_integral])
        previousF=flocal;previousP=plocal;macroF=F;macroP=pbar;prior_ep=ep
    if not rows:raise ValueError('No energy audit frames')
    a=np.array(rows);scale=max(float(np.max(np.abs(a[:,3]))),1e-14)
    macro_micro=float(np.max(np.abs(a[:,1]-a[:,2]))/scale)
    compress=a[:,0]<=1+1e-8
    result={'energy_units':'N mm','macro_work_Nmm':float(a[-1,1]),'micro_work_Nmm':float(a[-1,2]),
            'solver_internal_energy_Nmm':float(a[-1,3]),
            'macro_micro_max_relative_difference':macro_micro,
            'compression_work_internal_relative_difference':float(abs(a[compress][-1,2]-a[compress][-1,3])/max(abs(a[compress][-1,3]),1e-14)),
            'cycle_work_internal_relative_difference':float(abs(a[-1,2]-a[-1,3])/scale),
            'work_reference_peak_internal_energy_Nmm':scale,
            'plastic_equivalent_strain_monotone':plastic_monotone,
            'isotropic_hardening_integral_Nmm':hardening_integral,
            'note':'Hardening integral includes plastic work and stored hardening; not independently resolved dissipation. Unload boundary release may undersample the unloading path.',
            'criteria':{'macro_micro_relative':.01,'work_internal_relative':.02},
            'checks':{'finite_solver_energy':bool(np.all(np.isfinite(a[:,3]))),'macro_micro_work':macro_micro<.01,
                      'compression_work_balance':bool(abs(a[compress][-1,2]-a[compress][-1,3])<.02*abs(a[compress][-1,3])),
                      'cycle_work_balance':bool(abs(a[-1,2]-a[-1,3])<.02*scale),
                      'plastic_accumulation':plastic_monotone}}
    np.savetxt(output/'energy-history.csv',a,delimiter=',',header='step_time,macro_work_Nmm,micro_work_Nmm,solver_internal_energy_Nmm,hardening_integral_Nmm',comments='')
    (output/'energy-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
