"""All-accepted-step diagnostics, independent of visualization output cadence."""
import numpy as np


def whole_path_audit(rows,energy_scale):
    a=np.asarray(rows,float)
    loaded=a[:,3]>.01*np.max(a[:,3]);contact=loaded&(a[:,10]>0)
    change=abs(a[:,5]-a[0,5]);total=change+a[:,8]+a[:,9]
    result=dict(sampling='every_accepted_step_including_initial',rows=len(a),pressure_window='command > 1 percent of observed peak',energy_floor_Nmm=float(energy_scale*1e-8),normalization_note='Ratios near first loading/contact can be singular as internal energy approaches zero; peaks alone do not establish convergence.')
    for name,mask in (('loaded',loaded),('contact',contact)):
        indices=np.flatnonzero(mask)
        if not len(indices):result[name]=None;continue
        kin=a[indices,4];i=int(indices[np.argmax(kin)]);ratio=kin/np.maximum(total[indices],energy_scale*1e-8)
        result[name]=dict(steps=len(indices),peak_kinetic_Nmm=float(a[i,4]),peak_kinetic_time_s=float(a[i,0]),peak_kinetic_over_energy_scale=float(a[i,4]/energy_scale),peak_kinetic_internal_ratio=float(np.max(kin/np.maximum(change[indices],energy_scale*1e-8))),peak_kinetic_internal_contact_ratio=float(np.max(ratio)))
    return result
