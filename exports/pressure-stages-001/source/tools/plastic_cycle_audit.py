"""Independent face-traction force integration for unloaded contact-free states."""
import numpy as np
from packing_fem.explicit_material import response
from packing_fem.explicit_periodic import Periodic,deformation


def face_internal_force(points,tets,stress):
    p=np.asarray(points)[tets];local=np.zeros((len(tets),4,3))
    for opposite in range(4):
        face=[i for i in range(4) if i!=opposite];a,b,c=p[:,face].transpose(1,0,2);area=.5*np.cross(b-a,c-a)
        inward=np.einsum('ei,ei->e',area,p[:,opposite]-a)>0;area[inward]*=-1
        local[:,opposite]=np.einsum('eij,ej->ei',stress,area)/3
    force=np.zeros_like(points);np.add.at(force,np.asarray(tets).ravel(),local.reshape(-1,3));return force


def unloaded_face_audit(state,config,final_contacts):
    if final_contacts:
        return dict(performed=False,reason='Active friction history is not in last-state.npz; full force reconstruction would reset history and is not performed.')
    x=state['points'];t=state['tets'];X=state['reference_points'];sigma=response(state['elastic_b'],state['peeq'],config)[0]
    force=face_internal_force(x,t,sigma);periodic=Periodic(X,config['length_mm'],config.get('mass_frame','pinned'));reduced=periodic.force(force);reduced[-6:]/=periodic.L
    F=deformation(state['generalized_position'][-6:]);det=np.linalg.det((x[t][:,1:]-x[t][:,:1]).swapaxes(1,2));volume=det/6
    mean=np.einsum('e,eij->ij',volume,sigma)/(np.linalg.det(F)*periodic.L**3)
    scale=config['pressure_mpa']*periodic.L**2;ratio=float(np.linalg.norm(reduced)/scale);stress_ratio=float(np.max(abs(mean))/config['pressure_mpa'])
    return dict(performed=True,method='tetrahedral outward face tractions, each nodal force sigma*area_vector/3, periodic projection; existing constitutive stress response',force_reference_N=scale,free_force_norm_N=float(np.linalg.norm(reduced)),force_over_peak_load_scale=ratio,volume_mean_cauchy_MPa=mean.tolist(),mean_stress_over_peak=stress_ratio,checks=dict(free_force_below_2percent_peak=ratio<.02,mean_stress_below_2percent_peak=stress_ratio<.02),scope='endpoint only, no active contacts; not a whole-path static force audit')
