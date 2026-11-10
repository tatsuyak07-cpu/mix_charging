"""Replay nonlinear rigid/body contact with successively enlarged periodic tiles."""
import argparse,gc,json
from pathlib import Path
import numpy as np
import ipctk as ipc
from packing_fem.rigid_inclusion import make_periodic,make_contact
from packing_fem.explicit_periodic import deformation
from packing_fem.ipc_periodic import free_surface
from tools.audit_rigid_compaction import load,predecessor


def audit(run,step,out):
    run=Path(run);out=Path(out)
    if out.exists():raise FileExistsError(out)
    c=json.loads((run/'config.json').read_text());d=load(run/f'state-{step:04d}.npz')
    old,origin=predecessor(run,step,d,load(run/f'state-{step-1:04d}.npz'))
    p=make_periodic(d['X'],c['length_mm'],c);q=d['q'];inverse=np.linalg.inv(deformation(q[-6:]));records=[];forces=[]
    required=None
    for image_radius in [1,2]:
        contact=make_contact(d['X'],d['t'],p,c,radius=image_radius)
        x=contact.positions(q);value,force,_,count=contact.evaluate_coupled(q,old['q'])
        n=contact.last_normal;barrier=float(contact.barrier(n,contact.mesh,x));f,t=contact.rigid_resultant(q)
        if image_radius==1:
            # Every replicated surface has identical deformed fractional span;
            # inspect one body image and one unstrained rigid image separately.
            body=d['x'][np.unique(free_surface(d['X'],d['t'],p))]
            center,(rotation,_,_)=p.pose(q)
            rigid=contact.local[:contact.rigid_local_count]@rotation.T+center
            widths=np.ptp(np.vstack([body,rigid])@inverse.T/p.L,axis=0)
            widths+=c['ipc_dhat_mm']*np.linalg.norm(inverse,axis=1)/p.L
            required=max(1,int(np.floor(widths.max())))
            if required>1:raise ValueError('Production radius1 lacks geometric coverage; enlarged audit alone cannot repair the saved trajectory')
        records.append(dict(image_radius=image_radius,images=(2*image_radius+1)**3,contacts=count,rigid_contacts=contact.rigid_contact_count(),barrier_energy_Nmm=barrier,contact_potential_Nmm=float(value),rigid_force_N=f.tolist(),rigid_torque_Nmm=t.tolist(),intersections=bool(ipc.has_intersections(contact.mesh,x))))
        forces.append(force.copy());del contact,x,n;gc.collect()
    difference=float(abs(forces[1]-forces[0]).max());energy_error=abs(records[1]['barrier_energy_Nmm']-records[0]['barrier_energy_Nmm'])/max(abs(records[1]['barrier_energy_Nmm']),1e-15)
    passed=not any(r['intersections'] for r in records) and records[0]['contacts']==records[1]['contacts'] and records[0]['rigid_contacts']==records[1]['rigid_contacts'] and difference<1e-6 and energy_error<1e-7
    result=dict(passed=bool(passed),state=str(run/f'state-{step:04d}.npz'),predecessor=origin,required_relative_shift_bound=required,nonlinear_actual_displacement=True,generalized_force_max_difference_N=difference,barrier_energy_relative_difference=energy_error,tiles=records)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True);return passed


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--run',required=True);a.add_argument('--step',required=True,type=int);a.add_argument('--out',required=True);args=a.parse_args()
    if not audit(args.run,args.step,args.out):raise SystemExit(1)
