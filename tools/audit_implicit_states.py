"""Audit actual saved large-deformation states with an enlarged periodic tiling."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import ipctk as ipc
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_material import geometry,response
from packing_fem.ipc_periodic import PeriodicIPC,free_surface


def audit(paths,config,out):
    c=json.loads(Path(config).read_text());results=[]
    if Path(out).exists():raise FileExistsError('Audit output already exists')
    for path in paths:
        with np.load(path) as r:d=dict(r)
        X=d['X'];t=d['t'];p=Periodic(X,c['length_mm']);q=d['q'];parent=d.get('parent',np.arange(len(t)))
        volume=geometry(d['x'],t)[0];J=response(d['b'],d['ep'],c)[2]
        local=np.bincount(parent,weights=d['V0']*J,minlength=len(t))
        nodes=np.unique(free_surface(X,t,p));inv=np.linalg.inv(deformation(q[-6:]))
        fractional=d['x'][nodes]@inv.T/p.L
        widths=np.ptp(fractional,axis=0)+c['ipc_dhat_mm']*np.linalg.norm(inv,axis=1)/p.L
        required=max(1,int(np.floor(widths.max())))
        radius=max(2,required+1)
        contact1=PeriodicIPC(X,t,p,c,radius=1);contact2=PeriodicIPC(X,t,p,c,radius=radius)
        energies=[];gradients=[];counts=[];intersections=[]
        for a in [contact1,contact2]:
            x=a.positions(q);n=a.collisions(x)
            energies.append(a.barrier(n,a.mesh,x));gradients.append(np.asarray(a.R.T@a.barrier.gradient(n,a.mesh,x)).ravel());counts.append(len(n))
            intersections.append(bool(ipc.has_intersections(a.mesh,x)))
        row=dict(state=str(path),compression=float(-q[p.prescribed]),porosity=float(1-volume.sum()/(p.L**3*np.linalg.det(deformation(q[-6:])))),elements=len(t),history_points=len(parent),minimum_volume_mm3=float(volume.min()),material_volume_consistency=float(np.max(abs(local-volume)/volume)),required_relative_shift_bound=required,enlarged_radius=radius,contacts_radius1=counts[0],contacts_enlarged=counts[1],barrier_energy_relative_difference=float(abs(energies[1]-energies[0])/max(abs(energies[1]),1e-15)),barrier_gradient_max_difference=float(np.max(abs(gradients[1]-gradients[0]))),intersections_radius1=intersections[0],intersections_enlarged=intersections[1])
        row['passed']=bool(not any(intersections) and row['material_volume_consistency']<1e-7 and counts[0]==counts[1] and row['barrier_energy_relative_difference']<1e-7 and row['barrier_gradient_max_difference']<1e-6)
        results.append(row);print(json.dumps(row),flush=True)
        Path(out).write_text(json.dumps(results,indent=2)+'\n')
    return all(row['passed'] for row in results)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--config',required=True);a.add_argument('--out',required=True);a.add_argument('states',nargs='+');args=a.parse_args()
    if not audit(args.states,args.config,args.out):raise SystemExit(1)
