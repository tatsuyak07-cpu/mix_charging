"""Replay a saved failed contact trial for geometry diagnostics, not restart."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_solver import lumped_mass
from packing_fem.area_contact import AreaContact
from packing_fem.barrier_contact import feature,relative
from packing_fem.contact_audit import plain


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.source/'config.json').read_text());s=json.loads((a.source/'summary.json').read_text());capture=json.loads((a.source/'contact-audit.json').read_text());audit=np.loadtxt(a.source/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1)
    with np.load(a.source/'last-state.npz') as z:X=z['reference_points'];x=z['points'];t=z['tets'];V=z['reference_volume'];q=z['generalized_position'];v=z['generalized_velocity']
    dt=s['surface_resolution_attempts'][-1]['trial_dt_s'];lastdt=audit[-1,0]-audit[-2,0];pbc=Periodic(X,c['length_mm']);pbc.prescribed_active=False;pbc.free=np.arange(pbc.ndof);pbc.mass(lumped_mass(t,V,c['density_tonne_mm3'],len(X)))
    f=np.array(capture['solver_internal_generalized_force']);external=np.array(capture['external_generalized_force'])
    acceleration=pbc.acceleration(f+external,v,0.,c['damping_per_s'])/(1-.5*lastdt*c['damping_per_s'])
    trialq=q+dt*v+.5*dt*dt*acceleration;y=pbc.positions(trialq);H=deformation(trialq[-6:])*pbc.L
    contact=AreaContact(X,t,pbc,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'],max_candidates=c['contact_candidate_budget'],multiple_patches=c.get('multiple_patches',False),edge_stationarity_filter=c.get('edge_stationarity_filter',False),closest_tie_rule=c.get('closest_tie_rule','stop'))
    error=None
    try:contact.evaluate(y,y*0,H,H*0,0.)
    except ValueError as exc:error=dict(type=type(exc).__name__,message=str(exc),feature=getattr(exc,'feature',None))
    selected={}
    for key,ids,kind,shift,_,_ in contact.guard.neighbors:
        if kind!='pt':continue
        z=y[ids].copy();z[1:]+=H@shift;branch,w=feature(z,'pt');r=relative(z,'pt',branch);d=float(np.linalg.norm(r))
        if d>=contact.thickness:continue
        if contact.material_neighborhood.local(key,ids,'pt',shift,z,w,y,H):continue
        selected.setdefault(int(ids[0]),[]).append((d,key,ids,shift,z,r))
    failures=[]
    for node,candidates in selected.items():
        best=min(candidates,key=lambda v:(v[0],v[1]));d=best[0];tol=64*np.finfo(float).eps*max(contact.L,d);ptol=max(contact.L*1e-10,4*np.sqrt(2*d*tol));ties=[r for r in candidates if abs(r[0]-d)<=tol]
        distinct=[r for r in ties if np.linalg.norm(r[-1]-best[-1])>ptol]
        if distinct:
            rows=[]
            for r in ties:
                ids=r[2];ref=X[ids].copy();ref[1:]+=contact.L*np.asarray(r[3])
                rows.append(dict(key=list(r[1]),ids=ids.tolist(),shift=list(r[3]),distance_mm=r[0],reference_points=ref.tolist(),points=r[4].tolist(),offset=r[-1].tolist()))
            failures.append(dict(source_node=node,source_current=y[node].tolist(),source_reference=X[node].tolist(),tie_distance_tolerance_mm=tol,projection_tolerance_mm=ptol,tied_candidates=rows))
    result=plain(dict(scope=__doc__,source=str(a.source),accepted_time_s=float(audit[-1,0]),trial_time_s=float(audit[-1,0]+dt),trial_dt_s=dt,replayed_error=error,ambiguous_sources=failures,contact_force_changed=False,trial_is_solution=False))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(a.output/'trial-geometry.npz',reference_points=X,accepted_points=x,trial_points=y,tets=t,cell=H);(a.output/'diagnose_rve_stop.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
