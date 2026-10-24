"""Replay the last rejected contact evaluation with committed friction history."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
from packing_fem.area_contact import AreaContact
from packing_fem.explicit_periodic import Periodic,deformation
from packing_fem.explicit_solver import lumped_mass
from packing_fem.contact_audit import plain


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads((a.source/'config.json').read_text());s=json.loads((a.source/'summary.json').read_text())
    retries=json.loads((a.source/'contact-retries.json').read_text());last=retries[-1]
    capture=json.loads((a.source/'contact-audit.json').read_text());audit=np.loadtxt(a.source/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1)
    with np.load(a.source/'last-state.npz') as z:d={k:z[k] for k in z.files}
    X=d['reference_points'];t=d['tets'];q=d['generalized_position'];v=d['generalized_velocity']
    periodic=Periodic(X,c['length_mm']);periodic.prescribed_active=False;periodic.free=np.arange(periodic.ndof)
    periodic.mass(lumped_mass(t,d['reference_volume'],c['density_tonne_mm3'],len(X)))
    lastdt=audit[-1,0]-audit[-2,0];dt=last['trial_dt_s']
    f=np.array(capture['solver_internal_generalized_force']);external=np.array(capture['external_generalized_force'])
    acceleration=periodic.acceleration(f+external,v,0.,c['damping_per_s'])/(1-.5*lastdt*c['damping_per_s'])
    trialq=q+dt*v+.5*dt*dt*acceleration;x=periodic.positions(trialq);vhalf=v+.5*dt*acceleration
    H=deformation(trialq[-6:])*periodic.L;Hdot=(deformation(vhalf[-6:])-np.eye(3))*periodic.L
    contact=AreaContact(X,t,periodic,c['contact_penalty_mpa_mm'],c['friction'],c['contact_skin_mm'],thickness=c['barrier_thickness_mm'],max_candidates=c['contact_candidate_budget'],multiple_patches=c['multiple_patches'],edge_stationarity_filter=c['edge_stationarity_filter'],local_material_filter=c.get('local_material_filter',True),closest_tie_rule=c['closest_tie_rule'])
    contact.history={tuple(item['key']):{k:np.array(v) if isinstance(v,list) else v for k,v in item['state'].items()} for item in capture['committed_history']}
    # The certificate depends on the accepted separation and periodic images,
    # not just the force/history. Restore them from the accepted geometry/log.
    H0=deformation(q[-6:])*periodic.L
    contact.guard.prepare_geometry(d['points'],H0)
    contact.guard.accepted_geometry=(d['points'].copy(),H0.copy())
    contact.guard.accepted_shifts=tuple(contact.guard.trial_shifts)
    history=np.loadtxt(a.source/'history.csv',delimiter=',',skiprows=1)
    contact.guard.minimum_separation=float(history[-1,23])
    replayed=None
    try:contact.evaluate(x,periodic.velocities(vhalf),H,Hdot,dt)
    except ValueError as exc:replayed=str(exc)
    result=plain(dict(scope=__doc__,last_rejection=last,replayed_error=replayed,
                      same_accepted_time=abs(last['accepted_time_s']-audit[-1,0])<1e-18,
                      rejection_reproduced=replayed==last['reason'],retry_reason_counts=dict(Counter(r['reason'] for r in retries)),
                      solver_failure=s['failure'],time_advanced=False,physical_solution_verified=False))
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(a.output/'trial-geometry.npz',reference_points=X,trial_points=x,tets=t,cell=H)
    (a.output/'diagnose_last_contact_retry.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
