"""Replay a converged increment's constitutive and actual Coulomb force balance."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_material import geometry
from packing_fem.explicit_periodic import Periodic
from packing_fem.implicit_material import evaluate
from packing_fem.ipc_periodic import PeriodicIPC
from packing_fem.friction_audit import friction_report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--step',type=int,required=True);parser.add_argument('--out',required=True);parser.add_argument('--previous-state',help='Actual saved post-remesh predecessor, when the material chart changed');a=parser.parse_args()
    run=Path(a.run);out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    c=json.loads((run/'config.json').read_text())
    previous=Path(a.previous_state) if a.previous_state else run/f'state-{a.step-1:04d}.npz'
    with np.load(previous) as f:old=dict(f)
    with np.load(run/f'state-{a.step:04d}.npz') as f:new=dict(f)
    if any(not np.array_equal(old[key],new[key]) for key in ['X','t','parent','V0']):raise ValueError('Replay needs consecutive states with matching material chart; use the saved post-remesh predecessor if topology changed.')
    p=Periodic(new['X'],c['length_mm']);inverse=np.linalg.inv(geometry(old['x'],old['t'])[2])
    _,material,_,state=evaluate(new['x'],new['t'],old['b'],old['ep'],inverse,old['V0'],c,parent=old['parent'])
    contact=PeriodicIPC(new['X'],new['t'],p,c)
    _,contact_force,_,count=contact.evaluate_coupled(new['q'],old['q'])
    force=np.asarray(p.A.T@material.ravel()).ravel()+contact_force
    with (run/'history.csv').open() as f:row=next(r for r in csv.DictReader(f) if int(r['step'])==a.step)
    residual=float(np.max(abs(force[:p.nmicro]),initial=0.));pressure=float(-force[p.prescribed]/p.L**3)
    report=dict(step=a.step,contacts=count,free_residual_N=residual,saved_free_residual_N=float(row['free_residual_N']),nominal_pressure_MPa=pressure,pressure_replay_difference_MPa=abs(pressure-float(row['nominal_pressure_MPa'])),material_b_replay_max_error=float(np.max(abs(state['b']-new['b']))),plastic_strain_replay_max_error=float(np.max(abs(state['ep']-new['ep']))),force_acceptance_N=c['implicit_force_tolerance_N'])
    report.update(free_residual_l2_N=float(np.linalg.norm(force[:p.nmicro])),residual_l2_to_axial_reaction=float(np.linalg.norm(force[:p.nmicro])/max(abs(force[p.prescribed]/p.L),1e-12)),l2_replay_difference_N=abs(float(np.linalg.norm(force[:p.nmicro]))-float(row['free_residual_l2_N'])))
    gauge=-force[:p.nmicro].reshape(-1,3).sum(axis=0)
    report.update(predecessor_state=str(previous),gauge_reaction_N=gauge.tolist(),gauge_reaction_norm_N=float(np.linalg.norm(gauge)),gauge_reaction_to_axial_reaction=float(np.linalg.norm(gauge)/max(abs(force[p.prescribed]/p.L),1e-12)))
    report['passed']=bool(report['l2_replay_difference_N']<1e-8 and residual<c['implicit_force_tolerance_N'] and report['pressure_replay_difference_MPa']<1e-8 and report['material_b_replay_max_error']<1e-7 and report['plastic_strain_replay_max_error']<1e-7)
    report['coulomb']=friction_report(contact,new['q'],old['q'],c['friction'])
    report['passed']=report['passed'] and report['coulomb']['passed']
    out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
