"""Manufactured tip impact in a periodic solid with a cubic pore.

Exercises force feedback in the compression solver, NOT quasi-static compaction.
The tip initially touches no solid; edge guard remains enabled throughout.
"""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem import sintered
from packing_fem.explicit_material import oriented,hex_to_tet
from packing_fem.explicit_solver import solve


def assess(directory):
    data=json.loads((directory/'verification.json').read_text());runs=data['runs']
    checks={
        'active_contact_and_full_path':all(r['active_force_feedback'] and r['completed'] for r in runs.values()),
        'energy_under_2percent':all(r['energy_error']<.02 for r in runs.values()),
        'penetration_under_0_02mm':all(r['max_penetration_mm']<.02 for r in runs.values()),
        'energy_improves_with_half_step':runs['half_dt']['energy_error']<runs['base']['energy_error'],
        'material_and_force_audits':all(all(r['summary']['checks'][k] for k in ('positive_J','material_volume_consistency','closed_boundary_volume','coulomb_bound','reference_volume_conserved')) for r in runs.values())}
    assessment=dict(checks=checks,all_dynamic_functional_checks_passed=all(checks.values()),
        quasistatic_validated=False,edge_edge_response_implemented=False)
    (directory/'functional-checks.json').write_text(json.dumps(assessment,indent=2)+'\n')
    if not all(checks.values()):raise RuntimeError('Dynamic integration functional check failed')
    return assessment


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    mask=np.ones((3,3,3),bool);mask[1,1,1]=False
    X,h,_=sintered.mesh(mask,1.);t=hex_to_tet(X,h)
    tip=np.array([[1.4,1.43,1.005],[1.3,1.33,1.2],[1.5,1.33,1.2],[1.4,1.53,1.2]])
    ids=np.arange(len(X),len(X)+4);X=np.r_[X,tip];t=oriented(X,np.r_[t,ids[None]])
    velocity=np.zeros_like(X);velocity[-4:,2]=-1000.
    config=json.loads(Path('configs/explicit-prototype.json').read_text())
    config.update(contact_backend='tracked_guarded',compression_fraction=.0001,ramp_s=.000015,hold_s=0.,initial_velocity_mm_s=velocity.tolist(),remesh_every=0,output_every=1)
    results={}
    for name,cfl in [('base',.2),('half_dt',.1)]:
        c={**config,'cfl':cfl};r=solve(a.output/name,X,t,c)
        history=np.genfromtxt(a.output/name/'history.csv',names=True,delimiter=',')
        results[name]=dict(summary=r,max_contacts=int(history['contacts'].max()),
            max_penetration_mm=float(history['penetration_mm'].max()),
            energy_error=r['energy_relative_error'],completed=r['checks']['path_completed'],
            active_force_feedback=bool(history['contacts'].max()>0))
        print(json.dumps(dict(case=name,**{k:v for k,v in results[name].items() if k!='summary'})),flush=True)
    (a.output/'verification.json').write_text(json.dumps(dict(purpose=__doc__,runs=results,quasistatic_validated=False),indent=2)+'\n')
    assess(a.output)

if __name__=='__main__':main()
