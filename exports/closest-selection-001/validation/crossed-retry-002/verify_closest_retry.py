"""Manufactured moving crossed-tetra test of real EE guard/refine/retry."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_material import oriented,hex_to_tet
from packing_fem.explicit_solver import solve
from packing_fem.contact_audit import plain


def run(output,strategy='closest_quality'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    X=np.array([[-1.,0,0],[1,0,0],[-1,-.1,-.1],[-1,.1,-.1],
                [0,-1,.0200005],[0,1,.0200005],[-.1,-1,.1200005],[.1,-1,.1200005]])+2
    t=oriented(X,np.arange(8).reshape(2,4))
    # A disconnected pair wholly inside the cell leaves macro/micro DOFs
    # redundant. A distant periodic frame gives the cell a real macro basis.
    from packing_fem import sintered
    grid=np.indices((8,8,8));mask=np.sum((grid==0)|(grid==7),axis=0)>=2
    frame,hexes,_=sintered.mesh(mask,5/8)
    t=np.r_[hex_to_tet(frame,hexes),t+len(frame)];X=np.r_[frame,X]
    velocity=X*0;velocity[-4:,2]=-1.
    c=json.loads(Path('configs/explicit-prototype.json').read_text())
    c.update(length_mm=5.,contact_backend='area_guarded',loading_mode='cauchy_pressure',pressure_mpa=1e-12,
             compression_fraction=0.,ramp_s=2e-6,hold_s=1e-6,damping_per_s=0.,yield_mpa=1e6,
             remesh_every=0,max_steps=2000,output_every=1,initial_velocity_mm_s=velocity.tolist(),
             surface_remesh_on_edge_guard=True,surface_remesh_max_events=16,
             surface_remesh_strategy=strategy,multiple_patches=True,edge_stationarity_filter=True,
             closest_tie_rule='clarke',active_contact_timestep=True,save_contact_audit=True)
    result=solve(output/'calculation',X,t,c)
    attempts=result['surface_resolution_attempts'];events=result['remesh_events']
    history=np.loadtxt(output/'calculation/history.csv',delimiter=',',skiprows=1)
    repaired=[a for a in attempts if a['repaired']]
    checks=dict(real_edge_guard_triggered=bool(attempts),guard_repaired=bool(repaired),
                time_advanced_after_repair=bool(repaired) and history[-1,0]>repaired[0]['accepted_time_s'],
                path_completed=result['checks']['path_completed'],
                transfer_audits_pass=bool(events) and all(e['accepted'] and all(e.get('checks',{}).values()) for e in events),
                contacts_after_repair=bool(np.max(history[:,18])>0),
                no_edge_spring=not result['edge_edge_response_implemented'],
                coulomb_bound=result['checks']['coulomb_bound'])
    report=plain(dict(scope=__doc__,checks=checks,passed=all(checks.values()),solver=result,
                physical_quasistatic_solution_verified=False,
                interpretation='Initial velocity drives a geometric/contact retry test; this is not a pressure-compaction equilibrium test. A distant periodic frame removes redundant macro DOFs.'))
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'verify_closest_retry.py').write_bytes(Path(__file__).read_bytes())
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    report=run(a.output);print(json.dumps(dict(checks=report['checks'],passed=report['passed'],failure=report['solver']['failure']),indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
