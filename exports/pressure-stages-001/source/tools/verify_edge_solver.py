"""Interior edge response integration; feature handoff remains a red gate."""
import argparse,json
from pathlib import Path
import numpy as np
from packing_fem.explicit_solver import solve


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);root=Path('exports/explicit-fem-prototype-002/dynamic_impact')
    config=json.loads((root/'config.json').read_text())
    with np.load(root/'initial-mesh.npz') as mesh:X,t=mesh['points'],mesh['tets']
    results={}
    for name,scale,penalty in [('base',1.,1.),('half_dt',.5,1.),('edge_penalty2',1.,2.)]:
        c={**config,'contact_backend':'tracked_edges','cfl':config['cfl']*scale,'edge_penalty_scale':penalty,'output_every':1}
        r=solve(a.output/name,X,t,c);h=np.genfromtxt(a.output/name/'history.csv',names=True,delimiter=',')
        results[name]=dict(summary=r,max_edge_contacts=int(h['edge_contacts'].max()),
            max_penetration_mm=float(h['penetration_mm'].max()),last_time_s=float(h['time_s'][-1]),
            edge_activation_observed=bool(h['edge_contacts'].max()>0))
        print(json.dumps(dict(case=name,**{k:v for k,v in results[name].items() if k!='summary'},failure=r['failure'])),flush=True)
        (a.output/'partial-results.json').write_text(json.dumps(results,indent=2)+'\n')
    # Previously completed dynamic node-to-face test must remain usable.
    root=Path('exports/tracked-node-impact-002/base');c=json.loads((root/'config.json').read_text());c['contact_backend']='tracked_edges'
    with np.load(root/'initial-mesh.npz') as mesh:points,tets=mesh['points'],mesh['tets']
    r=solve(a.output/'node_regression',points,tets,c);h=np.genfromtxt(a.output/'node_regression/history.csv',names=True,delimiter=',')
    results['node_regression']=dict(summary=r,max_contacts=int(h['contacts'].max()),max_edge_contacts=int(h['edge_contacts'].max()))
    checks=dict(edge_activation=all(results[n]['edge_activation_observed'] for n in ('base','half_dt','edge_penalty2')),
                node_dynamic_regression=bool(r['checks']['path_completed'] and h['contacts'].max()>0 and r['energy_relative_error']<.02))
    summary=dict(runs=results,functional_checks=checks,feature_handoff_validated=False,quasistatic_compaction_validated=False)
    (a.output/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(checks),flush=True)

if __name__=='__main__':main()
