"""Compare completed loaded paths at common strains; incomplete runs never pass."""
import argparse,csv,json
from pathlib import Path
import numpy as np


def read_case(item):
    run=Path(item['run']);record=dict(item,completed=False,audit_passed=False,periodic_audit_passed=False)
    if not (run/'config.json').exists():record['reason']='Not run';return record,None,None
    c=json.loads((run/'config.json').read_text())
    status=json.loads((run/'status.json').read_text()) if (run/'status.json').exists() else {}
    record['status']=status
    if not (run/'history.csv').exists():record['reason']='No accepted loading increment';return record,None,c
    with (run/'history.csv').open() as f:rows=[r for r in csv.DictReader(f) if r['phase']=='compression']
    if not rows:record['reason']='No accepted loading increment';return record,None,c
    curve=np.array([[float(r[k]) for k in ['compression','porosity','nominal_pressure_MPa']] for r in rows])
    if np.any(np.diff(curve[:,0])<=0):raise ValueError('Compression curve is not strictly ordered')
    record.update(accepted_states=len(rows),last_compression=float(curve[-1,0]),last_porosity=float(curve[-1,1]))
    record['completed']=bool(status.get('status')=='loaded_target_reached' and c['target_final_porosity']==.1 and curve[-1,1]<=.1+c['porosity_tolerance'])
    for key,name in [('audit_passed','audit.json'),('periodic_audit_passed','periodic-audit.json')]:
        path=run/name;record[key]=bool(path.exists() and json.loads(path.read_text()).get('passed',False))
    return record,curve,c


def assess(spec,out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    definition=json.loads(Path(spec).read_text());families={}
    physical=['young_mpa','poisson','yield_mpa','hardening_mpa','friction','length_mm','rigid_diameter_mm','rigid_height_mm','rigid_segments','rigid_center_mm','transverse_boundary','ipc_dhat_mm','ipc_stiffness_mpa','ipc_slip_regularization_mm','implicit_force_tolerance_N','porosity_tolerance']
    for name,items in definition['families'].items():
        loaded=[read_case(item) for item in items];records=[r for r,_,_ in loaded];curves=[v for _,v,_ in loaded];configs=[c for _,_,c in loaded if c is not None]
        same_physics=bool(configs and all(all(c.get(k)==configs[0].get(k) for k in physical) for c in configs))
        family=dict(cases=records,same_physics=same_physics,three_levels_defined=len(items)>=3,three_levels_available=bool(len(items)>=3 and all(v is not None and len(v)>=2 for v in curves)),passed=False)
        if len(items)>=3 and all(v is not None and len(v)>=2 for v in curves):
            lo=max(v[0,0] for v in curves);hi=min(v[-1,0] for v in curves)
            if hi>lo:
                strains=np.linspace(lo,hi,definition['common_strain_samples']);samples=[np.column_stack([np.interp(strains,v[:,0],v[:,j]) for j in [1,2]]) for v in curves]
                differences=[]
                for first,second in zip(samples,samples[1:]):
                    differences.append(dict(porosity_maximum_absolute=float(abs(second[:,0]-first[:,0]).max()),pressure_maximum_relative=float(abs(second[:,1]-first[:,1]).max()/max(abs(second[:,1]).max(),1e-12))))
                tolerance=definition['developer_screening_thresholds'];last=differences[-1]
                trend=all(differences[-1][k]<=differences[-2][k]+1e-12 for k in last)
                family.update(common_compression_interval=[float(lo),float(hi)],adjacent_differences=differences,errors_decrease=trend)
                family['passed']=bool(same_physics and trend and all(r['completed'] and r['audit_passed'] and r['periodic_audit_passed'] for r in records) and last['porosity_maximum_absolute']<=tolerance['porosity_absolute'] and last['pressure_maximum_relative']<=tolerance['pressure_relative'])
        families[name]=family
    report=dict(passed=bool(families and all(f['passed'] for f in families.values())),threshold_source='Developer-defined screening thresholds, not a user material specification or mathematical error bound.',definition=definition,families=families)
    out.write_text(json.dumps(report,indent=2)+'\n');return report['passed']


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if not assess(a.spec,a.out):raise SystemExit(1)
