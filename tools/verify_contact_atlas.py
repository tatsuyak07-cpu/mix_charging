"""Recheck every retained RVE pose against a finer independent quadrature."""
import argparse,json,time
from pathlib import Path
import numpy as np
from packing_fem.contact_atlas import build,density,scales
from packing_fem import surface_patch as patch
from tools.verify_onset_budget import write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--face',type=int);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter();index=json.loads((a.fixtures/'index.json').read_text());rows=[];rules={}
    for face in sorted({f['face'] for f in index}):
        if a.face is not None and face!=a.face:continue
        fixtures=[f for f in index if f['face']==face];poses=[];previous=None
        for f in fixtures:
            with np.load(a.fixtures/f['file']) as z:pose=dict(source=z['source'].tolist(),targets=z['targets'].tolist(),mapping=z['mapping'].tolist(),groups=f['groups'])
            poses.append(pose);rule=build(poses,f['reference_area_mm2'],10.,.02,previous_bary=previous);previous=rule['bary']
            print(json.dumps(dict(face=face,poses=len(poses),points=len(rule['weights']),candidates=rule['candidate_points'])),flush=True)
        rules[str(face)]={k:rule[k].tolist() for k in ('bary','weights')};rules[str(face)]['atlas']=poses
        (a.output/'rules.json').write_text(json.dumps(rules)+'\n')
        for f,pose in zip(fixtures,poses):
            ref=patch.integrate(np.asarray(pose['source']),np.asarray(pose['targets']),f['reference_area_mm2'],kn=10.,vertex_map=np.asarray(pose['mapping']),groups=pose['groups'],pointwise=True,error_budget='normalized',tolerance=.001/64)
            value=rule['weights']@density(rule['bary'],pose,10.,.02);error=abs(value-ref['value'])/scales(ref['value'])
            rows.append(dict(face=face,extra_macro_compression=f['extra_macro_compression'],points=len(rule['weights']),energy_error=float(error[0]),force_error=float(error[1:].max()),passed=float(error.max())<.001))
            write_csv(a.output/'comparison.csv',rows)
    (a.output/'rules.json').write_text(json.dumps(rules)+'\n')
    result=dict(scope=__doc__,passed=all(r['passed'] for r in rows),cases=rows,wall_s=time.perf_counter()-start,
                limitations=['Normal cubature on retained geometry only; full coupled replay and tangential-history sensitivity are separate checks.','No certification of unseen poses; every coupled step still audits its applied normal rule.'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
