"""Geometry-only audit of retained contact cells; no coupled-history claim."""
import argparse,json,time
from pathlib import Path
import numpy as np
from packing_fem import surface_patch as patch
from tools.verify_onset_budget import capture_fixtures,errors,write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
    folder=capture_fixtures(a.output,a.run);fixtures=json.loads((folder/'index.json').read_text());rows=[]
    for face in sorted(set(f['face'] for f in fixtures)):
        selected=[f for f in fixtures if f['face']==face];partition=None;last=None
        for stage,f in enumerate(selected+[selected[0]]):
            with np.load(folder/f['file']) as z:source=z['source'];targets=z['targets'];mapping=z['mapping']
            begin=time.perf_counter()
            last=patch.integrate(source,targets,f['reference_area_mm2'],kn=10.,vertex_map=mapping,groups=f['groups'],pointwise=True,
                                 tolerance=.00025,error_budget='normalized',initial_partition=partition,compress_rule=False)
            partition=last['partition']
            print(json.dumps(dict(face=face,stage=stage,cells=len(partition),points=len(last['weights']),wall_s=time.perf_counter()-begin)),flush=True)
        np.savez_compressed(a.output/f'face-{face}-rule.npz',**{k:last[k] for k in ('bary','weights','partition')})
        for f in selected:
            with np.load(folder/f['file']) as z:source=z['source'];targets=z['targets'];mapping=z['mapping']
            common=dict(kn=10.,vertex_map=mapping,groups=f['groups'],pointwise=True,error_budget='normalized')
            ref=patch.integrate(source,targets,f['reference_area_mm2'],tolerance=.000015625,**common)
            value=last['weights']@patch.grouped_field(last['bary'],source,targets,f['groups'],10.,.02,mapping,True)[0]
            e=errors(value,ref['value']);rows.append(dict(face=face,extra_macro_compression=f['extra_macro_compression'],**e,passed=max(e['energy_error'],e['force_error'])<.001))
            write_csv(a.output/'comparison.csv',rows)
    result=dict(scope=__doc__,passed=all(r['passed'] for r in rows),cases=rows,wall_s=time.perf_counter()-start,
                limitations=['Saved-pose geometry checks, not a coupled advance, convergence or active-history remapping.','Finest adaptive integral is a numerical reference.'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:raise SystemExit(2)


if __name__=='__main__':main()
