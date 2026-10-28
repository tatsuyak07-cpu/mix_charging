"""Fixed-shape neck refinement; retain failures and references without reruns."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import shutil
from packing_fem.neck_validation import write_neck,solve_neck,beam_benchmark


def worker(out,c,level,kind,timeout):
    try:
        write_neck(out,c,level,kind)
        result=solve_neck(out,timeout)
        print(json.dumps({'finished':out.name,'status':result['status']}),flush=True)
        return result
    except Exception as exc:
        record={'error':repr(exc),'case':out.name}
        if out.exists():(out/'verification-error.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True);return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',type=Path,help='Optional already completed C3D8I r1 reference')
    parser.add_argument('--timeout',type=float,default=1200)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    os.environ['OMP_NUM_THREADS']='1'
    c=json.loads(Path('configs/plastic-smoke.json').read_text());c.update(initial_increment=.025,max_increment=.05)
    result={}
    for kind in ('C3D8','C3D8I'):
        result['beam_'+kind]=beam_benchmark(a.output/('beam_'+kind),kind)
    variants=[(level,kind) for level in (1,2,4) for kind in ('C3D8','C3D8I')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending={}
        for level,kind in variants:
            name=f'{kind}_r{level}'
            if a.pilot and level==1 and kind=='C3D8I':
                previous=json.loads((a.pilot/'metadata.json').read_text())
                if previous['config']!=c or previous['element_type']!=kind or previous['geometry']['refinement_factor']!=level:
                    raise ValueError('Pilot conditions do not match; refusing a mixed comparison')
                shutil.copytree(a.pilot,a.output/name)
                (a.output/name/'reused-reference.json').write_text(json.dumps({'source':str(a.pilot)},indent=2)+'\n')
                result[name]=json.loads((a.output/name/'summary.json').read_text());continue
            future=pool.submit(worker,a.output/name,c,level,kind,a.timeout);pending[future]=name
        for future in as_completed(pending):result[pending[future]]=future.result()
    (a.output/'verification-results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
