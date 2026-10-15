"""Run same-geometry periodic C3D8/C3D8I trials with observable restart states."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib,json,os,shutil
from pathlib import Path
import numpy as np
from packing_fem import sintered
from packing_fem.incompatible import write_model,run


def worker(source,out,base,kind,steps,timeout):
    try:
        out.mkdir(parents=True,exist_ok=False)
        c={**base,'element_type':kind,'compression_steps':steps,'control':'displacement'}
        for name in ('sintered-geometry.npz','estimate.json'):shutil.copyfile(source/name,out/name)
        with np.load(out/'sintered-geometry.npz') as g:p,e=g['points'],g['elements']
        est=json.loads((out/'estimate.json').read_text())
        (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
        write_model(out,c,p,e,np.rint(p/c['mesh_mm']).astype(int),est,True)
        (out/'provenance.json').write_text(json.dumps({'geometry_source':str(source),'geometry_sha256':hashlib.sha256((out/'sintered-geometry.npz').read_bytes()).hexdigest(),'input_sha256':hashlib.sha256((out/'plastic.inp').read_bytes()).hexdigest()},indent=2)+'\n')
        r=run(out,timeout)
        print(json.dumps({'finished':out.name,'status':r['status'],'checks':r['checks']}),flush=True)
        return r
    except Exception as exc:
        result={'case':out.name,'error':repr(exc)}
        if out.exists():(out/'verification-error.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--timeout',type=float,default=1800)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);os.environ['OMP_NUM_THREADS']='1'
    base=json.loads(Path('configs/plastic-smoke.json').read_text());base['length_mm']=25.
    source=a.output/'geometry';source.mkdir()
    points,elements,indices,estimate=sintered.prepare(source,base,True)
    (source/'config.json').write_text(json.dumps(base,indent=2)+'\n')
    (source/'size-selection.json').write_text(json.dumps({'purpose':'small periodic implementation verification, not RVE-size convergence','excluded_20mm':{'particles':6,'winding_rank':2,'reason':'does not percolate in three independent directions'},'selected_length_mm':25,'selected_winding_rank':estimate['winding_rank'],'original_100mm_specification_unchanged':True},indent=2)+'\n')
    result={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={pool.submit(worker,source,a.output/f'{kind}_n{steps}',base,kind,steps,a.timeout):f'{kind}_n{steps}'
              for kind,steps in [('C3D8',20),('C3D8I',20),('C3D8I',40)]}
        for job in as_completed(jobs):result[jobs[job]]=job.result()
    (a.output/'verification-results.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
