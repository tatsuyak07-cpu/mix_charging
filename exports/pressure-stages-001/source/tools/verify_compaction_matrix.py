"""Reproducible quasi-static pressure-contact sensitivity matrix.

Outputs are exclusive, all failures retained. Four subprocesses maximum, one
BLAS thread per process. A single manufactured slit is NOT packing validation.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.workers<=4:raise ValueError('Worker budget must be 1..4')
    a.output.mkdir(parents=True,exist_ok=False)
    jobs={'time8':[],'half_dt':['--cfl','.1'],'time16':['--time-scale','16'],
          'thickness_half':['--thickness','.01'],'penalty2':['--penalty-scale','2'],
          'damping_half':['--damping-scale','.5'],'mesh2_bulk':['--refinement','2','--refinement-pattern','bulk']}
    def run(name,extra):
        cmd=[sys.executable,'tools/run_barrier_compaction.py','--config','configs/explicit-prototype.json','--contact-backend','area_guarded','--output',str(a.output/name),'--time-scale','8','--timestep-method','assembled_mass_bound','--max-steps','180000',*extra]
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONPATH':'.'}
        with (a.output/f'{name}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
        summary=a.output/name/'summary.json'
        return dict(returncode=r.returncode,command=cmd,summary=json.loads(summary.read_text()) if summary.exists() else None)
    results={}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(run,name,extra):name for name,extra in jobs.items()}
        for future in as_completed(pending):
            name=pending[future]
            try:results[name]=future.result()
            except Exception as exc:results[name]={'error':repr(exc)}
            (a.output/'matrix.json').write_text(json.dumps(results,indent=2)+'\n');print(name,flush=True)

if __name__=='__main__':main()
