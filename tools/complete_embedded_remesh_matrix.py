"""Resume independent verification cases in separate bounded worker processes."""
import argparse,csv,json,shutil,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from tools.verify_embedded_remesh import verify


def worker(case,root):
    fraction,slip=case;target=Path(root)/f'fraction-{fraction}-slip-{slip}';target.mkdir()
    try:
        result,old,new,p,state=verify(fraction,slip,1,target)
        if fraction==.5 and slip==.01:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import numpy as np
            fig,axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
            for ax,contact,z,title in ((axes[0],old,state[1],'Before'),(axes[1],new,p['x'],'After nested bisection')):
                for ids in contact.faces:
                    if not np.all(np.isclose(contact.X[ids,2],.6)):continue
                    tri=z[np.r_[ids,ids[0]]];ax.plot(tri[:,0],tri[:,1],color='black',lw=1)
                nodes=np.flatnonzero((contact.inner.integration_area0>0)&np.isclose(contact.inner.X[:,2],.6));samples=contact.P@z
                ax.scatter(samples[nodes,0],samples[nodes,1],s=10,color='#267ba6');ax.set(title=title,xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
            fig.suptitle('Prescribed plane contact: retained material samples\nNot a packing compaction or quality-repair result');fig.savefig(target/'retained-samples.png',dpi=170);plt.close(fig)
    except Exception as exc:
        result=dict(fraction=fraction,slip_mm=slip,passed=False,error=repr(exc),traceback=traceback.format_exc())
        (target/'failure.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--partial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    results=[];done=set()
    for path in sorted(a.partial.glob('fraction-*/summary.json')):
        result=json.loads(path.read_text());key=(result['fraction'],result['slip_mm'])
        if key in done:raise ValueError('Duplicate completed case')
        done.add(key);results.append(result);shutil.copytree(path.parent,a.output/path.parent.name)
    source=a.output/'source';source.mkdir()
    for path in Path('packing_fem').glob('*.py'):shutil.copy2(path,source/path.name)
    for path in (Path(__file__),Path('tools/verify_embedded_remesh.py')):shutil.copy2(path,source/path.name)
    cases=[(f,s) for f in (.2,.5,.8) for s in (.0001,.01) if (f,s) not in done]
    with ProcessPoolExecutor(max_workers=3) as pool:
        tasks=[pool.submit(worker,case,a.output) for case in cases]
        for task in as_completed(tasks):
            result=task.result();results.append(result);print(json.dumps(dict(fraction=result['fraction'],slip_mm=result['slip_mm'],passed=result['passed'],checks=result.get('checks'),error=result.get('error'))),flush=True)
    results.sort(key=lambda r:(r['fraction'],r['slip_mm']))
    summary=dict(passed=len(results)==6 and all(r['passed'] for r in results),cases=results,reused_completed_cases=sorted(done),scope='Prescribed plane-contact nested refinement, not packing compaction.',full_compaction_completed=False)
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['fraction','slip_mm','passed','instantaneous_passed','continuation_passed','periodic_virtual_work_error','maximum_continued_force_error','maximum_continued_energy_error','maximum_continued_dissipation_error']
    with (a.output/'comparison.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for r in results:
            row=dict(fraction=r['fraction'],slip_mm=r['slip_mm'],passed=r['passed'],instantaneous_passed=r.get('checks',{}).get('instantaneous_audit'),continuation_passed=r.get('checks',{}).get('continued_path'),periodic_virtual_work_error=r.get('transfer',{}).get('periodic_virtual_work_relative_error'))
            for name,key in (('force','force_error'),('energy','energy_error'),('dissipation','dissipation_error')):row['maximum_continued_'+name+'_error']=max((c[key] for c in r.get('continuation',[])),default=None)
            w.writerow(row)
    image=a.output/'fraction-0.5-slip-0.01/retained-samples.png'
    if image.exists():shutil.copy2(image,a.output/'retained-samples.png')
    if not summary['passed']:raise SystemExit(2)


if __name__=='__main__':main()
