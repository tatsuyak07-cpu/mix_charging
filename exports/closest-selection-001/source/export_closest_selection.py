"""Archive audited edge selection, retry controls and the rapid RVE path."""
import argparse,csv,hashlib,json,re,shutil,zipfile
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    calc=a.input/'pressure-run/calculation';s=json.loads((calc/'summary.json').read_text())
    control=json.loads((a.input/'crossed-retry-002/summary.json').read_text())
    diagnosis=json.loads((a.input/'contact-stop-diagnosis-005/summary.json').read_text())
    assert diagnosis['rejection_reproduced'] and diagnosis['certificate_inequality_reproduced']
    log=(a.input/'full-tests.log').read_text();passed=re.search(r'(\d+) passed in ',log)
    assert passed and 'failed' not in log and 'skipped' not in log and control['passed']
    a.output.mkdir(parents=True,exist_ok=False);shutil.copytree(a.input,a.output/'validation')
    baseline=Path('exports/packing-rve-replay-001/validation/corrected-run/calculation')
    for name in ('summary.json','history.csv'):shutil.copy2(baseline/name,a.output/f'baseline-{name}')
    code=a.output/'source';code.mkdir()
    for f in Path('packing_fem').glob('*.py'):shutil.copy2(f,code/f.name)
    for name in ('tools/verify_closest_retry.py','tools/diagnose_last_contact_retry.py','tools/export_closest_selection.py','tests/test_closest_selection.py'):
        shutil.copy2(name,code/Path(name).name)
    result=dict(scope=__doc__,whole_suite_passed=int(passed.group(1)),crossed_retry_checks=control['checks'],
                rapid_pressure_run=s,last_retry_diagnosis=diagnosis,physical_compaction_verified=False,
                interpretation='Manufactured retry control and rapid BCC diagnostic are separate from quasi-static pressure acceptance.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    rows=[]
    for i,event in enumerate(s['remesh_events'],1):
        choice=event.get('candidate_selection',{}).get('candidate',{})
        rows.append(dict(event=i,time_s=event.get('time_s'),accepted=event['accepted'],side=choice.get('side'),fraction=event.get('fraction'),quality=event.get('minimum_quality_after'),reason=event.get('reason','')))
    with (a.output/'candidate-events.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=['event','time_s','accepted','side','fraction','quality','reason']);writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.2))
    for directory,label in ((baseline,'Previous midpoint strategy'),(calc,'Audited closest-point strategy')):
        h=np.loadtxt(directory/'history.csv',delimiter=',',skiprows=1)
        ax[0].plot(h[:,24],100*h[:,2],label=label)
        ax[1].plot(h[:,0]*1000,h[:,15],label=label)
    ax[0].set(xlabel='Command pressure [MPa]',ylabel='Porosity [%]')
    ax[1].set(xlabel='Time [ms]',ylabel='Minimum quality');ax[1].axhline(.03,color='red',ls=':',label='Unchanged quality floor')
    for z in ax:z.grid(alpha=.25);z.legend(fontsize=8)
    fig.suptitle('Rapid transient diagnostics — not quasi-static pressure solutions');fig.tight_layout();fig.savefig(a.output/'selection-comparison.png',dpi=180);plt.close(fig)
    from packing_fem.adaptive_tet import boundary
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    with np.load(calc/'last-state.npz') as z:
        x=z['points'];t=z['tets'];fig=plt.figure(figsize=(7,6));ax=fig.add_subplot(111,projection='3d')
        ax.add_collection3d(Poly3DCollection(x[boundary(t)],facecolor='#88bddd',edgecolor='#294459',linewidth=.12))
        for j,setter in enumerate((ax.set_xlim,ax.set_ylim,ax.set_zlim)):setter(x[:,j].min(),x[:,j].max())
        ax.set_box_aspect(np.ptp(x,axis=0));ax.set(xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title='Last accepted transient mesh — NOT equilibrium')
        fig.tight_layout();fig.savefig(a.output/'stopped-mesh.png',dpi=180);plt.close(fig)
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,str(Path(a.output.name)/f.relative_to(a.output)))
    print(json.dumps(dict(output=str(a.output),files=len(hashes))))


if __name__=='__main__':main()
