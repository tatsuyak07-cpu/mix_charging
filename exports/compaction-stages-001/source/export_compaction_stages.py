"""Publish all stage-4 attempts and explicitly unsuccessful pressure checks."""
import argparse,hashlib,json,re,shutil,zipfile
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assessment=json.loads((a.input/'assessment/summary.json').read_text())
    log=(a.input/'full-tests-002.log').read_text();match=re.search(r'(\d+) passed in ',log)
    assert match and 'failed' not in log and 'skipped' not in log
    assert json.loads((a.input/'crossed-control/summary.json').read_text())['passed']
    a.output.mkdir(parents=True,exist_ok=False);shutil.copytree(a.input,a.output/'validation')
    for name in ('common-prefix-comparison.csv','case-outcomes.csv'):shutil.copy2(a.input/'assessment'/name,a.output/name)
    result=dict(scope=__doc__,whole_suite_passed=int(match.group(1)),assessment=assessment,
                statement='All pressure attempts stopped before 0.1 MPa/hold. Prefix comparisons are not quasistatic acceptance results.')
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    code=a.output/'source';code.mkdir()
    for f in Path('packing_fem').glob('*.py'):shutil.copy2(f,code/f.name)
    names=('tests/test_edge_coverage.py','tests/test_closest_tie.py','tests/test_material_neighborhood.py',
           'tools/assess_compaction_stages.py','tools/export_compaction_stages.py','tools/diagnose_new_history.py',
           'tools/audit_rve_remesh_proposal.py','tools/diagnose_last_contact_retry.py','tools/verify_near_edge.py')
    for name in names:shutil.copy2(name,code/Path(name).name)
    snapshots={}
    for name in assessment['cases']:
        old=a.input/name/'source'
        snapshots[name]=[f.name for f in Path('packing_fem').glob('*.py') if not (old/f.name).exists() or f.read_bytes()!=(old/f.name).read_bytes()]
    (a.output/'runtime-source-differences.json').write_text(json.dumps(snapshots,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[('resolved-rapid-run','4 ms ramp'),('resolved-slow-run','8 ms ramp'),('resolved-half-run','8 ms, half timestep'),('resolution-half-run','4 ms, half resolution')]
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for name,label in selected:
        h=np.loadtxt(a.input/name/'calculation/history.csv',delimiter=',',skiprows=1)
        ax[0,0].plot(h[:,24],100*h[:,2],label=label)
    ax[0,0].set(xlabel='Command pressure [MPa]',ylabel='Transient porosity [%]');ax[0,0].legend(fontsize=8)
    labels=[label for _,label in selected];peak=[assessment['whole_loaded_kinetic_ratios'][name] for name,_ in selected]
    ax[0,1].bar(labels,peak,color=['#a65b45','#3a82ac','#5d9d8a','#bc9d4a']);ax[0,1].axhline(.01,color='red',ls=':',label='Required bound');ax[0,1].set(ylabel='Peak kinetic / internal+contact energy\n(loaded stopped prefix)');ax[0,1].tick_params(axis='x',labelrotation=16);ax[0,1].legend(fontsize=8)
    import csv
    with (a.output/'common-prefix-comparison.csv').open() as stream:rows=list(csv.DictReader(stream))
    for kind in ('rate','timestep','edge_resolution'):
        group=[r for r in rows if r['comparison']==kind];x=[float(r['pressure_MPa']) for r in group];d=[max(abs(float(r['porosity_first'])-float(r['porosity_second'])),1e-16) for r in group]
        ax[1,0].semilogy(x,d,label=kind)
    ax[1,0].set(xlabel='Common prefix pressure [MPa]',ylabel='Absolute porosity difference');ax[1,0].legend(fontsize=8)
    pressures=[assessment['cases'][name]['summary']['peak_command_pressure_MPa'] for name,_ in selected]
    ax[1,1].bar(labels,pressures,color=['#a65b45','#3a82ac','#5d9d8a','#bc9d4a']);ax[1,1].axhline(.1,color='red',ls=':',label='Requested pressure');ax[1,1].set(ylabel='Pressure reached before stop [MPa]',ylim=(0,.11));ax[1,1].tick_params(axis='x',labelrotation=16);ax[1,1].legend(fontsize=8)
    for z in ax.ravel():z.grid(alpha=.2)
    fig.suptitle('Stage 1–4 verification: stopped transients, NOT equilibrium compaction results');fig.tight_layout();fig.savefig(a.output/'stage4-comparison.png',dpi=180);plt.close(fig)
    from packing_fem.adaptive_tet import boundary
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    with np.load(a.input/'resolved-rapid-run/calculation/last-state.npz') as z:x=z['points'];t=z['tets']
    fig=plt.figure(figsize=(7,6));axis=fig.add_subplot(111,projection='3d');axis.add_collection3d(Poly3DCollection(x[boundary(t)],facecolor='#88bddd',edgecolor='#294459',linewidth=.12))
    for j,setter in enumerate((axis.set_xlim,axis.set_ylim,axis.set_zlim)):setter(x[:,j].min(),x[:,j].max())
    axis.set_box_aspect(np.ptp(x,axis=0));axis.set(xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title='Stopped mesh at ~0.05094 MPa — NOT equilibrium')
    fig.tight_layout();fig.savefig(a.output/'stopped-mesh.png',dpi=180);plt.close(fig)
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,str(Path(a.output.name)/f.relative_to(a.output)))
    print(json.dumps(dict(output=str(a.output),files=len(hashes),whole_suite_passed=int(match.group(1)))))


if __name__=='__main__':main()
