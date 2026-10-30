"""Reviewable engineering results for contact atlas and conservative search."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--coupled',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output;root.mkdir(parents=True,exist_ok=False)
    normal=json.loads((a.source/'saved-poses-004/summary.json').read_text());search=json.loads((a.source/'large-search/summary.json').read_text())
    shutil.copy2(a.source/'saved-poses-004/comparison.csv',root/'normal-comparison.csv')
    shutil.copy2(a.source/'large-search/summary.json',root/'search-summary.json')
    shutil.copytree(a.source/'saved-poses-004',root/'normal-atlas');shutil.copytree(a.source/'large-search',root/'search')
    shutil.copytree(Path('results/porosity-cycle-development-001/persistent-frozen-poses/fixtures'),root/'normal-fixtures')
    rows=normal['cases'];x=np.arange(len(rows));fig,ax=plt.subplots(figsize=(11,4),layout='constrained')
    ax.semilogy(x,np.maximum([r['energy_error']*100 for r in rows],1e-10),'o-',label='Normal energy')
    ax.semilogy(x,np.maximum([r['force_error']*100 for r in rows],1e-10),'s-',label='Normal force')
    ax.axhline(.1,color='red',ls='--',label='0.1% criterion');ax.set_xticks(x,[f"{r['face']}\n{r['extra_macro_compression']:g}" for r in rows]);ax.set(xlabel='Source face / additional macro compression',ylabel='Normalized integration error (%)',title='Retained geometry controls, not equilibrated packing states');ax.legend();fig.savefig(root/'normal-errors.png',dpi=170);plt.close(fig)
    old=search['legacy_counts'];new=search['aabb_counts'];old_count=old['point_triangle_broad_candidates']+old['edge_edge_broad_candidates'];new_count=new['point_triangle_broad_candidates']+new['edge_edge_broad_candidates']
    with (root/'search-candidates.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['method','point_triangle_candidates','edge_edge_candidates','retained_jobs']);w.writerow(['legacy',old['point_triangle_broad_candidates'],old['edge_edge_broad_candidates'],search['legacy_jobs']]);w.writerow(['aabb',new['point_triangle_broad_candidates'],new['edge_edge_broad_candidates'],search['aabb_jobs']])
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained');bars=ax.bar(['Legacy sphere search','Conservative AABB'],[old_count,new_count],color=['#777777','#267ba6']);ax.axhline(search['physical_candidate_cap'],color='red',ls='--',label='Unchanged production cap');ax.bar_label(bars,fmt='%d');ax.set(ylabel='Conservative broad candidate pairs',title=f"Same {search['aabb_jobs']:,} retained jobs; no missing pairs");ax.legend();fig.savefig(root/'search-candidates.png',dpi=170);plt.close(fig)
    coupled=json.loads((a.coupled/'summary.json').read_text());last=coupled['attempts'][-1];attempt=a.coupled/f"attempt-{last['attempt']:03d}"
    destination=root/'coupled';destination.mkdir()
    for name in ('summary.json','attempts.json','geometry.json'):shutil.copy2(a.coupled/name,destination/name)
    for path in a.coupled.glob('replay-rules-*.json'):shutil.copy2(path,destination/path.name)
    for name in ('summary.json','config.json','history.csv','progress.jsonl','last-state.npz','last-mesh.vtu','accepted-step-audit.csv.gz','contact-audit.json','contact-integration-budget.jsonl'):
        if (attempt/name).exists():shutil.copy2(attempt/name,destination/('last-'+name if name in ('summary.json','config.json') else name))
    progress=[json.loads(line) for line in (attempt/'progress.jsonl').read_text().splitlines()];progress=[r for r in progress if r['step']>=90207]
    with (root/'coupled-contact-prefix.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(progress[0]));w.writeheader();w.writerows(progress)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained');step=[r['step'] for r in progress]
    axes[0].plot(step,[r['porosity']*100 for r in progress],'o-');axes[0].set(xlabel='Accepted step',ylabel='Transient porosity (%)')
    axes[1].plot(step,[r['contacts'] for r in progress],'o-');axes[1].set(xlabel='Accepted step',ylabel='Active quadrature contacts');fig.suptitle('Coupled contact-onset prefix only; NOT a 10% compaction solution');fig.savefig(root/'coupled-prefix.png',dpi=170);plt.close(fig)
    for run in sorted(list(a.source.glob('coupled-onset*'))+list(a.source.glob('global-cycle-*'))):
        if not run.is_dir():continue
        trial=root/'trial-diagnostics'/run.name;trial.mkdir(parents=True)
        for path in run.glob('*.json'):shutil.copy2(path,trial/path.name)
        for path in run.glob('attempt-*/summary.json'):
            target=trial/path.parent.name;target.mkdir();shutil.copy2(path,target/path.name)
        if (run/'source').exists():shutil.copytree(run/'source',trial/'source')
    for name in ('global-sensitivity','global-sensitivity-002','fit-diagnosis','reference-self-audit','batch-invariance'):
        if (a.source/name).exists():shutil.copytree(a.source/name,root/name)
    for name in ('initial-virtual-support.json','initial-support-provenance.json','global-budget-0.json','global-budget-1.json','global-budget-2.json'):
        shutil.copy2(a.source/name,root/name)
    for i in range(3):
        run=a.source/f'global-cycle-{i}-v7';summary=json.loads((run/'summary.json').read_text());latest=run/f"attempt-{summary['attempts'][-1]['attempt']:03d}"
        target=root/'sensitivity-prefixes'/str(i);target.mkdir(parents=True)
        for name in ('config.json','summary.json','progress.jsonl','contact-integration-budget.jsonl'):
            shutil.copy2(latest/name,target/name)
    logs=root/'test-logs';logs.mkdir()
    for path in a.source.glob('*.log'):shutil.copy2(path,logs/path.name)
    source=root/'source';source.mkdir()
    for directory in ('packing_fem','tests'):
        target=source/directory;target.mkdir()
        for path in Path(directory).glob('*.py'):shutil.copy2(path,target/path.name)
    for directory in ('regression-source','face-frame-experiment-source','final-validation-source','final-validation-source-002'):
        shutil.copytree(a.source/directory,root/directory)
    checkpoints=sorted(attempt.glob('checkpoint-*/manifest.json'))
    if checkpoints:shutil.copytree(checkpoints[-1].parent,root/'last-checkpoint')
    from tools.assess_contact_passage import assess
    passage=assess(a.coupled)
    provenance=dict(contact_passage=passage,normal_budget_sensitivity=json.loads((a.source/'global-sensitivity-002/summary.json').read_text()),normal_fixture_source='exports/porosity-cycle-development-001.zip',coupled_source=str(a.coupled),normal_controls=normal,search=search,
                    coupled_completed=coupled['completed'],goal_porosity=.1,limitations=['Contact-onset diagnostic; no loaded 10% equilibrium, unloading or spatial convergence claim.','Positive normal cubature requires full inactive coupled replay for new histories.','Raw forces on periodic representatives must be compared through complete generalized forces, including image work.'])
    (root/'assessment.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (root/'README.md').write_text(Path('docs/contact-atlas-development.md').read_text().replace('../exports/contact-atlas-development-001/','./'))
    manifest={str(path.relative_to(root)):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in root.rglob('*') if path.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(str(root)+'.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob('*'):
            if path.is_file():archive.write(path,str(path.relative_to(root.parent)))


if __name__=='__main__':main()
