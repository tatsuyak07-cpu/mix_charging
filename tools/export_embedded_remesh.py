"""Export qualified and failed nested-refinement checks without a compaction claim."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    old=json.loads((a.source/'plane-matrix-final/summary.json').read_text());new=json.loads((a.source/'plane-matrix-stick-corrected/summary.json').read_text())
    solver=json.loads((a.source/'solver-transaction-stick-corrected/summary.json').read_text());packing=json.loads((a.source/'packing-eligibility/summary.json').read_text())
    shutil.copytree(a.source,root/'diagnostics')
    shutil.copy2(a.source/'plane-matrix-stick-corrected/comparison.csv',root/'comparison.csv')
    image=a.source/'plane-matrix-stick-corrected/retained-samples.png'
    if image.exists():shutil.copy2(image,root/'retained-samples.png')
    def errors(summary):return [max((r['force_error'] for r in c.get('continuation',[])),default=np.nan) for c in summary['cases']]
    x=np.arange(len(new['cases']));fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.bar(x-.18,errors(old),width=.36,label='Before stick/flow correction',color='#b95c42');ax.bar(x+.18,errors(new),width=.36,label='After correction',color='#267ba6')
    ax.axhline(1e-7,color='red',ls='--',label='Unchanged 1e-7 verification limit');ax.set_yscale('log');ax.set_xticks(x,[f"{c['fraction']:g} / {c['slip_mm']:g}" for c in new['cases']]);ax.set(xlabel='Edge fraction / prior slip (mm)',ylabel='Maximum periodic-force relative difference',title='Continued prescribed motion: refined versus unrefined contact');ax.legend();fig.savefig(root/'continued-force-errors.png',dpi=170);plt.close(fig)
    logs=root/'test-logs';logs.mkdir()
    for path in a.source.glob('*.log'):shutil.copy2(path,logs/path.name)
    current=root/'implementation';current.mkdir()
    for directory in ('packing_fem','tests','tools'):
        target=current/directory;target.mkdir()
        for path in Path(directory).glob('*.py'):shutil.copy2(path,target/path.name)
    checks=dict(planar_transfer_and_continuation=new['passed'],solver_initial_transfer=solver['checks']['active_transfer_committed'],solver_restart=solver['checks']['active_restart_exact'],rejected_solver_trajectory_unchanged=solver['checks']['rejected_trajectory_unchanged'],solver_three_step_continuation=solver['checks']['bounded_three_steps'],packing_eligible=packing['eligible_candidates']>0)
    assessment=dict(stage3_completed=False,full_compaction_completed=False,status='Partial, not qualified for packing large-deformation remeshing.',checks=checks,
                    baseline_matrix=old,corrected_matrix=new,solver=solver,packing_eligibility=packing,
                    limits=['Partition-aligned retained-node bisection only; no sample creation/coarsening or pose-atlas split.',
                            'Moving-state momentum/kinetic audit failed; no relaxation of acceptance limits.',
                            'A solver step after initial transfer encounters unresolved tied closest projections.',
                            'No mesh-quality recovery, loaded 10% equilibrium, unloading or spatial convergence result.'])
    (root/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    (root/'README.md').write_text(Path('docs/active-embedded-remesh.md').read_text().replace('../exports/active-embedded-remesh-001/','./').replace('../exports/active-embedded-remesh-001.zip','../active-embedded-remesh-001.zip'))
    manifest={str(path.relative_to(root)):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in root.rglob('*') if path.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(str(root)+'.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob('*'):
            if path.is_file():archive.write(path,str(path.relative_to(root.parent)))


if __name__=='__main__':main()
