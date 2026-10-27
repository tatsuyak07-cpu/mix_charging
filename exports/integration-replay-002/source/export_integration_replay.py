"""Archive history replay verification, including failed attempts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--verification',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); root=a.output
    if root.exists() or root.with_suffix('.zip').exists():raise FileExistsError(root)
    shutil.copytree(a.source,root/'validation')
    run=root/'validation'/a.verification
    summary=json.loads((run/'summary.json').read_text())
    # Closed-form work for the exact prescribed monotone segments: first
    # positive excursion .02, reverse .03, final positive excursion .015.
    # First yield costs one elastic limit; each reversal costs two.
    from packing_fem.barrier_contact import barrier
    _,normal,_=barrier(.015,20.,.02)
    traction=.3*normal; elastic_limit=traction/10.
    expected_work=traction*(.02+.03+.015-5*elastic_limit)
    assessment=dict(scope='Closed-form constant-normal flat-interface cycle only',
                    expected_final_force_x_N=-traction,
                    expected_friction_work_Nmm=expected_work,
                    maximum_final_force_error_N=max(abs(c['final_force_x_N']+traction) for c in summary['cases']),
                    maximum_work_error_Nmm=max(abs(c['total_friction_work_Nmm']-expected_work) for c in summary['cases']))
    assessment['passed']=bool(assessment['maximum_final_force_error_N']<1e-10 and assessment['maximum_work_error_Nmm']<1e-12)
    (root/'closed-form-check.json').write_text(json.dumps(assessment,indent=2)+'\n')
    if not summary['passed'] or not assessment['passed']:raise ValueError('Verification failed; preserve output for diagnosis')
    shutil.copy2(run/'summary.json',root/'summary.json')
    shutil.copy2(run/'trace.csv',root/'trace.csv')
    narrow=json.loads((root/'validation/narrow-integration/summary.json').read_text())
    if not narrow['passed']:raise ValueError('Narrow-strip integration verification failed')
    shutil.copy2(root/'validation/narrow-integration/summary.json',root/'narrow-summary.json')
    strip_rows=list(csv.DictReader((root/'validation/narrow-integration/integration.csv').open()))
    group=[r for r in strip_rows if float(r['center_mm'])==.501 and float(r['minimum_gap_mm'])==.0005]
    fig,axis=plt.subplots(figsize=(7,4),layout='constrained')
    for n in (16,32,64,128):
        values=sorted([r for r in group if r['method']=='uniform_midpoint' and float(r['resolution'])==n],key=lambda r:float(r['width_mm']))
        axis.semilogx([float(r['width_mm']) for r in values],[float(r['force_relative_error']) for r in values],'o-',label=f'Uniform midpoint n={n}')
    values=sorted([r for r in group if r['method']=='bound_adaptive' and float(r['resolution'])==1e-6],key=lambda r:float(r['width_mm']))
    axis.semilogx([float(r['width_mm']) for r in values],[float(r['force_relative_error']) for r in values],'kD--',label='Bound + adaptive Gauss')
    axis.set(xlabel='Active strip width (mm)',ylabel='Normal resultant relative error',title='Known quadratic-gap strip; not a 3-D FEM solve')
    axis.legend(fontsize=8);fig.savefig(root/'narrow-integration.png',dpi=170);plt.close(fig)
    rows=list(csv.DictReader((run/'trace.csv').open()))
    name=summary['cases'][-1]['name']; values=[r for r in rows if r['case']==name]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    slip=np.array([float(r['slip_mm']) for r in values])
    force=np.array([float(r['force_x_N']) for r in values])
    exact=np.array([float(r['analytic_force_x_N']) for r in values])
    axes[0].plot(slip,exact,label='Scalar Coulomb reference',lw=3,color='.7')
    axes[0].plot(slip,force,'o-',ms=3,label='Periodic surface contact')
    switch=summary['cases'][-1]['insertion_step']
    axes[0].plot(slip[switch],0.,'rx',ms=10,label='Late initialization')
    axes[0].plot(slip[switch],force[switch],'ks',mfc='none',ms=9,label='History replay')
    axes[0].set(xlabel='Tangential displacement (mm)',ylabel='Upper resultant (N)',title='Prescribed flat-interface cycle')
    axes[0].legend(fontsize=8)
    n=np.arange(len(summary['cases'])); cold=[c['cold_force_error_N'] for c in summary['cases']]
    axes[1].bar(n,cold,label='Late initialization error')
    axes[1].plot(n,[c['replay_maximum_absolute_error'] for c in summary['cases']],'ko',label='Replay error (zero)')
    axes[1].set(xlabel='Verification case',ylabel='Tangential resultant error (N)',title='Replay from known inactive checkpoint')
    axes[1].set_xticks(n,[str(i+1) for i in n]);axes[1].legend(fontsize=8)
    fig.suptitle('Contact-only verification; not a packing equilibrium result',fontsize=11)
    fig.savefig(root/'history-replay.png',dpi=170);plt.close(fig)
    source=root/'source';source.mkdir()
    for file in ('tools/verify_integration_replay.py','tools/export_integration_replay.py','tools/verify_narrow_integration.py','docs/integration-replay.md'):
        shutil.copy2(file,source/Path(file).name)
    for file in Path('packing_fem').glob('*.py'):
        target=source/'packing_fem'/file.name; target.parent.mkdir(exist_ok=True)
        shutil.copy2(file,target)
    for file in ('tests/test_material_quadrature.py','tests/test_contact_audit.py','AGENTS.md','pyproject.toml'):
        shutil.copy2(file,source/Path(file).name)
    (root/'provenance.json').write_text(json.dumps(dict(
        base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        production_contact_law_modified=False,
        verification_directory=a.verification,
        archived_failed_attempts=['verification.log','verification-002.log','verification-003.log'],
        tests='17 passed; full suite not rerun for verification-only changes'),indent=2)+'\n')
    manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob('*')) if f.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for name,digest in manifest.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    with zipfile.ZipFile(root.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(root.rglob('*')):
            if file.is_file():archive.write(file,str(file.relative_to(root.parent)))
    with zipfile.ZipFile(root.with_suffix('.zip')) as archive:assert archive.testzip() is None
    print(json.dumps(dict(files=len(manifest),verification_passed=summary['passed'],zip_bytes=root.with_suffix('.zip').stat().st_size)))


if __name__=='__main__':main()
