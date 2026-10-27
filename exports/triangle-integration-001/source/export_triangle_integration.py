"""Preserve finite-triangle integration checks and all development attempts."""
import argparse,csv,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output
    if root.exists() or root.with_suffix('.zip').exists():raise FileExistsError(root)
    shutil.copytree(args.source,root/'validation')
    final=root/'validation/verification-005';partition=root/'validation/partition-verification'
    summary=json.loads((final/'summary.json').read_text());part=json.loads((partition/'summary.json').read_text())
    if not summary['passed'] or not part['passed']:raise ValueError('Verification did not pass; preserve partial export')
    for filename in ('summary.json','comparison.csv'):shutil.copy2(final/filename,root/filename)
    shutil.copy2(partition/'summary.json',root/'partition-summary.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for axis,key in zip(axes,('force_relative_error','energy_relative_error')):
        for kind in ('edge','corner'):
            for tolerance in (1e-3,5e-4):
                widths=(.0002,.002,.02)
                errors=[max(c[key] for c in summary['cases'] if c['kind']==kind and c['width_mm']==w and c['tolerance']==tolerance) for w in widths]
                axis.loglog(widths,errors,'o-',label=f'{kind}, tolerance {tolerance}')
        axis.set(xlabel='Active parameter width (mm)',ylabel=key.replace('_',' '))
        axis.grid(True,alpha=.25);axis.legend(fontsize=8)
    fig.suptitle('Prescribed frictionless triangle pairs; independent 1-D reference')
    fig.savefig(root/'integration-errors.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    width=.002
    for axis,kind in zip(axes,('edge','corner')):
        row=next(c for c in summary['cases'] if c['kind']==kind and c['width_mm']==width and c['minimum_gap_mm']==.0005 and c['tolerance']==5e-4)
        with np.load(final/row['case']/'quadrature.npz') as z:bary=z['barycentric_points'];weights=z['area_weights']
        choice=np.linspace(0,len(bary)-1,min(2000,len(bary)),dtype=int)
        axis.scatter(bary[choice,1],bary[choice,2],s=4,c='tab:blue',label='Adaptive points (subset)')
        if kind=='edge':
            axis.axvline(width,color='tab:red',label='Activation boundary');axis.set_ylim(0,1)
        else:
            axis.plot([0,width],[width,0],color='tab:red',label='Activation boundary');axis.set_ylim(0,2*width);axis.set_aspect('equal')
        axis.set(xlim=(0,2*width),xlabel='Reference x (mm)',ylabel='Reference y (mm)',title=f'{kind}: {len(weights)} quadrature points')
        axis.legend(fontsize=8)
    fig.suptitle('Integration cells only; bulk FEM mesh is unchanged')
    fig.savefig(root/'quadrature-points.png',dpi=170);plt.close(fig)
    source=root/'source';source.mkdir()
    for f in ('tools/verify_triangle_integration.py','tools/verify_triangle_partition.py','tools/export_triangle_integration.py','tests/test_triangle_quadrature.py','docs/triangle-integration.md','AGENTS.md','pyproject.toml'):
        shutil.copy2(f,source/Path(f).name)
    for file in Path('packing_fem').glob('*.py'):
        target=source/'packing_fem'/file.name;target.parent.mkdir(exist_ok=True);shutil.copy2(file,target)
    provenance=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    pressure_solver_modified=False,friction_history_coupled=False,physical_solution_verified=False,
                    tests='30 passed; full suite not rerun',
                    attempts=dict(verification='Interrupted for excessive low-order work; completed case kept',
                                  verification_002='Interrupted before batch acceleration; completed cases kept',
                                  verification_003='Failed maximum depth; log and source kept',
                                  verification_004='Failed maximum depth with local budget distribution; log and source kept',
                                  verification_005='24 cases passed',partition_verification='12 cases passed'))
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob('*')) if f.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert all(hashlib.sha256((root/f).read_bytes()).hexdigest()==h for f,h in manifest.items())
    with zipfile.ZipFile(root.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as archive:
        for f in sorted(root.rglob('*')):
            if f.is_file():archive.write(f,str(f.relative_to(root.parent)))
    with zipfile.ZipFile(root.with_suffix('.zip')) as archive:assert archive.testzip() is None
    print(json.dumps(dict(files=len(manifest),zip_bytes=root.with_suffix('.zip').stat().st_size,passed=True)))


if __name__=='__main__':main()
