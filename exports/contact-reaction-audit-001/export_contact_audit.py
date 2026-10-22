"""Publish accepted contact witnesses, independent audits, and repeatability data."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.audit_contact_capture import reconstructed_spring


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--prior-audit',type=Path);a=p.parse_args()
    summary=json.loads((a.audit/'summary.json').read_text())
    if a.output.exists() or a.output.with_suffix('.zip').exists():raise FileExistsError('Exclusive output required')
    a.output.mkdir(parents=True);shutil.copytree(a.input,a.output/'simulation');shutil.copytree(a.audit,a.output/'independent-audit');baseline=Path('exports/plastic-contact-unloading-001/connected-plastic-unload8-001/run');target=a.output/'previous-baseline';target.mkdir()
    if a.prior_audit is not None:shutil.copytree(a.prior_audit,a.output/'prior-numeric-equality-audit')
    for name in ('accepted-step-audit.csv.gz','summary.json','config.json','last-state.npz'):shutil.copy2(baseline/name,target/name)
    with (a.output/'reaction-audit.csv').open('w') as f:
        rows=summary['finite_difference_tests'];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    data=json.loads((a.input/'run/contact-audit.json').read_text());b=np.loadtxt(a.input/'run/accepted-step-audit.csv.gz',delimiter=',',skiprows=1);old=np.loadtxt(baseline/'accepted-step-audit.csv.gz',delimiter=',',skiprows=1)
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    axs[0,0].plot(old[:,3],old[:,2]*100,label='Previous, capture disabled');axs[0,0].plot(b[:,3],b[:,2]*100,'--',label='Capture enabled');axs[0,0].set(xlabel='Commanded Cauchy pressure [MPa]',ylabel='Porosity [%]',title='Connected 4 mm fixture, initial porosity 0.25%')
    fd=summary['finite_difference_tests'];axs[0,1].loglog([v['step_mm'] for v in fd],[max(v['contact_force_relative_error'],1e-16) for v in fd],'o-');axs[0,1].axhline(2e-5,color='black',linestyle=':',label='Audit tolerance');axs[0,1].set(xlabel='Central difference step [mm]',ylabel='Contact-force maximum error / force scale',title='Independent contact potential differences')
    axs[1,0].plot([v['step_mm'] for v in fd],[v['total_generalized_force_max_error_N'] for v in fd],'o-',label='Assembly error');axs[1,0].plot([v['step_mm'] for v in fd],[v['free_static_residual_N'] for v in fd],'s-',label='Unloaded residual');axs[1,0].set(xlabel='Central difference step [mm]',ylabel='Force [N], macro components divided by L',title='Face-traction material + contact-force assembly')
    ratios=[r['kt']*np.linalg.norm(reconstructed_spring(r))/r['normal_force'] if r['mu'] else 0. for r in data['accepted']['records']];axs[1,1].bar(np.arange(len(ratios)),ratios,label='Derived spring / normal reaction');axs[1,1].axhline(.3,color='black',linestyle=':',label='mu = 0.3');axs[1,1].set(xlabel='Recorded contact integration point',ylabel='Constitutive tangential / normal ratio',title='Final state, no friction-history reset')
    for ax in axs.ravel():ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Endpoint contact audit on recorded active features; not search or mesh convergence');fig.tight_layout();fig.savefig(a.output/'contact-audit.png',dpi=180);plt.close(fig)
    for source in ('tools/audit_contact_capture.py','tools/export_contact_audit.py','tests/test_contact_audit.py','tools/plastic_cycle_audit.py'):shutil.copy2(source,a.output/Path(source).name)
    logs=a.output/'logs';logs.mkdir()
    for name in ('contact-audit-tests-001.log','contact-audit-tests-002.log','contact-audit-tests-003.log','contact-audit-full-tests-001.log','contact-audit-test-failed-001.py','connected-contact-audit-001.log','contact-reaction-independent-001.log','contact-reaction-independent-002.log'):
        source=Path('/tmp')/name
        if source.exists():shutil.copy2(source,logs/name)
    (a.output/'README.md').write_text('Contact audit of the existing 4 mm fixture, initial porosity 0.25%, E=10 MPa.\nRecorded force-evaluation state is separate from committed friction history. This is not an integrator restart checkpoint.\nIndependent audit uses central potential differences and face-traction material forces on the recorded active feature set. Collision search, contact-cycle CFL and mesh convergence are not verified by this audit.\nReproduction expects the full repository checkout and configured environment.\nOriginal rapid sliding test setup violated the motion certificate; corrected to resolved incremental motion without relaxing the guard. Failed test log and reconstructed initial test setup are retained.\n')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()};(a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(a.output.with_suffix('.zip')) as z:
        if z.testzip():raise RuntimeError('ZIP CRC failed')
    print(json.dumps(summary,indent=2));print('ZIP CRC passed')

if __name__=='__main__':main()
