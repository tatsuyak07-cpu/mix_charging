"""Export audited non-midpoint transfer proposals without a pressure solution."""
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    reports=[json.loads((a.input/name/'summary.json').read_text()) for name in ('source','target')]
    assert all(r['accepted'] and all(r['transaction_audit']['checks'].values()) for r in reports)
    full=(a.input/'full-tests-001.log').read_text()
    focused=(a.input/'weighted-tests-003.log').read_text()
    assert re.search(r'\b\d+ passed in ',full) and 'failed' not in full and 'skipped' not in full
    assert '9 passed in ' in focused
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copytree(a.input,a.output/'validation')
    saved=a.output/'input-state';saved.mkdir()
    for name in ('config.json','summary.json','last-state.npz','history.csv','contact-audit.json'):
        shutil.copy2(a.source/name,saved/name)
    code=a.output/'source';code.mkdir()
    for name in Path('packing_fem').glob('*.py'):shutil.copy2(name,code/name.name)
    for name in ('tools/audit_rve_remesh_proposal.py','tools/export_weighted_transfer.py','tests/test_weighted_surface_transfer.py'):
        shutil.copy2(name,code/Path(name).name)
    result=dict(scope=__doc__,material_and_contact_unchanged=True,
                accepted_state_transfer_only=True,time_advanced=False,
                automatic_solver_selection_implemented=False,
                pressure_path_verified=False,physical_compaction_verified=False,
                whole_suite_passed=int(re.search(r'(\d+) passed in ',full).group(1)),
                final_weighted_tests_passed=9,
                note='Whole suite collected before two additional focused tests; the final focused run contains all nine weighted-transfer tests.',
                source_and_target=reports)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,4.2))
    names=['Source closest point','Target closest point']
    ax[0].bar(names,[r['minimum_quality'] for r in reports],color=['#478bbd','#4b9a71'])
    ax[0].axhline(.03,color='red',ls=':',label='Unchanged quality floor')
    ax[0].set(ylabel='Minimum quality (accepted-state geometry)');ax[0].legend(fontsize=8)
    ax[1].bar(names,[max(r['transaction_audit']['scaled_errors'].values()) for r in reports],color=['#478bbd','#4b9a71'])
    ax[1].axhline(.001,color='red',ls=':',label='Unchanged audit tolerance')
    ax[1].set(yscale='log',ylabel='Largest scaled transfer error',ylim=(1e-14,.01));ax[1].legend(fontsize=8)
    for z in ax:z.grid(axis='y',alpha=.25);z.tick_params(axis='x',labelrotation=12)
    fig.suptitle('State/contact transfer audit only — no pressure integration')
    fig.tight_layout();fig.savefig(a.output/'weighted-transfer.png',dpi=180);plt.close(fig)
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,str(Path(a.output.name)/f.relative_to(a.output)))
    print(json.dumps(dict(output=str(a.output),files=len(hashes),whole_suite_passed=result['whole_suite_passed'])))


if __name__=='__main__':main()
