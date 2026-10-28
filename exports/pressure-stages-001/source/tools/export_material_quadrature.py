"""Export persistent quadrature/force-history conservation verification."""
import argparse,hashlib,json,re,shutil,zipfile
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    audit=json.loads((a.input/'rejected-proposal-003/summary.json').read_text());planes=json.loads((a.input/'plane-verification-002/summary.json').read_text());pressure=json.loads((a.input/'pressure-run/calculation/summary.json').read_text())
    log=(a.input/'full-tests-002.log').read_text();count=re.search(r'(\d+) passed in ',log)
    assert count and 'failed' not in log and 'skipped' not in log and audit['accepted'] and planes['passed']
    a.output.mkdir(parents=True,exist_ok=False);shutil.copytree(a.input,a.output/'validation')
    source=Path('exports/compaction-stages-001/validation/resolved-rapid-run/calculation');original=a.output/'input-state';original.mkdir()
    for name in ('config.json','summary.json','last-state.npz','history.csv','contact-audit.json'):shutil.copy2(source/name,original/name)
    shutil.copy2(sorted(source.glob('surface-remesh-attempt-*.npz'))[-1],original/'original-rejected-proposal.npz')
    legacy=json.loads(Path('exports/compaction-stages-001/validation/new-history-diagnosis/summary.json').read_text())
    result=dict(scope=__doc__,whole_suite_passed=int(count.group(1)),saved_proposal_transfer=audit,plane_verification=planes,
                legacy_nodal_quadrature_change=legacy,pressure_diagnostic=pressure,
                general_node_deletion_or_ALE_supported=False,adaptive_integration_creation_implemented=False,pressure_equilibrium_verified=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(a.input/'plane-verification-002/energy-comparison.csv',a.output/'energy-comparison.csv')
    code=a.output/'source';code.mkdir()
    for f in Path('packing_fem').glob('*.py'):shutil.copy2(f,code/f.name)
    for name in ('tools/verify_material_quadrature.py','tools/export_material_quadrature.py','tools/audit_rve_remesh_proposal.py','tests/test_material_quadrature.py','configs/packing-rve-material-quadrature.json'):shutil.copy2(name,code/Path(name).name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(11,4.4))
    values=[legacy['old_normal_energy_Nmm'],legacy['proposal_normal_energy_Nmm'],audit['normal_energy_after']]
    ax[0].bar(['Before refinement','Old nodal insertion','Persistent material points'],values,color=['#5687a4','#b46b4f','#4c967a']);ax[0].set(ylabel='Normal contact energy [N mm]',title='Same saved geometry / rejected proposal');ax[0].tick_params(axis='x',labelrotation=14)
    rows=planes['cases'];x=np.arange(len(rows));ax[1].bar(x-.17,[r['tangent_before'] for r in rows],width=.34,label='Before');ax[1].bar(x+.17,[r['tangent_after'] for r in rows],width=.34,label='After')
    ax[1].set(yscale='log',ylabel='Tangential elastic energy [N mm]',title='Sticking and sliding, three split positions');ax[1].set_xticks(x,[f"{r['slip_mm']} / {r['fraction']}" for r in rows]);ax[1].tick_params(axis='x',labelrotation=25);ax[1].set_xlabel('Prescribed slip [mm] / split fraction');ax[1].legend(fontsize=8)
    for z in ax:z.grid(axis='y',alpha=.25)
    fig.suptitle('Remeshing conservation tests — NOT a pressure equilibrium result');fig.tight_layout();fig.savefig(a.output/'energy-conservation.png',dpi=180);plt.close(fig)
    with np.load(a.input/'plane-verification-002/slip-0.01-fraction-0.2/quadrature.npz') as z:X=z['reference_points'];t=z['tets'];weights=z['area_weights']
    from packing_fem.adaptive_tet import boundary
    faces=boundary(t);faces=faces[np.all(np.isclose(X[faces,2],.6),axis=1)];nodes=np.unique(faces);positive=nodes[weights[nodes]>0];zero=nodes[weights[nodes]==0]
    fig,axis=plt.subplots(figsize=(6,5));axis.triplot(X[:,0],X[:,1],faces,color='#aaaaaa',lw=.7)
    axis.scatter(X[positive,0],X[positive,1],s=weights[positive]/weights[positive].max()*180,label='Retained material points (area weight)')
    axis.scatter(X[zero,0],X[zero,1],s=70,marker='D',facecolors='none',edgecolors='#bd5f3d',label='Added geometry nodes')
    axis.set(xlabel='Reference x [mm]',ylabel='Reference y [mm]',aspect='equal',title='Refined geometry with unchanged integration points');axis.legend(fontsize=8);fig.tight_layout();fig.savefig(a.output/'material-points.png',dpi=180);plt.close(fig)
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,str(Path(a.output.name)/f.relative_to(a.output)))
    print(json.dumps(dict(output=str(a.output),files=len(hashes),whole_suite_passed=int(count.group(1)))))


if __name__=='__main__':main()
