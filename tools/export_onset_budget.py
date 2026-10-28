"""Archive independent onset-integration verification and failed controls."""
import argparse,csv,hashlib,json,shutil,subprocess,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--verification',required=True);p.add_argument('--comparison',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    shutil.copytree(a.source,root/'validation')
    verification=root/'validation'/a.verification;comparison=root/'validation'/a.comparison
    s=json.loads((verification/'summary.json').read_text());old=json.loads((comparison/'summary.json').read_text())
    if not s['manufactured_normalized_passed'] or not s['rve_refinement_passed']:raise ValueError('Normalized quadrature verification did not pass; preserve output for diagnosis')
    shutil.copy2(verification/'summary.json',root/'summary.json')
    for name in ('manufactured.csv','rve.csv'):shutil.copy2(verification/name,root/name)
    shutil.copy2(comparison/'comparison.csv',root/'legacy-comparison.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for method,label in [('legacy_component','Legacy component budget'),('legacy_resultant','Legacy resultant budget'),
                         ('normalized','Normalized floor'),('normalized_quarter_tolerance','Normalized floor, tolerance / 4')]:
        rows=sorted([r for r in s['manufactured_cases'] if r['method']==method],key=lambda r:r['reference_energy_Nmm'])
        axes[0].loglog([r['reference_energy_Nmm'] for r in rows],
                       [100*max(r['energy_error'],r['force_error'],1e-16) for r in rows],'o-',ms=4,label=label)
    axes[0].axhline(.1,color='k',ls='--',label='Acceptance 0.1%')
    axes[0].set(xlabel='Independent reference energy (N mm)',ylabel='Maximum normalized integral error (%)',title='60-digit density + reduced 1-D triangle moments')
    axes[0].legend(fontsize=7)
    for tolerance,label in [(.00025,'Normalized floor'),(.0000625,'Tolerance / 4')]:
        rows=[r for r in s['rve_cases'] if r['method']=='normalized_refinement' and r['tolerance']==tolerance]
        axes[1].plot(range(1,len(rows)+1),[100*max(r['energy_error'],r['force_error']) for r in rows],'o-',label=label)
    axes[1].axhline(.1,color='k',ls='--',label='Acceptance 0.1%')
    axes[1].set(xlabel='RVE face / prescribed geometry case',ylabel='Error against refined reference (%)',title='12 geometry cases; reference tolerance / 16')
    axes[1].legend(fontsize=8)
    fig.suptitle('Normal surface integration only; not a pressure equilibrium calculation')
    fig.savefig(root/'integration-errors.png',dpi=180);plt.close(fig)
    fig,axis=plt.subplots(figsize=(7.5,4.5),layout='constrained')
    for face in sorted({r['face'] for r in s['rve_cases']}):
        rows=[r for r in s['rve_cases'] if r['method']=='frozen_onset_rule' and r['face']==face]
        axis.loglog([r['extra_macro_compression'] for r in rows],[100*max(r['energy_error'],r['force_error']) for r in rows],'o-',label=f'Face {face}')
    axis.axhline(.1,color='k',ls='--',label='Acceptance 0.1%')
    axis.set(xlabel='Prescribed additional macro compression (dimensionless)',ylabel='Maximum normalized integral error (%)',
             title=f"Frozen onset quadrature: {s['frozen_rule_failures']} of {s['frozen_rule_cases']} geometry cases failed")
    axis.legend(fontsize=8);fig.savefig(root/'frozen-rule-errors.png',dpi=180);plt.close(fig)
    source=root/'source';source.mkdir()
    for directory in ('packing_fem',):shutil.copytree(directory,source/directory,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('tools/experimental_surface_budget.py','tools/verify_onset_budget.py','tools/compare_onset_legacy.py',
                 'tools/export_onset_budget.py','tests/test_onset_budget.py','tests/test_surface_patch.py','docs/onset-budget.md',
                 'requirements-lock.txt','AGENTS.md'):
        target=source/name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(name,target)
    assessment=dict(manufactured_normalized_passed=s['manufactured_normalized_passed'],rve_refinement_passed=s['rve_refinement_passed'],
                    frozen_rule_cases=s['frozen_rule_cases'],frozen_rule_failures=s['frozen_rule_failures'],
                    legacy_failed_rows=old['failed_cases'],production_solver_modified=False,pressure_0_1_MPa_reached=False,
                    base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),tests=(a.source/'tests.log').read_text().strip(),
                    source_implementation_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('packing_fem').glob('*.py')})
    (root/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    (root/'README.md').write_text('# 接触開始時の積分誤差検証\n\n'
        '独立な平面試験18条件と実セル12姿勢の細分化比較に合格しました。\n'
        '接触開始時の積分点を固定した場合は8条件中5条件が不合格です。\n'
        '**圧縮ソルバーは変更しておらず、0.1 MPaの平衡解を得た結果ではありません。**\n\n'
        '[検証報告](../../docs/onset-budget.md) / [独立平面CSV](manufactured.csv) / [実セルCSV](rve.csv)\n\n'
        '[積分誤差PNG](integration-errors.png) / [固定積分点の誤差PNG](frozen-rule-errors.png)\n\n'
        'validation/に初回の列名重複による停止ログ、不合格の旧方式、コンパクトな実セル形状、全結果を収録しています。\n'
        '実セルの参照値は数値細分化による比較値であり、厳密解ではありません。source/に再現用ソースを保存しています。\n')
    manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob('*')) if f.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(root.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(root.rglob('*')):
            if file.is_file():z.write(file,str(file.relative_to(root.parent)))
    with zipfile.ZipFile(root.with_suffix('.zip')) as z:assert z.testzip() is None
    print(json.dumps(dict(files=len(manifest),zip_bytes=root.with_suffix('.zip').stat().st_size,assessment=assessment)))


if __name__=='__main__':main()
