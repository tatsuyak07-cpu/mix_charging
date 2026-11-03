"""Publish L1-budget verification, retaining failed attempts and exact data."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from packing_fem.checkpoint import load
from tools.export_pressure_stages import render


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();src,out=a.source,a.output;out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    matrix=read(src/'frozen-matrix-002/summary.json');plane=read(src/'plane-cycle/summary.json');moving=read(src/'moving-regression/summary.json')
    if not matrix['accepted_rule_admissible'] or not plane['passed'] or not moving['passed']:raise ValueError('Component verification failed')
    tests=(src/'final-tests-002.log').read_text()
    if '84 passed' not in tests:raise ValueError('Final regression incomplete')
    run=src/'packing-prefix';summary=read(run/'summary.json');attempt=run/f"attempt-{summary['attempts'][-1]['attempt']:03d}";rve=read(attempt/'summary.json')
    def table(name,rows):
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table('frozen-constraints.csv',[dict(block=c['block'],feasible=c['feasible'],weights_unchanged=c.get('weights_unchanged'),reason=c.get('reason','')) for c in matrix['cases']])
    rows=[]
    for c in plane['cases']:
        for i,e in enumerate(c['events']):
            r=e['report'];rows.append(dict(tolerance=c['tolerance'],initial_slip_mm=c['initial_slip_mm'],increment=i+1,weight_change=r['relative_area_weight_change'],work_error=r['accepted_work_error'],normal_energy_error=r['accepted_energy_errors']['normal_energy'],tangent_energy_error=r['accepted_energy_errors']['tangent_energy'],coulomb_ratio=e['coulomb_ratio'],checkpoint_error=c['checkpoint_force_error']))
    table('plane-cycle.csv',rows)
    progress=[json.loads(line) for line in (attempt/'progress.jsonl').read_text().splitlines()];table('packing-prefix.csv',progress)
    previous=list(csv.DictReader(Path('exports/runtime-parent-verification-001/packing-prefix.csv').open()));old={int(r['step']):r for r in previous if int(r['step'])>=90206};common=[r for r in progress if r['step'] in old]
    fields=('time_s','porosity','axial_compression','nominal_pressure_MPa','kinetic_Nmm','internal_Nmm')
    identity=dict(shared_steps=len(common),max_absolute_differences={k:max(abs(r[k]-float(old[r['step']][k])) for r in common) for k in fields});identity['exact']=all(v==0 for v in identity['max_absolute_differences'].values())
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    force=matrix['cases'][0]['report']['l1_audit']['groups']['accepted'];axes[0].bar(['Accepted excess','Audit limit'],[force['force_excess_L1'],force['force_limit']],color=['#168473','#cf5b29']);axes[0].set(ylabel='Force excess L1 (N)',title='Accepted rule retained, unchanged')
    for label,data in [('Previous',previous),('L1 projection',progress)]:
        selected=[r for r in data if int(r['step'])>=90206];axes[1].plot([int(r['step']) for r in selected],[100*float(r['porosity']) for r in selected],'.-',label=label)
    axes[1].axvline(90262,color='red',ls='--',label='Both stop');axes[1].ticklabel_format(useOffset=False,style='plain',axis='y');axes[1].set(xlabel='Accepted step',ylabel='Transient porosity (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Shared L1 budget fixed; packing stop NOT resolved; full compaction NOT completed');fig.savefig(out/'verification.png',dpi=170);plt.close(fig)
    state,metadata=load(rve['checkpoint_paths'][-1]);render(out/'packing-mesh.png',state,f"Actual transient mesh: porosity {100*rve['final_porosity']:.6f}% (NOT equilibrium)",metadata['config']['length_mm'])
    for file in src.rglob('*'):
        if not file.is_file() or '__pycache__' in file.parts or any(part.startswith('checkpoint-') for part in file.relative_to(src).parts):continue
        if file.suffix in ('.json','.log','.csv','.py') or file.name in ('progress.jsonl','contact-integration-budget.jsonl'):
            dst=out/'diagnostics'/file.relative_to(src);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dst)
    shutil.copytree(rve['checkpoint_paths'][-1],out/'last-accepted-checkpoint')
    inputs=out/'inputs';inputs.mkdir();shutil.copy2('results/restored-runtime/diagnostics/constraint-diagnosis-003/constraint-matrices.npz',inputs/'original-constraint-matrices.npz')
    for name in ('last-state.npz','last-mesh.vtu','contact-quadrature.npz'):
        if (attempt/name).exists():shutil.copy2(attempt/name,out/name)
    for folder in ('packing_fem','tests','tools'):
        dest=out/'source'/folder;dest.mkdir(parents=True)
        for file in Path(folder).glob('*.py'):shutil.copy2(file,dest/file.name)
    assessment=dict(accepted_rule_admissible=True,accepted_force_L1=force,regression_tests=84,plane_cases=6,moving_solver_passed=True,packing_steps=rve['steps'],packing_transient_porosity=rve['final_porosity'],failure=rve['failure'],last_resolution_attempt=rve['surface_resolution_attempts'][-1],previous_prefix_identity=identity,full_compaction_completed=False,convergence_verified=False)
    (out/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    (out/'README.md').write_text('''# L1誤差予算の検証

成分への均等配分を廃止し、親面/配置全体の超過合計に予算を直接適用。
受理済み状態は重み変更0で合格。次配置を含む保存射影は依然として実行不能。
実充填層は前回と同じ90,262刻みで停止。空隙率36.941697%は過渡値。
空隙率40%→10%・保持/除荷・収束確認は未完了。

- [検証図](verification.png)、[メッシュ](packing-mesh.png)
- [固定行列CSV](frozen-constraints.csv)、[平面CSV](plane-cycle.csv)、[充填層CSV](packing-prefix.csv)
- [判定JSON](assessment.json)、[報告](../../docs/l1-budget-verification.md)、[ZIP](../'''+out.name+'''.zip)

ZIPに実行時/最終ソース、入力行列、最後の受理チェックポイント、失敗した試行のログも保存。
元のresultsを上書きしていない。環境復元前のライブラリ不足による失敗ログも含む。
''')
    manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file()};(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(out.parent))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Corrupt ZIP')
        for name,digest in manifest.items():
            if hashlib.sha256(z.read(str(Path(out.name)/name))).hexdigest()!=digest:raise ValueError('ZIP hash mismatch')
    print(json.dumps(dict(assessment=assessment,archive=str(archive),bytes=archive.stat().st_size)))


if __name__=='__main__':main()
