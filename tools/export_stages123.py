"""Publish measured stage 1-3 outcomes, without promoting transient states."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from packing_fem.checkpoint import load,save
from tools.export_pressure_stages import render


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();src,out=a.source,a.output;out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    if '92 passed' not in (src/'final-tests-004.log').read_text():raise ValueError('Final regression incomplete')
    moving=read(src/'moving-preemptive/summary.json')
    if not moving['passed']:raise ValueError('Positive forecast integration control failed')
    required=.0464355684470197;cases=[];runs={};progress={}
    paths={'baseline_early':src/'baseline-early-002/attempt-000','rebuilt_99':src/'added-0-002/attempt-000','added_197':src/'driver-added-1/run-000/attempt-000','added_197_early':src/'driver-added-1-early/run-000/attempt-000'}
    for name,path in paths.items():
        d=read(path/'summary.json');s,_=load(d['checkpoint_paths'][-1]);rows=[json.loads(line) for line in (path/'progress.jsonl').read_text().splitlines()];progress[name]=rows;runs[name]=d
        forecasts=[e for e in d['surface_resolution_attempts'] if e.get('operation')=='preemptive_quadrature']
        cases.append(dict(case=name,last_step=d['steps'],time_s=s['time'],axial_compression=d['final_axial_compression'],transient_porosity=d['final_porosity'],old_failed_trial_time_crossed=s['time']>=required,history_entries=len(s['contact']['history']),early_attempts=len(forecasts),early_accepted=sum(e['repaired'] for e in forecasts),failure=d['failure']))
    for name in ('driver-added-1','driver-added-1-early'):
        if not read(src/name/'summary.json')['short_interval_passed']:raise ValueError('Coupled old-interval replay failed')
    def table(name,rows):
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table('case-comparison.csv',cases)
    for name,rows in progress.items():table(name+'-progress.csv',rows)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for name,rows in progress.items():
        selected=[r for r in rows if r['step']>=90206];style='--' if name.endswith('early') else '-'
        axes[0].plot([(r['time_s']-required)*1e6 for r in selected],[100*r['porosity'] for r in selected],style,label=name)
    axes[0].axvline(0,color='red',ls=':',label='Old rejected trial');axes[0].ticklabel_format(useOffset=False,style='plain',axis='y');axes[0].set(xlabel='Time relative to old rejected trial (microseconds)',ylabel='Transient porosity (%)');axes[0].legend(fontsize=7)
    axes[1].barh([c['case'] for c in cases],[(c['time_s']-required)*1e6 for c in cases],color=['#b44a37' if not c['old_failed_trial_time_crossed'] else '#237c75' for c in cases]);axes[1].axvline(0,color='red',ls=':');axes[1].set(xlabel='Accepted physical time beyond old rejected trial (microseconds)')
    fig.suptitle('Stages 1-3: old stop interval crossed; new integration stop remains; NOT equilibrium');fig.savefig(out/'comparison.png',dpi=170);plt.close(fig)
    original=read(Path('results/l1-budget-verification-001/certified-coarse-rules.json'));chosen=read(src/'fit-1-002/rules.json');point_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(9,8),layout='constrained')
    for axis,(face,rule) in zip(axes.ravel(),chosen.items()):
        B=np.asarray(rule['bary']);old=np.asarray(original[face]['bary']);oldset=set(map(tuple,old));area=sum(rule['weights'])
        axis.plot([0,1,0,0],[0,0,1,0],color='gray');axis.scatter(old[:,1],old[:,2],s=17,marker='x',color='magenta',label='Old locations');axis.scatter(B[:,1],B[:,2],s=7,color='#057fa5',label='Replayed locations')
        axis.set(xlabel='Material barycentric b1',ylabel='Material barycentric b2',title=f'Face {face}: {len(old)} -> {len(B)} points',xlim=(-.03,1.03),ylim=(-.03,1.03));axis.set_aspect('equal')
        for b,w in zip(B,rule['weights']):point_rows.append(dict(face=face,b0=b[0],b1=b[1],b2=b[2],reference_weight_mm2=w,area_fraction=w/area,old_location=tuple(b) in oldset))
    axes.ravel()[0].legend(fontsize=7);fig.suptitle('Material integration points on original parent faces; not the FE mesh');fig.savefig(out/'quadrature-points.png',dpi=170);plt.close(fig);table('quadrature-points.csv',point_rows)
    result=runs['added_197'];checkpoint=Path(result['checkpoint_paths'][-1]);state,metadata=load(checkpoint)
    render(out/'packing-mesh.png',state,f"Actual transient mesh: porosity {100*result['final_porosity']:.6f}% (NOT equilibrium)",metadata['config']['length_mm'])
    forecasts=[dict(case=name,**event) for name,d in runs.items() for event in d['surface_resolution_attempts'] if event.get('operation')=='preemptive_quadrature']
    assessment=dict(regression_tests=92,old_rejected_trial_time_s=required,cases=cases,forecast_attempts=forecasts,moving_forecast_control=moving['checks'],stage1_diagnosis_performed=True,stage2_old_interval_passed=True,stage3_timing_comparison_performed=True,packing_early_redistribution_effect_demonstrated=False,new_integration_stop_step=result['steps'],full_compaction_completed=False,convergence_verified=False)
    (out/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    # Preserve every failure/proposal and execution source; duplicate bulk
    # checkpoint arrays stay in the original results, not in every ZIP folder.
    for file in src.rglob('*'):
        if not file.is_file() or '__pycache__' in file.parts or any(part.startswith('checkpoint-') for part in file.relative_to(src).parts):continue
        if file.suffix in ('.json','.log','.csv','.py') or file.name in ('progress.jsonl','contact-integration-budget.jsonl'):
            target=out/'diagnostics'/file.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
    shutil.copytree(checkpoint,out/'last-accepted-checkpoint')
    for name in ('last-state.npz','last-mesh.vtu','contact-quadrature.npz'):
        if (paths['added_197']/name).exists():shutil.copy2(paths['added_197']/name,out/name)
    inputs=out/'inputs';inputs.mkdir();shutil.copytree('results/porosity-cycle-development-001/rve-cycle/attempt-002/checkpoint-00090206',inputs/'free-checkpoint')
    shutil.copy2(src/'fit-1-002/rules.json',inputs/'chosen-rules.json');shutil.copy2('results/l1-budget-verification-001/certified-coarse-rules.json',inputs/'seed-rules.json')
    for name in ('added-1','added-1-early'):shutil.copy2(src/f'{name}-config.json',inputs/f'{name}-config.json')
    controls=[src/'baseline-early/attempt-000/checkpoint-00090250',src/'added-1/attempt-000/checkpoint-00090252',Path('results/l1-budget-verification-001/packing-prefix/attempt-000/checkpoint-00090262'),src/'added-1-002/attempt-000/checkpoint-00090273']
    for index,path in enumerate(controls):
        s,meta=load(path);geometry={name:s[name] for name in ('X','x','q','v','a','time','prescribed_active','porosity_cycle_state')};geometry.update(contact_free_prefix=False,contact=dict(history={'geometry_control_only':{}}))
        dest=inputs/f'geometry-control-{index}';dest.mkdir()
        save(dest/'checkpoint',geometry,dict(config=meta['config'],scope='Geometry input only, NOT a solver restart and no history transferred',source_checkpoint=str(path),source_state_sha256=hashlib.sha256((path/'state.json').read_bytes()).hexdigest()))
        original_summary=path.parent/'summary.json'
        if original_summary.exists():
            context=read(original_summary)
            (dest/'summary.json').write_text(json.dumps(dict(scope='Geometric rejected-trial input only',surface_resolution_attempts=context['surface_resolution_attempts'],source_sha256=hashlib.sha256(original_summary.read_bytes()).hexdigest()),indent=2)+'\n')
    for folder in ('packing_fem','tests','tools'):
        target=out/'source'/folder;target.mkdir(parents=True)
        for file in Path(folder).glob('*.py'):shutil.copy2(file,target/file.name)
    (out/'README.md').write_text('''# 工程1〜3の開発・比較検証

関連92テスト合格。対象4親面の埋込積分点を197点とし、接触前から全連成再計算して旧停止区間を通過。
90,262→90,273刻み、物理時刻でも旧棄却試行を越えた。その先の積分誤差は未解消。
早期再配分単独の改善効果は充填層では未確認。大変形・平衡保持/除荷・空隙率10%・収束は未達成。

- [比較図](comparison.png)、[点配置](quadrature-points.png)、[実メッシュ](packing-mesh.png)
- [条件比較CSV](case-comparison.csv)、[点配置CSV](quadrature-points.csv)
- [判定JSON](assessment.json)、[詳細報告](../../docs/stages123-development.md)、[全データZIP](../'''+out.name+'''.zip)

ZIPのdiagnosticsに失敗した案、実行時ソース、全試行の判定/ログを保存。
last-accepted-checkpointは最後の受理済み過渡状態。inputsに接触前チェックポイントと選択した積分則を収録。
geometry-control-*は配置生成専用で、計算再開用ではない。元のresultsは上書きしていない。
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
