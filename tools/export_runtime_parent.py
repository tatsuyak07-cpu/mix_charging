"""Export bounded runtime-quadrature verification without claiming equilibrium."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from packing_fem.checkpoint import load
from tools.export_pressure_stages import render


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    src,out=a.source,a.output;out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    plane=read(src/'plane-cycle/summary.json')
    if not plane['passed']:raise ValueError('Plane cycle verification failed')
    coulomb=read(src/'plane-coulomb-audit.json')
    if not coulomb['all_coulomb_bounds'] or not coulomb['no_negative_dissipation']:raise ValueError('Plane friction audit failed')
    run=src/'packing-prefix-002';summary=read(run/'summary.json');attempt=run/f"attempt-{summary['attempts'][-1]['attempt']:03d}"
    rve=read(attempt/'summary.json');repairs=[r for r in rve['surface_resolution_attempts'] if 'runtime_redistribution' in r]
    def table(name,rows):
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rows=[]
    for c in plane['cases']:
        for index,e in enumerate(c['events']):
            r=e['report'];rows.append(dict(tolerance=c['tolerance'],initial_slip_mm=c['initial_slip_mm'],increment=index+1,dx_mm=e['dx_mm'],weight_change=r['relative_area_weight_change'],work_error=r['accepted_work_error'],normal_energy_error=r['accepted_energy_errors']['normal_energy'],tangent_energy_error=r['accepted_energy_errors']['tangent_energy'],coulomb_ratio=e['coulomb_ratio'],checkpoint_force_error=c['checkpoint_force_error']))
    table('plane-cycle.csv',rows)
    progress=[json.loads(line) for line in (attempt/'progress.jsonl').read_text().splitlines()]
    table('packing-prefix.csv',progress)
    events=[]
    for e in rve['surface_resolution_attempts']:
        if (e.get('feature') or {}).get('kind')=='parent_quadrature':
            r=e.get('runtime_redistribution',{});events.append(dict(accepted_time_s=e['accepted_time_s'],trial_time_s=e['trial_time_s'],repaired=e['repaired'],weight_change=r.get('relative_area_weight_change'),work_error=r.get('accepted_work_error'),reason=e.get('repair_stop','')))
    if events:table('runtime-attempts.csv',events)
    diagnosis=read(src/'constraint-diagnosis-003/summary.json')
    budget=read(src/'constraint-diagnosis-003/accepted-budget-consistency.json')
    if diagnosis['physical_trajectory_advanced']:raise ValueError('Reduced diagnostic altered physical trajectory')
    table('constraint-blocks.csv',[dict(block=c['block'],feasible=c['feasible'],response_blocks=c['normal_response_blocks'],samples=c['samples'],conservation_rows=c['conservation_rows'],accepted_as_physical_result=c['accepted_as_physical_result']) for c in diagnosis['cases']])
    (out/'accepted-budget-consistency.json').write_text(json.dumps(budget,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for slip in (.0001,.01):
        selected=[r for r in rows if r['initial_slip_mm']==slip and r['increment']==1]
        axes[0].loglog([r['tolerance'] for r in selected],[max(r['work_error'],1e-18) for r in selected],'o-',label=f'initial slip {slip} mm')
    axes[0].set(xlabel='Relative integration tolerance',ylabel='Accepted-force conservation error');axes[0].legend(fontsize=8)
    old=Path('results/parent-hybrid-development-001/packing-prefix/attempt-002/progress.jsonl')
    previous=[json.loads(l) for l in old.read_text().splitlines()]
    old_by_step={r['step']:r for r in previous if r['step']>=90206}
    common=[r for r in progress if r['step'] in old_by_step and r['step']<=90262]
    fields=('time_s','porosity','axial_compression','nominal_pressure_MPa','kinetic_Nmm','internal_Nmm')
    identity=dict(shared_accepted_steps=len(common),last_shared_step=max(r['step'] for r in common),max_absolute_differences={field:max(abs(r[field]-old_by_step[r['step']][field]) for r in common) for field in fields})
    identity['exact']=all(value==0 for value in identity['max_absolute_differences'].values())
    (out/'prefix-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    for label,data in [('Previous',previous),('Runtime repair',progress)]:
        selected=[r for r in data if r['step']>=90206]
        axes[1].plot([r['step'] for r in selected],[100*r['porosity'] for r in selected],'.-',label=label)
    axes[1].axvline(90262,color='red',ls='--',label='Previous stop');axes[1].ticklabel_format(useOffset=False,style='plain',axis='y')
    axes[1].set(xlabel='Accepted packing step',ylabel='Transient porosity (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Runtime quadrature verification; full compaction and convergence NOT completed')
    fig.savefig(out/'verification.png',dpi=170);plt.close(fig)
    state,metadata=load(rve['checkpoint_paths'][-1]);render(out/'packing-mesh.png',state,f"Actual transient mesh: porosity {100*rve['final_porosity']:.6f}% (NOT equilibrium)",metadata['config']['length_mm'])
    # All attempt summaries/logs and execution source snapshots, including pilot failures.
    for file in src.rglob('*'):
        if not file.is_file() or '__pycache__' in file.parts or any(part.startswith('checkpoint-') for part in file.relative_to(src).parts):continue
        if file.suffix in ('.json','.log','.csv','.py') or file.name in ('progress.jsonl','contact-integration-budget.jsonl','constraint-matrices.npz'):
            dst=out/'diagnostics'/file.relative_to(src);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dst)
    shutil.copytree(rve['checkpoint_paths'][-1],out/'last-accepted-checkpoint')
    for name in ('last-state.npz','last-mesh.vtu','contact-quadrature.npz'):
        if (attempt/name).exists():shutil.copy2(attempt/name,out/name)
    for folder in ('packing_fem','tests','tools'):
        dst=out/'source'/folder;dst.mkdir(parents=True)
        for file in Path(folder).glob('*.py'):shutil.copy2(file,dst/file.name)
    assessment=dict(plane_cycle_passed=plane['passed'],plane_cases=len(plane['cases']),runtime_repairs=len(repairs),packing_last_accepted_step=rve['steps'],packing_transient_porosity=rve['final_porosity'],failure=rve['failure'],parent_attempts=events,previous_prefix_identity=identity,accepted_budget_consistency=budget,full_compaction_completed=False,loaded_target_equilibrated=False,convergence_verified=False)
    (out/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    (out/'README.md').write_text('''# 時間発展中の保存再配分の検証

受理済み配置で力・エネルギー・摩擦履歴を保持し、次の試行配置に合わせて既存点の重みを再配分する。
平面接触の3精度×2初期滑り条件で正逆増分と接触状態の保存/再開を検査。
実充填層の停止理由・再配分件数は[判定JSON](assessment.json)を参照。
停止状態の空隙率は過渡値。40%→10%の圧密・保持/除荷・収束確認は未完了。

- [比較画像](verification.png)、[実メッシュ](packing-mesh.png)
- [平面CSV](plane-cycle.csv)、[充填層CSV](packing-prefix.csv)
- [制約切り分けCSV](constraint-blocks.csv)、[受理済み状態と予算配分](accepted-budget-consistency.json)
- [報告](../../docs/runtime-parent-verification.md)、[ZIP](../'''+out.name+'''.zip)

ZIPのdiagnosticsは中断・失敗した試行も含む。重複チェックポイント・大きな生の接触台帳はresultsに保持。
last-accepted-checkpointは受理済みの状態。sourceは最終実装、各実行内sourceは実行時の実装。
''')
    manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob('*')) if f.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(out.parent))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Corrupt ZIP')
        for name,digest in manifest.items():
            if hashlib.sha256(z.read(str(Path(out.name)/name))).hexdigest()!=digest:raise ValueError('ZIP checksum mismatch')
    print(json.dumps(dict(assessment=assessment,archive=str(archive),bytes=archive.stat().st_size)))


if __name__=='__main__':main()
