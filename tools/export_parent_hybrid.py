"""Export actual parent-contact diagnostics, including retained failed runs."""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from packing_fem.checkpoint import load
from tools.export_pressure_stages import render


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();src=a.source;out=a.output;out.mkdir(parents=True,exist_ok=False)
    read=lambda path:json.loads(path.read_text())
    plane=read(src/'plane-three-levels-ipm/summary.json')
    coupled=[read(src/f'hybrid-cfl-{tag}-ipm/summary.json') for tag in ('020','010','005')]
    if not plane['passed'] or not all(d['passed'] for d in coupled):raise ValueError('Final component verification failed')
    def csvfile(name,rows):
        with (out/name).open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rows=[]
    for c in plane['cases']:
        t=c['transfer'];rows.append(dict(tolerance=c['tolerance'],slip_mm=c['slip_mm'],passed=c['passed'],weight_change=t['redistribution']['relative_area_weight_change'],work_error=t['redistribution_work_error'],normal_energy_error=t['redistribution_energy_errors']['normal_energy'],tangent_energy_error=t['redistribution_energy_errors']['tangent_energy']))
    csvfile('plane-comparison.csv',rows)
    crows=[]
    for cfl,d in zip((.2,.1,.05),coupled):
        t=d['results']['enabled']['events'][0]['retained_embedded_samples'];crows.append(dict(cfl=cfl,passed=d['passed'],weight_change=t['redistribution']['relative_area_weight_change'],work_error=t['redistribution_work_error'],normal_energy_error=t['redistribution_energy_errors']['normal_energy'],tangent_energy_error=t['redistribution_energy_errors']['tangent_energy']))
    csvfile('coupled-cfl.csv',crows)
    brows=[]
    for scale,name in ((1.,'frozen-packing'),(.5,'frozen-packing-half-budget'),(.25,'frozen-packing-quarter-budget')):
        d=read(src/name/'summary.json')
        for c in d['cases']:
            brows.append(dict(budget_scale=scale,edge='-'.join(map(str,c['edge'])),passed=c['passed'],histories=c['transfer']['active_histories'],work_error=c['transfer']['periodic_virtual_work_relative_error'],new_material_samples=c['transfer']['new_material_samples']))
    csvfile('frozen-packing-budgets.csv',brows)
    run=src/'packing-prefix';summary=read(run/'summary.json');attempt=run/f"attempt-{summary['attempts'][-1]['attempt']:03d}"
    result=read(attempt/'summary.json');progress=[json.loads(line) for line in (attempt/'progress.jsonl').read_text().splitlines()]
    csvfile('packing-prefix.csv',progress)
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for slip in sorted({r['slip_mm'] for r in rows}):
        selected=[r for r in rows if r['slip_mm']==slip]
        axes[0].loglog([r['tolerance'] for r in selected],[r['weight_change'] for r in selected],'o-',label=f'slip {slip} mm')
    axes[0].set(xlabel='Parent relative tolerance',ylabel='Area-weighted relative weight change');axes[0].legend(fontsize=8)
    for field in ('work_error','normal_energy_error','tangent_energy_error'):
        axes[1].semilogy([r['cfl'] for r in crows],[max(r[field],1e-18) for r in crows],'o-',label=field)
    axes[1].set(xlabel='CFL (three-step transaction only)',ylabel='Relative conservation error');axes[1].legend(fontsize=7)
    resumed=[r for r in progress if r['step']>=90206]
    axes[2].plot([r['step'] for r in resumed],[100*r['porosity'] for r in resumed],'o-',markersize=2)
    axes[2].axvline(90250,color='green',ls='--',label='Accepted remesh');axes[2].set(xlabel='Accepted packing step',ylabel='Transient porosity (%)');axes[2].legend(fontsize=8)
    fig.suptitle('Parent audit + conservative reweighting: component checks; full compaction NOT completed')
    fig.savefig(out/'verification.png',dpi=170);plt.close(fig)
    checkpoint=Path(result['checkpoint_paths'][-1]);state,metadata=load(checkpoint)
    render(out/'packing-mesh.png',state,f"Actual transient mesh: porosity {100*result['final_porosity']:.6f}% (NOT equilibrium)",metadata['config']['length_mm'])
    # Keep lightweight summaries/source for every successful AND failed run.
    # Large duplicate checkpoints stay in the original result directory.
    for path in src.rglob('*'):
        if not path.is_file() or '__pycache__' in path.parts or any(part.startswith('checkpoint-') for part in path.relative_to(src).parts):continue
        if path.suffix in ('.json','.log','.csv','.py','.pkl') or path.name in ('progress.jsonl',):
            dest=out/'diagnostics'/path.relative_to(src);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    shutil.copytree(checkpoint,out/'last-accepted-checkpoint')
    for name in ('last-state.npz','last-mesh.vtu','contact-quadrature.npz','surface-remesh-attempt-0001.npz'):
        if (attempt/name).exists():shutil.copy2(attempt/name,out/name)
    for folder in ('packing_fem','tests','tools'):
        dest=out/'source'/folder;dest.mkdir(parents=True)
        for path in Path(folder).glob('*.py'):shutil.copy2(path,dest/path.name)
    assessment=dict(component_checks_passed=True,packing_remesh_accepted=result['remesh_events'][0]['accepted'],packing_last_accepted_step=result['steps'],packing_transient_porosity=result['final_porosity'],packing_failure=result['failure'],full_compaction_completed=False,general_repeated_remesh_completed=False,stage3_completed=False,tests_log='diagnostics/final-tests-ipm.log')
    (out/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    (out/'README.md').write_text('''# 親面監査・保存再配分の検証データ

平面接触6条件、CFL 3水準の3刻み移行、固定充填層2辺×誤差予算3水準を検証。
実充填層は90250刻みで接触中の再メッシュを承認し、90262刻みまで継続後、親面精度不足で停止。
空隙率36.941697%は過渡値。空隙率10%・保持/除荷完走・収束確認は未達成。

- [検証図](verification.png)、[実メッシュ](packing-mesh.png)
- [平面CSV](plane-comparison.csv)、[CFL比較CSV](coupled-cfl.csv)、[固定配置CSV](frozen-packing-budgets.csv)、[実充填層の過渡CSV](packing-prefix.csv)
- [判定JSON](assessment.json)、[詳細報告](../../docs/parent-hybrid-development.md)、[ダウンロードZIP](../'''+out.name+'''.zip)

ZIP内のdiagnosticsには失敗した試行も含む。last-accepted-checkpointは最後の受理状態。
重複する中間チェックポイントと大きな生の接触台帳は元のresultsに保持する。
sourceは最終実装、各試行内sourceはその実行時の実装。実充填層では重み再配分を使っていない。
''')
    manifest={str(path.relative_to(out)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.rglob('*')) if path.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(out.parent))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Corrupt ZIP')
        for name,digest in manifest.items():
            if hashlib.sha256(z.read(str(Path(out.name)/name))).hexdigest()!=digest:raise ValueError('ZIP hash mismatch')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,assessment=assessment)))


if __name__=='__main__':main()
