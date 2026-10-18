"""Archive completed layered contact trials, including failed path diagnostics.

The whole-contact-window kinetic diagnostic is sampled and gives a lower bound
on the true maximum. Passing a sampled maximum is never an all-step proof.
"""
import argparse,csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def contact_window(h,energy_scale):
    active=(h['contacts']>0)&(h['command_pressure_MPa']>.01*np.max(h['command_pressure_MPa']))
    scale=np.maximum(abs(h['internal_Nmm']-h['internal_Nmm'][0]),energy_scale*1e-8)
    ratio=h['kinetic_Nmm']/scale
    return active,ratio


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mechanism',type=Path,required=True);p.add_argument('--connected',type=Path,required=True);p.add_argument('--connectivity',type=Path,required=True);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.zip').exists():raise FileExistsError('Exclusive output required')
    for root in a.inputs:
        for name in ('summary.json','endpoint-audit.json','run/history.csv'):
            if not (root/name).is_file():raise ValueError(f'Incomplete run: {root/name}')
    mechanism=json.loads((a.mechanism/'summary.json').read_text())
    connected=json.loads((a.connected/'summary.json').read_text());connectivity=json.loads((a.connectivity/'summary.json').read_text())
    a.output.mkdir(parents=True);shutil.copytree(a.mechanism,a.output/'mechanism');shutil.copytree(a.connected,a.output/'connected-initial-stiffness');shutil.copytree(a.connectivity,a.output/'periodic-connectivity');rows=[];histories={};comparisons=[]
    for root in a.inputs:
        shutil.copytree(root,a.output/root.name)
        s=json.loads((root/'summary.json').read_text());r=json.loads((root/'run/summary.json').read_text());e=json.loads((root/'endpoint-audit.json').read_text());h=np.atleast_1d(np.genfromtxt(root/'run/history.csv',delimiter=',',names=True));histories[root.name]=(s,r,e,h)
        active,ratio=contact_window(h,r['energy_scale_Nmm']);total_internal_contact=abs(h['internal_Nmm']-h['internal_Nmm'][0])+h['normal_contact_Nmm']+h['tangent_contact_Nmm'];total_ratio=h['kinetic_Nmm']/np.maximum(total_internal_contact,r['energy_scale_Nmm']*1e-8);peak=float(np.max(ratio[active])) if np.any(active) else None
        row=dict(case=root.name,pressure_MPa=s['pressure_mpa'],ramp_ms=s['ramp_s']*1000,cfl=s['cfl'],steps=r['steps'],final_time_ms=float(h['time_s'][-1]*1000),elapsed_s=s['elapsed_s'],path_completed=r['checks']['path_completed'],solver_and_endpoint_checks_passed=s['solver_and_endpoint_checks_passed'],diagonal_branch_checks_passed=s['diagonal_branch_checks_passed'],initial_porosity=r['initial_porosity'],final_porosity=r['final_porosity'],axial_compression=r['final_axial_compression'],current_gap_mm=e['current_gap_mm'],stress_relative_error=e['stress_relative_error'],macro_shear_norm=e['macro_shear_norm'],loaded_kinetic_ratio=r['kinetic_internal_ratio_loaded'],sampled_contact_window_kinetic_ratio_lower_bound=peak,sampled_contact_window_kinetic_over_internal_plus_contact_lower_bound=float(np.max(total_ratio[active])) if np.any(active) else None,contact_window_exceeds_1percent=peak is not None and peak>.01,energy_relative_error=r['energy_relative_error'],coulomb_peak=float(np.max(h['coulomb_ratio'])),peak_peeq=e['peak_peeq'],independent_porosity_difference=abs(e['independent_geometric_porosity']-e['independent_material_porosity']),failure=r['failure'])
        rows.append(row)
    for T in sorted({r['ramp_ms'] for r in rows}):
        group=[r for r in rows if r['ramp_ms']==T]
        if len(group)!=2:continue
        coarse,fine=sorted(group,key=lambda r:r['cfl'],reverse=True)
        hc=histories[coarse['case']][3];hf=histories[fine['case']][3]
        times=np.linspace(0,min(hc['time_s'][-1],hf['time_s'][-1]),401)
        comparisons.append(dict(kind='CFL_halving_same_rate',ramp_ms=T,coarse=coarse['case'],fine=fine['case'],endpoint_porosity_difference=abs(coarse['final_porosity']-fine['final_porosity']),endpoint_axial_difference=abs(coarse['axial_compression']-fine['axial_compression']),max_sampled_porosity_trace_difference=float(np.max(abs(np.interp(times,hc['time_s'],hc['porosity'])-np.interp(times,hf['time_s'],hf['porosity'])))),comparison_is_mesh_convergence=False))
    for cfl in sorted({r['cfl'] for r in rows}):
        group=sorted([r for r in rows if r['cfl']==cfl],key=lambda r:r['ramp_ms'])
        if len(group)==2:
            comparisons.append(dict(kind='rate_halving_same_CFL',cfl=cfl,fast=group[0]['case'],slow=group[1]['case'],endpoint_porosity_difference=abs(group[0]['final_porosity']-group[1]['final_porosity']),endpoint_axial_difference=abs(group[0]['axial_compression']-group[1]['axial_compression']),comparison_is_rate_convergence=False))
    with (a.output/'comparison.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=dict(connected_initial_stiffness_checks=connected['checks'],periodic_connectivity_checks=connectivity['checks'],prepared_packing_stiffness_verified=False,mechanism_checks=mechanism['checks'],initial_fixture_has_unstiffened_closing_mode=all(mechanism['checks'].values()),scope='actual_1mm_20percent_layered_pore_contact_transients_not_particle_packing',cases=len(rows),solver_and_endpoint_pass_count=sum(r['solver_and_endpoint_checks_passed'] for r in rows),diagonal_branch_pass_count=sum(r['diagonal_branch_checks_passed'] for r in rows),sampled_contact_window_exceedance_count=sum(r['contact_window_exceeds_1percent'] for r in rows),contact_window_diagnostic='contacts>0 and command>1percent of observed peak; max KE/internal-change from output snapshots; lower bound on full-step maximum',all_step_contact_window_audit_available=False,quasistatic_contact_path_validated=False,mesh_convergence_verified=False,particle_compaction_completed=False,comparisons=comparisons)
    (a.output/'verification-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for name,(s,r,e,h) in histories.items():
        label=f"T={s['ramp_s']*1000:g} ms, CFL={s['cfl']}";phase=h['time_s']/s['ramp_s'];active,ratio=contact_window(h,r['energy_scale_Nmm'])
        axes[0].plot(phase,100*h['porosity'],label=label)
        axes[1].semilogy(phase,np.where(active,np.maximum(ratio,1e-10),np.nan),label=label)
        axes[2].plot(phase,h['energy_residual_Nmm']/r['energy_scale_Nmm'],label=label)
    axes[0].set(ylabel='Porosity [%]');axes[1].set(ylabel='Sampled contact-window kinetic / internal change');axes[1].axhline(.01,color='black',linestyle=':',label='1% criterion');axes[1].axvline(.5,color='gray',linestyle='--')
    axes[2].set(ylabel='Energy residual / energy scale')
    for ax in axes:ax.set_xlabel('Time / ramp duration');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Actual periodic plane-pore collapse: E=10 MPa, target 0.02 MPa (diagnostic trials)');fig.tight_layout();fig.savefig(a.output/'layered-contact-verification.png',dpi=180);plt.close(fig)
    chosen=sorted(rows,key=lambda r:(-r['ramp_ms'],r['cfl']))[0];root=a.output/chosen['case'];state=np.load(root/'run/last-state.npz');X=state['reference_points'];x=state['points'];t=state['tets']
    edges=sorted({tuple(sorted((int(tet[i]),int(tet[j])))) for tet in t for i in range(4) for j in range(i+1,4)})
    lo=np.minimum(X.min(axis=0),x.min(axis=0))-.04;hi=np.maximum(X.max(axis=0),x.max(axis=0))+.04
    fig=plt.figure(figsize=(10,5))
    for panel,pts,title in ((1,X,'Initial: 20% plane pore'),(2,x,f"Diagnostic final state: porosity {100*chosen['final_porosity']:.4f}%")):
        ax=fig.add_subplot(1,2,panel,projection='3d');ax.add_collection3d(Line3DCollection(pts[np.array(edges)],colors='navy',linewidths=.8));ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title=title);ax.set_box_aspect(hi-lo)
    fig.suptitle(f"Actual mesh: {len(X)} nodes / {len(t)} tets, scale 1; NOT sphere packing");fig.tight_layout();fig.savefig(a.output/'layered-mesh.png',dpi=180);plt.close(fig);state.close()
    logdir=a.output/'logs';logdir.mkdir()
    for log in sorted(Path('/tmp').glob('layered-cauchy-*.log')):shutil.copy2(log,logdir/log.name)
    shutil.copy2(__file__,a.output/'summarize_layered_pressure.py')
    (a.output/'README.md').write_text('''# 層状空隙セルの接触過渡検証\n\n1 mm検証セル、初期空隙率20%、空隙は平面状。球充填体ではありません。\nE=10 MPa、μ=0.3、目標Cauchy圧縮0.02 MPa。\n4 ms / 8 ms、CFL 0.2 / 0.1を比較し、全ケースと不合格判定を保存しています。\ncomparison.csv が集計、verification-summary.json が判定の区別です。\n各ケースの source と verification-plan.json は計算時のソースと事前の条件です。\n\n追加の接触期間KE診断は、接触数>0かつ指定圧力>観測最大圧力の1%の出力点を対象とします。\n出力点だけの最大値は全ステップ最大値の下限です。後半だけの判定とは区別し、\n sampled ratioが1%を超えれば、その接触期間を準静的に検証済みとは扱いません。\n対角解析枝との差を一般的な平衡解の不成立と混同しません。\nmechanism は、接触前に固体をひずませず空隙を閉じる無剛性モードの独立検査です。\nこの層状セルを連結した焼結骨格の準静的な圧密の代理には使いません。\nconnected-initial-stiffness は4 mm連結検証セルの初期接線の検査です。\nperiodic-connectivity は25 mm準備済み充填体を含む位相検査で、剛性や圧密の計算ではありません。\n\n表面再メッシュ・一般的な辺接触・球充填層の空隙率40%→10%の圧密は未完了です。\n''')
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()};(a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    archive=a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('ZIP CRC failure')
    print(json.dumps(summary,indent=2));print(json.dumps(dict(archive=str(archive),zip_crc='passed')))
if __name__=='__main__':main()
