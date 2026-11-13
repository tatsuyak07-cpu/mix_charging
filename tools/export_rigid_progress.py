"""Export an audited, explicitly incomplete rigid-inclusion restart path."""
import argparse,csv,gc,hashlib,json,shutil
from pathlib import Path
import numpy as np
import ipctk as ipc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from packing_fem.rigid_inclusion import make_periodic,make_contact,cylinder_surface,porosities,rigid_volume
from packing_fem.explicit_material import geometry,response
from packing_fem.explicit_periodic import deformation
from tools.audit_rigid_compaction import load
from tools.export_pressure_stages import render
from tools.export_rigid_compaction import sections


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export(segments,out,force_audit,audit_state,resume_config,guess,mesh):
    out=Path(out);c=json.loads((Path(segments[-1][0])/'config.json').read_text())
    physics=['length_mm','young_mpa','poisson','yield_mpa','hardening_mpa','friction','transverse_boundary','ipc_dhat_mm','ipc_stiffness_mpa','ipc_slip_regularization_mm','rigid_inclusion','rigid_diameter_mm','rigid_height_mm','rigid_segments','rigid_center_mm','implicit_force_tolerance_N']
    rows=[];frames=[];previous=None;elapsed=0.;sources=[]
    for directory,last in segments:
        directory=Path(directory);config=json.loads((directory/'config.json').read_text())
        if any(config.get(k)!=c.get(k) for k in physics):raise ValueError('Physical model changed across selected segments')
        initial=directory/'state-0000.npz'
        if previous is None:frames.append((initial,None))
        else:
            a=load(previous);b=load(initial)
            if set(a)!=set(b) or any(not np.array_equal(a[k],b[k]) for k in a):raise ValueError('Restart arrays do not match exactly')
        with (directory/'history.csv').open() as stream:selected=[r for r in csv.DictReader(stream) if int(r['step'])<=last]
        if not selected or int(selected[-1]['step'])!=last:raise ValueError('Selected accepted prefix unavailable')
        for saved in selected:
            step=int(saved['step']);row=dict(saved,source_run=str(directory),source_step=step,source_wall_s=saved['wall_s'])
            row['step']=len(rows)+1;row['wall_s']=elapsed+float(saved['wall_s']);rows.append(row)
            previous=directory/f'state-{step:04d}.npz';frames.append((previous,row))
        elapsed=float(rows[-1]['wall_s']);sources.append(directory)
    audit=json.loads(Path(force_audit).read_text())
    if not audit['passed'] or digest(previous)!=digest(audit_state):raise ValueError('The independent force audit does not identify this checkpoint')
    if abs(audit['states'][-1]['matrix_region_porosity']-float(rows[-1]['porosity']))>1e-10:raise ValueError('Audited porosity mismatch')
    initial=load(frames[0][0]);loaded=load(previous);reference=float(initial['V0'].sum());local,faces=cylinder_surface(c);reference_distances=pdist(local)
    records=[];contact=None;p=None;cached_t=None
    for index,(path,row) in enumerate(frames):
        d=load(path)
        if p is None or not np.array_equal(d['X'],p.X) or not np.array_equal(d['t'],cached_t):
            contact=None;gc.collect();p=make_periodic(d['X'],c['length_mm'],c);cached_t=d['t'].copy();contact=make_contact(d['X'],d['t'],p,c)
        H=deformation(d['q'][-6:]);volumes=geometry(d['x'],d['t'])[0];J=response(d['b'],d['ep'],c)[2]
        integrated=np.bincount(d['parent'],weights=d['V0']*J,minlength=len(d['t']))
        phi,total=porosities(volumes.sum(),p.L**3*np.linalg.det(H),c);center,(R,_,_)=p.pose(d['q']);rotated=local@R.T
        record=dict(step=index,source=str(path),sha256=digest(path),compression=float(-d['q'][p.prescribed]),matrix_region_porosity=phi,total_porosity=total,reference_volume_relative_error=abs(float(d['V0'].sum()/reference)-1),minimum_element_volume_mm3=float(volumes.min()),quadrature_volume_relative_error=float(np.max(abs(integrated-volumes)/volumes)),fixed_lattice_error=float(abs(H-np.diag([1,1,H[2,2]])).max()),saved_position_error_mm=float(abs(d['x']-p.positions(d['q'])).max()),rigid_pair_distance_error_mm=float(abs(pdist(rotated)-reference_distances).max()),surface_intersections=bool(ipc.has_intersections(contact.mesh,contact.positions(d['q']))))
        record['csv_porosity_error']=0. if row is None else abs(phi-float(row['porosity']))
        record['passed']=bool(volumes.min()>0 and record['reference_volume_relative_error']<1e-7 and record['quadrature_volume_relative_error']<1e-7 and record['fixed_lattice_error']<1e-12 and record['saved_position_error_mm']<1e-8 and record['rigid_pair_distance_error_mm']<1e-9 and not record['surface_intersections'] and record['csv_porosity_error']<1e-10 and 0<=total<=phi<=1)
        records.append(record)
        if not record['passed']:raise ValueError('Saved geometry audit failed: '+str(path))
        if index%10==0:print(json.dumps(dict(audited_geometry_states=index+1,total=len(frames))),flush=True)
    del contact;gc.collect()
    out.mkdir(exist_ok=False,parents=True)
    (out/'geometry-path-audit.json').write_text(json.dumps(dict(passed=True,force_audit_scope='Only the endpoint has the separate full force/material/friction replay; this file audits geometry at every selected saved state.',states=records),indent=2)+'\n')
    shutil.copy2(force_audit,out/'endpoint-force-audit.json')
    for source,name in [(frames[0][0],'initial-state.npz'),(previous,'current-state.npz'),(resume_config,'resume-config.json'),(guess,'resume-initial-iterate.npz'),(mesh,'original-mesh.npz')]:shutil.copy2(source,out/name)
    (out/'config.json').write_text(json.dumps(c,indent=2)+'\n')
    for index,source in enumerate(sources,1):shutil.copy2(source/'history.csv',out/f'original-history-part-{index:02d}.csv')
    with (out/'history.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));writer.writeheader();writer.writerows(rows)
    for name,state in [('initial',initial),('current',loaded)]:
        p=make_periodic(state['X'],c['length_mm'],c);center,(R,_,_)=p.pose(state['q']);phi=records[0 if name=='initial' else -1]['matrix_region_porosity']
        render(out/f'{name}-mesh.png',state,f'Floating D4 x H4 mm rigid cylinder:16-grain prototype\nMatrix-region porosity {100*phi:.4f}%',c['length_mm'],footer='Actual FEM coordinates; displacement scale1x; orange = rigid\nPartial compression: target10% NOT reached; mesh/increment convergence unverified',rigid_surface=(local@R.T+center,faces))
    sections(initial,loaded,c,out/'pore-sections.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axes[0].plot([100*float(r['compression']) for r in rows],[100*float(r['porosity']) for r in rows],'.-',label='Matrix region');axes[0].plot([0],[100*records[0]['matrix_region_porosity']],'o',mfc='none',label='Initial geometry');axes[0].axhline(10,color='gray',ls='--',label='Target (not reached)');axes[0].set(xlabel='Axial compression (%)',ylabel='Porosity (%)');axes[0].legend();axes[0].grid(alpha=.3)
    axes[1].plot([100*float(r['porosity']) for r in rows],[float(r['nominal_pressure_MPa']) for r in rows],'.-');axes[1].set(xlabel='Matrix-region porosity (%)',ylabel='Nominal pressure (MPa)');axes[1].grid(alpha=.3);fig.suptitle('Accepted partial compression path; fixed transverse cell');fig.savefig(out/'compression-path.png',dpi=170);plt.close(fig)
    summary=dict(target_reached=False,initial_matrix_porosity=records[0]['matrix_region_porosity'],checkpoint_matrix_porosity=records[-1]['matrix_region_porosity'],checkpoint_total_porosity=records[-1]['total_porosity'],compression=float(rows[-1]['compression']),pressure_MPa=float(rows[-1]['nominal_pressure_MPa']),accepted_steps=len(rows),geometry_audited_states=len(records),rigid_volume_mm3=rigid_volume(c),prototype_only=True,full_scale_D20_H20_L100_calculated=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'provenance.json').write_text(json.dumps(dict(exact_restart_array_checks_passed=True,segments=[dict(run=str(run),last_step=last) for run,last in segments],endpoint_source=str(previous),endpoint_sha256=digest(previous),endpoint_audit_state_sha256=digest(audit_state),wall_time='Cumulative selected-segment wall_s; excludes other trial branches, external initial-guess preparation and failures after the selected prefix.'),indent=2)+'\n')
    (out/'README.md').write_text(f'''# 浮遊剛体円柱：圧縮計算の再開元\n\n**空隙率10%は未達です。このフォルダは再開元の収束済み状態です。**\n\n16粒子、球径10mm、焼結ネック径2mm、剛体円柱は直径4mm・高さ4mmの数値試験モデル。\n最終仕様の直径20mm・高さ20mm／100mm角セルとは区別します。E10MPa、ν0.33、摩擦0.3。\n円柱は自由並進・自由回転で、両横周期寸法24.140718mmを固定しています。\n\n充填層領域の空隙率{100*summary['initial_matrix_porosity']:.4f}%→{100*summary['checkpoint_matrix_porosity']:.4f}%。\n軸圧縮率{100*summary['compression']:.4f}%、公称圧力{summary['pressure_MPa']:.6f}MPa。\nセル全体の空隙率は{100*summary['checkpoint_total_porosity']:.4f}%。\n\n[現在メッシュPNG](current-mesh.png)・[初期メッシュ](initial-mesh.png)・[剛体中心断面](pore-sections.png)・[圧縮曲線](compression-path.png)・[統合CSV](history.csv)。\n[全{len(records)}状態の体積・周期・剛体形状・非交差監査](geometry-path-audit.json)・[最終状態の力・材料履歴・摩擦の独立再計算](endpoint-force-audit.json)。\n力とトルク／半径は各成分0.005N未満を確認します。ベクトルのノルムとは区別します。\n\n骨格の現在体積には弾性圧縮を含み、基準材料体積と剛体形状を保存します。\n初期NPZは数値緩和前の幾何形状です。単一試作で、メッシュ・増分収束、剛体の有無の定量比較、保持・除荷は未認定です。\nCSVの時間は選択した区間の累積で、別試行や初期推定の準備費用を含みません。\n\n## 再開用データ\n\n[収束済み状態](current-state.npz)・[元メッシュ](original-mesh.npz)・[再開設定](resume-config.json)・[初期推定だけに使う未収束配置](resume-initial-iterate.npz)。\n初期推定の材料履歴は受理せず、収束済み状態の履歴から計算します。出力先は未使用の名前にします。\n\n```bash\nPYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/run_implicit_compaction.py \\\n  --config {out}/resume-config.json \\\n  --mesh {out}/original-mesh.npz \\\n  --restart {out}/current-state.npz \\\n  --initial-iterate {out}/resume-initial-iterate.npz \\\n  --out results/new-rigid-continuation\n```\n''')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--segment',nargs=2,action='append',required=True);parser.add_argument('--out',required=True);parser.add_argument('--force-audit',required=True);parser.add_argument('--audit-state',required=True);parser.add_argument('--resume-config',required=True);parser.add_argument('--guess',required=True);parser.add_argument('--mesh',required=True);args=parser.parse_args()
    export([(run,int(last)) for run,last in args.segment],args.out,args.force_audit,args.audit_state,args.resume_config,args.guess,args.mesh)
