"""Export only accepted full-path prototype states, plus all failed run logs."""
import argparse,csv,json,shutil,zipfile,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.export_pressure_stages import render,boundary
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from packing_fem.explicit_periodic import deformation


def render_cycle_comparison(root,diagnostic_dir,path):
    with (diagnostic_dir/'history.csv').open() as stream:rows=list(csv.DictReader(stream))
    items=[('Initial; porosity 41.162%',root/'implicit-002/state-0000.npz')]
    for prefix,row in [('Loaded',rows[0]),('Zero axial load',rows[-1])]:
        title=f"{prefix}; porosity {100*float(row['porosity']):.3f}%\nAxial compression {100*float(row['compression']):.3f}%"
        items.append((title,diagnostic_dir/f"state-{int(row['step']):04d}.npz"))
    states=[]
    for title,source in items:
        with np.load(source) as data:state=dict(data)
        if 'parent' in state:
            parent=state['parent'];weights=state['V0']
            state['ep']=np.bincount(parent,weights=weights*state['ep'],minlength=len(state['t']))/np.bincount(parent,weights=weights,minlength=len(state['t']))
        states.append(state)
    cloud=np.concatenate([r['x'] for r in states]);lo=np.minimum(cloud.min(0),0)-.5;hi=np.maximum(cloud.max(0),12.1)+.5
    norm=plt.Normalize(0,max(r['ep'].max() for r in states));cmap=plt.get_cmap('viridis');fig=plt.figure(figsize=(15,5.7))
    for index,((title,source),state) in enumerate(zip(items,states)):
        ax=fig.add_subplot(1,3,index+1,projection='3d');faces,owners=boundary(state['t'])
        ax.add_collection3d(Poly3DCollection(state['x'][faces],facecolors=cmap(norm(state['ep'][owners])),edgecolors='#263e4a',linewidths=.1))
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=title)
        ax.set_box_aspect(hi-lo);ax.view_init(elev=23,azim=-53)
    fig.subplots_adjust(left=.02,right=.91,bottom=.15,top=.85,wspace=.05);bar=fig.add_axes([.94,.25,.012,.5])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=bar,label='Equivalent plastic strain')
    fig.suptitle('Large-deformation FEM: compression, hold and zero-load recovery',fontsize=15)
    fig.text(.5,.035,'Actual calculated mesh, displacement scale 1x; identical axis scales. E=10 MPa, friction=0.3.\nPrototype: requested initial porosity 40% and mesh / increment convergence remain unverified.',ha='center',fontsize=10)
    fig.savefig(path,dpi=170);plt.close(fig)


def export(root,destination):
    root=Path(root);out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    chain=['implicit-002','implicit-003','implicit-004','implicit-005','implicit-006','implicit-007','implicit-008']
    records=[]
    for name in chain:
        path=root/name/'history.csv'
        if not path.exists():continue
        config=json.loads((root/name/'config.json').read_text())
        with path.open() as f:
            for row in csv.DictReader(f):
                record={k:(v if k=='phase' else float(v)) for k,v in row.items()}
                record.update(run=name,state=f"{name}/state-{int(record['step']):04d}.npz",residual_acceptance_N=config['implicit_force_tolerance_N'])
                with np.load(root/record['state']) as checkpoint:
                    x=checkpoint['x'];tet=checkpoint['t'];H=deformation(checkpoint['q'][-6:]);a=x[tet]
                    solid=float(np.linalg.det((a[:,1:]-a[:,:1]).swapaxes(1,2)).sum()/6)
                    cell=float(config['length_mm']**3*np.linalg.det(H))
                    if abs((1-solid/cell)-record['porosity'])>1e-9:raise ValueError('CSV and mesh porosity mismatch')
                    record.update(solid_volume_mm3=solid,cell_volume_mm3=cell,cell_volume_ratio=float(np.linalg.det(H)),transverse_area_ratio=float(np.linalg.norm(np.cross(H[:,0],H[:,1]))),lateral_stretch_x=float(np.linalg.norm(H[:,0])),lateral_stretch_y=float(np.linalg.norm(H[:,1])))
                records.append(record)
    if not records:raise ValueError('No accepted states')
    fields=list(dict.fromkeys(k for row in records for k in row))
    with (out/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
    initial=root/'implicit-002/state-0000.npz'
    choices=[('initial',initial,None)]
    loading=[r for r in records if r['phase']=='compression']
    for phi in [.4,.3,.2,.1]:
        row=min(loading,key=lambda r:abs(r['porosity']-phi))
        if abs(row['porosity']-phi)<.015:choices.append((f'porosity-{round(100*phi):02d}',root/row['state'],row))
    if records[-1]['phase']=='unload':choices.append(('recovered',root/records[-1]['state'],records[-1]))
    best=min(loading,key=lambda row:row['porosity'])
    choices.append(('minimum-porosity',root/best['state'],best))
    choices.append(('latest',root/records[-1]['state'],records[-1]))
    for label,path,row in choices:
        with np.load(path) as data:state=dict(data)
        if 'parent' in state:
            parent=state['parent'];weights=state['V0']
            state['ep']=np.bincount(parent,weights=weights*state['ep'],minlength=len(state['t']))/np.bincount(parent,weights=weights,minlength=len(state['t']))
        if row is None:title='Initial faceted mesh: porosity 41.162%'
        else:title=f"{row['phase']}; porosity {100*row['porosity']:.3f}%\nAxial compression {100*row['compression']:.2f}%; nominal stress {row['nominal_pressure_MPa']:.5f} MPa"
        render(out/(label+'-mesh.png'),state,'Unvalidated finite-strain FEM prototype\n'+title,12.1)
        shutil.copy2(path,out/(label+'-mesh.npz'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for phase,color in [('compression','#1267a5'),('hold','#c57a00'),('unload','#a52c48')]:
        rows=[r for r in records if r['phase']==phase]
        if rows:
            axes[0].plot([100*r['compression'] for r in rows],[100*r['porosity'] for r in rows],'.-',color=color,label=phase)
            axes[1].plot([r['nominal_pressure_MPa'] for r in rows],[100*r['porosity'] for r in rows],'.-',color=color,label=phase)
    for ax in axes:ax.axhline(10,color='gray',ls='--',lw=.8);ax.set_ylabel('Measured 3D porosity (%)');ax.grid(alpha=.25);ax.legend()
    axes[0].set_xlabel('Axial compression (%)');axes[1].set_xlabel('Nominal axial compression stress (MPa)')
    if max(abs(r['nominal_pressure_MPa']) for r in records)>2:axes[1].set_xscale('symlog',linthresh=.01)
    fig.suptitle('Actual accepted states; prototype, mesh / increment convergence unverified')
    fig.savefig(out/'porosity-path.png',dpi=180);plt.close(fig)
    diagnostic=None
    diagnostic_dir=root/'hold-unload-002'
    if (diagnostic_dir/'status.json').exists():
        diagnostic=json.loads((diagnostic_dir/'status.json').read_text())
        shutil.copy2(diagnostic_dir/'history.csv',out/'demonstration-hold-unload.csv')
        shutil.copy2(diagnostic_dir/'status.json',out/'demonstration-hold-unload.json')
        render_cycle_comparison(root,diagnostic_dir,out/'demonstration-cycle.png')
        with (diagnostic_dir/'history.csv').open() as f:release=list(csv.DictReader(f))
        for label,row in [('demonstration-loaded',release[0]),('demonstration-recovered',release[-1])]:
            path=diagnostic_dir/f"state-{int(row['step']):04d}.npz"
            with np.load(path) as data:state=dict(data)
            if 'parent' in state:
                weights=state['V0'];parent=state['parent']
                state['ep']=np.bincount(parent,weights=weights*state['ep'],minlength=len(state['t']))/np.bincount(parent,weights=weights,minlength=len(state['t']))
            render(out/(label+'-mesh.png'),state,f"Diagnostic {row['phase']}; porosity {100*float(row['porosity']):.3f}%\nNot a 10% target result",12.1)
            shutil.copy2(path,out/(label+'-mesh.npz'))
    final=records[-1];status=json.loads((root/final['run']/'status.json').read_text())
    summary=dict(status=status,latest=final,initial_porosity=0.4116199849746327,requested_initial_porosity=.4,validated=False,archive_root=str(root),accepted_states=len(records),minimum_loaded_porosity_state=best,loaded_target_reached=bool(best['porosity']<=.101),demonstration_cycle=diagnostic,note='No failed iterate is included in the curve. Retrospective development chain; see each run configuration and termination/failure logs.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name in ['tests.log','tests-final.log','refinement-probe.log','cavity-probe.log','remesh-probe.log','state-audit-001.json','state-audit-001.log','state-audit-002.json','state-audit-002.log','contact-vectorization.log','tests-complete.log','path-state-audit.json','path-state-audit.log','state-audit-final.json','state-audit-final.log','provenance.json']:
        if (root/name).exists():shutil.copy2(root/name,out/name)
    (out/'README.md').write_text('# 大変形を先に計算する試作結果\n\n実際に釣り合い基準を満たして保存した状態だけをCSVと曲線に含めています。\n初期メッシュの測定空隙率は41.162%で、指定の40%と差があります。\n材料・荷重増分・摩擦正則化・メッシュの収束を認定した結果ではありません。\n\n`summary.json`に到達状態、`history.csv`に空隙率・公称応力・残差、\n`*-mesh.png`に変形倍率1倍の計算メッシュ、`*-mesh.npz`に節点・四面体・材料履歴を保存しています。\n圧縮・保持・除荷は別のphaseです。\n\n完全な入力・ソーススナップショット・失敗試行は隣接ZIPに保存しています。\n')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():z.write(path,'raw/'+str(path.relative_to(root)))
        for path in sorted(out.iterdir()):
            if path.is_file():z.write(path,'exports/'+path.name)
        for folder in ['packing_fem','tools']:
            for path in sorted(Path(folder).glob('*.py')):z.write(path,'final-source/'+str(path))
        for name in ['configs/implicit-ipc-compaction.json','docs/large-deformation-first.md','requirements-implicit.txt','requirements-lock.txt','pyproject.toml','AGENTS.md']:
            path=Path(name)
            if path.exists():z.write(path,'final-source/'+str(path))
    (out/'archive-sha256.txt').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();export(a.root,a.out)
