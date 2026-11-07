"""Export fixed-transverse-cell results without replacing historical free-cell data."""
import argparse,csv,json,shutil,zipfile,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.export_pressure_stages import render
from packing_fem.explicit_periodic import deformation,Periodic
from packing_fem.explicit_material import geometry


def export(root,out,run_name="run-001",caption=None,make_archive=True):
    root=Path(root);run=root/run_name;out=Path(out);out.mkdir(parents=True,exist_ok=False)
    c=json.loads((run/'config.json').read_text());status=json.loads((run/'status.json').read_text())
    with (run/'history.csv').open() as f:rows=list(csv.DictReader(f))
    initial_porosity=float(status['initial_porosity'])
    with np.load(run/'state-0000.npz') as initial_state:
        initial_compression=float(-initial_state['q'][-4])
        initial_porosity=float(1-geometry(initial_state['x'],initial_state['t'])[0].sum()/(c['length_mm']**3*np.linalg.det(deformation(initial_state['q'][-6:]))))
    errors=[];pair_errors=[]
    for row in rows:
        path=run/f"state-{int(row['step']):04d}.npz"
        with np.load(path) as r:
            H=deformation(r['q'][-6:]);X=r['X'];x=r['x'];p=Periodic(X,c['length_mm'])
            for cl in np.unique(p.classes):
                ids=np.flatnonzero(p.classes==cl)
                if len(ids)<2:continue
                delta=X[ids]-X[ids[0]];horizontal=abs(delta[:,2])<1e-8
                if np.any(horizontal):pair_errors.append(float(np.max(abs(x[ids[horizontal]]-x[ids[0]]-delta[horizontal]))))
        errors.append(float(np.max(abs(H-np.diag([1.,1.,H[2,2]])))))
    if max(pair_errors,default=0.)>1e-8:raise ValueError('Paired transverse boundary vectors changed')
    if max(errors)>1e-12:raise ValueError('Transverse cell changed')
    verification=dict(accepted_states=len(rows),maximum_paired_boundary_vector_error_mm=max(pair_errors,default=0.),maximum_transverse_macro_error=max(errors),lateral_length_mm=c['length_mm'],transverse_area_ratio=1.,recovery='zero axial load with transverse constraints retained',validated=False)
    (out/'boundary-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    shutil.copy2(run/'history.csv',out/'history.csv');shutil.copy2(run/'status.json',out/'summary.json')
    compression=[r for r in rows if r['phase']=='compression'];holds=[r for r in rows if r['phase']=='hold']
    loaded=holds[-1] if holds else compression[-1]
    selections=[('initial',None),('loaded',loaded),('latest',rows[-1])]
    for name,row in selections:
        path=run/('state-0000.npz' if row is None else f"state-{int(row['step']):04d}.npz")
        with np.load(path) as r:state=dict(r)
        parent=state['parent'];weight=state['V0']
        state['ep']=np.bincount(parent,weights=weight*state['ep'],minlength=len(state['t']))/np.bincount(parent,weights=weight,minlength=len(state['t']))
        title=f'Initial mesh: measured porosity {100*initial_porosity:.3f}%' if row is None else f"{row['phase']}: porosity {100*float(row['porosity']):.3f}%\nAxial compression {100*float(row['compression']):.3f}%"
        render(out/(name+'-mesh.png'),state,(caption or 'Fixed transverse periodic cell: prototype')+'\n'+title,c['length_mm'],footer=None if caption is None else 'Actual FEM mesh; displacement scale 1x\n'+caption+'; mesh / increment convergence unverified')
        shutil.copy2(path,out/(name+'-mesh.npz'))
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for phase,color in [('compression','#1267a5'),('hold','#c57a00'),('unload','#a52c48')]:
        selected=[r for r in rows if r['phase']==phase]
        if selected:
            compression_values=[100*float(r['compression']) for r in selected];porosity_values=[100*float(r['porosity']) for r in selected]
            if phase=='compression' and initial_compression<float(selected[0]['compression']):
                compression_values.insert(0,100*initial_compression);porosity_values.insert(0,100*initial_porosity)
            axes[0].plot(compression_values,porosity_values,'.-',color=color,label=phase)
    axes[0].axhline(10,ls='--',color='gray',lw=.8);axes[0].set(xlabel='Axial compression (%)',ylabel='Measured porosity (%)');axes[0].legend();axes[0].grid(alpha=.25)
    for key,label in [('lateral_length_x_mm','Lx'),('lateral_length_y_mm','Ly')]:axes[1].plot([int(r['step']) for r in rows],[float(r[key]) for r in rows],label=label)
    axes[1].set(xlabel='Accepted step',ylabel='Periodic cell length (mm)',ylim=(.95*c['length_mm'],1.05*c['length_mm']));axes[1].legend();axes[1].grid(alpha=.25)
    fig.suptitle('Fixed transverse periodic vectors; mesh / increment convergence unverified');fig.savefig(out/'porosity-and-cell-size.png',dpi=180);plt.close(fig)
    (out/'README.md').write_text(f'''# 横方向の周期セル寸法を固定した計算\n\nX・Y方向の周期ベクトルを初期値に固定し、軸方向だけ圧縮します。\n両横寸法は全{len(rows)}状態で {c['length_mm']} mm、巨視的横方向変形の最大誤差は {max(errors):.3g} でした。\n周期的な微視的変位の揺らぎは許しています。横方向反力は拘束反力です。\n\n初期メッシュの測定空隙率は{100*initial_porosity:.3f}%（指定40%に対する粗い試行）です。\n保持時／最終載荷時空隙率 {100*float(loaded['porosity']):.4f}%、最終記録 {100*float(rows[-1]['porosity']):.4f}%。\n状態: `{status['status']}`。メッシュ・増分収束は未認定です。\n除荷は横方向拘束を維持したゼロ軸荷重回復であり、全方向無拘束ではありません。\n\n[メッシュ](loaded-mesh.png)・[最終メッシュ](latest-mesh.png)・[空隙率と横寸法](porosity-and-cell-size.png)・[CSV](history.csv)・[判定](summary.json)・[全データZIP](../fixed-transverse-001.zip)。\n''')
    if not make_archive:return
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for path in sorted(root.rglob('*')):
            if path.is_file():z.write(path,'raw/'+str(path.relative_to(root)))
        for path in out.iterdir():
            if path.is_file():z.write(path,'exports/'+path.name)
        for name in ['tools/run_implicit_compaction.py','tools/export_fixed_cell.py','configs/implicit-ipc-fixed-cell.json','docs/fixed-transverse.md']:
            z.write(name,'source/'+name)
    (out/'archive-sha256.txt').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();export(a.root,a.out)
