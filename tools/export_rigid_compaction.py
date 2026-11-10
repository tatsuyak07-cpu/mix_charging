"""Export actual rigid-inclusion FEM checkpoints, sections and audit evidence."""
import argparse,csv,json,shutil,zipfile,hashlib
from pathlib import Path
from itertools import product
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
from packing_fem.rigid_inclusion import make_periodic,cylinder_surface,porosities
from packing_fem.explicit_periodic import deformation
from packing_fem.explicit_material import geometry
from tools.export_pore_sections import cut_tetrahedron
from tools.export_pressure_stages import render
from tools.audit_rigid_compaction import load


def rigid_section(vertices,faces,normal,level,axes):
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0);points=[]
    for i,j in edges:
        di=vertices[i,normal]-level;dj=vertices[j,normal]-level
        if abs(di)<1e-9:points.append(vertices[i,list(axes)])
        if di*dj<0:points.append((vertices[i]+di/(di-dj)*(vertices[j]-vertices[i]))[list(axes)])
    if len(points)<3:return None
    points=np.unique(np.round(points,10),axis=0)
    if len(points)<3:return None
    return points[ConvexHull(points).vertices]


def sections(initial,loaded,c,path):
    L=c['length_mm'];local,faces=cylinder_surface(c);fig,grid=plt.subplots(2,2,figsize=(11,10),layout='constrained')
    for col,(name,state) in enumerate([('Initial geometry',initial),('Loaded',loaded)]):
        p=make_periodic(state['X'],L,c);center,(R,_,_)=p.pose(state['q']);lengths=L*np.diag(deformation(state['q'][-6:]));rigid=local@R.T+center;tet=state['x'][state['t']];phi,total=porosities(geometry(state['x'],state['t'])[0].sum(),np.prod(lengths),c)
        for row,normal in enumerate([2,1]):
            ax=grid[row,col];axes=tuple(i for i in range(3) if i!=normal);level=center[normal]%lengths[normal];polygons=[];rigid_polygons=[]
            lo=np.ceil(-np.maximum(state['x'].max(axis=0),rigid.max(axis=0))/lengths).astype(int);hi=np.floor((lengths-np.minimum(state['x'].min(axis=0),rigid.min(axis=0)))/lengths).astype(int)
            for shift in product(*(range(a,b+1) for a,b in zip(lo,hi))):
                offset=np.asarray(shift)*lengths;minimum=tet.min(axis=1)+offset;maximum=tet.max(axis=1)+offset;mask=(minimum[:,normal]<=level)&(maximum[:,normal]>=level)
                for axis in axes:mask&=(minimum[:,axis]<=lengths[axis])&(maximum[:,axis]>=0)
                for vertices in tet[mask]+offset:
                    poly=cut_tetrahedron(vertices,normal,level,axes)
                    if poly is not None:polygons.append(poly)
                poly=rigid_section(rigid+offset,faces,normal,level,axes)
                if poly is not None:rigid_polygons.append(poly)
            boundary=Rectangle((0,0),lengths[axes[0]],lengths[axes[1]],fill=False,edgecolor='#b32929',zorder=4);ax.add_patch(boundary)
            for polys,color in [(polygons,'#5b839f'),(rigid_polygons,'#d99b31')]:
                collection=PolyCollection(polys,facecolors=color,edgecolors='#263e4a',linewidths=.15);collection.set_clip_path(boundary);ax.add_collection(collection)
            if lengths[axes[1]]<L:ax.add_patch(Rectangle((0,lengths[axes[1]]),L,L-lengths[axes[1]],facecolor='#e5e5e5',hatch='///',edgecolor='#bdbdbd'))
            ax.set(xlim=(0,L),ylim=(0,L),xlabel='XYZ'[axes[0]]+' (mm)',ylabel='XYZ'[axes[1]]+' (mm)',title=f'{name}: matrix porosity {100*phi:.3f}%\n'+'XYZ'[normal]+f'={level:.3f} mm');ax.set_aspect('equal')
    fig.suptitle('Floating rigid cylinder in periodic solid FEM\nCuts through the moving rigid center; displacement scale 1x');fig.get_layout_engine().set(rect=(0,.06,1,.94));fig.text(.5,.012,'Blue: solid skeleton. Orange: rigid cylinder. White: void. Gray: outside current cell.\nSame mm scale; titles show3D matrix-region porosity. Mesh / increment convergence unverified.',ha='center',fontsize=10);fig.savefig(path,dpi=170);plt.close(fig)


def export(root,run,out):
    root=Path(root);run=Path(run);out=Path(out);c=json.loads((run/'config.json').read_text());status=json.loads((run/'status.json').read_text())
    if status.get('status')!='loaded_target_reached':raise ValueError('Final export requires a reached loaded target')
    audit=json.loads((root/'final-audit.json').read_text());assert audit['passed']
    with (run/'history.csv').open() as f:rows=list(csv.DictReader(f))
    final=rows[-1];initial=load(run/'state-0000.npz');loaded_path=run/f"state-{int(final['step']):04d}.npz";loaded=load(loaded_path);local,faces=cylinder_surface(c)
    if np.max(abs(initial['q'][-6:]))>1e-12:raise ValueError('Export the full accepted path from the initial geometry')
    out.mkdir(exist_ok=False,parents=True)
    for name,state,path in [('initial',initial,run/'state-0000.npz'),('loaded',loaded,loaded_path)]:
        p=make_periodic(state['X'],c['length_mm'],c);center,(R,_,_)=p.pose(state['q']);cell=c['length_mm']**3*np.linalg.det(deformation(state['q'][-6:]));phi,total=porosities(geometry(state['x'],state['t'])[0].sum(),cell,c)
        render(out/(name+'-mesh.png'),state,f'Floating D{c["rigid_diameter_mm"]:g} x H{c["rigid_height_mm"]:g} mm rigid cylinder:{c["particle_count"]}-grain prototype\nMatrix-region porosity {100*phi:.3f}%; orange = rigid',c['length_mm'],footer='Actual nodal positions; displacement scale 1x\nFixed transverse cell; mesh / increment convergence unverified',rigid_surface=(local@R.T+center,faces));shutil.copy2(path,out/(name+'-mesh.npz'))
        np.savez_compressed(out/(name+'-rigid.npz'),points=local@R.T+center,faces=faces,center=center,rotation=R)
    sections(initial,loaded,c,out/'pore-sections.png');shutil.copy2(run/'history.csv',out/'history.csv');shutil.copy2(run/'config.json',out/'config.json');shutil.copy2(run/'status.json',out/'summary.json')
    for name in ['final-audit.json','periodic-contact-audit.json','input/placement.json']:
        if (root/name).exists():shutil.copy2(root/name,out/Path(name).name)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');axs[0].plot([100*float(r['compression']) for r in rows],[100*float(r['porosity']) for r in rows],'.-',label='Matrix region');axs[0].plot([100*float(r['compression']) for r in rows],[100*float(r['total_porosity']) for r in rows],'.-',label='Whole cell');axs[0].plot([0],[100*audit['states'][0]['matrix_region_porosity']],'o',mfc='none',label='Initial geometry');axs[0].axhline(10,ls='--',c='gray');axs[0].set(xlabel='Axial compression (%)',ylabel='Porosity (%)');axs[0].legend();axs[0].grid(alpha=.3)
    axs[1].plot([100*float(r['porosity']) for r in rows],[float(r['nominal_pressure_MPa']) for r in rows],'.-');axs[1].set(xlabel='Matrix-region porosity (%)',ylabel='Nominal pressure (MPa)');axs[1].grid(alpha=.3);fig.suptitle('Rigid-inclusion prototype; mesh / increment convergence unverified');fig.savefig(out/'compression-path.png',dpi=170);plt.close(fig)
    with (out/'rigid-history.csv').open('w',newline='') as f:
        fields=['step','compression','matrix_region_porosity','total_porosity','pressure_MPa','rigid_contacts','rigid_force_max_N','rigid_torque_over_radius_max_N'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(r for r in audit['states'] if 'pressure_MPa' in r)
    (out/'README.md').write_text(f'''# 剛体円柱：工程1〜3の数値結合試験\n\n16粒子、球径10mm、焼結ネック径2mm、E10MPa、ν0.33、摩擦0.3。\n試験円柱は直径4mm・高さ4mm、自由並進・自由回転、非接着。最終仕様の20x20mm／100mmセルとは区別します。\n\n充填層領域の空隙率{100*audit['states'][0]['matrix_region_porosity']:.4f}%→{100*float(final['porosity']):.4f}%。\n軸圧縮率{100*float(final['compression']):.4f}%、公称圧力{float(final['nominal_pressure_MPa']):.6f}MPa。\n横周期寸法を固定。初期状態は数値緩和前の幾何学形状です。\n\n[最終メッシュ](loaded-mesh.png)・[初期形状](initial-mesh.png)・[剛体中心断面](pore-sections.png)・[圧縮曲線](compression-path.png)・[元CSV](history.csv)・[剛体力／接触CSV](rigid-history.csv)・[独立監査](final-audit.json)。\n\n剛体体積を除いた領域を分母とする空隙率を目標判定に使い、セル全体の空隙率も併記します。\n骨格の現在体積は弾性圧縮を含み、剛体体積は一定です。保持・除荷と剛体なしとの定量比較は未実施。\n単一標本・三角形面による円柱近似で、メッシュ・増分・円柱分割収束は未認定です。\n''')
    items=[(p,'raw/'+str(p.relative_to(root))) for p in sorted(root.rglob('*')) if p.is_file()];items +=[(p,'exports/'+p.name) for p in sorted(out.iterdir()) if p.is_file()]
    for folder in ['packing_fem','tools','tests']:items.extend((p,'source/'+str(p)) for p in sorted(Path(folder).glob('*.py')))
    items.extend((p,'source/'+str(p)) for p in Path('configs').glob('rigid-cylinder*.json'))
    for p in [Path('AGENTS.md'),Path('docs/rigid-inclusion-development.md'),Path('requirements-implicit.txt'),Path('requirements-lock.txt')]:items.append((p,'source/'+str(p)))
    archives=[];z=None
    def close():
        nonlocal z
        if z is not None:
            p=Path(z.filename);z.close();archives.append(dict(file=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()));z=None
    for path,name in items:
        if z is None:z=zipfile.ZipFile(out.parent/f'{out.name}-part-{len(archives)+1:02d}.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=5)
        z.write(path,name)
        if z.fp.tell()>40*1024**2:close()
    close();assert all(a['bytes']<100*1024**2 for a in archives);(out/'archives.json').write_text(json.dumps(archives,indent=2)+'\n')
    with (out/'README.md').open('a') as f:
        f.write('\n## 全入力・実行ソース・不採用試行\n\n')
        for a in archives:f.write(f"- [{a['file']}](../{a['file']})\n")

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();export(a.root,a.run,a.out)
