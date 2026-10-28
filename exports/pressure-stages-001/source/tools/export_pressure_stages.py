"""Export checked stage checkpoints and preserve every calculation attempt."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from packing_fem.checkpoint import load
from packing_fem.explicit_periodic import deformation


def boundary(t):
    faces=t[:,[[1,2,3],[0,3,2],[0,1,3],[0,2,1]]].reshape(-1,3)
    _,inverse,count=np.unique(np.sort(faces,axis=1),axis=0,return_inverse=True,return_counts=True)
    keep=count[inverse]==1
    return faces[keep],np.repeat(np.arange(len(t)),4)[keep]


def render(path,state,title,L):
    x=state['x'];faces,owners=boundary(state['t'])
    colors=plt.get_cmap('viridis')(plt.Normalize(0,max(float(state['ep'].max()),1e-6))(state['ep'][owners]))
    fig=plt.figure(figsize=(9,8),layout='constrained')
    fig.get_layout_engine().set(rect=(0,.07,1,.93))
    axis=fig.add_subplot(projection='3d')
    axis.add_collection3d(Poly3DCollection(x[faces],facecolors=colors,edgecolors='#263e4a',linewidths=.12))
    corners=np.array([[a,b,c] for a in (0.,L) for b in (0.,L) for c in (0.,L)])
    moved=corners@deformation(state['q'][-6:]).T
    for i in range(8):
        for j in range(i+1,8):
            if np.count_nonzero(corners[i]!=corners[j])==1:
                axis.plot(*corners[[i,j]].T,color='gray',ls='--',lw=.6)
                axis.plot(*moved[[i,j]].T,color='#b32929',lw=.8)
    lo=np.minimum(x.min(axis=0),np.minimum(corners.min(axis=0),moved.min(axis=0)))-.3
    hi=np.maximum(x.max(axis=0),np.maximum(corners.max(axis=0),moved.max(axis=0)))+.3
    axis.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='X (mm)',ylabel='Y (mm)',zlabel='Z (mm)',title=title)
    axis.set_box_aspect(hi-lo);axis.view_init(elev=23,azim=-53)
    sm=plt.cm.ScalarMappable(norm=plt.Normalize(0,max(float(state['ep'].max()),1e-6)),cmap='viridis')
    fig.colorbar(sm,ax=axis,shrink=.55,label='Equivalent plastic strain')
    fig.text(.5,.015,'Actual nodal positions, displacement scale 1x; no smoothing\nSmall BCC test cell; mesh convergence unverified',ha='center',fontsize=9)
    fig.savefig(path,dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--tests-log',required=True)
    p.add_argument('--assessment',type=Path,help='Explicit postmortem assessment when the runner was interrupted')
    a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
    assessment=a.assessment or a.run/'summary.json'
    summary=json.loads(assessment.read_text())
    final_attempt=a.run/f"attempt-{summary['attempts'][-1]['attempt']:03d}"
    result=json.loads((final_attempt/'summary.json').read_text())
    records=result['pressure_stages'];selected={}
    with np.load(a.run/'model.npz') as model:
        initial=dict(x=model['X'],t=model['t'],q=np.zeros(6),ep=np.zeros(len(model['t'])))
        render(root/'initial-mesh.png',initial,'Initial mesh: no load',json.loads((final_attempt/'config.json').read_text())['length_mm'])
    for manifest in sorted(a.source.glob('rve-run*/attempt-*/checkpoint-*/manifest.json')):
        payload=json.loads((manifest.parent/'state.json').read_text())
        saved_time=dict(payload['state']['dict'])['time']
        if not any(row['passed'] and abs(saved_time-row['time_s'])<1e-13 for row in records):continue
        state,metadata=load(manifest.parent)
        for row in records:
            if row['passed'] and abs(state['time']-row['time_s'])<1e-13:
                selected[row['pressure_MPa']]=(manifest.parent,state,metadata)
    table=[]
    for row in records:
        pressure=row['pressure_MPa'];entry=dict(row)
        if row['passed']:
            if pressure not in selected:raise ValueError(f'Missing exact equilibrated checkpoint at {pressure}')
            path,state,metadata=selected[pressure];entry['checkpoint']=str(path.relative_to(a.source))
            entry['checkpoint_state_sha256']=json.loads((path/'manifest.json').read_text())['state_sha256']
            entry['axial_compression']=float(state['audit_rows'][-1][1])
            tag=f"stage-{pressure:g}-MPa";entry['mesh_image']=tag+'.png'
            render(root/(tag+'.png'),state,f"Equilibrated stage: Cauchy pressure {pressure:g} MPa\nPorosity {row['porosity']*100:.3f}%",metadata['config']['length_mm'])
            np.savez_compressed(root/(tag+'.npz'),points=state['x'],reference_points=state['X'],tets=state['t'],peeq=state['ep'])
        table.append(entry)
    (root/'stages.json').write_text(json.dumps(table,indent=2)+'\n')
    with (root/'stages.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['pressure_MPa','time_s','porosity','force_ratio','kinetic_ratio','passed','axial_compression','checkpoint','mesh_image'],extrasaction='ignore')
        writer.writeheader();writer.writerows(table)
    shutil.copy2(final_attempt/'history.csv',root/'history.csv')
    shutil.copy2(final_attempt/'summary.json',root/'solver-summary.json')
    shutil.copy2(assessment,root/'calculation-summary.json')
    h=np.genfromtxt(root/'history.csv',delimiter=',',names=True)
    if result['failure']:
        with np.load(final_attempt/'last-state.npz') as saved:
            diagnostic=dict(x=saved['points'],t=saved['tets'],q=saved['generalized_position'],ep=saved['peeq'])
        render(root/'last-accepted-diagnostic.png',diagnostic,
               f"Stopped calculation: NOT an equilibrium solution\nCommanded Cauchy pressure {h['command_pressure_MPa'][-1]:.6g} MPa",
               json.loads((final_attempt/'config.json').read_text())['length_mm'])
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].plot(h['time_s'],100*h['porosity'],label='Accepted transient path')
    passed=[row for row in table if row['passed']]
    axes[0].plot([r['time_s'] for r in passed],[100*r['porosity'] for r in passed],'ko',label='Passed equilibrium hold')
    axes[0].set(xlabel='Numerical time (s)',ylabel='Porosity (%)');axes[0].legend(fontsize=8)
    axes[1].plot(h['command_pressure_MPa'],100*h['porosity'],color='.6',label='Accepted transient path')
    axes[1].plot([r['pressure_MPa'] for r in passed],[100*r['porosity'] for r in passed],'ko-',label='Passed equilibrium hold')
    axes[1].set(xlabel='Commanded Cauchy pressure (MPa)',ylabel='Porosity (%)');axes[1].legend(fontsize=8)
    fig.suptitle('Small BCC prototype; transient points are not equilibrium solutions')
    fig.savefig(root/'pressure-porosity.png',dpi=180);plt.close(fig)
    source=root/'source';source.mkdir()
    for name in ('packing_fem','tools','tests','configs','native'):
        shutil.copytree(name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ('AGENTS.md','requirements-lock.txt','pyproject.toml','docs/pressure-stages.md'):
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,target)
    provenance=dict(base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),calculation_directory=str(a.run),
                    tests_log=a.tests_log,source_implementation_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('packing_fem').glob('*.py')},
                    initial_porosity=json.loads((a.run/'geometry.json').read_text())['initial_porosity'],
                    limitations=['Small periodic BCC, not a representative 100 mm random/isotropic packing.',
                                 'Mesh, loading-rate and relaxation convergence not established.',
                                 'Trial uncalibrated plastic material and numerical density/damping.',
                                 'Inactive-only topology refinement; general active remeshing/ALE unsupported.'])
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    # Each part is an independently readable ZIP; even failed runs and source
    # snapshots are retained. Avoid GitHub's per-file upload limit.
    files=sorted(f for f in a.source.rglob('*') if f.is_file());parts=[];group=[];size=0
    for f in files:
        if group and size+f.stat().st_size>40_000_000:parts.append(group);group=[];size=0
        group.append(f);size+=f.stat().st_size
    if group:parts.append(group)
    archive_manifest={}
    for i,group in enumerate(parts,1):
        path=root/f'validation-part-{i:02d}.zip'
        with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
            for f in group:
                name=str(f.relative_to(a.source));z.write(f,'validation/'+name)
                archive_manifest[name]=dict(part=path.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),bytes=f.stat().st_size)
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:raise ValueError(f'Archive integrity failed: {path}')
        if path.stat().st_size>=95_000_000:raise ValueError(f'Archive part exceeds GitHub file budget: {path}')
    (root/'validation-manifest.json').write_text(json.dumps(archive_manifest,indent=2)+'\n')
    lines=['# 工程1〜5の実装・検証と段階圧縮',
           '', '**0.1 MPaの平衡解は未取得です。** 接触開始直後の再積分要求が続いたため中断しました。',
           '小さなBCC試験セルであり、100 mm充填層・メッシュ収束・40%から10%の圧密を保証しません。',
           '', '| 平衡Cauchy応力 MPa | 空隙率 % | メッシュ |', '|---:|---:|---|']
    for row in passed:lines.append(f"| {row['pressure_MPa']:g} | {100*row['porosity']:.6f} | [PNG]({row['mesh_image']}) |")
    lines.extend(['', '[平衡段階CSV](stages.csv) / [過渡履歴CSV](history.csv) / [圧力・空隙率PNG](pressure-porosity.png)',
                  '', '[計算・検証報告](../../docs/pressure-stages.md) / [事後集計](calculation-summary.json)',
                  '', 'validation-part-*.zipは独立したZIPです。すべて同じ場所に展開してvalidation/を復元してください。',
                  '元の失敗・中断・不完全チェックポイントも保持しています。使用可否は事後集計と報告を参照してください。',
                  'validation-manifest.jsonに収録先・SHA256を記録しています。source/に最終のプログラムと設定を保存しています。'])
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    manifest={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob('*')) if f.is_file()}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(export=str(root),passed_stages=len(passed),archive_parts=len(parts),files=len(manifest),solver_checks=result['checks'])))


if __name__=='__main__':main()
