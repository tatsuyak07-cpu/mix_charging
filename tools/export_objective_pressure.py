"""Export immutable verification data, plots and logs for Codex Web users."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--sensitivity',type=Path,nargs='*',default=[]);a=p.parse_args()
    if a.output.exists() or a.output.with_suffix('.zip').exists():raise FileExistsError('Export destination already exists')
    shutil.copytree(a.input,a.output)
    for source in a.sensitivity:shutil.copytree(source,a.output/'sensitivity'/source.name)
    logdir=a.output/'logs';logdir.mkdir()
    for log in sorted(Path('/tmp').glob('objective-pressure-*.log')):shutil.copy2(log,logdir/log.name)
    rows=np.genfromtxt(a.input/'transient-cubes.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for P in (.02,.1):
        name=f'cube-n1-cfl0.1-P{P}';h=np.genfromtxt(a.input/name/'history.csv',delimiter=',',names=True)
        label=f'{P} MPa'+(' (diagnostic; low-KE failed)' if P==.1 else '');axes[0].plot(1000*h['time_s'],h['command_pressure_MPa'],label=label)
        axes[1].plot(1000*h['time_s'],h['axial_compression'],label=label)
        row=next(r for r in rows if r['case']==name);axes[1].axhline(1-row['analytic_axial_stretch'],linestyle=':',color='gray')
    axes[0].set(xlabel='Time [ms]',ylabel='Commanded Cauchy compression [MPa]')
    axes[1].set(xlabel='Time [ms]',ylabel='Axial compression (1 - stretch)')
    labels=[f"n={r['refinement']}\nCFL={r['cfl']}\nP={r['cauchy_pressure_MPa']}" for r in rows]
    axes[2].semilogy(range(len(rows)),[r['axial_error'] for r in rows],'o',label='Axial stretch error')
    axes[2].set(xticks=range(len(rows)),xticklabels=labels,ylabel='Absolute error vs analytic limit')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.suptitle('Actual periodic FEM: homogeneous 1 mm cube, E=10 MPa (no pores/contact)');fig.tight_layout();fig.savefig(a.output/'pressure-verification.png',dpi=180);plt.close(fig)
    name='cube-n1-cfl0.1-P0.1'
    with np.load(a.input/name/'last-state.npz') as state:X=state['reference_points'];x=state['points'];t=state['tets'];ep=state['peeq']
    fig=plt.figure(figsize=(10,5))
    for panel,points,title in ((1,X,'Reference'),(2,x,f'Diagnostic 0.1 MPa; max PEEQ={ep.max():.6f}')):
        ax=fig.add_subplot(1,2,panel,projection='3d');edges=sorted({tuple(sorted((int(tet[i]),int(tet[j])))) for tet in t for i in range(4) for j in range(i+1,4)})
        ax.add_collection3d(Line3DCollection(points[np.array(edges)],colors='navy',linewidths=.8));ax.set(xlim=(0,1.3),ylim=(0,1.3),zlim=(0,1.3),xlabel='x [mm]',ylabel='y [mm]',zlabel='z [mm]',title=title);ax.set_box_aspect((1,1,1))
    fig.suptitle('Diagnostic mesh: low-KE check FAILED; 8 nodes / 6 tets, scale 1 (not packing)');fig.tight_layout();fig.savefig(a.output/'mesh-at-0.1MPa.png',dpi=180);plt.close(fig)
    diag=json.loads((a.input/'historical-load-diagnostic.json').read_text());fig,ax=plt.subplots(figsize=(7,4))
    old=np.array(diag['old_work_equivalent_cauchy_MPa']);new=np.array(diag['new_work_equivalent_cauchy_MPa']);ind=np.arange(3)
    ax.bar(ind-.18,[old[0,2],old[1,2],old[0,1]],width=.36,label='Old generalized nominal load')
    ax.bar(ind+.18,[new[0,2],new[1,2],new[0,1]],width=.36,label='Cauchy follower load');ax.set(xticks=ind,xticklabels=['xz','yz','xy'],ylabel='Work-equivalent shear stress [MPa]',title='Same historical sheared cell; applied-load diagnostic only');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(a.output/'load-definition-comparison.png',dpi=180);plt.close(fig)
    hashes={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (a.output/'README.md').write_text('''# 真応力による周期一軸圧縮制御の検証\n\npressure-verification.png / transient-cubes.csv は空隙のない1 mm立方体の実際の陽解法FEMです。\nmesh-at-0.1MPa.png はその実メッシュであり、球充填体ではありません。\ncontact-equilibrium.csv は接触を含む解析的な平衡状態の残差検査で、過渡計算ではありません。\nload-definition-comparison.png は過去のセル形状を固定して外力の定義だけを比較しています。過去の解析の再計算ではありません。\n\n従来の公称荷重方式は保持し、新方式は loading_mode=cauchy_pressure で明示的に選びます。\n層状空隙の過渡試験、一般的な辺接触の反力、表面再メッシュ、40%→10%の充填体圧密は未完了です。\n初回の実荷重試験は0.02 MPaの4ケースが合格、0.1 MPaの2ケースは運動エネルギー・解析枝・せん断の検査に不合格です。\n追加の質量フレーム・増加時間比較は sensitivity に保存し、全ての判定を保持します。\n単一の細分化比較は一般的なメッシュ収束を証明しません。\n''')
    archive=a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(a.output.parent))
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f'ZIP CRC failure: {bad}')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,zip_crc='passed')))
if __name__=='__main__':main()
