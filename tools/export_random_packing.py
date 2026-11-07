"""Export the random-geometry FEM demonstration, splitting large data archives."""
import argparse,csv,json,hashlib,shutil,zipfile
from pathlib import Path
from tools.export_fixed_cell import export


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--run',default='compression-001');a=p.parse_args()
    root=Path(a.root);out=Path(a.out);run=root/a.run
    packing=json.loads((root/'packing.json').read_text());mesh=json.loads((root/'mesh-007/metadata.json').read_text());status=json.loads((run/'status.json').read_text())
    export(root,out,run_name=a.run,caption=f"Random sintered packing: {packing['particle_count']} particles (prototype)",make_archive=False)
    for name in ['centers.csv','bonds.csv','packing.json','tests.log','loading-only-tests.log','final-tests.log','volume-audit-tests.log','acceleration-tests.log','consistent-warm-tests-002.log','full-tangent-tests.log','affine-hybrid-tests.log','lbfgs-tests.log','contact-audit.json','volume-history-audit.json','input-reproduction.json','increment-comparison.json','acceleration-comparison.json']:
        if (root/name).exists():shutil.copy2(root/name,out/name)
    (out/'mesh-metadata.json').write_text(json.dumps(mesh,indent=2)+'\n')
    with (run/'history.csv').open() as stream:rows=list(csv.DictReader(stream))
    final=rows[-1]
    assessment=dict(stage1='seeded random nonoverlapping centers generated',stage2='periodic sinter graph and CAD volume verified',stage3='periodic mesh within 0.5 percentage point initial porosity tolerance',stage4=status['status'],initial_porosity=mesh['initial_porosity'],final_porosity=float(final['porosity']),final_compression=float(final['compression']),loaded_target_reached=bool(status.get('target_reached',False)),holding_and_unloading_performed=False,statistical_isotropy_verified=False,mesh_and_increment_convergence_verified=False)
    (out/'stages-1-4.json').write_text(json.dumps(assessment,indent=2)+'\n')
    readme=f'''# ランダム充填層の工程1〜4\n\n16粒子、シード{packing['packing_seed']}、56本の焼結ネック。\n球径10 mm、ネック径2 mm、E=10 MPa、ν=0.33、摩擦0.3。\n初期CAD空隙率40%、メッシュ空隙率{100*mesh['initial_porosity']:.4f}%。\n初期セル長{mesh['cell_length_mm']:.6f} mm。横方向の周期ベクトルは圧縮中に固定。\n\n圧縮率{100*float(final['compression']):.4f}%、空隙率{100*float(final['porosity']):.4f}%。\n判定: `{status['status']}`。今回の範囲は工程4までで、保持・除荷は未実施です。\n単一16粒子の試作で、統計的等方性・メッシュ・増分収束は未認定です。\n\n[初期メッシュ](initial-mesh.png)・[載荷メッシュ](loaded-mesh.png)・[空隙率曲線と横寸法](porosity-and-cell-size.png)・[CSV](history.csv)・[粒子位置](centers.csv)・[焼結接続](bonds.csv)・[工程判定](stages-1-4.json)・[報告](../../docs/random-packing-development.md)。\n\n乱数シードの不採用配置、メッシュ作成の失敗・精度不足の試行もデータZIPに保存しています。\n球配置の最適化は初期形状を作る処理であり、荷重計算は3次元四面体の有限変形FEMです。\n'''
    (out/'README.md').write_text(readme)
    items=[(path,'raw/'+str(path.relative_to(root))) for path in sorted(root.rglob('*')) if path.is_file()]
    items += [(path,'exports/'+path.name) for path in sorted(out.iterdir()) if path.is_file()]
    for directory in ['packing_fem','tools','tests']:
        items += [(path,'source/'+str(path)) for path in sorted(Path(directory).glob('*.py'))]
    source_files=list(Path('configs').glob('random-periodic-compaction*.json'))
    source_files += [Path(name) for name in ['docs/random-packing-development.md','requirements-implicit.txt','requirements-lock.txt','AGENTS.md','README.md']]
    for path in source_files:items.append((path,'source/'+str(path)))
    archives=[];z=None;count=0
    def finish():
        nonlocal z
        if z is not None:
            path=Path(z.filename);z.close();archives.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()));z=None
    try:
        for path,arcname in items:
            if z is None:
                count+=1;z=zipfile.ZipFile(out.parent/f'{out.name}-part-{count:02d}.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=5)
            z.write(path,arcname)
            if z.fp.tell()>64*1024**2:finish()
    finally:finish()
    if any(r['bytes']>100*1024**2 for r in archives):raise ValueError('Archive exceeds GitHub file size limit')
    (out/'archives.json').write_text(json.dumps(archives,indent=2)+'\n')
    with (out/'README.md').open('a') as stream:
        stream.write('\n## 全入力・状態・ログ・ソース\n\n')
        for entry in archives:stream.write(f"- [{entry['file']}](../{entry['file']}) ({entry['bytes']/1024**2:.1f} MiB)\n")

if __name__=='__main__':main()
