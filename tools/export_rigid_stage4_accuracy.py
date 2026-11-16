"""Publish completed stage4 diagnostics without promoting them to convergence."""
import argparse,hashlib,json,shutil
from pathlib import Path
from tools.compare_rigid_mesh_accuracy import read_study


def export(root,out):
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    strict,records=read_study(root/'strict-mesh-screen-001')
    small,small_records=read_study(root/'small-load-screen-001')
    read_study(root/'baseline-replayed-001')
    accuracy=json.loads((root/'accuracy-comparison-004/assessment.json').read_text())
    regimes=json.loads((root/'load-regimes-001/assessment.json').read_text())
    neck=json.loads((root/'neck-sections-002/assessment.json').read_text())
    preload=json.loads((root/'initial-contact-001/assessment.json').read_text())
    launches=[]
    for study in ['strict-mesh-screen-001','small-load-screen-001']:
        for level in range(3):
            runtime=root/study/'runtime'/f'level-{level:03d}'
            receipt=json.loads((runtime/'launch.json').read_text());manifest=json.loads((runtime/'manifest.json').read_text())
            if receipt['child_return_code']!=0 or not receipt['runtime_hashes_unchanged']:raise ValueError('Calculation did not finish cleanly')
            for name,digest in manifest.items():
                if hashlib.sha256((runtime/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed')
            launches.append(dict(study=study,level=level,child_return_code=0,frozen_source_unchanged=True))
    files={
        'accuracy-assessment.json':'accuracy-comparison-004/assessment.json',
        'accuracy-comparison.csv':'accuracy-comparison-004/comparison.csv',
        'mesh-accuracy.png':'accuracy-comparison-004/mesh-accuracy.png',
        'load-regimes.json':'load-regimes-001/assessment.json',
        'load-regimes.csv':'load-regimes-001/comparison.csv',
        'load-regimes.png':'load-regimes-001/load-regimes.png',
        'strict-mesh-assessment.json':'strict-mesh-screen-001/assessment.json',
        'small-load-assessment.json':'small-load-screen-001/assessment.json',
        'neck-sections.json':'neck-sections-002/assessment.json',
        'neck-sections.csv':'neck-sections-002/sections.csv',
        'initial-contact.json':'initial-contact-001/assessment.json',
        'initial-contact.csv':'initial-contact-001/contact-preload.csv',
        'original-physics.json':'strict-mesh-screen-001/input/original-physics.json',
        'plan.json':'plan.json','small-load-plan.json':'small-load-plan.json',
        'tests-001.log':'tests-001.log','tests-003.log':'tests-003.log',
        'baseline-copy-provenance.json':'baseline-replayed-001/copy-provenance.json',
        'neck-input-rejection.json':'neck-sections-001-failure.json',
        'unbound-audit-rejection.json':'accuracy-comparison-001-failure.json'}
    for study,label in [('strict-mesh-screen-001','strict'),('small-load-screen-001','small'),('baseline-replayed-001','baseline')]:
        for level in range(3):
            for name in ['audit.json','history.csv','config.json']:
                files[f'{label}-level-{level:03d}-{name}']=f'{study}/level-{level:03d}/{name}'
    for target,source in files.items():shutil.copy2(root/source,out/target)
    summary=dict(stage4_qualified=False,completed_new_loading_cases=6,original_physics_audits_passed=True,
        original_baseline_states_reaudited=3,full_loaded_path_mesh_convergence_verified=False,
        full_loaded_path_increment_convergence_verified=False,launches=launches,
        maximum_observed_pressure_solve_change_relative=accuracy['maximum_observed_pressure_solve_change_relative'],
        strict_mesh_pressure_difference_relative=strict['production_coarsest_vs_finest']['pressure_relative'],
        small_load_mesh_pressure_difference_relative=small['production_coarsest_vs_finest']['pressure_relative'],
        small_load_all_meshes_elastic=regimes['small_load_is_elastic_verified_from_saved_history'],
        initial_contact_preload=preload['rows'],neck_area_summary=neck['summaries'],
        physical_scope='16grains, freeD4/H4mm cylinder,24.1407176351mm fixed-transverse cell. Not full100mm/D20mm system.',
        previous_loaded_target='exports/rigid-recovery-loaded-001/README.md')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# 工程4の継続：数値探索精度・初期メッシュ依存性',
        '', '16粒子D10 mm、ネックD2 mm、自由6DOF円柱D4/H4 mm、セル24.1407 mm角の統合試験です。E10 MPa、ν0.33、摩擦0.3、横周期寸法固定を維持しました。100 mm角・円柱D20/H20 mmの本番系ではありません。',
        '', '**工程4は未合格です。** 以前の[空隙率40.4057%→10.0652%の到達計算](../rigid-recovery-loaded-001/README.md)を保持しています。今回の追加計算は初期0.5%と0.01%圧縮の計6ケースで、全40%→10%経路の再計算・収束認定ではありません。',
        '', '## 完了した実計算', '',
        '| T4要素数 | 0.5%圧縮の圧力 MPa | 厳密化による圧力差 | 0.01%圧縮の圧力 MPa |',
        '|---:|---:|---:|---:|']
    for a,b,row in zip(records,small_records,accuracy['rows'],strict=True):
        lines.append(f"| {a['tetrahedra']} | {a['pressure_MPa']:.10g} | {100*row['pressure_change_relative']:.5f}% | {b['pressure_MPa']:.10g} |")
    lines.extend(['',f"元の合格基準0.005 Nを維持し、0.5%圧縮では反復停止を0.0001 Nへ厳密化。内側停止値も0.0001→0.00002 Nへ変更し、同等性検証済みPARDISOを使用しました。これは複数の数値探索設定を変えた感度比較で、単一変数の誤差上限ではありません。圧力の変化は最大{100*accuracy['maximum_observed_pressure_solve_change_relative']:.5f}%ですが、最粗／最細差は{100*strict['production_coarsest_vs_finest']['pressure_relative']:.4f}%残りました。隣接差6.2206%→6.9721%は減少せず、従来の5%目安に不合格です。",
        '', '![反復精度とメッシュ依存性](mesh-accuracy.png)',
        '', '[数値CSV](accuracy-comparison.csv)・[実状態SHAに結び付けた比較](accuracy-assessment.json)・[厳密化した3メッシュ判定](strict-mesh-assessment.json)。独立監査は元の物理設定で接触・摩擦・材料履歴・体積・剛体釣合い・非交差を再計算しています。',
        '', '## 切り分けで分かったこと', '',
        f"0.01%圧縮でも最粗／最細の圧力差は{100*small['production_coarsest_vs_finest']['pressure_relative']:.4f}%です。停止条件2e-6 Nで全3ケースの元物理監査に合格しました。ただし最細の最大相当塑性ひずみは0.0525548、塑性化した骨格基準体積比は7.05145e-8です。最粗・中間は塑性ひずみ0でした。全メッシュが弾性という比較条件は成立しておらず、『塑性化前から同じ差がある』とは断定しません。",
        '', '![低荷重と0.5%圧縮の比較](load-regimes.png)',
        '', '[低荷重比較CSV](load-regimes.csv)・[履歴からの判定](load-regimes.json)。',
        '', '初期形状は材料応力ゼロですが、接触バリアの局所最大力は最粗0.00743、中間0.00493、最細0.18592 Nでした。対応する初期公称圧力は9.82e-9、6.50e-9、2.24e-7 MPaで、局所力と巨視的圧力を区別します。初期形状を釣合い済み解としては扱いません。低荷重の局所塑性との因果関係は未検証です。[初期接触CSV](initial-contact.csv)・[判定](initial-contact.json)。',
        '', '他の球・ネックと交差せず周期面にも掛からない40か所のネック中央断面を、四面体と平面の交差多角形から積分しました。CAD円断面π mm²に対する平均面積不足は3.4186%→1.2837%→0.9893%です。形状近似に差があることは確認できましたが、圧力差の単独原因とする証拠ではありません。[断面CSV](neck-sections.csv)・[対象と除外理由](neck-sections.json)。',
        '', '## 次の開発で解消する点', '',
        'まずゼロ軸ひずみでの初期緩和と接触位置を調べ、最細メッシュの局所塑性の起源を特定します。次に同じ面形状を保つ要素細分化で、CAD形状近似と固体要素の離散化の影響を分けます。採用する接触・メッシュの仕様を決めた後、40%→10%全経路で3段階の荷重刻み・メッシュ比較を実行します。ロッキングや接触バリアを唯一の原因と断定せず、物理条件の変更で差を消すことはしません。',
        '', '## 検証と再現', '',
        '関連179テスト合格（[32件](tests-001.log)＋[147件](tests-003.log)）。厳密停止、許容値を緩める指定の拒否、解析解付き断面積分、既存の有限変形・周期・再メッシュ・履歴・終点制御を含みます。',
        '', '6ケースは実行ソースを固定し、正常終了と全ソースSHA不変を確認しました。旧比較の監査は保存状態SHAを持たないため比較から一度拒否し、旧状態をビット一致で複写して3ケース全ての元物理監査を再実行しました。[複写記録](baseline-copy-provenance.json)・[拒否記録](unbound-audit-rejection.json)。異なる旧セルの充填配置を誤って渡した断面監査も幾何整合検査が拒否し、正しい円柱用配置でやり直しました。[入力拒否記録](neck-input-rejection.json)。不採用試行も全保存データに含めています。',
        '', '[集約判定](summary.json)・[実行計画](plan.json)・[低荷重計画](small-load-plan.json)。今回の拡大周期像監査は実施していません。以前の10%終点の周期像監査は以前の保存状態についての結果です。',
        '', 'アーカイブを展開して下記の相対パスを復元した後、最細0.5%圧縮状態の元物理監査を独立再実行できます。出力先は未使用のファイルを指定します。',
        '', '```bash', 'PYTHONPATH=. python tools/audit_rigid_compaction.py \\', '  --run results/rigid-stage4-001/strict-mesh-screen-001/level-002 \\', '  --physics-config results/rigid-stage4-001/strict-mesh-screen-001/input/original-physics.json \\', '  --local-predecessors-only --out /tmp/stage4-replay.json', '```', ''])
    (out/'README.md').write_text('\n'.join(lines))
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();export(a.root,a.out)
