# 浮遊剛体円柱：圧縮計算の再開元

**空隙率10%は未達です。このフォルダは再開元の収束済み状態です。**

16粒子、球径10mm、焼結ネック径2mm、剛体円柱は直径4mm・高さ4mmの数値試験モデル。
最終仕様の直径20mm・高さ20mm／100mm角セルとは区別します。E10MPa、ν0.33、摩擦0.3。
円柱は自由並進・自由回転で、両横周期寸法24.140718mmを固定しています。

充填層領域の空隙率40.4057%→24.6724%。
軸圧縮率21.1369%、公称圧力0.049881MPa。
セル全体の空隙率は24.5608%。

[現在メッシュPNG](current-mesh.png)・[初期メッシュ](initial-mesh.png)・[剛体中心断面](pore-sections.png)・[圧縮曲線](compression-path.png)・[統合CSV](history.csv)。
[全88状態の体積・周期・剛体形状・非交差監査](geometry-path-audit.json)・[最終状態の力・材料履歴・摩擦の独立再計算](endpoint-force-audit.json)。
力とトルク／半径は各成分0.005N未満を確認します。ベクトルのノルムとは区別します。

骨格の現在体積には弾性圧縮を含み、基準材料体積と剛体形状を保存します。
初期NPZは数値緩和前の幾何形状です。単一試作で、メッシュ・増分収束、剛体の有無の定量比較、保持・除荷は未認定です。
CSVの時間は選択した区間の累積で、別試行や初期推定の準備費用を含みません。

## 再開用データ

[収束済み状態](current-state.npz)・[元メッシュ](original-mesh.npz)・[再開設定](resume-config.json)・[次増分の初期推定用配置](resume-initial-iterate.npz)。
初期推定の材料履歴は受理せず、収束済み状態の履歴から計算します。出力先は未使用の名前にします。

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/run_implicit_compaction.py \
  --config exports/rigid-recovery-001/resume-config.json \
  --mesh exports/rigid-recovery-001/original-mesh.npz \
  --restart exports/rigid-recovery-001/current-state.npz \
  --initial-iterate exports/rigid-recovery-001/resume-initial-iterate.npz \
  --out results/new-rigid-continuation
```

## 今回の工程1〜4の状態

工程1の停止点診断と、工程2の解法修正を実施しています。工程2の24%通過、工程3の10%到達、工程4の圧縮応答の収束は、この固定スナップショットでは未完了です。進行中の試行をこのフォルダの採用結果へ混ぜていません。

[診断](diagnosis.json)・[同じCADの3段階メッシュ監査](mesh-preflight.json)・[未完了を不合格とする収束判定](convergence-preliminary-001.json)。形状の体積誤差が減っただけでは圧縮応答のメッシュ収束とは判定しません。

[周期像27→125の監査](probe-first-periodic-audit.json)は、回復最初の増分（空隙率24.9350%）で合格しました。現在の24.6724%状態の拡大周期像監査ではありません。現在状態を含む回復3増分の力・材料履歴・摩擦は、元の物理設定から独立再計算して合格しています。全経路88状態の幾何監査と区別します。

[関連108件の回帰試験](tests-009.log)と、保存間隔を修正した後の[関連20件](tests-010.log)に合格。未採用の完全固定・対角正則化比較は[上限超過の記録](tests-007.log)を残しています。全ての数値設定が収束したという意味ではありません。重複試行の意図的中断は記録し、解法の収束失敗と区別します。

再開の初期推定NPZの配置は現在の採用済み配置と同じです。次の荷重へ写像して使い、材料履歴は `current-state.npz` から読みます。[前回採用状態](previous-state.npz)は微小滑りの初期推定にだけ使用します。新しい物理的な接線ばね履歴を追加するものではありません。

[開発報告](../../docs/rigid-recovery-stages-1-4.md)・[以前の途中結果とアーカイブ](../rigid-inclusion-progress-002/README.md)。

## 実行ソースと終了済み試行の全保存データ

各tar.xzは単独で展開できます。全パートを同じ保存先へ展開します。計算中のprobe-001, continuation-005, continuation-006は除外しています。
[全ファイルSHA-256](raw-archive-files.json)・[アーカイブ検証](archive-verification.json)。

- [rigid-recovery-001-part-01.tar.xz](../rigid-recovery-001-part-01.tar.xz)
- [rigid-recovery-001-part-02.tar.xz](../rigid-recovery-001-part-02.tar.xz)
- [rigid-recovery-001-part-03.tar.xz](../rigid-recovery-001-part-03.tar.xz)
- [rigid-recovery-001-part-04.tar.xz](../rigid-recovery-001-part-04.tar.xz)
- [rigid-recovery-001-part-05.tar.xz](../rigid-recovery-001-part-05.tar.xz)
- [rigid-recovery-001-part-06.tar.xz](../rigid-recovery-001-part-06.tar.xz)
- [rigid-recovery-001-part-07.tar.xz](../rigid-recovery-001-part-07.tar.xz)
- [rigid-recovery-001-part-08.tar.xz](../rigid-recovery-001-part-08.tar.xz)
- [rigid-recovery-001-part-09.tar.xz](../rigid-recovery-001-part-09.tar.xz)
- [rigid-recovery-001-part-10.tar.xz](../rigid-recovery-001-part-10.tar.xz)
- [rigid-recovery-001-part-11.tar.xz](../rigid-recovery-001-part-11.tar.xz)
- [rigid-recovery-001-part-12.tar.xz](../rigid-recovery-001-part-12.tar.xz)
- [rigid-recovery-001-part-13.tar.xz](../rigid-recovery-001-part-13.tar.xz)
- [rigid-recovery-001-part-14.tar.xz](../rigid-recovery-001-part-14.tar.xz)

展開後の`source/`に実行コードと画像・CSV、`raw/`に終了済み計算の入力・保存状態・不採用試行が入っています。元の相対パスを復元する場合は、`source/results/rigid-stages-1-4-001/`を作成し`raw/`の内容をコピーしてから`source/`で実行します。参照する過去の保存データは対応する公開アーカイブも展開します。

[最終の配布整合性検証](delivery-final-verification.json)で、公開PNG・NPZ・CSV・監査と全14アーカイブのSHA-256を確認しました。`delivery-verification.json`はアーカイブ作成前の検証記録です。配布の合格は、空隙率10%への到達やメッシュ・荷重刻み収束の合格を意味しません。
