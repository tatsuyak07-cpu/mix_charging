# 浮遊剛体円柱：圧縮計算の再開元

**空隙率10%は未達です。このフォルダは再開元の収束済み状態です。**

16粒子、球径10mm、焼結ネック径2mm、剛体円柱は直径4mm・高さ4mmの数値試験モデル。
最終仕様の直径20mm・高さ20mm／100mm角セルとは区別します。E10MPa、ν0.33、摩擦0.3。
円柱は自由並進・自由回転で、両横周期寸法24.140718mmを固定しています。

充填層領域の空隙率40.4057%→25.0287%。
軸圧縮率20.7540%、公称圧力0.048237MPa。
セル全体の空隙率は24.9161%。

[現在メッシュPNG](current-mesh.png)・[初期メッシュ](initial-mesh.png)・[剛体中心断面](pore-sections.png)・[圧縮曲線](compression-path.png)・[統合CSV](history.csv)。
[全82状態の体積・周期・剛体形状・非交差監査](geometry-path-audit.json)・[最終状態の力・材料履歴・摩擦の独立再計算](endpoint-force-audit.json)。
力とトルク／半径は各成分0.005N未満を確認します。ベクトルのノルムとは区別します。

骨格の現在体積には弾性圧縮を含み、基準材料体積と剛体形状を保存します。
初期NPZは数値緩和前の幾何形状です。単一試作で、メッシュ・増分収束、剛体の有無の定量比較、保持・除荷は未認定です。
CSVの時間は選択した区間の累積で、別試行や初期推定の準備費用を含みません。

## 再開用データ

[収束済み状態](current-state.npz)・[元メッシュ](original-mesh.npz)・[再開設定](resume-config.json)・[初期推定だけに使う未収束配置](resume-initial-iterate.npz)。
初期推定の材料履歴は受理せず、収束済み状態の履歴から計算します。出力先は未使用の名前にします。

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/run_implicit_compaction.py \
  --config exports/rigid-inclusion-progress-002/resume-config.json \
  --mesh exports/rigid-inclusion-progress-002/original-mesh.npz \
  --restart exports/rigid-inclusion-progress-002/current-state.npz \
  --initial-iterate exports/rigid-inclusion-progress-002/resume-initial-iterate.npz \
  --out results/new-rigid-continuation
```

## 周期画像の比較と検証

[関連97テスト](tests.log)・[27→125画像の力・エネルギー・摩擦仕事の監査](periodic-contact-audit.json)に合格しました。
候補数は935対934で、同一辺対の符号付き積分重みが2項から1項に集約される違いです。
合計重みは一致し、その組の力・エネルギー・摩擦仕事を元の割当ごとに再計算して保存性を確認しています。
[候補数だけによる不合格](periodic-count-only-failed.json)と[ゼロ寄与仮説の不合格](periodic-inert-only-failed.json)も保持しています。
