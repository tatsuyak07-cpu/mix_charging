# Periodic sphere-packing compression — solid FEM

追加する接触積分点の前提を検証：**接触前からの摩擦履歴再生12条件、狭い帯の積分18条件に合格**。
[検証報告](docs/integration-replay.md)・[履歴比較図](exports/integration-replay-002/history-replay.png)・[積分比較図](exports/integration-replay-002/narrow-integration.png)・[全データZIP](exports/integration-replay-002.zip)。
関連17テスト合格。規定運動の平面接触・既知形状の帯での単体検証です。2次元面積分、圧縮ソルバーの巻き戻し・接続、0.1 MPaの平衡到達は未完了です。

摩擦履歴・接触積分を保持する `material` モードを実装：**過去の拒否提案で面積・法線エネルギーを保持し、10項目の移行監査に合格**。
[検証報告](docs/material-quadrature.md)・[比較図](exports/material-quadrature-001/energy-conservation.png)・[新設定](configs/packing-rve-material-quadrature.json)・[全データZIP](exports/material-quadrature-001.zip)。
全体300テスト合格。固着・滑りの履歴と力評価状態を分けて移行します。積分点の適応追加は未実装で、0.1 MPaの完走は未達成です。

工程1〜4を比較検証：**辺対の接触確認を修正し、7条件を計算。0.1 MPa載荷・保持と準静的条件は未合格**。
[検証報告](docs/compaction-stages-1-4.md)・[比較図](exports/compaction-stages-001/stage4-comparison.png)・[条件別CSV](exports/compaction-stages-001/case-outcomes.csv)・[全データZIP](exports/compaction-stages-001.zip)。
全体284テスト合格。実充填セルでは新しい接触点の履歴不足と接触積分の変化で停止し、載荷速度による空隙率差も残っています。停止時の値を平衡圧密の結果として扱っていません。

ほぼゼロ離隔の辺対を検証：**丸め誤差による誤った除外判定を修正し、端点接触だけでは不十分な確認条件も特定**。
[検証報告](docs/near-edge.md)・[判定比較図](exports/near-edge-001/near-edge-checks.png)・[局所形状図](exports/near-edge-001/near-edge-geometry.png)・[全データZIP](exports/near-edge-001.zip)。
33条件の座標感度と局所三角形の非交差を確認しました。辺対ごとの接触確認条件は未修正で、今回の圧縮再計算は行っていません。

分割候補の自動選択と棄却試行の再実行を接続：**前回の停止区間を4回の監査付き分割で通過**。
[検証報告](docs/closest-selection.md)・[比較画像](exports/closest-selection-001/selection-comparison.png)・[停止時メッシュ](exports/closest-selection-001/stopped-mesh.png)・[全データZIP](exports/closest-selection-001.zip)。
全体267テスト合格。再計算は約0.05872 MPaで幾何チェックによる刻み下限に達して停止しました。空隙率約33.87%は未平衡の途中状態で、0.1 MPaの圧密解ではありません。

非中点分割の状態・摩擦履歴を検証：**保存状態の両分割案が移行監査に合格**。
[検証報告](docs/weighted-surface-transfer.md)・[比較画像](exports/weighted-transfer-001/weighted-transfer.png)・[全データZIP](exports/weighted-transfer-001.zip)。
追加接触節点への重み付き摩擦履歴の補間と、その後の滑りも確認しました。圧縮ソルバーへの自動選択・再試行の組み込みは未完了です。

表面分割の集中を検証：**最後の停止候補では中点分割が接触範囲を外れることを確認**。
[検証報告](docs/refinement-diagnostics.md)・[分割位置の比較図](exports/refinement-diagnostics-001/insertion-comparison.png)・[全データZIP](exports/refinement-diagnostics-001.zip)。
関連31テスト・幾何診断16項目に合格。相手側の辺を最短位置で分割する案は品質を保ちますが、状態・摩擦履歴の移行と圧縮計算への組み込みは未検証です。

周期変位移行修正後の全履歴を再検証：**移行エラーは解消したが、表面分割16回の上限で停止**。
[検証報告](docs/packing-rve-replay.md)・[比較図](exports/packing-rve-replay-001/replay-comparison.png)・[停止時メッシュ](exports/packing-rve-replay-001/stopped-mesh.png)・[全データZIP](exports/packing-rve-replay-001.zip)。
関連35テスト合格。約0.05674 MPa・空隙率34.27%は未平衡の途中状態であり、0.1 MPaの準静的解ではありません。

工程4の小周期セルを実装：**10 mm球・2 mmネックのBCC焼結セルで低荷重の周期圧縮を検証**。
[開発・検証報告](docs/packing-rve-development.md)・[比較図](exports/packing-rve-001/packing-rve-validation.png)・[実メッシュ](exports/packing-rve-001/actual-packing-mesh.png)・[全データZIP](exports/packing-rve-001.zip)。
初期空隙率39.84375%。0.001 MPaの対照を独立な線形FEM解・力の再積分・刻み半減と比較しました。
接触停止からの辺分割・再試行、等距離最短点の劣勾配、周期変位の直接移行、検証付きC++要素カーネルを追加。
**0.1 MPaまでの準静的大変形、表面の品質回復、メッシュ収束、40%→10%圧密は未完了**です。大荷重の保存状態を平衡解として扱っていません。

工程1〜3の試験実装：**参照材料近傍の幾何条件、別パッチの複数接触、周期表面の辺分割と状態・摩擦履歴の移行を追加**。
[開発・検証報告](docs/surface-development.md)・[分割前後の画像](exports/surface-bisection-view-001/surface-bisection.png)・[判定結果](exports/surface-development-003/summary.json)・[全データZIP](exports/surface-development-003.zip)。
全体224テスト・診断10項目に合格。小さな圧力対照で分割後の継続と解析解を検証し、接触専用試験で摩擦履歴・エネルギー・巨視的接触力の移行を確認。
通常設定は維持しています。一般の表面再生成、真の辺接触の力積分、接触積分のメッシュ収束は未完了で、40%→10%圧密の結果ではありません。

変形写像の非重複条件を検証：**保存形状を補間した5段階すべてで、形状だけの十分条件が合格**。
[検証報告](docs/structured-extension.md)・[画像](exports/structured-extension-001/geometry-extension.png)・[CSV](exports/structured-extension-001/geometry-certificate.csv)・[全データZIP](exports/structured-extension-001.zip)。
空隙内の補助写像を用い、材料節点・物理体積・接触力を保持しました。関連17テストと診断10項目が合格。
接触力の削除や停止ガード解除には接続していません。細分化した圧力一周期・メッシュ収束・40%→10%圧密は未完了です。

点対面の源・相手を分けて検証：**源側の距離減少だけで正常な積分点を除外する案を反例で棄却**。
[検証報告](docs/point-stationarity.md)・[診断画像](exports/point-stationarity-001/point-stationarity.png)・[積分点CSV](exports/point-stationarity-001/wave-quadrature.csv)・[全データZIP](exports/point-stationarity-001.zip)。
全体テスト197件と診断7項目が合格しました。追加は診断専用で、接触力・停止ガードは維持しています。
変形後の細分化4・6、圧力一周期のメッシュ収束、40%→10%圧密は未解決です。

辺候補の表面最短条件を検証：**試験機能で初期形状の全5段階が評価可能、変形後の4・6は引き続き停止**。
[検証報告](docs/edge-stationarity.md)・[候補画像](exports/edge-stationarity-002/edge-stationarity.png)・[CSV](exports/edge-stationarity-002/mesh-status.csv)・[全データZIP](exports/edge-stationarity-002.zip)。
全体テスト189件に合格。真の辺接近の停止ガードと周期平面接触の反力を維持しました。試験機能の通常設定は無効です。
細分化した圧力一周期・メッシュ収束・40%→10%圧密は未完了です。

局所材料近傍の候補判定を追加：**角部の点対面候補を区別し、未解決の辺接近では停止を維持**。
[検証報告](docs/local-material-neighborhood.md)・[候補画像](exports/local-material-neighborhood-001/local-material-candidates.png)・[停止理由CSV](exports/local-material-neighborhood-001/error-comparison.csv)・[全データZIP](exports/local-material-neighborhood-001.zip)。
全体179件・追加を含む重点8件のテストに合格。周期接触と保存済み空隙壁接触で反力の一致を確認しました。
細分化4・6の事前検査は引き続き不合格です。細分化した圧力一周期・メッシュ収束・40%→10%圧密は未完了です。

メッシュ細分化の事前検査：**細分化4・6で初期形状の接触候補が解決できず、不合格**。
[検証報告](docs/contact-mesh-preflight.md)・[実メッシュと候補の画像](exports/contact-mesh-preflight-002/mesh-preflight.png)・[CSV](exports/contact-mesh-preflight-002/mesh-comparison.csv)・[診断データZIP](exports/contact-mesh-preflight-002.zip)。
同じ変形形状でも粗い3段階の法線接触エネルギーは1、1/4、1/9と変化しました。形状・体積・周期性の移送検査と関連テスト4件は合格。
局所的な空隙壁の角部を接触候補から区別する修正が必要です。細分化した圧力一周期・メッシュ収束・40%→10%圧密は未完了です。

接触・塑性・除荷一周期のCFL半減を検証：**0.2→0.1で全比較項目が合格**。
[検証報告](docs/contact-cycle-cfl.md)・[比較PNG](exports/contact-cycle-cfl-001/cfl-comparison.png)・[差分CSV](exports/contact-cycle-cfl-001/path-differences.csv)・[全結果ZIP](exports/contact-cycle-cfl-001.zip)。
残留ひずみ差5.61e-9、全過程の最大空隙率差1.14e-8。細刻み終点の独立反力監査と関連テスト4件も合格しました。
4 mm・初期空隙率0.25%の検証セルでの一回の刻み半減です。メッシュ収束・球充填体の40%→10%圧密は未完了です。

接触反力を独立監査：**摩擦履歴と反力評価時の状態を保存し、エネルギーの数値微分・材料面力積分で終点反力を照合**。
[検証報告](docs/contact-reaction-audit.md)・[比較PNG](exports/contact-reaction-audit-001/contact-audit.png)・[監査CSV](exports/contact-reaction-audit-001/reaction-audit.csv)・[全結果ZIP](exports/contact-reaction-audit-001.zip)。
4 mm検証セルの同条件再計算で全ステップ数値・最終8配列が前回と完全一致。独立反力監査と全テスト169件に合格しました。接触探索・接触一周期のCFL感度・メッシュ収束・球充填体の40%→10%圧密は未完了です。

塑性・接触後の除荷を検証：**0.1 MPaから除荷後、残留圧縮約22.0712%、空隙率約0.22173%**。
[検証報告](docs/plastic-contact-unloading.md)・[比較PNG](exports/plastic-contact-unloading-001/plastic-unloading.png)・[実メッシュ](exports/plastic-contact-unloading-001/residual-mesh.png)・[CSV・全結果ZIP](exports/plastic-contact-unloading-001.zip)。
4 mm角・初期空隙率0.25%の検証用セルで、除荷8/16 msの両条件が既存検査に合格しました。関連テスト27件合格。この保存結果は独立反力監査に必要な摩擦履歴を含みません。接触一周期の刻み・メッシュ収束、球充填体の40%→10%圧密は未完了です。

除荷判定を追加検証：**有限変形弾性解と刻み・速度比較で、ゼロ荷重近傍の瞬間KE比の分母の影響を確認**。
[検証報告](docs/unload-quasistatic-verification.md)・[比較PNG](exports/unload-quasistatic-verification-001/unload-sensitivity.png)・[CSV・全計算ZIP](exports/unload-quasistatic-verification-001.zip)。
関連テスト24件合格。急速荷重を棄却できることも確認しました。補助検査と既存の総合判定を区別し、塑性・接触後の除荷と球充填体の圧密は未検証です。

連結セルの追加検証：**初期剛性を3段階で細分化し、弾性圧縮・除荷の対照試験を実施**。
[結果・判定](docs/connected-refinement-unloading.md)・[比較PNG](exports/connected-refinement-unloading-002/refinement-unloading.png)・[全データZIP](exports/connected-refinement-unloading-002.zip)。
形状は復元しましたが、除荷終盤の既存KE比判定は不合格のため診断データとして保持しています。塑性除荷・球充填体の40%→10%圧密は未完了です。

連結多孔質セルで真応力圧縮を検証：**全採用ステップの監査CSVを追加**。
[条件・判定・適用限界](docs/connected-pressure-verification.md)・[比較PNG](exports/connected-pressure-verification-001/connected-pressure-verification.png)・[全データZIP](exports/connected-pressure-verification-001.zip)。
4 mm角・初期空隙率0.25%の検証用形状です。球充填層の40%→10%圧密は未完了です。

接触過渡を追加検証：**層状セルに接触前の無剛性モードを確認**。
[検証条件・判定](docs/layered-contact-transient.md)・[比較PNG](exports/layered-contact-transient-001/layered-contact-verification.png)・[全データZIP](exports/layered-contact-transient-001.zip)。
接触後の終点の釣合いと、全過程の準静的性を分けて評価します。球充填体の圧密完了を意味しません。

大変形の圧力指定を改善：**一軸Cauchy真応力の追従荷重 `cauchy_pressure` を追加**。
[式・検証・適用限界](docs/objective-pressure.md)・[比較PNG](exports/objective-pressure-001/pressure-verification.png)・[結果ZIP](exports/objective-pressure-001.zip)。
従来の公称荷重方式は保持しています。今回の実荷重試験は空隙のない検証立方体です。
接触過渡と球充填層の40%→10%圧密の完了を意味しません。

接触積分を修正：**参照面積による `area_guarded` を追加し、平面32ケースの位置依存検査に合格**。
[実装・検証・適用限界](docs/area-contact-fix.md)・[比較PNG](exports/area-contact-fix-002/area-contact-verification.png)・[結果ZIP](exports/area-contact-fix-002.zip)。
辺だけの接触・複数パッチは解像不足として停止します。100 mm充填層の40%→10%圧密は未完了です。

従来方式の検証記録：**平面接触のメッシュ位置依存を確認し、物理検証に不合格**。
同じ間隙でも接触力が約4.5倍異なるため、従来方式の圧密値を物理予測に採用しません。
[結果・原因・修正に必要な条件](docs/plane-contact-verification.md)・[比較PNG](exports/plane-contact-verification-001/plane-contact-dependence.png)・[検証データZIP](exports/plane-contact-verification-001.zip)。

最新の開発：**周期自己接触を含む陽解法FEMの圧力・空隙率制御**。[実装・検証・適用限界](docs/barrier-compaction.md)。
4 mmの検証セルで0.1 MPaまで計算。100 mmの球充填層／空隙率40%→10%の計算は未完了です。
[結果ZIP](exports/barrier-compaction-002.zip)・[圧力と空隙率](exports/barrier-compaction-002/pressure-porosity.png)・[実メッシュ](exports/barrier-compaction-002/mesh-at-pressure.png)。
以下には過去の条件・計算記録も保持しています。

追加検証：**焼結ネックの形状固定細分化とC3D8/C3D8I比較**。[条件と判定](docs/neck-element-verification.md)。
C3D8Iの部品応答は最細分比較で2%基準内。局所塑性ひずみ・周期充填層全体は未検証です。

追加検証：**刻み幅・材料感度・メッシュ・仕事の6ケース比較**。[結果と適用限界](docs/plastic-verification.md)。
刻み幅の影響は小さい一方、メッシュ収束は未達です。

追加モデル：**温度依存性なし・大変形弾塑性の試験モデル**。[仮の物性・圧縮除荷手順・適用範囲](docs/plastic-trial.md)。
専用コマンドは `python -m packing_fem.plastic`。自己接触と0.1 MPa到達の検証は今後の段階です。

最新の追加計算：**E=10 MPa・圧縮応力0.1 MPa** は小ひずみ基準に不合格の診断用です。[計算記録と適用限界](docs/sintered-10mpa.md)。

最新の追加計算：**E=1 GPa・圧縮応力0.1 MPa**。[変形後メッシュ画像と結果](docs/sintered-1gpa.md)。
設定は `configs/sintered-1gpa.json`、従来の70 GPa設定は比較用に保持しています。

従来の静解析モデルは直径10 mmの球と直径2 mmの焼結ネックを一体化した試験骨格です。
CalculiX 2.20の静的有限要素法で圧縮と空隙率変化を計算します。円柱剛体は含みません。
DEMやHertzばねモデルで有限要素法を代替していません。

初期の静解析仕様（最新の圧密仕様は上記リンク）：100 mm角、ネック込みの初期空隙率60%（離散形状の許容差±0.2ポイント）、
骨格E=70 GPa、ν=0.33、粒子間は焼結接合、
一軸圧縮の目標Cauchy応力0.1 MPa、横方向平均応力ゼロ、空隙内圧ゼロ。
焼結モデルと限界は [docs/sintered.md](docs/sintered.md)。旧非接着モデルは
[docs/model.md](docs/model.md)、旧検証記録は
[docs/validation.md](docs/validation.md) に記載します。

## 実行

クラウド環境では以下を使用します。セットアップは計算と別に実行します。

```bash
cd /workspace/mix_charging
source /workspace/environment/mix_charging/.venv/bin/activate
export MPLCONFIGDIR=/workspace/environment/mix_charging/mplconfig
export XDG_CACHE_HOME=/workspace/environment/mix_charging/cache
export MPLBACKEND=Agg

# 本仕様の粒子配置と計算規模を確認。有限要素の本計算はまだ開始しない。
python -m packing_fem.cli estimate --config configs/baseline.json --output results/estimate-001

# 30 mm角の焼結骨格で動作検証。100 mm角の本仕様とは区別する。
python -m packing_fem.cli run --config configs/sintered-smoke.json --output results/sintered-test-001

# 本仕様の有限要素入力を生成。大規模モデルなので明示的な指定が必要。
python -m packing_fem.cli prepare --config configs/baseline.json --output results/baseline-001 --allow-large

# 入力生成後に実計算。時間制限は秒単位で、1回のsolver実行に適用される。
python -m packing_fem.cli solve results/baseline-001 --timeout 600

python -m pytest -q
```

既存出力フォルダは拒否します。別名で実行してください。計算中のネットワーク通信や
依存インストールはありません。`--solver` で別のCalculiX実行ファイルを指定できます。
通常のLinuxではBLAS/LAPACK、ARPACK、SPOOLESが必要です。

## 出力と判定

焼結モデルは `sintered.inp/.log/.dat`、`sintered-summary.json`、
`sintered-deformed.vtu`、`sintered-linear-history.csv/.png` を生成します。
履歴は静的線形解の比例荷重によるものです。ボクセル近似・ネック形状と
メッシュ収束の限界は [焼結モデル仕様](docs/sintered.md) を確認してください。

以下は保持している旧非接着モデルの出力です。

- `metadata.json`：物理条件、数値設定、球とメッシュの初期体積誤差。
- `model.inp`、`cycle-XX.inp`：実際に実行する有限要素入力と荷重履歴。
- `cycle-XX.log/.dat/.sta/.frd`：ソルバーのログと元の結果。
- `cycle-XX-history.csv/.png`：応力、セル寸法、空隙率、エネルギーの履歴。
- `cycle-XX-last-accepted.vtu`：最後にソルバーが受理した増分の粒子変形。
  このファイルがあるだけでは平衡解の成立を意味しません。ParaView等で表示できます。
- `cycle-XX-summary.json`：収束、応力、貫入、体積、運動、接触候補の検査。

元のソルバー応力は周期コピー数で分割した材料の値です。粒子の局所物理応力には
コピー数を掛ける必要があります。CSVの巨視的応力は全コピーを集計して補正済みです。

`diagnostics_only` は物理的な結果として採用しません。CLIも終了コード2を返します。
全検査を通った場合でも `equilibrium_candidate_requires_refinement` とし、
メッシュ・接触ペナルティ・粒子配置・緩和設定への依存性は別途検証します。
完全な検証済み解を自動的に宣言する状態は設けていません。

## 設定

`configs/baseline.json` は10 mm焼結モデル、`configs/sintered-smoke.json` は小規模試験用。
球だけなら764個、接合体積込みで空隙率を合わせた現試験骨格は750個です。

以下は旧5 mm非接着モデルの設定説明です。旧本仕様は
`configs/baseline-5mm-unbonded.json`、旧動作検証用は `configs/smoke.json`。
球体積と整数粒子数から、本仕様では6,112粒子、解析的空隙率59.9970535%になります。
曲面を近似する二次四面体C3D10を既定とし、理想球と有限要素メッシュの空隙率は
別々に表示します。`mesh_order: 1` によるC3D4も診断用に使用できます。

数値的な密度、セル制御の慣性、減衰、緩和時間、接触ペナルティは物性の追加指定では
ありません。準静的平衡を探すための開発者定義の設定で、測定密度や実際の時間とは
解釈しません。結果採用前に変更して解が変わらないことを確かめる必要があります。

## GitHubから結果を取得

CSV・PNG・計算記録は [exports](exports/README.md) に収録しています。
クラウドの環境構築には `bash tools/setup_cloud.sh` を使用します（uvが必要）。

周期C3D8Iの内部モード・仕事・圧縮分割数の追加検証は
[検証報告](docs/periodic-incompatible-verification.md)に記載しています。
25 mm角・1%圧縮の実験用モデルです。100 mm角・0.1 MPaやメッシュ収束は未検証です。

荷重を増加させた[空隙率の試験計算](docs/pressure-ramp.md)と
[CSV・PNG](exports/README.md#荷重増加と空隙率探索計算)を追加しました。
体積評価の監査条件を超えて停止しており、0.1 MPaまでの検証済み曲線は得られていません。

[初期40%から目標10%を目指す圧縮試算](docs/porosity40-to10.md)を追加しました。
設定は [porosity40-to10-trial.json](configs/porosity40-to10-trial.json)。
実際の初期メッシュ空隙率は39.8592%、体積監査で停止しており10%には未到達です。

空隙率40%から10%を扱うための[大変形計算手法の変更案](docs/large-deformation-method-review.md)をまとめました。
準静的陽解法FEM・周期自己接触・適応再メッシュを第一候補とし、MPM等との比較と導入時の検査を記載しています。

[準静的陽解法FEM・周期自己接触・局所再メッシュの研究用初版](docs/explicit-prototype.md)を実装しました。
[検証画像・CSV・準備済みモデル](exports/explicit-fem-prototype-002/)と[ZIP](exports/explicit-fem-prototype-002.zip)を収録しています。
内部再接続は実装済みですが、自由表面の全面再生成や40%→10%の圧密計算は未検証です。

次段階の検討: [周期接触の不連続・計算負荷と検証順序](docs/explicit-contact-next-review.md)。接触辺の通過時に生じるエネルギーの跳びを特定し、時間刻み半減でも残ることを確認しました。

接触の次段階検証: [面・凹辺・凹頂点の最近点処理と力・エネルギーの検証](docs/contact-feature-verification.md)。閉じた表面の試験用処理であり、圧縮ソルバーへの組込みは未実施です。

[周期切断面の内外判定と既知の接触領域の摩擦履歴の検証](docs/periodic-friction-verification.md)を追加しました。試験用演算子の検証であり、圧縮ソルバーへの組込みと接触領域の自動追跡は未実施です。

[接触追跡・接触中の内部再接続・研究用圧縮ソルバー結合の検証](docs/tracked-solver-integration.md)を追加しました。動的な節点接触の力の結合は確認しましたが、一般のスリット試験は未実装の辺―辺反力を必要とするため停止します。

[辺―辺の内部接触の反力・摩擦とソルバー結合](docs/interior-edge-contact.md)を追加しました。辺の反力は有効になりましたが、節点―面へのエネルギー整合な切替えで停止する研究用実装です。
