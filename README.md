# Periodic sphere-packing compression — solid FEM

追加検証：**焼結ネックの形状固定細分化とC3D8/C3D8I比較**。[条件と判定](docs/neck-element-verification.md)。
C3D8Iの部品応答は最細分比較で2%基準内。局所塑性ひずみ・周期充填層全体は未検証です。

追加検証：**刻み幅・材料感度・メッシュ・仕事の6ケース比較**。[結果と適用限界](docs/plastic-verification.md)。
刻み幅の影響は小さい一方、メッシュ収束は未達です。

追加モデル：**温度依存性なし・大変形弾塑性の試験モデル**。[仮の物性・圧縮除荷手順・適用範囲](docs/plastic-trial.md)。
専用コマンドは `python -m packing_fem.plastic`。自己接触と0.1 MPa到達の検証は今後の段階です。

最新の追加計算：**E=10 MPa・圧縮応力0.1 MPa** は小ひずみ基準に不合格の診断用です。[計算記録と適用限界](docs/sintered-10mpa.md)。

最新の追加計算：**E=1 GPa・圧縮応力0.1 MPa**。[変形後メッシュ画像と結果](docs/sintered-1gpa.md)。
設定は `configs/sintered-1gpa.json`、従来の70 GPa設定は比較用に保持しています。

現在の既定モデルは直径10 mmの球と直径2 mmの焼結ネックを一体化した試験骨格です。
CalculiX 2.20の静的有限要素法で圧縮と空隙率変化を計算します。円柱剛体は含みません。
DEMやHertzばねモデルで有限要素法を代替していません。

仕様：100 mm角、ネック込みの初期空隙率60%（離散形状の許容差±0.2ポイント）、
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
