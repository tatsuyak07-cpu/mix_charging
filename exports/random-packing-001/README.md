# ランダム充填層の工程1〜4

16粒子、シード20261109、56本の焼結ネック。
球径10 mm、ネック径2 mm、E=10 MPa、ν=0.33、摩擦0.3。
初期CAD空隙率40%、メッシュ空隙率40.4037%。
初期セル長24.109757 mm。横方向の周期ベクトルは圧縮中に固定。

圧縮率34.6854%、空隙率9.9870%。
判定: `loaded_target_reached`。今回の範囲は工程4までで、保持・除荷は未実施です。
単一16粒子の試作で、統計的等方性・メッシュ・増分収束は未認定です。

[初期メッシュ](initial-mesh.png)・[載荷メッシュ](loaded-mesh.png)・[空隙率曲線と横寸法](porosity-and-cell-size.png)・[CSV](history.csv)・[粒子位置](centers.csv)・[焼結接続](bonds.csv)・[工程判定](stages-1-4.json)・[報告](../../docs/random-packing-development.md)。

乱数シードの不採用配置、メッシュ作成の失敗・精度不足の試行もデータZIPに保存しています。
球配置の最適化は初期形状を作る処理であり、荷重計算は3次元四面体の有限変形FEMです。

最終公称圧力 0.162071 MPa、最大自由残差 0.004732 N。
[公称圧力と空隙率](pressure-porosity.png)・[中央断面の空隙比較](pore-sections.png)。降伏0.05 MPa・硬化0.2 MPaは開発用の仮値です。

ZIPをすべて同じ場所に展開し、`raw/` を `results/random-packing-001/` に、`source/` の内容をプロジェクト直下に配置すると、報告書の再実行コマンドで保存経路から再開できます。

## 全入力・状態・ログ・ソース

- [random-packing-001-part-01.zip](../random-packing-001-part-01.zip) (67.3 MiB)
- [random-packing-001-part-02.zip](../random-packing-001-part-02.zip) (65.9 MiB)
- [random-packing-001-part-03.zip](../random-packing-001-part-03.zip) (65.6 MiB)
- [random-packing-001-part-04.zip](../random-packing-001-part-04.zip) (68.3 MiB)
- [random-packing-001-part-05.zip](../random-packing-001-part-05.zip) (66.7 MiB)
- [random-packing-001-part-06.zip](../random-packing-001-part-06.zip) (66.2 MiB)
- [random-packing-001-part-07.zip](../random-packing-001-part-07.zip) (65.1 MiB)
- [random-packing-001-part-08.zip](../random-packing-001-part-08.zip) (67.9 MiB)
- [random-packing-001-part-09.zip](../random-packing-001-part-09.zip) (69.0 MiB)
- [random-packing-001-part-10.zip](../random-packing-001-part-10.zip) (64.4 MiB)
- [random-packing-001-part-11.zip](../random-packing-001-part-11.zip) (66.5 MiB)
- [random-packing-001-part-12.zip](../random-packing-001-part-12.zip) (66.9 MiB)
- [random-packing-001-part-13.zip](../random-packing-001-part-13.zip) (64.0 MiB)
- [random-packing-001-part-14.zip](../random-packing-001-part-14.zip) (67.5 MiB)
- [random-packing-001-part-15.zip](../random-packing-001-part-15.zip) (64.7 MiB)
- [random-packing-001-part-16.zip](../random-packing-001-part-16.zip) (69.0 MiB)
- [random-packing-001-part-17.zip](../random-packing-001-part-17.zip) (27.2 MiB)
