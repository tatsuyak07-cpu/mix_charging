# ヤング率1 GPaでの再計算

ユーザー指定により E=1 GPa（1000 MPa）へ変更し、圧縮応力0.1 MPaまで計算。
設定は `configs/sintered-1gpa.json`。従来の70 GPa設定と結果は保持した。
100 mm角、ν=0.33、直径10 mmの750球、直径2 mmの焼結ネック、周期境界、
円柱剛体なし。旧計算の形状NPZをそのまま再利用し、E以外の条件が同一であることを確認した。
1 mmのC3D8要素400,706個、物理節点590,681個。

## 結果

CalculiX 2.20 / SPOOLESによる静的線形弾性解析は429.747秒で終了。
初期面積・初期体積基準の小変形応力制御で、幾何学的非線形や追随荷重は扱わない。

| 指標 | 結果 |
|---|---:|
| 平均軸方向圧縮応力 | 0.100000000005 MPa |
| 軸方向圧縮ひずみ | 0.001539235（0.1539235%） |
| セルの軸方向短縮量 | 0.1539235 mm |
| 初期空隙率 | 59.9294% |
| 変形後空隙率 | 59.89856494% |
| 空隙率の減少 | 0.03083506パーセントポイント |
| 局所ひずみ成分の絶対最大値 | 0.009257859（約0.926%） |

全ての実行時検査（応力目標、他の平均応力ゼロ、正の要素Jacobian、正のセル体積、
物理的な空隙率、材料ひずみ1%未満）が合格。状態は
`equilibrium_candidate_requires_refinement`。ネックが2要素幅の粗い試験形状であり、
メッシュ収束は未検証。局所ひずみは1%の確認基準に近く、線形モデルの限界に注意する。

## 独立した照合

形状NPZのSHA256は新旧とも
`85f66bea949be6a4f1fd50c3d40da0310e599944687214e9f442663e699d0709`。
70 GPaの計算との節点変位比は線形弾性の予測70倍に一致し、最大相対誤差4.65e-7。
軸圧縮ひずみ比は70.00000455。周期対応節点の変位差と巨視ひずみとの最大残差は
7.4e-8 mm。検査スクリプトは `tools/check_modulus_comparison.py`。

等方線形弾性の骨格体積ひずみと巨視ひずみによる一次評価では空隙率変化は
-0.0003071715、変形節点の体積積分では-0.0003083506。
差-1.179e-6は有限体積評価の高次項を含むため記録する。
これは幾何学的非線形解析やメッシュ収束の検証を意味しない。

## 画像とダウンロード

[画像・CSV・検査結果](../exports/sintered-1gpa-001/)、
[画像・CSV ZIP](../exports/sintered-1gpa-images-csv-001.zip)、
[使用メッシュNPZ・変形後VTU ZIP](../exports/sintered-1gpa-mesh-001.zip)。
実倍率の `mesh_actual_1x.png` は実際の変形節点座標。
`mesh_deformation_20x.png` と `mesh_cutaway_20x.png` は節点変位のみ20倍表示。
後者は初期X<50 mmの要素を残した切断図で、切断面も表示する。
破線は未変形セル、赤線は巨視ひずみによる変形セル。形状の平滑化や再メッシュなし。
CSVは線形解の比例荷重履歴であり、非線形の増分計算ではない。

## 再実行

出力先は毎回未使用の名前を指定する。

```bash
python -m packing_fem.cli run --config configs/sintered-1gpa.json --output results/sintered-1gpa-new --allow-large --timeout 900
python tools/export_deformed_mesh.py results/sintered-1gpa-new results/sintered-1gpa-images-new
```

今回の計算は旧NPZを再利用して形状を完全一致させた。上記の再生成でも同じseedと
設定を使用するが、比較時にはNPZ・節点・要素の一致を確認する。
環境内の元結果は `results/sintered-1gpa-001/` と `results/sintered-1gpa-images-001/`。
大容量のDAT/FRD/INPはクラウド内に保持し、GitHubには収録しない。
