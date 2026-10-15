# ダウンロード用の結果

- [CSV・PNG一式](sintered-results-001.zip)：100 mm角の焼結モデルの履歴CSV/PNG、全体・断面・ネックのメッシュ画像。
- [計算・検証記録とメッシュ](calculation-records-001.zip)：これまでの計算のCSV、JSON、ログ、終了記録、および `sintered-baseline-03` の使用メッシュNPZと変形後VTU。失敗した試行も含みます。

画像・CSVは同フォルダの `images/` と `data/` から個別にも取得できます。GitHubではPNGをクリックして表示し、CSVやZIPは Raw / Download raw file から取得してください。

ソルバーの大容量 `.dat/.frd/.inp` などの生出力はクラウドの `results/` に保持し、Gitには収録していません。結果はメッシュ収束未検証の候補解です。ネック直径2 mmに対して要素幅1 mmであり、定量的な採用には細分化検証が必要です。

## 1 GPa・0.1 MPaの追加計算

- [変形後メッシュPNG・履歴CSV・検査結果](sintered-1gpa-001/)
- [画像・CSV ZIP](sintered-1gpa-images-csv-001.zip)
- [使用メッシュNPZ・変形後VTU ZIP](sintered-1gpa-mesh-001.zip)
- [計算条件と結果の説明](../docs/sintered-1gpa.md)

画像の変形倍率はファイル名と画像内に明記しています（実倍率／20倍）。

## 10 MPa・0.1 MPa：診断用

[計算記録・CSV](sintered-10mpa-diagnostics-001/) / [ZIP](sintered-10mpa-diagnostics-001.zip)。
局所ひずみ約92.6%で小ひずみ基準に不合格。変形・空隙率は物理的な予測として採用できません。

## 大変形弾塑性の初期モデル

[30 mm角の圧縮・除荷CSVとPNG、入力、メッシュ](plastic-trial-001/) / [ZIP](plastic-trial-002.zip)。
未校正・自己接触未実装。1%圧縮後の永久変形を確認する試験で、0.1 MPa到達の計算ではありません。

## 大変形弾塑性の追加検証

[6ケースの比較CSV・PNG・全記録](plastic-verification-001/) / [ZIP](plastic-verification-001.zip)。
メッシュ変更で応力が約20%変わり、収束は未達です。[詳しい判定](../docs/plastic-verification.md)。

## ネックの形状固定細分化・要素形式の比較

[比較CSV・グラフ・使用メッシュ・全記録](neck-elements-001/) / [ZIP](neck-elements-001.zip)。
C3D8Iの部品反力・残留変形差は最細分で2%基準内。充填層全体や局所塑性ひずみの収束とは区別します。

## 周期C3D8Iの内部モードと圧縮分割数

[比較PNG・CSV・入力・メッシュ・検査記録](periodic-modes-001/) / [ZIP](periodic-modes-001.zip)。
25 mm角・1%圧縮後に除荷。C3D8Iの20→40分割で最大応力差0.019%、残留ひずみ差0.018%。
[検証報告](../docs/periodic-incompatible-verification.md)。メッシュ収束・自己接触は未検証です。

## 荷重増加と空隙率（探索計算）

[粗い刻みのCSV・PNG・記録](pressure-ramp-001/) / [ZIP](pressure-ramp-001.zip)。
[細かい刻みのCSV・PNG・記録](pressure-ramp-002/) / [ZIP](pressure-ramp-002.zip)。
[荷重刻みの比較](pressure-ramp-comparison-001/) / [計算報告](../docs/pressure-ramp.md)。
25 mm角の試験モデル。体積監査の条件を超えて停止した経路であり、0.1 MPaには未到達。
自己接触・材料校正・メッシュ収束は未検証です。
