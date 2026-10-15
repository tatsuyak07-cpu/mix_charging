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
