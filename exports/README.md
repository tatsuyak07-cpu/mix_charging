# ダウンロード用の結果

- [CSV・PNG一式](sintered-results-001.zip)：100 mm角の焼結モデルの履歴CSV/PNG、全体・断面・ネックのメッシュ画像。
- [計算・検証記録とメッシュ](calculation-records-001.zip)：これまでの計算のCSV、JSON、ログ、終了記録、および `sintered-baseline-03` の使用メッシュNPZと変形後VTU。失敗した試行も含みます。

画像・CSVは同フォルダの `images/` と `data/` から個別にも取得できます。GitHubではPNGをクリックして表示し、CSVやZIPは Raw / Download raw file から取得してください。

ソルバーの大容量 `.dat/.frd/.inp` などの生出力はクラウドの `results/` に保持し、Gitには収録していません。結果はメッシュ収束未検証の候補解です。ネック直径2 mmに対して要素幅1 mmであり、定量的な採用には細分化検証が必要です。
