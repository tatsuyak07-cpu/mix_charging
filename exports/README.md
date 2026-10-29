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

## 初期40%から目標10%の圧縮試算

[グラフ・CSV・使用メッシュ画像・全記録](porosity40-to10-001/) / [ZIP](porosity40-to10-001.zip) /
[計算報告](../docs/porosity40-to10.md)。
実際の初期メッシュ空隙率39.8592%。軸圧縮9.8%・空隙率36.3672%で体積監査に不合格となり停止。
**10%には未到達**です。25 mm角の未校正・自己接触未実装の探索モデルです。

## 陽解法・周期自己接触・局所再メッシュの研究用初版

[検証画像・CSV・準備済み40%モデル・診断記録](explicit-fem-prototype-002/) / [ZIP](explicit-fem-prototype-002.zip) /
[実装・検証報告](../docs/explicit-prototype.md)。
全60テスト合格。小規模の準静的・除荷・再接続試験と、不合格の動的接触試験を区別して保存しています。
再メッシュは内部2→3再接続に限定。40%→10%の圧密計算は未実行です。

- [explicit-contact-review-001](explicit-contact-review-001): 接触有効化の不連続、時間刻み半減比較、計算負荷の診断。[ZIP](explicit-contact-review-001.zip)。準静的圧密の合格結果ではありません。

- [contact-features-verification-002](contact-features-verification-002): 面・凹辺・凹頂点に対する摩擦なし接触形状演算子の解析式比較。[ZIP](contact-features-verification-002.zip)。周期自己接触・圧密計算への組込みは未実施。001は単位表記分離前の記録を保持。

- [periodic-friction-verification-001](periodic-friction-verification-001): 周期境界での独立試験点の内外判定、既知の接触領域の摩擦履歴と仕事の収支。[ZIP](periodic-friction-verification-001.zip)。圧縮ソルバーへの組込み・接触領域の自動追跡は未実施。

- [tracked-solver-verification-003](tracked-solver-verification-003) と [tracked-node-impact-002](tracked-node-impact-002): 研究用ソルバー結合、接触履歴・内部再接続・動的な反力試験。[ZIP一式](tracked-solver-integration-002.zip)。元のスリット試験は辺―辺反力が未実装のため不合格。001は過剰な停止判定があった初版を保持。

中間記録のtracked-solver-verification-002とtracked-node-impact-001、および最初のZIPも、仕事積分補正の修正前の記録として保持しています。

- [edge-pair-verification-001](edge-pair-verification-001)、[edge-solver-verification-001](edge-solver-verification-001): 内部の辺―辺反力・摩擦・モーメントと結合検証。[ZIP一式](edge-contact-development-001.zip)。元のスリットは辺から節点―面への切替えで停止し、一般圧密は未検証。

## 工程1〜5：埋め込み積分・履歴再生・段階圧縮

[全記録](pressure-stages-001/) / [計算・検証報告](../docs/pressure-stages.md)。
0.001・0.01 MPaで平衡保持に合格しました。接触開始直後に再積分要求が続いたため中断し、0.03 MPa保持と0.1 MPa到達は未確認です。
小さなBCC周期セルの研究用試験で、100 mm充填層やメッシュ収束を保証しません。

- [平衡段階CSV](pressure-stages-001/stages.csv) / [過渡履歴CSV](pressure-stages-001/history.csv)
- [圧力・空隙率PNG](pressure-stages-001/pressure-porosity.png)
- [初期メッシュPNG](pressure-stages-001/initial-mesh.png) / [0.01 MPaの平衡メッシュPNG](pressure-stages-001/stage-0.01-MPa.png)
- [事後集計と中断の記録](pressure-stages-001/calculation-summary.json) / [SHA256一覧](pressure-stages-001/manifest.json)

フォルダー内の `validation-part-*.zip` はそれぞれ独立したZIPです。すべて同じ場所へ展開すると、元の試行・ソース・失敗記録を含む `validation/` が復元されます。収録先とファイルのSHA256は `validation-manifest.json` に記録しています。不完全な最後のチェックポイントも保存し、使用可能な状態と明確に区別しています。

## 接触開始時の積分誤差床と固定積分点

[検証報告](../docs/onset-budget.md) / [CSV・PNG・形状・全記録](onset-budget-001/) / [ZIP](onset-budget-001.zip)。
誤差配分の見直し案は平面9形状×2許容値と実セル12姿勢の細分化比較に合格しました。
初期積分点の固定利用は8条件中5条件で不合格です。圧縮ソルバーは変更しておらず、0.1 MPaの平衡結果ではありません。

- [独立平面CSV](onset-budget-001/manufactured.csv) / [実セルCSV](onset-budget-001/rve.csv) / [旧方式との比較CSV](onset-budget-001/legacy-comparison.csv)
- [積分誤差PNG](onset-budget-001/integration-errors.png) / [固定積分点の誤差PNG](onset-budget-001/frozen-rule-errors.png)

## 大変形への到達可能性の検討

[検討書](../docs/large-deformation-feasibility.md) / [判定・概算JSON](large-deformation-feasibility-001/assessment.json) /
[開発工程CSV](large-deformation-feasibility-001/development-gates.csv) / [根拠資料・全記録ZIP](large-deformation-feasibility-001.zip)。
有限変形FEMを第一候補としつつ、現行コードの接触・摩擦履歴・活性再メッシュの統合が未完成であることを評価しました。
空隙率40%→10%の体積変化、100 mm化の幾何的規模、段階的な継続判断を記録しています。新しいFEM計算・到達の合格結果ではありません。
