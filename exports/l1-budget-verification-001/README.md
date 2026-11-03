# L1誤差予算の検証

成分への均等配分を廃止し、親面/配置全体の超過合計に予算を直接適用。
受理済み状態は重み変更0で合格。次配置を含む保存射影は依然として実行不能。
実充填層は前回と同じ90,262刻みで停止。空隙率36.941697%は過渡値。
空隙率40%→10%・保持/除荷・収束確認は未完了。

- [検証図](verification.png)、[メッシュ](packing-mesh.png)
- [固定行列CSV](frozen-constraints.csv)、[平面CSV](plane-cycle.csv)、[充填層CSV](packing-prefix.csv)
- [判定JSON](assessment.json)、[報告](../../docs/l1-budget-verification.md)、[ZIP](../l1-budget-verification-001.zip)

ZIPに実行時/最終ソース、入力行列、最後の受理チェックポイント、失敗した試行のログも保存。
元のresultsを上書きしていない。環境復元前のライブラリ不足による失敗ログも含む。
