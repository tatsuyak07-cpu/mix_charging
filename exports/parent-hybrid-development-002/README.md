# 親面監査・保存再配分の検証データ

平面接触6条件、CFL 3水準の3刻み移行、固定充填層2辺×誤差予算3水準を検証。
実充填層は90250刻みで接触中の再メッシュを承認し、90262刻みまで継続後、親面精度不足で停止。
空隙率36.941697%は過渡値。空隙率10%・保持/除荷完走・収束確認は未達成。

- [検証図](verification.png)、[実メッシュ](packing-mesh.png)
- [平面CSV](plane-comparison.csv)、[CFL比較CSV](coupled-cfl.csv)、[固定配置CSV](frozen-packing-budgets.csv)、[実充填層の過渡CSV](packing-prefix.csv)
- [判定JSON](assessment.json)、[詳細報告](../../docs/parent-hybrid-development.md)、[ダウンロードZIP](../parent-hybrid-development-002.zip)

ZIP内のdiagnosticsには失敗した試行も含む。last-accepted-checkpointは最後の受理状態。
重複する中間チェックポイントと大きな生の接触台帳は元のresultsに保持する。
sourceは最終実装、各試行内sourceはその実行時の実装。実充填層では重み再配分を使っていない。
