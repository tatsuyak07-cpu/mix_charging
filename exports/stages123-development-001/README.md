# 工程1〜3の開発・比較検証

関連92テスト合格。対象4親面の埋込積分点を197点とし、接触前から全連成再計算して旧停止区間を通過。
90,262→90,273刻み、物理時刻でも旧棄却試行を越えた。その先の積分誤差は未解消。
早期再配分単独の改善効果は充填層では未確認。大変形・平衡保持/除荷・空隙率10%・収束は未達成。

- [比較図](comparison.png)、[点配置](quadrature-points.png)、[実メッシュ](packing-mesh.png)
- [条件比較CSV](case-comparison.csv)、[点配置CSV](quadrature-points.csv)
- [判定JSON](assessment.json)、[詳細報告](../../docs/stages123-development.md)、[全データZIP](../stages123-development-001.zip)

ZIPのdiagnosticsに失敗した案、実行時ソース、全試行の判定/ログを保存。
last-accepted-checkpointは最後の受理済み過渡状態。inputsに接触前チェックポイントと選択した積分則を収録。
geometry-control-*は配置生成専用で、計算再開用ではない。元のresultsは上書きしていない。
