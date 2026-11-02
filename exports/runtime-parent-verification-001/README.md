# 時間発展中の保存再配分の検証

受理済み配置で力・エネルギー・摩擦履歴を保持し、次の試行配置に合わせて既存点の重みを再配分する。
平面接触の3精度×2初期滑り条件で正逆増分と接触状態の保存/再開を検査。
実充填層の停止理由・再配分件数は[判定JSON](assessment.json)を参照。
停止状態の空隙率は過渡値。40%→10%の圧密・保持/除荷・収束確認は未完了。

- [比較画像](verification.png)、[実メッシュ](packing-mesh.png)
- [平面CSV](plane-cycle.csv)、[充填層CSV](packing-prefix.csv)
- [制約切り分けCSV](constraint-blocks.csv)、[受理済み状態と予算配分](accepted-budget-consistency.json)
- [報告](../../docs/runtime-parent-verification.md)、[ZIP](../runtime-parent-verification-001.zip)

ZIPのdiagnosticsは中断・失敗した試行も含む。重複チェックポイント・大きな生の接触台帳はresultsに保持。
last-accepted-checkpointは受理済みの状態。sourceは最終実装、各実行内sourceは実行時の実装。
