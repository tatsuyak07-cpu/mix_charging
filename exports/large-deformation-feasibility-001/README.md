# 充填層の大変形への到達可能性

2026-10-29（日本時間）の設計検討です。新たなFEM圧縮計算は行っていません。

**有限変形FEMの方式は候補として妥当ですが、現行コードの局所修正だけで大変形到達を見込む方針は推奨しません。**
接触・摩擦履歴・ゆがみ回復を一体で通す小規模実証を先に行い、継続／エンジン移行を判断します。

[検討書](../../docs/large-deformation-feasibility.md) / [判定・概算JSON](assessment.json) / [開発工程CSV](development-gates.csv)

0.1 MPa保持と空隙率10%は別の目標です。どちらも新たに到達したという記録ではありません。
100 mm化の要素数は幾何外挿で、実行時間の測定値ではありません。

validation/evidence/に検討時点のソース・既存報告・設定・小さな判定データを保存しています。
validation/input-manifest.jsonは入力SHA256、manifest.jsonはこの成果物のSHA256です。
