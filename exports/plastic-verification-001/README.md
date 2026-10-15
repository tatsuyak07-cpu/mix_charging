# 大変形弾塑性の追加検証

30 mm角の未校正モデル6ケース。刻み幅、仮の材料値、メッシュ、仕事を比較しました。
刻み幅・仕事検査は基準内ですが、1 mm→0.75 mmでピーク応力が約20%低下し、メッシュ収束は未達です。
自己接触未実装、100 mm角・0.1 MPaまでの計算は未実施です。
比較CSV、PNG、全判定JSON、各ケースの入力・設定・メッシュ・最終VTUを収録しています。
energy-summary-original.jsonは元の正規化、energy-summary-peak-normalized.jsonはサイクルのピーク内部エネルギーによる正規化です。
詳細はリポジトリの docs/plastic-verification.md を参照してください。
