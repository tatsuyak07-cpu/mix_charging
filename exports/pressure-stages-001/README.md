# 工程1〜5の実装・検証と段階圧縮

**0.1 MPaの平衡解は未取得です。** 接触開始直後の再積分要求が続いたため中断しました。
小さなBCC試験セルであり、100 mm充填層・メッシュ収束・40%から10%の圧密を保証しません。

| 平衡Cauchy応力 MPa | 空隙率 % | メッシュ |
|---:|---:|---|
| 0.001 | 39.832238 | [PNG](stage-0.001-MPa.png) |
| 0.01 | 39.625659 | [PNG](stage-0.01-MPa.png) |

[平衡段階CSV](stages.csv) / [過渡履歴CSV](history.csv) / [圧力・空隙率PNG](pressure-porosity.png)

[計算・検証報告](../../docs/pressure-stages.md) / [事後集計](calculation-summary.json)

validation-part-*.zipは独立したZIPです。すべて同じ場所に展開してvalidation/を復元してください。
元の失敗・中断・不完全チェックポイントも保持しています。使用可否は事後集計と報告を参照してください。
validation-manifest.jsonに収録先・SHA256を記録しています。source/に最終のプログラムと設定を保存しています。
