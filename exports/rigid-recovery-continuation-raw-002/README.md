# 終了済みの継続計算・未採用試行・工程4データ

空隙率17.2847%まで原物理条件で独立監査に合格しました。目標10%のcontinuation-010は計算中で、このスナップショットから除外しています。受理状態と未収束試行を混同しないでください。工程4の初期メッシュ比較は未合格です。

[検証の結果](../rigid-recovery-verification-001/README.md)・[先に展開する以前のアーカイブ](../rigid-recovery-001/README.md)。今回の保存には、その後終了したprobe-001、continuation-005〜009および011、3メッシュの実圧縮データを含みます。以前と同じ内容のファイルはSHA-256照合して再利用します。

停止理由はintentional-interruptions各JSONに残しています。009はより速く受理状態へ達した010を優先して停止、011の20回の予備エネルギー探索は今回の増分の元の連成残差を増やしたため採用しませんでした。ソースと未採用配置を保存しています。10%への到達や全経路収束を示す保存データではありません。

## 実行ソースと終了済み試行の全保存データ

各tar.xzは単独で展開できます。全パートを同じ保存先へ展開します。計算中のcontinuation-010は除外しています。
[全ファイルSHA-256](raw-archive-files.json)・[アーカイブ検証](archive-verification.json)。

- [rigid-recovery-continuation-raw-002-part-01.tar.xz](../rigid-recovery-continuation-raw-002-part-01.tar.xz)
- [rigid-recovery-continuation-raw-002-part-02.tar.xz](../rigid-recovery-continuation-raw-002-part-02.tar.xz)
- [rigid-recovery-continuation-raw-002-part-03.tar.xz](../rigid-recovery-continuation-raw-002-part-03.tar.xz)
- [rigid-recovery-continuation-raw-002-part-04.tar.xz](../rigid-recovery-continuation-raw-002-part-04.tar.xz)
- [rigid-recovery-continuation-raw-002-part-05.tar.xz](../rigid-recovery-continuation-raw-002-part-05.tar.xz)
- [rigid-recovery-continuation-raw-002-part-06.tar.xz](../rigid-recovery-continuation-raw-002-part-06.tar.xz)
- [rigid-recovery-continuation-raw-002-part-07.tar.xz](../rigid-recovery-continuation-raw-002-part-07.tar.xz)
- [rigid-recovery-continuation-raw-002-part-08.tar.xz](../rigid-recovery-continuation-raw-002-part-08.tar.xz)
- [rigid-recovery-continuation-raw-002-part-09.tar.xz](../rigid-recovery-continuation-raw-002-part-09.tar.xz)
- [rigid-recovery-continuation-raw-002-part-10.tar.xz](../rigid-recovery-continuation-raw-002-part-10.tar.xz)

以前のアーカイブとSHA-256が一致した547ファイルは重複格納していません。`raw-archive-files.json`の`previous_manifest`が示す以前の全パートを先に展開し、その後に今回の全パートを展開します。再利用したファイルとハッシュも同JSONに記録しています。

展開後の`source/`に実行コードと画像・CSV、`raw/`に終了済み計算の入力・保存状態・不採用試行が入っています。元の相対パスを復元する場合は、`source/results/rigid-stages-1-4-001/`を作成し`raw/`の内容をコピーしてから`source/`で実行します。参照する過去の保存データは対応する公開アーカイブも展開します。

このスナップショット作成後にcontinuation-012を開始したため、012の計算中データも含みません。保存処理の最初の試行は自分自身のログを入力に含めてハッシュ監査で不合格となり、未検証アーカイブは別途保管しました。外部ログを使った今回の再試行は全2763メンバーのSHA-256検査に合格しています。
