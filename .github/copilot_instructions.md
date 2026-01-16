# GitHub Copilot 用リポジトリ指示ファイル

このリポジトリで Copilot を使う際のガイドラインと、プロジェクト固有の前提条件をまとめています。

目的
- Planarian（プラナリア）行動解析用のスクリプト群を補完・編集する際の指針を提供します。

重要な前提
- 画像は1枚あたり10秒で取得されています（frame interval = 10秒）。
- GUI（tools）で入力される時間設定は `time_config.json` に保存される場合があります。主に以下のキーを参照します:
  - `day_start_time` (例: "07:00")
  - `night_start_time` (例: "19:00")
  - `measurement_start_time` (例: "09:00:00")
- CSV（results.csv）には少なくとも `filename`, `centroid_x`, `centroid_y`, `major_axis`, `minor_axis` の列が存在します。
- 時刻ログ（CSV内の `datetime` 列）は不正確な場合があるため、GUI入力の測定開始時刻とファイル名の順序（フレーム順）に基づいてタイムスタンプを再生成することを優先します。

スタイルとベストプラクティス
- 日本語コメント／文字列を用いて説明する。変数名は英語で一貫させる。
- 日時は pandas の `Datetime` 型で管理する。プロット軸や集計は `datetime` 列が datetime64[ns] であることを前提とする。
- ファイル名のソートは自然順（numeric-aware）で行い、フレーム番号が正しく昇順になるようにする。
- 可能な限り既存の関数やクラスを壊さない形で変更する。

よくあるタスクの例
- CSV読み込み時に `datetime` を再生成する: GUIからの `measurement_start_time` と1フレーム=10秒を使って、ファイル名の自然順に基づいてタイムスタンプを付与する。
- 時間ビン作成の際、最終ビンがデータを含めるように終端を十分に伸ばす（time_interval に応じて余裕を持たせる）。
- 昼夜判定は `day_start_time` / `night_start_time` を用いる。Constant darkness の場合（昼=夜時刻）には背景表示等を無視する。

編集の際の注意点
- 既存の unit tests はないが、編集後は `behavior_analysis.py` が import 可能で簡単なロード実行ができることを確認すること。
- 大きな構造変更は事前に設計を提示してから行う。
- 余計なテストデータの追加や新しいpyファイルの追加は基本しない
- ファイル追加が必要なら伺いを立ててから行う (勝手に追加しない)

接触先
- レポジトリ内の `README.md` を参照してドメイン知識を確認してください。

条件
- 個体は常に1匹であると仮定します。
- 昼の画像と夜の画像があり、夜間はかなり検出しづらいです。

コード修正後に以下のエラーが出がちなので、回避
発生場所 行:1 文字:48
+ cd C:\Users\Shinichi\PycharmProjects\Planarian && python -c "with ope ...
+                                                ~~
トークン '&&' は、このバージョンでは有効なステートメント区切りではありません。
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : InvalidEndOfLine


---

以上を基に Copilot を使って補完や変更を行ってください。問題が不明瞭な場合は、まず小さい修正を提案してから適用することを推奨します。

