# 解析済みフォルダのスキップ機能 - 実装サマリー

バッチ解析における解析済みフォルダの自動スキップ機能と関連機能の実装

---

## 📋 実装内容

### 1. 解析済みフォルダの自動検出とスキップ

#### 機能概要
- `segmentation_analysis/`フォルダの存在を自動検出
- 必須ファイルの完全性をチェック
- 解析済みフォルダは自動的にスキップし、既存データを統合解析に利用
- 部分的な解析結果（ファイル欠損）は警告を出して再解析

#### 必須ファイルのチェックリスト
```
segmentation_analysis/
├── analysis_results.csv
├── detailed_immobility_analysis.csv
├── aggregated_immobility_analysis.csv
├── day_night_summary.csv
└── time_config.json
```

#### 実装詳細

**新規メソッド**: `check_folder_analyzed(folder_path)`
- 解析ディレクトリの存在確認
- 必須ファイルの完全性チェック
- 戻り値: `(is_analyzed, is_complete)` タプル

**処理フロー**:
1. フォルダ検索時に各フォルダの解析状態をチェック
2. `detected_folders`に解析済みフラグを追加（5要素タプル）
3. バッチ実行時に解析済みフォルダをスキップ
4. 既存の`aggregated_immobility_analysis.csv`を統合解析に含める

---

### 2. 強制再解析オプション

#### 機能概要
- GUIにチェックボックスを追加: 「解析済みフォルダも強制再解析」
- チェックON: 解析済みフォルダも再度解析
- チェックOFF（デフォルト）: 解析済みフォルダをスキップ

#### 使用シーン
- 解析パラメータ（時間間隔、昼夜の開始時刻）を変更した場合
- 解析アルゴリズムを改善した後に全データを再計算
- モデルを更新してセグメンテーション結果を再取得

#### 変数
- `self.force_reanalysis`: BooleanVar型
- デフォルト値: `False`

---

### 3. 統合解析結果の出力先指定

#### 機能概要
- 統合解析結果（`batch_summary`）を任意のフォルダに保存可能
- 未指定時は従来通り親フォルダ直下の`batch_summary/`に保存

#### UI要素
- 「統合結果出力先」入力欄と参照ボタンを追加
- プレースホルダー: "※ 未指定時は親フォルダ/batch_summary"

#### 変数
- `self.batch_output_dir`: StringVar型
- デフォルト値: 空文字列（親フォルダのbatch_summaryを使用）

#### 実装詳細
```python
# 統合解析の出力先を決定
if self.batch_output_dir.get():
    summary_output_dir = self.batch_output_dir.get()
else:
    # デフォルト: 親フォルダ直下にbatch_summary/を作成
    parent_path = self.parent_dir.get()
    summary_output_dir = os.path.join(parent_path, 'batch_summary')
```

---

### 4. 解析状態の視覚化

#### Treeviewの拡張
- 「解析状態」列を追加（5列構成に変更）
- 色分け表示:
  - **✓ 解析済み**（緑色）: 完全な解析結果が存在
  - **未解析**（赤色）: まだ解析されていない

#### ステータスメッセージの改善
```
検出: 5個のフォルダ (解析済み: 3, 未解析: 2)
```

---

### 5. バッチ実行結果の分類表示

#### 結果の分類
- **成功**: 新規に解析を実行して完了
- **スキップ**: 既存データを使用（解析済みフォルダ）
- **失敗**: エラーが発生

#### 結果ウィンドウの改善
- 統計表示: `成功: X | スキップ: Y | 失敗: Z`
- 色分け:
  - 成功: 緑色
  - スキップ: 青色
  - 失敗: 赤色

---

## 🔧 技術詳細

### データ構造の変更

#### `detected_folders`リストの拡張
**旧**: 4要素タプル
```python
(folder_path, image_count, measurement_date, measurement_start_time)
```

**新**: 5要素タプル
```python
(folder_path, image_count, measurement_date, measurement_start_time, is_analyzed)
```

### 既存データの読み込み

**新規メソッド**: `_load_existing_analysis_info(output_dir, meas_time, measurement_date)`

機能:
- 既存の`aggregated_immobility_analysis.csv`を検出
- `time_config.json`から`time_interval`を読み込み
- 統合解析用のデータ構造を返す

戻り値:
```python
{
    'output_dir': str,
    'aggregated_csv': str,
    'detailed_csv': str,
    'summary_csv': str,
    'time_interval': int
}
```

### time_config.jsonの拡張

**追加項目**: `time_interval`

目的: 既存解析結果の時間間隔を保存・復元

```json
{
    "day_start_time": "12:00",
    "night_start_time": "23:00",
    "measurement_start_time": "09:00:00",
    "measurement_date": "2026-02-10",
    "time_interval": 10
}
```

---

## 📊 処理フロー

### フォルダ検索時
```
1. 親フォルダを走査
2. 各フォルダで画像数をカウント
3. check_folder_analyzed()で解析状態をチェック
4. (folder_path, image_count, date, time, is_analyzed)のタプルを作成
5. Treeviewに色分けして表示
```

### バッチ実行時
```
1. detected_foldersをループ
2. 各フォルダで:
   a. is_analyzed == True かつ force_reanalysis == False
      → スキップして_load_existing_analysis_info()を呼び出し
   b. それ以外
      → 通常の解析を実行
3. 全フォルダの解析結果（新規+既存）を収集
4. 2つ以上の実験がある場合、統合解析を実行
5. 指定された出力先に結果を保存
```

---

## 💡 利点

### 1. 時間の節約
- 大規模なバッチ解析の再実行時に、既に解析済みのフォルダをスキップ
- 一部のフォルダのみ追加で解析する場合に便利

### 2. データの一貫性
- 既存の解析結果を統合解析に含めることができる
- パラメータ変更時のみ強制再解析を使用

### 3. 柔軟性
- 統合解析結果を任意の場所に保存可能
- プロジェクト構造に応じた出力先を選択

### 4. エラーハンドリング
- 部分的な解析結果（ファイル欠損）を自動検出
- 警告を出して自動的に再解析

---

## 🧪 テスト方法

### テストケース1: 解析済みフォルダのスキップ
1. 一部のフォルダを事前に解析
2. バッチ実行（強制再解析OFF）
3. 解析済みフォルダが緑色で表示され、スキップされることを確認
4. 統合解析結果に既存データが含まれることを確認

### テストケース2: 強制再解析
1. 全フォルダを事前に解析
2. 強制再解析をONにしてバッチ実行
3. 全フォルダが再解析されることを確認

### テストケース3: 部分的な解析結果
1. あるフォルダの`segmentation_analysis/`から必須ファイルを1つ削除
2. フォルダ検索を実行
3. 警告メッセージが表示されることを確認
4. バッチ実行時に自動的に再解析されることを確認

### テストケース4: 統合結果の出力先指定
1. 統合結果出力先に別フォルダを指定
2. バッチ実行
3. 指定したフォルダに`batch_summary/`が作成されることを確認

---

## 📝 コード変更箇所

### inference_analysis_gui.py

#### `__init__()` メソッド
- `self.batch_output_dir` 追加
- `self.force_reanalysis` 追加
- `self.detected_folders` のコメント更新（5要素タプル）

#### `create_batch_mode_widgets()` メソッド
- 統合結果出力先の入力欄と参照ボタンを追加
- 強制再解析チェックボックスを追加
- Treeviewに「解析状態」列を追加

#### 新規メソッド
- `browse_batch_output()`: 統合結果出力先の選択
- `check_folder_analyzed()`: 解析済みフォルダのチェック
- `_load_existing_analysis_info()`: 既存解析データの読み込み

#### 修正メソッド
- `_search_folders_thread()`: 解析状態のチェックを追加
- `_update_folder_list()`: 色分け表示と統計情報の追加
- `edit_folder_time()`: 5要素タプルに対応
- `remove_selected_folder()`: 5要素タプルに対応
- `_run_batch_analysis_thread()`: スキップロジックの追加
- `_run_batch_onnx_analysis()`: time_config.jsonにtime_interval追加
- `_run_onnx_inference_and_analysis()`: time_config.jsonにtime_interval追加
- `_on_batch_completion()`: スキップカウントの追加
- `_show_batch_result_detail()`: スキップの色分け表示

---

## 🔄 互換性

### 既存データとの互換性
- 古いバージョンで作成された`segmentation_analysis/`フォルダも正しく検出
- `time_config.json`に`time_interval`が無い場合はデフォルト値を使用

### 後方互換性
- 統合結果出力先を指定しない場合、従来通りの動作（親フォルダ/batch_summary）
- 強制再解析をOFFにした場合、従来の動作と同等（全て実行）

---

## 📚 関連ドキュメント

- [EXCEL_BATCH_ANALYSIS_GUIDE.md](./EXCEL_BATCH_ANALYSIS_GUIDE.md): ユーザー向け使用ガイド
- [EXCEL_BATCH_IMPLEMENTATION.md](./EXCEL_BATCH_IMPLEMENTATION.md): Excelバッチ解析の実装詳細

---

## 🚀 今後の拡張案

1. **並列処理**: 複数フォルダを並列に解析（現在は順次実行）
2. **解析履歴の記録**: どのパラメータでいつ解析したかを記録
3. **差分解析**: パラメータ変更後に影響を受けるフォルダのみ再解析
4. **解析キャッシュ**: より細かい粒度でのキャッシュ管理

---

実装日: 2026-02-10

