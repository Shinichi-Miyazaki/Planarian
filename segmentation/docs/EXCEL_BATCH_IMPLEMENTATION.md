# Excelベースのバッチ解析 + 複数日統合解析機能 - 実装サマリー

実装日: 2026-02-10

---

## 📋 実装内容

### 1. 新規ファイル作成

#### `batch_summary_analysis.py`
複数日データの統合解析モジュール

**主要機能**:
- 複数実験のaggregated CSVを読み込み
- 明期開始(12:00:00)基準で相対時刻に変換
- z-score正規化による標準化
- 時系列プロット（3指標: 移動量、不動性割合、体長）
- 明期 vs 暗期のjitter + boxplot比較
- R解析用の統合CSV出力

**クラス**:
- `BatchSummaryAnalyzer`: 統合解析の主要クラス
  - `load_and_combine_data()`: データ読み込みと統合
  - `normalize_data()`: z-score正規化
  - `plot_timeseries()`: 時系列プロット作成
  - `plot_light_dark_comparison()`: 明暗期比較プロット作成
  - `save_combined_csv()`: R解析用CSV出力
  - `run_full_analysis()`: 全解析ステップの実行

**注意点**:
- 測定開始が12:00:00以降の場合、12:00:00から測定開始までは欠損値として扱う
- z-score正規化は全区間データで実施（個体差・日間変動を補正）

#### `docs/EXCEL_BATCH_ANALYSIS_GUIDE.md`
Excelベースのバッチ解析機能の詳細ガイド

**内容**:
- Excelファイルのフォーマット仕様
- 使い方の詳細説明
- 出力ファイル構成
- 統合解析の詳細
- トラブルシューティング

#### `create_sample_excel.py`
サンプルExcelファイル作成スクリプト

**用途**:
- テンプレートExcelファイルの生成
- テスト用

---

### 2. 既存ファイルの修正

#### `inference_analysis_gui.py`

**追加機能**:
1. Excelファイル選択UI
   - `self.excel_file`: Excelファイルパスの変数
   - `browse_excel_file()`: Excelファイル選択メソッド

2. Excel読み込み機能
   - `_search_folders_thread()`: Excel指定時の処理分岐を追加
   - `pd.read_excel()`: Excelからdir_name, start_date, start_timeを読み込み
   - フォルダ名マッチング: 親フォルダ配下でdir_nameと一致するフォルダを検索
   - 日付フォーマット変換: yyyyMMdd → YYYY-MM-DD

3. 統合解析の実行
   - `_run_batch_analysis_thread()`: 個別解析結果を収集
   - 2つ以上の実験が成功した場合、`batch_summary_analysis.run_batch_summary_analysis()`を呼び出し
   - 統合解析結果を親フォルダ直下の`batch_summary/`に出力

**修正箇所**:
- `__init__()`: `self.excel_file`変数を追加
- `create_batch_mode_widgets()`: Excelファイル選択UIを追加
- `search_folders()`: excel_fileパラメータを追加
- `_search_folders_thread()`: Excel読み込みロジックを追加
- `_run_batch_analysis_thread()`: 統合解析の実行ロジックを追加
- `_run_single_analysis_thread()`: 戻り値を受け取るように変更

#### `run_inference_analysis.py`

**追加機能**:
- `run_inference_and_analysis()`の戻り値を追加
- 解析結果のパス情報を返却

**戻り値**:
```python
{
    'output_dir': str,          # 出力ディレクトリ
    'aggregated_csv': str,      # 集約データCSVのパス
    'detailed_csv': str,        # 詳細データCSVのパス
    'summary_csv': str,         # サマリーCSVのパス
    'time_interval': int        # 時間間隔（分）
}
```

#### `requirements.txt`

**追加パッケージ**:
- `seaborn>=0.11.0`: jitter + boxplotの作成
- `openpyxl>=3.0.0`: Excelファイルの読み込み

---

## 🔧 技術的な詳細

### z-score正規化

**計算式**:
```python
z-score = (x - mean) / std
```

**実装理由**:
- 個体差や測定日間のベースライン変動を補正
- 異なる実験間での相対的な変動パターンを比較可能
- 平均0、標準偏差1に標準化

### 明期開始基準の時刻変換

**明期開始時刻**: 12:00:00（固定）

**変換ロジック**:
1. 測定開始 < 12:00:00の場合:
   - 相対時刻 = データポイントの経過時間 - (12:00:00 - 測定開始)
   - 例: 08:00:00開始 → 12:00:00は+4h

2. 測定開始 >= 12:00:00の場合:
   - 相対時刻 = データポイントの経過時間 + (測定開始 - 12:00:00)
   - 0.00h ～ (測定開始 - 12:00:00)は欠損値
   - 例: 13:30:00開始 → 0.00h～1.50hは欠損

### 明期・暗期の判定

- **明期**: 12:00:00 - 24:00:00 (相対時刻: 0h - 12h)
- **暗期**: 00:00:00 - 12:00:00 (相対時刻: 12h - 24h)

```python
period = 'Light' if 0 <= (hours_from_light_start % 24) < 12 else 'Dark'
```

---

## 📊 出力ファイル

### 個別フォルダの出力
各画像フォルダ内の`segmentation_analysis/`:
- `analysis_results.csv`: セグメンテーション結果
- `detailed_immobility_analysis.csv`: 詳細行動解析
- `aggregated_immobility_analysis.csv`: 時間ビン集約データ（統合解析で使用）
- `day_night_summary.csv`: 昼夜別統計
- `time_config.json`: 時間設定
- `*.png`: グラフ

### 統合解析の出力
親フォルダ直下の`batch_summary/`:
- `timeseries_comparison.png`: 時系列プロット（3指標）
- `light_dark_comparison.png`: 明期 vs 暗期比較
- `combined_data_for_R.csv`: R解析用統合データ
- `summary_statistics.txt`: サマリー統計

---

## 🎨 グラフスタイル

### カラーパレット
指定の色リストから選択:
```python
COLOR_PALETTE = [
    '#4169E1',  # blue_base
    '#DC143C',  # red_base
    '#228B22',  # green_base
    '#FFA500',  # orange_base
    '#800080',  # purple_base
    '#FF00FF',  # magenta_base
    '#808080',  # gray_base
]
```

### プロットスタイル
- **時系列**: 線グラフ + マーカー、暗期背景をグレー表示
- **明暗期比較**: boxplot（明期=金色、暗期=青色）+ jitter plot（黒色、透明度40%）

---

## ✅ テスト方法

### 1. サンプルExcelファイルを作成

```powershell
cd C:\Users\Shinichi\PycharmProjects\Planarian\segmentation
python create_sample_excel.py
```

### 2. GUIでテスト

1. `python inference_analysis_gui.py`
2. 「バッチフォルダ解析」タブを選択
3. 親フォルダとExcelファイルを選択
4. 「フォルダを検索」→ フォルダが検出されることを確認
5. 「バッチ実行」→ 個別解析と統合解析が実行されることを確認

### 3. 出力確認

- 各フォルダ内に`segmentation_analysis/`が作成されているか
- 親フォルダ直下に`batch_summary/`が作成されているか
- 統合プロットが正しく表示されているか
- R解析用CSVが出力されているか

---

## 🚨 既知の制限事項

1. **Excelフォーマット**
   - 1枚目のシートのみ読み込み
   - 列名は固定（`dir_name`, `start_date`, `start_time`）

2. **統合解析の実行条件**
   - 2つ以上の実験が成功した場合のみ実行
   - 1つのみの場合は個別解析のみ

3. **日付・時刻フォーマット**
   - 厳密なフォーマット要求（yyyyMMdd, HH:mm:ss）
   - Excel のセル書式に注意が必要

4. **測定開始時刻が12:00:00以降**
   - 12:00:00から測定開始までは欠損値として扱われる
   - プロットで線が途切れて表示される

---

## 🔄 今後の拡張可能性

1. **Excelフォーマットの柔軟化**
   - 列名のカスタマイズ
   - 複数シートの対応

2. **正規化方法の選択**
   - z-score以外の正規化手法（min-max, robust scalingなど）
   - UI上での選択機能

3. **統計検定の追加**
   - 明期 vs 暗期のt検定
   - ANOVA（複数日間の比較）
   - 自動的にp値を計算・表示

4. **プロットのカスタマイズ**
   - 色・スタイルの設定UI
   - グラフサイズの調整

---

## 📚 参考資料

- 指定の色リスト: global-copilot-instructions
- グラフテーマ: classic theme with specified colors
- 統計手法: z-score normalization for cross-day comparison

---

## ✨ まとめ

この実装により、以下が可能になりました:

1. ✅ Excelファイルからの実験条件読み込み
2. ✅ 指定フォルダのみの自動検出・解析
3. ✅ 複数日データの統合と時刻の統一
4. ✅ z-score正規化による標準化
5. ✅ 時系列プロットと明暗期比較プロット
6. ✅ R解析用CSV出力
7. ✅ 測定開始が12:00:00以降の場合の欠損値処理

効率的で柔軟なバッチ解析ワークフローが実現されました！

