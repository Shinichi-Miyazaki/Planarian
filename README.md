# Planarian Detection & Behavior Analysis System

プラナリアの自動検出・追跡・行動解析システム。画像解析により個体の位置・面積・活動量を自動測定し、行動解析を行います。

## システム概要

- **セグメンテーション推論**: ディープラーニング（U-Net）による高精度検出
- **行動解析**: 移動量・活動リズム・昼夜比較の統計解析
- **可視化**: 時系列グラフ・統計レポート・動画生成

---

## ⚙️ インストール

```powershell
# 仮想環境を作成（推奨）
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 必要なパッケージをインストール
pip install -r requirements.txt
```

---

## 📁 プロジェクト構成

```
Planarian/
├── behavior_analysis.py          # 行動解析スクリプト（CSVから解析）
├── requirements.txt               # パッケージ依存関係
├── README.md                      # このファイル
├── segmentation/                  # セグメンテーションモジュール
│   ├── run_inference_analysis.py  # 統合スクリプト（推論+解析）
│   ├── inference_analysis_gui.py  # GUIランチャー
│   ├── inference.py               # セグメンテーション推論
│   ├── train.py                   # モデル学習
│   ├── labeling_gui.py            # ラベリングツール
│   ├── config.py                  # 設定ファイル
│   ├── README.md                  # セグメンテーションガイド
│   ├── data/                      # 学習データ
│   ├── models/                    # 学習済みモデル
│   ├── outputs/                   # 出力結果
│   ├── docs/                      # ドキュメント
│   └── legacy/                    # 旧バージョン・Colab用スクリプト
├── docs/                          # システムドキュメント
│   ├── GPU_COMPATIBILITY_ISSUE.md
│   └── install_pytorch_gpu.md
└── legacy/                        # 旧スクリプト
    ├── animal_detector_gui.py     # 従来型画像処理GUI
    ├── check_cuda.py
    └── check_gpu.py
```

---

## 🚀 クイックスタート

### 1. セグメンテーション推論 + 行動解析（推奨）

**画像フォルダを指定して、一括で解析を実行:**

```powershell
cd segmentation

# コマンドライン版
python run_inference_analysis.py --images <画像フォルダ> --output <出力フォルダ>

# GUI版
python inference_analysis_gui.py
```

**出力:**
- セグメンテーション結果CSV
- 行動解析データ（移動量・不動性）
- 時系列グラフ
- 統計レポート（昼夜別）
- セグメンテーション動画（オプション）

**詳細:** `segmentation/README.md`

---

### 2. CSVから行動解析のみ

**既にCSVがある場合:**

```powershell
python behavior_analysis.py
# → GUIでCSVファイルを選択
```

**出力:**
- 移動量・不動性の時系列グラフ
- 昼夜別統計CSV
- サマリーレポート

---

## 📖 ドキュメント

### メインドキュメント
- **セグメンテーション**: `segmentation/README.md`
- **行動解析**: このREADMEの後半

### 追加ドキュメント（docs/）
- `GPU_COMPATIBILITY_ISSUE.md`: GPU互換性の問題
- `install_pytorch_gpu.md`: GPU版PyTorchインストール

### セグメンテーションドキュメント（segmentation/docs/）
- `COLAB_TRAINING_GUIDE.md`: Google Colabでの学習ガイド
- `COLAB_SINGLE_CELL_GUIDE.md`: 単一セル学習（シンプル版）
- `PATH_CONFIGURATION_GUIDE.md`: パス設定ガイド
- `TEST_INFERENCE_VISUALIZATION.md`: 推論結果の可視化テスト

---

## 🔧 セグメンテーションモデルの学習

### オプション1: Google Colab（推奨）

**無料GPUを使用した高速学習:**

1. `segmentation/legacy/train_colab.ipynb`をGoogle Colabで開く
2. ガイドに従ってデータをアップロード
3. 学習を実行（約30分〜1時間）
4. 学習済みモデルをダウンロード

**詳細:** `segmentation/docs/COLAB_TRAINING_GUIDE.md`

### オプション2: ローカル学習

```powershell
cd segmentation

# 1. ラベリング（100枚以上推奨）
python labeling_gui.py

```

**詳細:** `segmentation/README.md`

---

## 📊 行動解析

### CSVから行動解析を実行

```powershell
python behavior_analysis.py
# → GUIでCSVファイルを選択
```

### 機能

1. **移動量計算**: フレーム間の重心移動距離
2. **不動性判定**: 移動量3ピクセル以下を不動とカウント
3. **時間集約**: 10分間隔で集計
4. **昼夜比較**: 昼夜サイクルに基づく統計解析

### 出力

- **グラフ（PNG）**:
  - 移動量の時系列
  - 不動性割合の時系列
  - 体長の時系列
  - 昼夜別の比較

- **CSV**:
  - `detailed_immobility_analysis.csv`: フレームごとの詳細データ
  - `aggregated_immobility_analysis.csv`: 時間集約データ
  - `day_night_summary.csv`: 昼夜別統計サマリー

### 時間設定

測定開始時刻と昼夜サイクルは`time_config.json`で設定:

```json
{
  "day_start_time": "07:00",
  "night_start_time": "19:00",
  "measurement_start_time": "09:00:00",
  "measurement_date": "2026-02-06"
}
```

---

## 🛠️ トラブルシューティング

### パッケージインストールエラー

```powershell
# 基本パッケージのインストール
pip install pandas matplotlib scipy opencv-python numpy torch torchvision segmentation-models-pytorch tqdm
```

**注意:** 推論時は**albumentationsは不要**です。ビルドエラーが出ても無視してOKです。

---

### albumentationsのビルドエラー（無視してOK）

```
error: Microsoft Visual C++ 14.0 or greater is required
```

**解決:** 推論時はalbumentations不要なので無視してください。

学習時のみ必要ですが、Google Colabで学習する場合は問題ありません。

---

### GPU互換性

- RTX 5070 Ti (sm_120)は現在のPyTorchで未対応
- **Google Colabの使用を推奨**（無料GPU）
- 詳細: `docs/GPU_COMPATIBILITY_ISSUE.md`

### モデルファイルが見つからない

```
FileNotFoundError: モデルファイルが見つかりません
```

**解決方法:**
1. Google Colabで学習済みモデルを生成
2. `segmentation/models/best_unet.pth`に配置
3. または`--model`オプションでパスを指定

---

## 📝 ライセンス・引用

このプロジェクトを使用する場合は、適切な引用をお願いします。

---

## 🤝 貢献

バグ報告や機能提案は、GitHubのIssueまでお願いします。

---

## 📞 サポート

詳細なドキュメントは各ディレクトリのREADMEを参照してください：
- セグメンテーション: `segmentation/README.md`
- GPU問題: `docs/GPU_COMPATIBILITY_ISSUE.md`
- Colab学習: `segmentation/docs/COLAB_TRAINING_GUIDE.md`
# → models/best_unet.pth に保存
# → outputs/training_history.png で学習曲線を確認

# 3. 推論
python inference.py --images <入力フォルダ> --output <出力フォルダ>
# → outputs/analysis_results.csv（behavior_analysis.py互換）
# → outputs/segmentation_video.avi（動画）
```

**設定ファイル (`config.py`):**
```python
# モデル設定
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
ENCODER_NAME = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'

# 学習設定
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# 推論設定
CONFIDENCE_THRESHOLD = 0.5
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 50000
```

**長所:**
- 高精度（夜間画像でも検出可能）
- パラメータ調整不要（学習済みモデルを使用）
- 新しいデータにも対応可能（追加学習）

**短所:**
- GPU推奨（RTX 5070 Ti推奨）
- 訓練データ作成が必要（100枚以上）

---

### 2. 行動解析システム (`behavior_analysis.py`)

**機能:**
- **時系列データ処理**: CSVから時系列データを生成
- **活動量解析**: 移動距離、不動時間、活動リズムの解析
- **統計解析**: 昼夜比較、外れ値除去、統計的検定
- **可視化**: 時系列グラフ、ヒートマップ、活動パターンの可視化

**使い方:**
```python
from behavior_analysis import BehaviorAnalyzer
from datetime import date

# 初期化
analyzer = BehaviorAnalyzer(
    csv_path='results.csv',
    time_interval_minutes=10,  # ビン幅（分）
    day_start_time='07:00',  # 昼の開始時刻
    night_start_time='19:00',  # 夜の開始時刻
    measurement_start_time='09:00:00',  # 測定開始時刻
    measurement_date=date(2026, 2, 5)  # 測定開始日
)

# データ読み込み
analyzer.load_data()

# 解析実行
analyzer.analyze_movement(
    threshold=10,  # 不動閾値（pixels）
    save_path='movement_analysis.png'
)

# 統計情報を出力
stats = analyzer.get_statistics()
print(stats)
```

**入力CSVフォーマット:**
```csv
filename,centroid_x,centroid_y,major_axis,minor_axis,area,circularity
00001.jpg,320.5,240.8,45.2,30.1,1200,0.85
00002.jpg,322.1,241.5,46.0,29.8,1210,0.83
...
```

**重要な前提:**
- 画像は1枚あたり10秒で取得（`FRAME_INTERVAL_SECONDS = 10`）
- 個体は常に1匹と仮定
- 測定開始時刻とフレーム順からタイムスタンプを生成（CSVの時刻は使用しない）

**出力:**
- 時系列グラフ（面積・移動量の変化）
- 昼夜周期の可視化（背景色で区別）
- 統計情報（平均・標準偏差・最大・最小）
- 昼夜比較（t検定またはMann-Whitney U検定）

**Constant Darkness対応:**
- `day_start_time == night_start_time` の場合、恒常暗条件と判定
- 昼夜の背景色表示をスキップ

---

## 📈 主な機能

### 画像検出機能（従来手法）

#### 二値化方式
- **relative（推奨）**: 画像の平均輝度を基準に相対的に閾値を設定
  - `relative_thresh=0.15` で平均輝度の85%を閾値とする
  - 照明条件の変動に強く、夜間の誤検出を抑制
- **adaptive**: 適応的二値化（局所的に閾値を設定）
- **fixed**: 固定閾値（非推奨）

#### 背景除去機能
- **Temporal Median Filter**: 複数フレームの中央値から背景を作成して除去
  - 動きの少ないノイズ除去に有効
  - フレーム数100枚程度を推奨
- **自動背景補正**: 各画像で自動的に背景を推定して除去
  - **adaptive（推奨）**: ガウシアンブラーによる高速処理
  - **morphological**: モルフォロジー処理による背景除去
  - **rolling_ball**: Rolling Ball法（軽量化済み、radius=15）

#### 検出精度向上機能
- **ROI（関心領域）設定**: 円形ROIで検出範囲を制限
- **エッジ強調**: Sobelフィルタで輪郭を強調（重み調整可能）
- **コントラスト調整**: CLAHE（局所適応ヒストグラム平坦化）による夜間画像の強調
- **ターゲット選択**: プラナリアをクリックして最適パラメータを自動推定

---

### 行動解析機能（behavior_analysis.py）

#### データ処理
- **タイムスタンプ生成**: 測定開始時刻とフレーム間隔から正確な時刻を生成
- **移動量計算**: フレーム間の重心座標から移動距離を算出
- **不動割合計算**: 設定閾値以下の移動を不動と判定
- **外れ値除去**: IQR法またはZ-score法で異常値を除外

#### 統計解析
- **時間ビン集約**: 指定間隔（デフォルト10分）でデータを集約
- **昼夜比較**: 昼夜の活動量・移動距離の統計的比較
- **活動リズム解析**: 周期性の検出、ピーク時刻の特定
- **Constant Darkness対応**: 恒常暗条件での解析にも対応

#### 可視化機能
- **時系列グラフ**: 面積・移動量の時系列変化を可視化
- **昼夜周期の表示**: グラフ上に昼夜を色分けして表示
- **統計情報の表示**: 平均値・標準偏差・最大値・最小値を表示
- **測定開始時刻マーカー**: グラフ上に測定開始時刻を表示

---

## 🚀 クイックスタート

### 1. 従来手法で検出

```powershell
# 1. GUIを起動
python animal_detector_gui.py

# 2. 画像フォルダを選択

# 3. パラメータを調整してプレビュー確認

# 4. 解析開始
# → results.csv が出力される
```

### 2. ディープラーニングで検出

```powershell
cd segmentation

# 1. ラベリング（100枚以上）
python labeling_gui.py

# 2. 学習
python train.py

# 3. 推論
python inference.py --images <入力> --output <出力>
# → outputs/analysis_results.csv が出力される
```

### 3. 行動解析

```python
from behavior_analysis import BehaviorAnalyzer
from datetime import date

# 解析器を初期化
analyzer = BehaviorAnalyzer(
    csv_path='results.csv',
    time_interval_minutes=10,
    day_start_time='07:00',
    night_start_time='19:00',
    measurement_start_time='09:00:00',
    measurement_date=date(2026, 2, 5)
)

# データ読み込み
analyzer.load_data()

# 移動量解析
analyzer.analyze_movement(threshold=10, save_path='movement.png')

# 統計情報
stats = analyzer.get_statistics()
print(stats)
```

---

## 📂 ファイル構成

```
Planarian/
├── animal_detector_gui.py       # 従来手法GUI
├── behavior_analysis.py         # 行動解析モジュール
├── requirements.txt             # 依存パッケージ
├── README.md                    # 本ファイル
├── segmentation/                # ディープラーニング手法
│   ├── config.py                # 設定ファイル
│   ├── dataset.py               # データローダー
│   ├── unet_model.py            # U-Netモデル定義
│   ├── train.py                 # 学習スクリプト
│   ├── inference.py             # 推論スクリプト
│   ├── labeling_gui.py          # ラベリングツール
│   ├── utils.py                 # ユーティリティ関数
│   ├── data/                    # データセット
│   │   ├── images/              # 元画像
│   │   └── labels/              # ラベル（マスク）
│   ├── models/                  # 学習済みモデル
│   │   └── best_unet.pth        # ベストモデル
│   └── outputs/                 # 推論結果
│       ├── analysis_results.csv
│       ├── training_history.png
│       └── segmentation_video.avi
└── .github/
    └── copilot_instructions.md  # Copilot指示ファイル
```

---

## 💡 推奨ワークフロー

### シナリオ1: 照明条件が安定している場合
→ **従来手法（`animal_detector_gui.py`）** を推奨

```
1. GUIを起動
2. パラメータ設定: relative, 閾値0.15, 背景補正OFF
3. 解析実行
4. behavior_analysis.py で行動解析
```

### シナリオ2: 夜間画像が多い / 低コントラスト
→ **ディープラーニング手法（`segmentation/`）** を推奨

```
1. ラベリングGUIで訓練データ作成（100枚以上）
2. train.py で学習
3. inference.py で推論
4. behavior_analysis.py で行動解析
```

### シナリオ3: 初めてのデータセット
→ **両方試して比較**

```
1. 従来手法で検出（高速）
2. 検出精度が不十分な場合、ディープラーニング手法に切り替え
3. behavior_analysis.py で行動解析
```

---

## ⚠️ トラブルシューティング

### 問題1: 夜間の誤検出が多い（従来手法）
**解決策:**
- 相対閾値を `0.20-0.30` に上げる
- ROIを設定して検出範囲を制限
- Temporal Median Filterを使用（フレーム数100）

### 問題2: ディープラーニングの学習が進まない
**解決策:**
- 訓練データを100枚以上に増やす
- データ拡張パラメータを調整（`config.py`）
- 学習率を下げる（`LEARNING_RATE = 1e-5`）

### 問題3: 推論が遅い
**解決策:**
- GPUを使用（`DEVICE = 'cuda'` in `config.py`）
- バッチサイズを増やす（`BATCH_SIZE = 16`）
- 画像サイズを縮小（`IMAGE_HEIGHT = 256, IMAGE_WIDTH = 256`）

### 問題4: Rolling Ballが重い（従来手法）
**解決策:**
- `AUTO_BG_CORRECTION_RADIUS = 15` に設定済み（軽量化）
- 代わりに `adaptive` または `morphological` を使用

### 問題5: behavior_analysis.pyでエラー
**解決策:**
- CSVに `filename, centroid_x, centroid_y, major_axis, minor_axis` が含まれているか確認
- `measurement_start_time` と `measurement_date` を正しく設定
- タイムスタンプ生成は自動（CSVの時刻は無視）

---

## 📝 引用・ライセンス

このシステムは研究用途で開発されました。使用する際は適切に引用してください。

**使用ライブラリ:**
- OpenCV: https://opencv.org/
- PyTorch: https://pytorch.org/
- Segmentation Models PyTorch: https://github.com/qubvel/segmentation_models.pytorch
- Albumentations: https://albumentations.ai/

---

## 🔧 開発メモ

- **フレーム間隔**: 1枚あたり10秒（`FRAME_INTERVAL_SECONDS = 10`）
- **個体数**: 常に1匹と仮定
- **昼夜判定**: `day_start_time` / `night_start_time` を使用
- **Constant Darkness**: 昼夜時刻が同じ場合、背景色表示をスキップ
- **タイムスタンプ**: 測定開始時刻 + フレーム順から生成（CSVの時刻は使用しない）

---

## 📧 お問い合わせ

問題が発生した場合は、`.github/copilot_instructions.md` を参照してください。
