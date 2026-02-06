# セマンティックセグメンテーション - セットアップと使い方

U-Netベースのディープラーニングによる高精度プラナリア検出システム

---

## 🚀 クイックスタート（推奨）

### セグメンテーション推論 + 行動解析を一括実行

**学習済みモデルがある場合、画像フォルダを指定するだけで全て自動実行:**

```powershell
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

**オプション:**
```powershell
python run_inference_analysis.py \
  --images "C:\data\exp001\images" \
  --output "C:\data\exp001\results" \
  --model "models\best_unet.pth" \
  --day-start "07:00" \
  --night-start "19:00" \
  --measurement-start "09:00:00" \
  --measurement-date "2026-02-06" \
  --no-video
```

---

## ⚠️ GPU互換性について

RTX 5070 Ti (sm_120)は現在のPyTorchで未対応です。

### 解決策

1. **Google Colabで学習**（推奨・無料GPU）→ `docs/COLAB_TRAINING_GUIDE.md`
2. **CPUモードで実行**（遅いが動作）→ 現在の設定

---

## 📋 セットアップ手順

### 1. ラベリング（100枚以上推奨）

```powershell
python labeling_gui.py
```

- `data/images/` と `data/labels/` を選択
- **ショートカット**: N=次へ、P=前へ、S=保存、C=クリア、D=描画、E=消しゴム

### 2. モデル学習

#### 方法A: Google Colab（推奨）

1. データをZIP圧縮:
   ```powershell
   python legacy\create_data_zip.py
   ```

2. Google Colabで学習:
   - `legacy/train_colab.ipynb`をColabで開く
   - GPUランタイムを選択
   - すべてのセルを実行
   - `best_unet.pth`をダウンロード → `models/`に配置

**詳細:** `docs/COLAB_TRAINING_GUIDE.md`

#### 方法B: ローカル（CPUモード）

```powershell
python train.py
```

**注意:** CPUは非常に遅い（GPU比で10-50倍）

### 3. 推論のみ実行

```powershell
python inference.py --images <入力> --output <出力>
```

**出力:**
- `analysis_results.csv`（behavior_analysis.py互換）
- `segmentation_video.avi`（動画、オプション）

---

## 📁 ファイル構成

```
segmentation/
├── run_inference_analysis.py  # 統合スクリプト（推論+解析）
├── inference_analysis_gui.py  # GUIランチャー
├── inference.py               # セグメンテーション推論のみ
├── train.py                   # モデル学習
├── labeling_gui.py            # ラベリングツール
├── config.py                  # 設定ファイル
├── dataset.py                 # データローダー
├── unet_model.py              # U-Netモデル定義
├── utils.py                   # ユーティリティ関数
├── README.md                  # このファイル
├── data/                      # 学習データ
│   ├── images/                # 元画像
│   └── labels/                # ラベル（マスク）
├── models/                    # 学習済みモデル
│   └── best_unet.pth          # ベストモデル
├── outputs/                   # 出力結果
├── docs/                      # ドキュメント
│   ├── COLAB_TRAINING_GUIDE.md
│   ├── COLAB_SINGLE_CELL_GUIDE.md
│   ├── PATH_CONFIGURATION_GUIDE.md
│   └── TEST_INFERENCE_VISUALIZATION.md
└── legacy/                    # 旧スクリプト・Colab用
    ├── train_colab.ipynb
    ├── train_colab_single.py
    └── create_data_zip.py
```

---

## ⚙️ 設定（config.py）

主要パラメータ:

```python
# モデル設定
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
ENCODER_NAME = 'resnet34'

# 学習設定
BATCH_SIZE = 4  # CPUモード用
MAX_EPOCHS = 100
LEARNING_RATE = 1e-4

# 推論設定
CONFIDENCE_THRESHOLD = 0.5
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 50000
```

---

## 🛠️ トラブルシューティング

### ✅ albumentationsは不要です！

**推論時にはalbumentationsは不要です。**

inference.pyは完全にalbumentations非依存で実装されています。
- 正規化: numpyで実装
- テンソル変換: torch.from_numpy()を使用

**注意:** 学習時（train.py）のみalbumentationsが必要ですが、Google Colabで学習する場合は問題ありません。

---

### ModuleNotFoundError: No module named 'pandas'など

**原因:** 基本パッケージがインストールされていません。

**解決方法:**

```powershell
cd ..
pip install pandas matplotlib scipy
```

または

```powershell
cd ..
pip install -r requirements.txt
```

**注意:** albumentationsの警告が出ても無視してOKです（推論では不要）。

---

### モデルファイルが見つからない

```
FileNotFoundError: モデルファイルが見つかりません
```

**解決方法:**
1. Google Colabで学習済みモデルを生成（`docs/COLAB_TRAINING_GUIDE.md`参照）
2. `models/best_unet.pth`に配置
3. または`--model`オプションでパスを指定

### その他のパッケージエラー

```powershell
# 親ディレクトリのrequirements.txtからインストール
cd ..
pip install -r requirements.txt

# または個別にインストール
pip install pandas matplotlib scipy
```

### GPU互換性

詳細: `../docs/GPU_COMPATIBILITY_ISSUE.md`

---

## 📚 詳細ドキュメント

- **Colab学習ガイド**: `docs/COLAB_TRAINING_GUIDE.md`
- **パス設定ガイド**: `docs/PATH_CONFIGURATION_GUIDE.md`
- **推論可視化テスト**: `docs/TEST_INFERENCE_VISUALIZATION.md`

---

## 次のステップ

1. ラベリングツールで訓練データを作成
2. Google Colabでモデルを学習
3. `run_inference_analysis.py`で画像を解析
4. 生成されたグラフとCSVで行動パターンを分析
    ├── training_history.png
    └── segmentation_video.avi
```

---

**詳細なドキュメントは親ディレクトリの `README.md` を参照してください。**
