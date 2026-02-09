# セグメンテーションガイド

U-Netによる高精度プラナリア検出システムの使い方

---

## 📝 3ステップで使える

### 1. ラベリング（学習データ作成）

```powershell
python labeling_gui.py
```

- 画像フォルダと保存先を選択
- マウスでプラナリアをなぞる
- `S`キーで保存、`N`キーで次へ
- **100枚以上推奨**

---

### 2. モデル学習

#### Google Colab（推奨・無料GPU）

```powershell
# 1. データをZIP化
python legacy\create_data_zip.py

# 2. Google Colabで開く
# → legacy/train_colab.ipynb
# → GPUランタイムを選択して実行

# 3. best_unet.pth をダウンロード
# → models/ に配置
```

詳細: `docs/COLAB_TRAINING_GUIDE.md`

#### ローカル学習（CPU・遅い）

```powershell
python legacy\train.py
```

---

### 3. 推論+解析（メイン機能）

#### GUI版（推奨）

```powershell
python inference_analysis_gui.py
```

**2つのモード:**

1. **単一フォルダ解析**: 1つのフォルダを指定して解析
2. **バッチフォルダ解析**: 親フォルダ配下を走査して複数フォルダを一括解析

**バッチモードの使い方:**
1. 「バッチフォルダ解析」タブを選択
2. 親フォルダを選択（画像フォルダを含む上位フォルダ）
3. 画像数閾値を設定（デフォルト: 1000枚以上）
4. 「フォルダを検索」をクリック
5. 検出されたフォルダごとに測定日時を編集（ダブルクリック）
6. 「バッチ実行」で一括解析開始
7. 各フォルダ内に`segmentation_analysis`フォルダが作成され、結果が保存される

#### コマンドライン版

```powershell
python run_inference_analysis.py --images <画像フォルダ> --output <出力フォルダ>
```


**出力:**
- セグメンテーション結果CSV
- 行動解析データ（移動量・不動性）
- 時系列グラフ、統計レポート
- 動画（オプション）

---

## 🎮 ONNX Runtime（GPU推論・RTX 5070 Ti対応）

PyTorchが未対応のGPUでも高速推論が可能。

### 使い方

```powershell
# 1. パッケージインストール
pip install onnx onnxruntime-directml

# 2. モデル変換
python export_onnx.py --model models\best_unet.pth

# 3. GPU推論
python inference_onnx.py --images <画像フォルダ> --output <出力フォルダ> --directml
```

**効果:** CPU比で10-30倍高速化

詳細: `ONNX_RUNTIME_GUIDE.md`

---

## ⚙️ 設定

### config.py

```python
# 画像サイズ
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# 推論設定
CONFIDENCE_THRESHOLD = 0.5
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 50000

# デバイス
DEVICE = 'cpu'  # PyTorch使用時
```

---

## 🛠️ トラブルシューティング

### モデルファイルが無い

**解決:** Google Colabで学習して`models/best_unet.pth`に配置

### albumentationsエラー

**解決:** 推論では不要。無視してOK

### GPU使えない

**解決:** ONNX Runtime + DirectMLを使用（上記参照）

---

## 📚 詳細ドキュメント

### 基本機能
- **このREADME**: 基本的な使い方（ラベリング・学習・推論）

### 高度な機能
- **[バッチフォルダ解析ガイド](docs/BATCH_ANALYSIS_GUIDE.md)**: 複数フォルダの一括解析方法
- **[ONNX Runtime GPU推論ガイド](ONNX_RUNTIME_GUIDE.md)**: GPU高速化の方法

### 学習関連
- **[Google Colab学習ガイド](docs/COLAB_TRAINING_GUIDE.md)**: 無料GPUでモデル学習
- **[Google Colab単一セル版](docs/COLAB_SINGLE_CELL_GUIDE.md)**: 1セルで完結する学習

### トラブルシューティング
- **[GPU互換性問題](../docs/GPU_COMPATIBILITY_ISSUE.md)**: RTX 5070 Ti等の新GPU対応
- **[PyTorch GPUインストール](../docs/install_pytorch_gpu.md)**: GPU版PyTorchの導入

---

## 📁 ファイル構成

```
segmentation/
├── labeling_gui.py            # ★ ラベリング
├── inference_analysis_gui.py  # ★ 推論+解析GUI（バッチ対応）
├── export_onnx.py             # ★ モデル変換
├── inference_onnx.py          # ★ ONNX推論（GPU対応）
├── inference.py               # PyTorch推論
├── run_inference_analysis.py  # CLI版
├── unet_model.py
├── config.py
├── utils.py
├── data/
│   ├── images/
│   └── labels/
├── docs/
│   ├── BATCH_ANALYSIS_GUIDE.md       # バッチ解析ガイド（NEW）
│   ├── IMPLEMENTATION_SUMMARY.md     # 実装サマリー（NEW）
│   ├── COLAB_TRAINING_GUIDE.md
│   └── ...
├── legacy/
│   ├── train.py
│   └── ...
└── models/
    └── best_unet.pth          # 学習済みモデル
```
├── models/
│   └── best_unet.pth          # 学習済みモデル
└── legacy/
    ├── train.py               # ローカル学習
    ├── train_colab.ipynb      # Colab学習
    └── create_data_zip.py
```

**★ = よく使うファイル**

---

## 📚 詳細ドキュメント

- **Colab学習**: `docs/COLAB_TRAINING_GUIDE.md`
- **ONNX Runtime**: `ONNX_RUNTIME_GUIDE.md`
- **パス設定**: `docs/PATH_CONFIGURATION_GUIDE.md`
