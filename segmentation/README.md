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

```powershell
# GUI版（推奨）
python inference_analysis_gui.py

# コマンドライン版
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

## 📁 ファイル構成

```
segmentation/
├── labeling_gui.py            # ★ ラベリング
├── inference_analysis_gui.py  # ★ 推論+解析GUI
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
