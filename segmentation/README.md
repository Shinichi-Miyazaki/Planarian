# セマンティックセグメンテーション - Deep Learning手法

U-Netベースのディープラーニングによる高精度プラナリア検出システム

**詳細は親ディレクトリの `README.md` を参照してください。**

---

## ⚠️ 重要: GPU互換性について（RTX 5070 Ti使用者向け）

RTX 5070 Ti (CUDA Compute Capability sm_120) は、現在のPyTorch 2.6/2.7では**未対応**です。

### 解決策: 2つの選択肢

#### ✅ オプション1: Google Colabで学習（推奨）
- **無料でGPU使用可能**（T4、V100など）
- **互換性問題なし**
- **詳細は `COLAB_TRAINING_GUIDE.md` を参照**

#### ⚠️ オプション2: ローカルでCPUモード
- 現在の設定では**CPUモード**で動作（遅いが動作します）
- PyTorch 2.8以降でsm_120サポート追加予定
- 詳細は `../install_pytorch_gpu.md` を参照

---

## クイックスタート

### 1. ラベリング（100枚以上推奨）
```powershell
python labeling_gui.py
```
- `data/images/` と `data/labels/` を選択
- ショートカット: N=次へ, P=前へ, S=保存, C=クリア, D=描画, E=消しゴム

### 2. 学習

#### 🌟 方法A: Google Colabで学習（推奨・GPU使用）

**オプションA-1: 1セル実行版（最も簡単）**
1. データをZIP圧縮: `python create_data_zip.py`
2. Google Colabで新規ノートブックを作成
3. `train_colab_single.py` の内容を1つのセルにコピー&ペースト
4. セルを実行（ZIPをアップロード）
5. 学習済みモデルをダウンロード
**詳細: `COLAB_SINGLE_CELL_GUIDE.md`**

**オプションA-2: ノートブック版**
1. データをZIP圧縮: `python create_data_zip.py`
2. `train_colab.ipynb` をGoogle Colabで開く
3. GPUランタイムを選択
4. すべてのセルを実行
5. 学習済みモデルをダウンロード
**詳細: `COLAB_TRAINING_GUIDE.md`**

#### 方法B: ローカルで学習（CPUモード）

```powershell
python train.py
```
- `models/best_unet.pth` に保存
- `outputs/training_history.png` で学習曲線を確認
- **注意**: CPUモードは非常に遅い（GPU比で10-50倍）

### 3. 推論
```powershell
python inference.py --images <入力> --output <出力>
```
- `outputs/analysis_results.csv`（behavior_analysis.py互換）
- `outputs/segmentation_video.avi`（動画）

---

## 設定ファイル

`config.py` でパラメータを調整：
- モデル設定（画像サイズ、エンコーダー）
- 学習設定（バッチサイズ、学習率、エポック数）
- 推論設定（信頼度閾値、面積フィルタ）

---

## ファイル構成

```
segmentation/
├── config.py          # 設定ファイル（すべてのパラメータ）
├── dataset.py         # データローダー
├── unet_model.py      # U-Netモデル定義
├── train.py           # 学習スクリプト
├── inference.py       # 推論スクリプト
├── labeling_gui.py    # ラベリングツール
├── utils.py           # ユーティリティ関数
├── data/
│   ├── images/        # 元画像
│   └── labels/        # ラベル（マスク）
├── models/
│   └── best_unet.pth  # ベストモデル
└── outputs/
    ├── analysis_results.csv
    ├── training_history.png
    └── segmentation_video.avi
```

---

**詳細なドキュメントは親ディレクトリの `README.md` を参照してください。**
