# セマンティックセグメンテーション - Deep Learning手法

U-Netベースのディープラーニングによる高精度プラナリア検出システム

**詳細は親ディレクトリの `README.md` を参照してください。**

---

## クイックスタート

### 1. ラベリング（100枚以上推奨）
```powershell
python labeling_gui.py
```
- `data/images/` と `data/labels/` を選択
- ショートカット: N=次へ, P=前へ, S=保存, C=クリア, D=描画, E=消しゴム

### 2. 学習
```powershell
python train.py
```
- `models/best_unet.pth` に保存
- `outputs/training_history.png` で学習曲線を確認

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
