# Google Colabでトレーニングする手順

## なぜGoogle Colabを使うのか？

✅ **無料でGPU使用可能**（T4、V100など）
✅ **RTX 5070 Ti (sm_120) の互換性問題を回避**
✅ **PyTorchが最初から設定済み**
✅ **ローカルPCのリソースを消費しない**

---

## 手順

### 1. データの準備

#### 方法A: ZIPファイルを作成（簡単）

PowerShellで実行:
```powershell
cd C:\Users\Miya\PycharmProjects\Planarian\segmentation
Compress-Archive -Path data\* -DestinationPath data.zip -Force
```

または、Pythonスクリプトで:
```powershell
python create_data_zip.py
```

これで `segmentation/data.zip` が作成されます。

#### 方法B: Google Driveを使用（大量データ向け）

1. `segmentation/data/` フォルダをGoogle Driveにアップロード
2. Google Driveのパスをメモ（例: `MyDrive/Planarian/segmentation/data/`）

---

### 2. Google Colabでノートブックを開く

1. https://colab.research.google.com/ にアクセス
2. **ファイル > ノートブックをアップロード**
3. `segmentation/train_colab.ipynb` をアップロード

または：

1. Google Driveに `train_colab.ipynb` をアップロード
2. Google Driveから右クリック > **アプリで開く > Google Colaboratory**

---

### 3. GPUランタイムを設定

1. **ランタイム > ランタイムのタイプを変更**
2. **ハードウェアアクセラレータ: GPU** を選択
3. **GPU タイプ: T4** を選択（無料プランで利用可能）
4. **保存**

---

### 4. データをアップロード

#### 方法A: ZIPファイル（推奨・小規模データセット向け）

ノートブックの「2. データアップロード」セクションで、**方法2: ZIPファイルアップロード**のセルを実行:

```python
from google.colab import files
import zipfile

uploaded = files.upload()  # data.zip を選択
```

#### 方法B: Google Drive（大規模データセット向け）

ノートブックの「2. データアップロード」セクションで、**方法1: Google Drive マウント**のセルを実行:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Googleアカウントの認証を求められたら許可してください。

---

### 5. トレーニング実行

1. **ランタイム > すべてのセルを実行**、または
2. 各セルを上から順番に実行（Shift+Enter）

**所要時間の目安:**
- 33枚の画像、100エポック: 約30-60分（T4 GPU）
- 100枚の画像、100エポック: 約1-2時間（T4 GPU）

---

### 6. モデルのダウンロード

トレーニング完了後、最後のセルを実行:

```python
from google.colab import files
files.download('/content/models/best_unet.pth')
files.download('/content/outputs/training_history.png')
```

ブラウザに以下がダウンロードされます:
- `best_unet.pth` - 学習済みモデル
- `training_history.png` - 学習曲線

---

### 7. ローカルで推論を実行

1. ダウンロードした `best_unet.pth` を `segmentation/models/` に配置
2. PowerShellで実行:

```powershell
cd C:\Users\Miya\PycharmProjects\Planarian\segmentation
python inference.py --images <入力動画またはフォルダ> --output <出力フォルダ>
```

例:
```powershell
python inference.py --images "C:\Videos\planarian_test.mp4" --output "C:\Output"
```

---

## トラブルシューティング

### ⚠️ セッションがタイムアウトする

Google Colabの無料版は最大12時間までセッションが続きます。
長時間のトレーニングの場合:
- Early Stoppingが有効なので、通常は数十エポックで終了します
- `config.MAX_EPOCHS` を減らす（例: 50エポック）

### ⚠️ GPUメモリ不足（Out of Memory）

ノートブックの「3. 設定ファイル」セクションで、`BATCH_SIZE` を減らす:

```python
BATCH_SIZE = 4  # デフォルト: 8
```

### ⚠️ データがアップロードできない

データサイズが大きい場合は、Google Drive方式を使用してください。

### ⚠️ ダウンロードが始まらない

ブラウザのポップアップブロックを無効にしてください。

---

## Google Colab Pro（有料版）の検討

より高速なトレーニングが必要な場合:
- **Google Colab Pro**: 月額1,179円
  - より高速なGPU（V100、A100）
  - より長いセッション時間
  - バックグラウンド実行

---

## CPUモードとの比較

| 項目 | CPUモード (ローカル) | GPUモード (Google Colab) |
|------|---------------------|------------------------|
| 速度 | 1x（非常に遅い） | 10-50x（高速） |
| コスト | 無料 | 無料（Pro版は有料） |
| セットアップ | 簡単 | やや複雑 |
| データ転送 | 不要 | 必要 |
| 推論 | ローカルで実行 | ローカルで実行 |

---

## まとめ

**推奨ワークフロー:**

1. **ラベリング**: ローカルで実行（`labeling_gui.py`）
2. **トレーニング**: Google Colabで実行（GPU使用）
3. **推論**: ローカルで実行（`inference.py`）

このハイブリッド方式により、RTX 5070 Tiの互換性問題を回避しつつ、効率的にモデルを学習できます！

---

## 参考リンク

- Google Colab公式: https://colab.research.google.com/
- Google Colab使い方: https://colab.research.google.com/notebooks/intro.ipynb
