# Google Colab 1セル実行版 - 使い方

## 📋 概要

`train_colab_single.py` は、Google Colabで**1つのセルで完結**するトレーニングスクリプトです。
複数のセルに分かれたノートブックではなく、全コードを1つのセルにコピー&ペーストするだけで実行できます。

---

## 🚀 使い方

### ステップ1: データを準備

**方法A: ZIPファイル（推奨）**
```powershell
cd C:\Users\Miya\PycharmProjects\Planarian\segmentation
python create_data_zip.py
```
→ `data.zip` が作成されます

**方法B: Google Driveにアップロード**
- `segmentation/data/` フォルダをGoogle Driveにアップロード

---

### ステップ2: Google Colabで実行

1. **Google Colabを開く**
   - https://colab.research.google.com/ にアクセス
   - **ファイル > 新しいノートブック**

2. **GPUランタイムを設定**
   - **ランタイム > ランタイムのタイプを変更**
   - **ハードウェアアクセラレータ: GPU**
   - **GPU タイプ: T4** を選択
   - **保存**

3. **コードをコピー**
   - `train_colab_single.py` を開く
   - **すべてのコードをコピー** (Ctrl+A → Ctrl+C)

4. **Colabのセルにペースト**
   - 新規セルにコードをペースト (Ctrl+V)

5. **設定を編集（必要に応じて）**
   ```python
   # 🔧 設定: ここを編集してください
   
   # データパスの設定
   USE_ZIP = True  # True: ZIP使用, False: Google Drive使用
   ZIP_FILENAME = 'data.zip'
   
   # 学習設定
   MAX_EPOCHS = 100
   BATCH_SIZE = 8  # メモリ不足の場合は4に減らす
   ```

6. **セルを実行** (Shift+Enter)
   - ZIPファイルをアップロードするプロンプトが表示されたら `data.zip` を選択
   - トレーニングが開始されます（30分～2時間）

7. **完了後、モデルをダウンロード**
   - `best_unet.pth` と `training_history.png` が自動でダウンロードされます

---

### ステップ3: ローカルで推論

1. ダウンロードした `best_unet.pth` を `segmentation/models/` に配置

2. 推論を実行:
```powershell
cd C:\Users\Miya\PycharmProjects\Planarian\segmentation
python inference.py --images <入力動画> --output <出力フォルダ>
```

---

## ⚙️ 設定のカスタマイズ

スクリプトの冒頭にある設定セクションを編集:

```python
# ============================================================================
# 🔧 設定: ここを編集してください
# ============================================================================

# ============================================================================
# パス設定
# ============================================================================

# ベースディレクトリ（すべてのデータとモデルの保存場所）
BASE_DIR = '/content/planarian'  # ここを変更すれば全体の保存先が変わります

# データディレクトリ（ベースディレクトリからの相対パス）
DATA_DIR_NAME = 'data'           # データフォルダ名
MODELS_DIR_NAME = 'models'       # モデル保存フォルダ名
OUTPUTS_DIR_NAME = 'outputs'     # 出力フォルダ名

# 自動生成されるパス（通常は変更不要）
# DATA_DIR = BASE_DIR/data
# MODELS_DIR = BASE_DIR/models
# OUTPUTS_DIR = BASE_DIR/outputs

# ============================================================================
# データソース
# ============================================================================

# データパス
USE_ZIP = True  # ZIP使用の場合
# または
USE_ZIP = False  # Google Drive使用の場合
GOOGLE_DRIVE_IMAGES_DIR = '/content/drive/MyDrive/...'
GOOGLE_DRIVE_LABELS_DIR = '/content/drive/MyDrive/...'

# 学習パラメータ
MAX_EPOCHS = 100           # エポック数（Early Stoppingで早期終了あり）
BATCH_SIZE = 8             # バッチサイズ（メモリ不足なら4に減らす）
LEARNING_RATE = 1e-4       # 学習率
EARLY_STOPPING_PATIENCE = 15  # Early Stopping
IMAGE_SIZE = 512           # 画像サイズ
```

### パス設定の例

#### 例1: デフォルト（/content/planarian）
```python
BASE_DIR = '/content/planarian'
```
結果:
```
/content/planarian/
├─ data/
│  ├─ images/
│  └─ labels/
├─ models/
│  └─ best_unet.pth
└─ outputs/
   └─ training_history.png
```

#### 例2: Google Driveに保存（永続化）
```python
BASE_DIR = '/content/drive/MyDrive/planarian_project'
```
→ セッション終了後もファイルが残ります（推奨）

#### 例3: カスタムパス
```python
BASE_DIR = '/content/my_work'
DATA_DIR_NAME = 'training_data'
MODELS_DIR_NAME = 'saved_models'
```
結果:
```
/content/my_work/
├─ training_data/
├─ saved_models/
└─ outputs/
```

---

## 📊 実行の流れ

```
[1/6] ライブラリをインストール中...
  ✓ segmentation-models-pytorch
  ✓ albumentations

[2/6] 環境確認中...
  ✓ GPU: Tesla T4
  ✓ VRAM: 15.0 GB

[3/6] データを準備中...

  📁 ベースディレクトリ: /content/planarian
     ├─ データ: /content/planarian/data
     ├─ モデル: /content/planarian/models
     └─ 出力: /content/planarian/outputs

  'data.zip' をアップロードしてください...
  ↓ ファイル選択
  
  data.zip を解凍中...
  ✓ 解凍完了: /content/planarian/data
  
  ✓ データディレクトリを確認:
    画像フォルダ: /content/planarian/data/images
    ラベルフォルダ: /content/planarian/data/labels
    画像数: 33 枚
    ラベル数: 33 枚

[4/6] データローダーを作成中...
  ✓ 学習データ: 26 サンプル
  ✓ 検証データ: 7 サンプル

[5/6] モデルを構築中...
  ✓ U-Net (ResNet34)
  ✓ パラメータ数: 24,436,369

[6/6] トレーニング開始
  Epoch 1/100
  Training: 100%|██████████| 4/4 [00:05<00:00]
  Validation: 100%|██████████| 1/1 [00:01<00:00]
  
  結果:
    Train Loss: 0.3524 | Train Dice: 0.7821
    Val Loss:   0.3102 | Val Dice:   0.8145
    ✓ ベストモデルを保存しました
  
  ...
  
  ✓ トレーニング完了！
  
  学習曲線を作成中...
  ✓ 学習曲線を保存
  [学習曲線グラフ表示]
  
  テスト推論（セグメンテーション結果の確認）
  テスト画像: 00012.jpg
  ✓ テスト推論結果を保存
  [推論結果表示: 元画像・正解ラベル・予測マスク・重ね合わせ]
  
  推論結果の統計:
    - 検出面積、IoUなど
  
  モデルと学習履歴をダウンロード
  ✓ best_unet.pth
  ✓ training_history.png
  ✓ test_inference_result.png
```

---

## 🔍 トラブルシューティング

### ⚠️ メモリ不足（Out of Memory）

設定を変更:
```python
BATCH_SIZE = 4  # 8 → 4 に減らす
```

### ⚠️ データが見つからない

パスを確認:
```python
# ZIP解凍後のパス
IMAGES_DIR = '/content/data/images'  # data/ の構造を確認
LABELS_DIR = '/content/data/labels'

# または Google Drive のパス
GOOGLE_DRIVE_IMAGES_DIR = '/content/drive/MyDrive/...'  # 正しいパスに変更
```

### ⚠️ セッションタイムアウト

- Google Colab無料版: 最大12時間
- 通常、Early Stoppingにより1-2時間で完了します
- 長時間かかる場合は `MAX_EPOCHS` を減らす

---

## 💡 Tips

### データ確認

スクリプト実行中に表示されるデータ数を確認:
```
✓ 画像: 33 枚
✓ ラベル: 33 枚
```

数が合わない場合は、データパスやZIPの構造を確認してください。

### 学習の中断

セルを停止 (■ボタン) すると、その時点までの学習が中断されます。
最後に保存されたベストモデルは残ります。

### 再実行

同じセルを再度実行すると、新しいトレーニングが開始されます。
前回のモデルは上書きされます。

---

## 📚 参考

- 詳細な手順: `COLAB_TRAINING_GUIDE.md`
- GPU互換性: `../GPU_COMPATIBILITY_ISSUE.md`
- 従来のノートブック版: `train_colab.ipynb`

---

## ✨ まとめ

**1セル実行版の利点:**
- ✅ 簡単: コピー&ペーストだけ
- ✅ 高速: セル間の移動不要
- ✅ 完全: すべてのコードが1箇所に
- ✅ 確実: セルの実行順序を気にしない

Google Colabでの快適なトレーニングをお楽しみください！ 🚀
