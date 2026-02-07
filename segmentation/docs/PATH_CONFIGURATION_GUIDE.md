# ✅ パス設定の改善完了！

## 📋 変更内容

Google Colab用トレーニングスクリプトで、**ベースディレクトリを冒頭で指定**できるようになりました。

---

## 🎯 主な改善点

### 1. ベースディレクトリの一元管理

**冒頭で1箇所設定するだけ**で、すべてのパスが自動生成されます：

```python
# ============================================================================
# パス設定
# ============================================================================

# ベースディレクトリ（すべてのデータとモデルの保存場所）
BASE_DIR = '/content/planarian'  # ← ここを変更するだけ！

# データディレクトリ（ベースディレクトリからの相対パス）
DATA_DIR_NAME = 'data'           # データフォルダ名
MODELS_DIR_NAME = 'models'       # モデル保存フォルダ名
OUTPUTS_DIR_NAME = 'outputs'     # 出力フォルダ名

# 自動生成されるパス（通常は変更不要）
DATA_DIR = f'{BASE_DIR}/{DATA_DIR_NAME}'      # /content/planarian/data
MODELS_DIR = f'{BASE_DIR}/{MODELS_DIR_NAME}'  # /content/planarian/models
OUTPUTS_DIR = f'{BASE_DIR}/{OUTPUTS_DIR_NAME}'  # /content/planarian/outputs
```

### 2. 実行時のパス表示

トレーニング開始時に、使用されるパスがわかりやすく表示されます：

```
[3/6] データを準備中...

📁 ベースディレクトリ: /content/planarian
   ├─ データ: /content/planarian/data
   ├─ モデル: /content/planarian/models
   └─ 出力: /content/planarian/outputs

'data.zip' をアップロードしてください...
```

### 3. ZIPファイルの構造改善

`create_data_zip.py` で作成されるZIPの構造が最適化され、Colab側で正しく展開されます：

```
data.zip の構造:
├─ images/
│  ├─ 00012.jpg
│  ├─ 00024.jpg
│  └─ ...
└─ labels/
   ├─ 00012.png
   ├─ 00024.png
   └─ ...
```

Colab側で解凍すると:
```
BASE_DIR/
└─ data/
   ├─ images/
   └─ labels/
```

---

## 💡 使用例

### 例1: デフォルト設定（一時的な作業）

```python
BASE_DIR = '/content/planarian'
```

**結果:**
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

**注意:** Colabセッション終了後、ファイルは削除されます

---

### 例2: Google Driveに保存（永続化・推奨）

```python
BASE_DIR = '/content/drive/MyDrive/planarian_project'
```

**メリット:**
- ✅ セッション終了後もファイルが残る
- ✅ 複数回のトレーニングで結果を比較可能
- ✅ モデルを安全に保管

**注意:** Google Driveをマウントする必要があります（スクリプトが自動で対応）

---

### 例3: カスタムフォルダ構成

```python
BASE_DIR = '/content/my_experiment'
DATA_DIR_NAME = 'training_data'
MODELS_DIR_NAME = 'saved_models'
OUTPUTS_DIR_NAME = 'results'
```

**結果:**
```
/content/my_experiment/
├─ training_data/
│  ├─ images/
│  └─ labels/
├─ saved_models/
│  └─ best_unet.pth
└─ results/
   └─ training_history.png
```

---

## 📂 ディレクトリ構造の全体像

### ローカル環境

```
Planarian/
└─ segmentation/
   ├─ data/
   │  ├─ images/          ← ラベリングした画像
   │  └─ labels/          ← ラベリングしたマスク
   ├─ models/             ← Colabからダウンロードしたモデル
   ├─ create_data_zip.py  ← ZIP作成スクリプト
   ├─ data.zip            ← 作成されたZIP
   ├─ train_colab_single.py  ← Colab用スクリプト
   └─ inference.py        ← 推論実行
```

### Google Colab環境

```
/content/planarian/          ← BASE_DIR
├─ data/                     ← DATA_DIR (ZIPから解凍)
│  ├─ images/
│  │  ├─ 00012.jpg
│  │  └─ ...
│  └─ labels/
│     ├─ 00012.png
│     └─ ...
├─ models/                   ← MODELS_DIR (学習中に作成)
│  └─ best_unet.pth
└─ outputs/                  ← OUTPUTS_DIR (学習中に作成)
   └─ training_history.png
```

---

## 🔄 ワークフロー

### ステップ1: ローカルでデータ準備

```powershell
cd C:\Users\Miya\PycharmProjects\Planarian\segmentation
python create_data_zip.py
```

→ `data.zip` が作成されます（構造: images/, labels/）

### ステップ2: Google Colabでトレーニング

1. `train_colab_single.py` をコピー
2. 冒頭で `BASE_DIR` を設定:
   ```python
   BASE_DIR = '/content/drive/MyDrive/planarian_project'  # 永続化する場合
   ```
3. セルを実行
4. `data.zip` をアップロード
5. トレーニング完了後、`best_unet.pth` をダウンロード

### ステップ3: ローカルで推論

```powershell
# best_unet.pth を models/ に配置
python inference.py --images <入力> --output <出力>
```

---

## 🎨 パス設定のベストプラクティス

### 推奨: Google Driveを活用

```python
# 永続化 + 整理しやすい構造
BASE_DIR = '/content/drive/MyDrive/projects/planarian/experiment_001'
```

**メリット:**
- プロジェクトごとにフォルダ分け
- 実験ごとに結果を保存
- セッション終了後も保持

### プロジェクトの整理例

```
Google Drive/
└─ MyDrive/
   └─ projects/
      └─ planarian/
         ├─ experiment_001/      ← 最初の実験
         │  ├─ models/
         │  └─ outputs/
         ├─ experiment_002/      ← パラメータ変更版
         │  ├─ models/
         │  └─ outputs/
         └─ experiment_003/      ← データ追加版
            ├─ models/
            └─ outputs/
```

---

## 📚 更新されたファイル

1. **`train_colab_single.py`**
   - ベースディレクトリ設定を冒頭に追加
   - パス表示を視覚化
   - ZIP解凍先を BASE_DIR/data に変更

2. **`create_data_zip.py`**
   - ZIP構造を最適化（images/, labels/ のみ）
   - 構造の説明を追加
   - 使用方法のガイドを追加

3. **`COLAB_SINGLE_CELL_GUIDE.md`**
   - パス設定の詳細説明を追加
   - 実行例を追加
   - ベストプラクティスを追加

---

## ✨ まとめ

**パス設定が簡単になりました！**

- ✅ **1箇所で設定**: `BASE_DIR` を変更するだけ
- ✅ **視覚的**: 実行時にパス構造が表示される
- ✅ **柔軟**: ローカル/Google Drive/カスタムパスに対応
- ✅ **整理しやすい**: プロジェクト/実験ごとにフォルダ分け可能

これで、Google Colabでのトレーニングがさらに快適になりました！🚀
