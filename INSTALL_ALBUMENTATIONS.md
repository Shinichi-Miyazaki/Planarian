# albumentationsインストールガイド

## エラー内容

```
ModuleNotFoundError: No module named 'albumentations'
```

このエラーは、画像データ拡張ライブラリ`albumentations`がインストールされていない場合に発生します。

---

## 解決方法

### 方法1: no-build-isolationオプション付きでインストール（推奨）

```powershell
pip install albumentations --no-build-isolation
```

このオプションにより、ビルド時の依存関係エラーを回避できます。

---

### 方法2: プリビルド版を優先してインストール

```powershell
pip install --prefer-binary albumentations
```

---

### 方法3: requirements.txtから一括インストール

```powershell
# プロジェクトルートで実行
pip install -r requirements.txt
```

---

### 方法4: Visual C++ Build Toolsをインストール

ビルドエラーが出る場合の根本的な解決策：

1. [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)をダウンロード
2. インストール時に「C++によるデスクトップ開発」を選択
3. インストール後、以下を実行：

```powershell
pip install albumentations
```

---

## インストール確認

```powershell
python -c "import albumentations; print('albumentations version:', albumentations.__version__)"
```

成功すると、バージョン番号が表示されます（例: `albumentations version: 2.0.8`）

---

## 依存パッケージ

albumentationsは以下のパッケージに依存します：
- numpy
- opencv-python または opencv-python-headless
- scipy
- PyYAML
- albucore
- pydantic

これらも自動的にインストールされますが、エラーが出る場合は個別にインストールしてください：

```powershell
pip install numpy opencv-python scipy pyyaml
```

---

## トラブルシューティング

### stringzillaのビルドエラー

```
error: Microsoft Visual C++ 14.0 or greater is required
```

**解決方法:**
- Visual C++ Build Toolsをインストール（方法4参照）
- または`--no-build-isolation`オプションを使用（方法1）

### その他のエラー

```powershell
# 最新版のpipにアップグレード
python -m pip install --upgrade pip

# キャッシュをクリアして再インストール
pip cache purge
pip install albumentations --no-cache-dir --no-build-isolation
```

---

## 実行確認

インストール後、統合スクリプトを実行してみてください：

```powershell
cd segmentation
python run_inference_analysis.py --help
```

エラーが出なければ成功です！
