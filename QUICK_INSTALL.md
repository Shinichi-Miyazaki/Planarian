# 簡易インストールガイド

## 必要なパッケージ

推論（実行）には以下のパッケージが必要です：

```powershell
pip install opencv-python numpy pandas matplotlib scipy pillow torch torchvision segmentation-models-pytorch tqdm
```

## ✅ albumentationsは不要です！

**推論時にはalbumentationsは不要**です。ビルドエラーが出ても無視してOKです。

- inference.pyは完全にalbumentations非依存で実装されています
- 学習時のみalbumentationsが必要（Google Colabで学習推奨）

## インストール手順

### PyCharmを使用している場合

PyCharmのターミナル（下部）で：

```powershell
# プロジェクトルートで実行
pip install opencv-python numpy pandas matplotlib scipy pillow
pip install torch torchvision segmentation-models-pytorch tqdm
```

または

```powershell
pip install -r requirements.txt
```

**注意:** albumentationsの警告が出ても無視してください。

### 確認

```powershell
cd segmentation
python run_inference_analysis.py --help
```

ヘルプが表示されれば成功です！

## よくあるエラー

### エラー: No module named 'pandas'

```powershell
pip install pandas matplotlib scipy
```

### エラー: No module named 'torch'

```powershell
pip install torch torchvision
```

### エラー: albumentationsのビルドエラー

→ **無視してOK**です。推論では使用しません。

## 実行方法

### GUIで実行

```powershell
cd segmentation
python inference_analysis_gui.py
```

### コマンドラインで実行

```powershell
cd segmentation
python run_inference_analysis.py --images <画像フォルダ> --output <出力フォルダ>
```

---

**これで完了です！** albumentationsなしで動作します。
