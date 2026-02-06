# GUIエラー「エラー None」の解決方法

## 改善した内容

### 1. エラーメッセージの詳細表示

**変更前:**
```python
except Exception as e:
    self.root.after(0, lambda: self._on_completion(False, f"エラー:\n{str(e)}"))
```

**変更後:**
```python
except Exception as e:
    import traceback
    error_msg = f"エラーが発生しました:\n\n{type(e).__name__}: {str(e)}\n\n詳細:\n{traceback.format_exc()}"
    print(error_msg)  # コンソールにも出力
    self.root.after(0, lambda: self._on_completion(False, error_msg))
```

### 2. 詳細なエラーウィンドウ

長いエラーメッセージはスクロール可能な別ウィンドウで表示されるようになりました。

### 3. 入力バリデーション強化

- 画像フォルダに画像ファイルがあるか確認
- モデルファイルの存在確認
- 出力フォルダの作成権限確認

---

## 使い方

### 1. GUIを起動

PyCharmのターミナルで：

```powershell
cd segmentation
python inference_analysis_gui.py
```

### 2. エラーが出た場合

**エラーメッセージを確認:**
- エラー詳細ウィンドウが表示されます
- PyCharmのコンソール（下部）にも詳細が出力されます

**コンソール出力を確認:**
- PyCharmの下部「Run」タブでエラーの詳細を確認できます
- トレースバックで問題の原因がわかります

---

## よくあるエラーと解決方法

### エラー: RuntimeError - state_dict loading error

**症状:**
```
RuntimeError: Error(s) in loading state_dict for UNetModel:
Missing key(s) in state_dict: "model.encoder.conv1.weight"...
Unexpected key(s) in state_dict: "encoder.conv1.weight"...
```

**原因:** モデルの保存形式と読み込み形式が一致していません

**解決済み:** `unet_model.py`の`load_model`関数を修正しました
- キー名の自動調整機能を追加
- `encoder.*` → `model.encoder.*` に変換
- `strict=False`で柔軟に読み込み

**対処:** 最新の`unet_model.py`を使用してください

---

### エラー: FileNotFoundError

**原因:** モデルファイルが見つかりません

**解決方法:**
1. Google Colabで学習済みモデルを生成
2. `segmentation/models/best_unet.pth`に配置
3. またはGUIで正しいモデルパスを指定

### エラー: No module named 'pandas'

**原因:** 必要なパッケージがインストールされていません

**解決方法:**
```powershell
pip install pandas matplotlib scipy
```

### エラー: No module named 'torch'

**原因:** PyTorchがインストールされていません

**解決方法:**
```powershell
pip install torch torchvision segmentation-models-pytorch
```

### エラー: 画像ファイルが見つかりません

**原因:** 画像フォルダに対応形式の画像がありません

**解決方法:**
- 対応形式: .png, .jpg, .jpeg, .bmp, .tif, .tiff
- 画像フォルダのパスを確認

---

## デバッグ方法

### 1. PyCharmのコンソール出力を確認

GUIを実行すると、PyCharmのコンソール（下部）に詳細なエラー情報が出力されます。

### 2. エラー詳細ウィンドウを確認

エラーが発生すると、スクロール可能なウィンドウにトレースバックが表示されます。

### 3. ターミナルで直接実行してテスト

```powershell
cd segmentation
python run_inference_analysis.py --images "画像フォルダ" --output "出力フォルダ"
```

コマンドラインで直接実行すると、より詳細なエラーメッセージが表示されます。

---

## 修正完了

エラーメッセージが「エラー None」ではなく、詳細な情報が表示されるようになりました。

**次回エラーが出たら:**
1. エラー詳細ウィンドウのメッセージを確認
2. PyCharmのコンソール出力を確認
3. エラーメッセージをコピーして対処方法を確認

これで原因が特定しやすくなりました！
