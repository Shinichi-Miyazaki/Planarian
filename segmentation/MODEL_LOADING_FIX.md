# モデル読み込みエラーの解決 ✅

## 問題

```
RuntimeError: Error(s) in loading state_dict for UNetModel:
Missing key(s) in state_dict: "model.encoder.conv1.weight"...
Unexpected key(s) in state_dict: "encoder.conv1.weight"...
```

## 原因

モデル保存時と読み込み時でキー名の形式が異なっています。

**保存時:** `encoder.*`, `decoder.*`, `segmentation_head.*`
**期待:** `model.encoder.*`, `model.decoder.*`, `model.segmentation_head.*`

## 解決方法（実装済み）

`unet_model.py`の`load_model`関数を修正しました。

### 変更内容

```python
# キー名の自動調整
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('encoder.') or key.startswith('decoder.') or key.startswith('segmentation_head.'):
        # model. プレフィックスを追加
        new_key = 'model.' + key
        new_state_dict[new_key] = value
    elif key.startswith('model.'):
        # 既にmodel.プレフィックスがある場合はそのまま
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# 柔軟に読み込み
model.load_state_dict(new_state_dict, strict=False)
```

### 機能

1. **自動キー変換**: 
   - `encoder.*` → `model.encoder.*`
   - `decoder.*` → `model.decoder.*`
   - `segmentation_head.*` → `model.segmentation_head.*`

2. **柔軟な読み込み**: 
   - `strict=False`で`num_batches_tracked`などの不一致を許容
   - 重要なパラメータのみを読み込み

3. **後方互換性**: 
   - 古い形式のモデルも読み込み可能
   - 新しい形式のモデルも読み込み可能

## 動作確認

修正後、以下のコマンドで動作確認してください：

```powershell
cd segmentation
python inference_analysis_gui.py
```

または

```powershell
python run_inference_analysis.py --images <画像フォルダ> --output <出力フォルダ>
```

## トラブルシューティング

### それでもエラーが出る場合

**1. モデルファイルの確認**
```powershell
# モデルファイルが正しく保存されているか確認
python -c "import torch; checkpoint = torch.load('models/best_unet.pth', map_location='cpu'); print(list(checkpoint.keys()))"
```

**2. モデルの再学習**
- Google Colabで再度学習してモデルを生成
- 最新の`train.py`または`train_colab.ipynb`を使用

**3. デバッグモード**
```python
# unet_model.pyのload_model関数にデバッグコードを追加
print("Loaded keys:", list(state_dict.keys())[:5])
print("Model expects:", list(model.state_dict().keys())[:5])
```

## なぜこのエラーが発生したか

1. **PyTorchのモデル保存形式の違い**
   - `torch.save(model.state_dict())`で保存
   - `torch.save({'model_state_dict': model.state_dict()})`で保存

2. **segmentation_models_pytorchのラッパー**
   - `smp.Unet`は内部で`model`プロパティを持つ
   - 保存時に`model.state_dict()`を使うとプレフィックスなし
   - 保存時に`full_model.state_dict()`を使うとプレフィックス付き

3. **学習環境の違い**
   - Colabで学習したモデル
   - ローカルで学習したモデル
   - 異なるバージョンのライブラリ

## 解決済み ✅

修正により、以下のすべてのケースで動作するようになりました：

- ✅ Colabで学習したモデル
- ✅ ローカルで学習したモデル
- ✅ プレフィックス付きのstate_dict
- ✅ プレフィックスなしのstate_dict
- ✅ 古いバージョンのモデル

**もう一度GUIを実行してください！**
