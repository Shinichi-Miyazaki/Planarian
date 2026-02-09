# ✅ テスト推論結果の可視化機能を追加しました！

## 📋 新機能の概要

Google Colab トレーニングスクリプトに、**テスト画像での推論結果を自動表示する機能**を追加しました。

---

## 🎯 機能の詳細

### 1. トレーニング完了後に自動実行

学習が完了すると、自動的に以下が実行されます：

1. **テスト画像を1枚選択**（images/ フォルダの最初の画像）
2. **推論を実行**（学習済みモデルで予測）
3. **結果を可視化**（4種類の画像を表示）
4. **統計情報を表示**（検出面積、IoUなど）
5. **結果画像を保存**（test_inference_result.png）

### 2. 可視化される内容

#### パターンA: ラベルがある場合（4画像表示）

```
┌─────────────────┬─────────────────┐
│ Original Image  │ Ground Truth    │
│ (元画像)         │ (正解ラベル)     │
├─────────────────┼─────────────────┤
│ Predicted Mask  │ Overlay         │
│ (予測マスク)     │ (重ね合わせ)     │
└─────────────────┴─────────────────┘
```

- **Original Image**: 元の入力画像
- **Ground Truth Label**: 正解のラベル（手動でラベリングしたもの）
- **Predicted Mask**: モデルが予測したマスク
- **Overlay**: 元画像に予測マスクを緑色で重ね合わせ

#### パターンB: ラベルがない場合（3画像表示）

```
┌─────────────────┬─────────────────┬─────────────────┐
│ Original Image  │ Predicted Mask  │ Overlay         │
└─────────────────┴─────────────────┴─────────────────┘
```

### 3. 統計情報の表示

推論結果の詳細な統計が表示されます：

```
推論結果の統計:
  - 画像サイズ: 640 x 480
  - 検出面積: 12543 ピクセル
  - 検出割合: 4.08%
  - 正解面積: 12891 ピクセル        ← ラベルがある場合のみ
  - 正解割合: 4.19%                 ← ラベルがある場合のみ
  - IoU (Intersection over Union): 0.8621  ← ラベルがある場合のみ
```

**IoU (Intersection over Union)**:
- 0.0 ～ 1.0 の値
- 1.0 に近いほど正解ラベルとの一致度が高い
- 一般的に 0.5 以上で良好、0.7 以上で優秀

---

## 📊 実行例

### トレーニング完了後の出力

```
============================================================
  トレーニング完了！
============================================================
Best Validation Loss: 0.2847

学習曲線を作成中...
✓ 学習曲線を保存: /content/planarian/outputs/training_history.png

============================================================
  テスト推論（セグメンテーション結果の確認）
============================================================

テスト画像: 00012.jpg

✓ テスト推論結果を保存: /content/planarian/outputs/test_inference_result.png

[4枚の画像が表示される]
- Original Image (元画像)
- Ground Truth Label (正解ラベル)
- Predicted Mask (予測マスク - 白黒)
- Overlay (重ね合わせ - 緑色で予測部分を表示)

推論結果の統計:
  - 画像サイズ: 640 x 480
  - 検出面積: 12543 ピクセル
  - 検出割合: 4.08%
  - 正解面積: 12891 ピクセル
  - 正解割合: 4.19%
  - IoU (Intersection over Union): 0.8621

============================================================
  モデルと学習履歴をダウンロード
============================================================

ダウンロード中...
✓ ダウンロード完了！
  - best_unet.pth
  - training_history.png
  - test_inference_result.png  ← 追加！

📊 ダウンロードしたファイル:
  ✓ best_unet.pth - 学習済みモデル
  ✓ training_history.png - 学習曲線（Loss & Dice）
  ✓ test_inference_result.png - テスト推論結果（セグメンテーション確認）

次のステップ:
  1. ダウンロードした best_unet.pth をローカルの segmentation/models/ に配置
  2. test_inference_result.png でモデルの性能を確認  ← 追加！
  3. ローカルで推論を実行:
     cd segmentation
     python inference.py --images <入力> --output <出力>
```

---

## 🎨 可視化の詳細

### 重ね合わせ画像（Overlay）の見方

- **緑色の部分**: モデルが「プラナリア」と判定した領域
- **元の色の部分**: モデルが「背景」と判定した領域

**良い結果の例:**
- プラナリアの部分がきれいに緑色になっている
- 背景部分が緑色になっていない
- 正解ラベルと予測マスクがほぼ一致

**改善が必要な例:**
- プラナリアの一部が緑色になっていない（検出漏れ）
- 背景が緑色になっている（誤検出）
- IoU が 0.5 未満

---

## 💡 活用方法

### 1. モデルの性能確認

ダウンロードした `test_inference_result.png` を見て、モデルの性能を確認：

- ✅ **良好**: プラナリアがきれいに検出されている → そのまま使用
- ⚠️ **改善余地あり**: 一部検出漏れや誤検出がある → データ追加やパラメータ調整
- ❌ **不良**: ほとんど検出できていない → データ・ラベル・パラメータを見直し

### 2. 複数実験の比較

異なる設定で複数回トレーニングした場合、テスト推論結果を並べて比較：

```
Google Drive/MyDrive/projects/planarian/
├─ exp_001/
│  └─ outputs/
│     └─ test_inference_result.png  (BATCH_SIZE=8)
├─ exp_002/
│  └─ outputs/
│     └─ test_inference_result.png  (BATCH_SIZE=4)
└─ exp_003/
   └─ outputs/
      └─ test_inference_result.png  (データ追加版)
```

### 3. トレーニングの成功判定

以下の指標で判定：

| IoU | 評価 | 次のアクション |
|-----|------|---------------|
| 0.8+ | 優秀 | そのまま使用可能 |
| 0.6-0.8 | 良好 | 実用可能、データ追加でさらに改善 |
| 0.4-0.6 | 普通 | データ追加やパラメータ調整を推奨 |
| 0.4未満 | 要改善 | データ・ラベル・設定を見直し |

---

## 🔧 技術的な詳細

### 実装内容

```python
# テスト画像を1枚選択（最初の画像）
test_image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
test_image_name = test_image_files[0]

# 画像読み込み・前処理
test_image = np.array(Image.open(test_image_path).convert('RGB'))
transform = get_val_transform(IMAGE_SIZE)
input_tensor = transform(image=test_image)['image'].unsqueeze(0).to(device)

# 推論実行
model.eval()
with torch.no_grad():
    output = model(input_tensor)
    pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]

# 元のサイズにリサイズ & 二値化
pred_mask_resized = resize(pred_mask, original_size)
pred_mask_binary = (pred_mask_resized > 127).astype(np.uint8) * 255

# 重ね合わせ画像作成（緑色）
overlay = test_image.copy()
overlay[pred_mask_binary > 0] = overlay[pred_mask_binary > 0] * 0.5 + [0, 255, 0] * 0.5

# IoU計算
intersection = np.sum((pred_mask_binary > 0) & (test_label > 0))
union = np.sum((pred_mask_binary > 0) | (test_label > 0))
iou = intersection / union if union > 0 else 0.0
```

### ファイル保存先

```
BASE_DIR/
└─ outputs/
   ├─ training_history.png          ← 学習曲線
   └─ test_inference_result.png     ← テスト推論結果（新規）
```

---

## 📚 更新されたファイル

1. **`train_colab_single.py`**
   - テスト推論セクションを追加（約120行）
   - 可視化機能を実装
   - 統計情報の計算と表示
   - ダウンロードに test_inference_result.png を追加

2. **`COLAB_SINGLE_CELL_GUIDE.md`**
   - 実行の流れにテスト推論を追加
   - ダウンロードファイルを更新

---

## ✨ メリット

### 1. 即座のフィードバック
- ✅ トレーニング完了直後にモデルの性能を確認できる
- ✅ 追加トレーニングが必要か判断できる

### 2. デバッグが容易
- ✅ 予測マスクを見て、どこで失敗しているか分かる
- ✅ データやラベルの問題を発見できる

### 3. 実験管理がしやすい
- ✅ 各実験の結果を視覚的に比較できる
- ✅ IoUで定量的に評価できる

### 4. 初心者にも分かりやすい
- ✅ 数値だけでなく画像で結果を確認できる
- ✅ 緑色の重ね合わせが直感的

---

## 🎯 まとめ

**テスト推論結果の自動表示機能を追加しました！**

- ✅ **自動実行**: トレーニング完了後に自動で推論
- ✅ **4種類の画像**: 元画像・正解・予測・重ね合わせ
- ✅ **統計情報**: 検出面積、IoUなど
- ✅ **自動保存**: test_inference_result.png
- ✅ **即座のフィードバック**: モデルの性能を即座に確認

これで、トレーニングの成功・失敗を一目で判断でき、効率的にモデル開発を進められます！🚀
