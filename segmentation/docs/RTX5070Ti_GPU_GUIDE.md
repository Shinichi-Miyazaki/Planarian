# RTX 5070 Ti GPU推論ガイド

RTX 5070 Ti等の新しいNVIDIA GPUでONNX Runtime + CUDAを使用して高速推論を実行する方法

---

## 🎯 概要

RTX 5070 TiはPyTorchのCUDA 12.1では完全にサポートされていない場合がありますが、ONNX Runtime + CUDAを使用することで高速GPU推論が可能です。

**性能向上:**
- CPU比で **20-50倍** の高速化
- PyTorch CPU推論比で大幅な改善

---

## 📦 インストール手順

### ステップ1: 基本パッケージのインストール

```powershell
pip install onnx
```

### ステップ2: CUDA対応ONNX Runtimeのインストール

**重要**: 既存の`onnxruntime`や`onnxruntime-directml`がインストールされている場合は、先にアンインストールしてください。

```powershell
# 既存のonnxruntimeをアンインストール（該当する場合のみ）
pip uninstall onnxruntime onnxruntime-directml

# CUDA対応版をインストール
pip install onnxruntime-gpu
```

### ステップ3: CUDA Toolkitの確認

ONNX Runtime GPU版は内部でCUDAライブラリを使用しますが、通常は自動的にバンドルされています。

**確認方法:**
```powershell
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

**期待される出力:**
```
['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

`CUDAExecutionProvider`が含まれていればOKです。

---

## 🚀 使い方

### 方法1: GUIから使用（推奨）

#### 単一フォルダモード

1. `inference_analysis_gui.py`を起動
   ```powershell
   cd C:\Users\Shinichi\PycharmProjects\Planarian\segmentation
   python inference_analysis_gui.py
   ```

2. 「単一フォルダ解析」タブで設定:
   - ✅ **ONNX Runtime (GPU対応)** にチェック
   - デバイス: **CUDA (NVIDIA GPU)** を選択
   - 画像フォルダ、モデル、出力フォルダを指定
   - 「実行」をクリック

#### バッチフォルダモード

1. 「バッチフォルダ解析」タブで設定:
   - 親フォルダを指定
   - Excelファイル（オプション）を指定
   - ✅ **ONNX Runtime (GPU対応)** にチェック
   - デバイス: **CUDA (NVIDIA GPU)** を選択
   - 「フォルダを検索」→ 「バッチ実行」

### 方法2: コマンドラインから使用

#### モデル変換（初回のみ）

```powershell
python export_onnx.py --model models\best_unet.pth
```

これにより`models\best_unet.onnx`が生成されます。

#### GPU推論実行

```powershell
python inference_onnx.py --images <画像フォルダ> --output <出力フォルダ> --device cuda
```

---

## ⚡ パフォーマンス比較

### 想定される処理速度（10,000枚の画像）

| 環境 | 処理時間 | 相対速度 |
|-----|---------|---------|
| PyTorch CPU | 5-10時間 | 1x |
| ONNX Runtime CPU | 3-7時間 | 1.5-2x |
| ONNX Runtime CUDA (RTX 5070 Ti) | 10-30分 | **20-50x** |

**注意**: 実際のパフォーマンスは画像サイズ、解像度、バッチサイズに依存します。

---

## 🔧 トラブルシューティング

### CUDAExecutionProviderが利用できない

**症状:**
```
⚠ CUDAが利用できません。CPUにフォールバック
```

**対処法:**

1. **onnxruntime-gpuが正しくインストールされているか確認**
   ```powershell
   pip list | findstr onnxruntime
   ```
   
   `onnxruntime-gpu`が表示されるべき。`onnxruntime`や`onnxruntime-directml`が表示される場合はアンインストール。

2. **CUDA対応を確認**
   ```powershell
   python -c "import onnxruntime as ort; print(ort.get_available_providers())"
   ```

3. **再インストール**
   ```powershell
   pip uninstall onnxruntime onnxruntime-directml onnxruntime-gpu
   pip install onnxruntime-gpu
   ```

### GPUメモリ不足エラー

**症状:**
```
CUDA out of memory
```

**対処法:**

1. 他のGPUを使用するアプリケーションを終了
2. バッチサイズを小さくする（現在は1枚ずつ処理なので通常は問題なし）
3. 画像解像度を下げる


---

## 📊 ベンチマーク方法

実際のパフォーマンスをテストする方法:

```powershell
# テスト画像フォルダを準備（例: 100枚）
mkdir test_images

# CPU推論
$start = Get-Date
python inference_onnx.py --images test_images --output output_cpu --device cpu --no-video
$cpu_time = (Get-Date) - $start
Write-Host "CPU: $($cpu_time.TotalSeconds)秒"

# GPU推論（CUDA）
$start = Get-Date
python inference_onnx.py --images test_images --output output_gpu --device cuda --no-video
$gpu_time = (Get-Date) - $start
Write-Host "GPU: $($gpu_time.TotalSeconds)秒"

# 高速化率
Write-Host "高速化: $($cpu_time.TotalSeconds / $gpu_time.TotalSeconds)倍"
```

---

## 🎓 推奨ワークフロー

### 初回セットアップ

1. **ONNX Runtime GPUをインストール**
   ```powershell
   pip uninstall onnxruntime onnxruntime-directml
   pip install onnxruntime-gpu
   ```

2. **CUDA対応を確認**
   ```powershell
   python -c "import onnxruntime as ort; print('CUDAExecutionProvider' in ort.get_available_providers())"
   ```
   → `True`が表示されればOK

3. **モデルを変換（.pth → .onnx）**
   ```powershell
   python export_onnx.py --model models\best_unet.pth
   ```

### 大量画像の解析

1. **GUIを起動**
   ```powershell
   python inference_analysis_gui.py
   ```

2. **バッチモードで設定:**
   - 親フォルダ: 複数の実験フォルダを含む親フォルダ
   - Excelファイル: 実験条件リスト（オプション）
   - ✅ ONNX Runtimeを使用
   - デバイス: CUDA (NVIDIA GPU)
   - フォルダを検索 → バッチ実行

3. **結果を確認:**
   - 各フォルダ内: `segmentation_analysis/`
   - 親フォルダ直下: `batch_summary/`（統合解析）

---

## 💡 よくある質問

### Q1: .pthモデルしかない場合は？

GUIで「ONNX Runtime (GPU対応)」にチェックを入れると、自動的に.onnxに変換されます。

### Q2: 既存のPyTorch推論と結果は同じ？

はい、ONNX Runtime推論はPyTorch推論と同じ結果を生成します。ただし、浮動小数点演算の違いによりわずかな差異が生じる場合があります（通常は無視できる程度）。

### Q3: バッチ処理で一部のフォルダだけGPU推論したい

全フォルダに対してGPU推論が適用されます。一部のみ変更したい場合は、単一フォルダモードで個別に実行してください。

---

## 📚 関連ドキュメント

- [ONNX_RUNTIME_GUIDE.md](ONNX_RUNTIME_GUIDE.md): ONNX Runtime全般のガイド
- [EXCEL_BATCH_ANALYSIS_GUIDE.md](EXCEL_BATCH_ANALYSIS_GUIDE.md): Excelベースのバッチ解析
- [README.md](README.md): セグメンテーション全般のガイド

---

## ✅ まとめ

RTX 5070 TiでGPU推論を使うには:

1. ✅ `pip install onnxruntime-gpu`
2. ✅ GUIで「ONNX Runtime + CUDA」を選択
3. ✅ 20-50倍の高速化を実現

**最終更新: 2026年2月10日**

