# ONNX Runtime GPU推論ガイド

RTX 5070 Ti等の新しいGPUで高速推論を実現するためのガイドです。

---

## 🎯 概要

ONNX Runtimeを使用することで、PyTorchが未対応のGPU（RTX 5070 Ti等）でもGPU推論が可能になります。

**メリット:**
- ✅ 最新GPU対応（RTX 5070 Ti, RTX 50シリーズ等）
- ✅ 最適化されたパフォーマンス
- ✅ DirectML対応（Windows）
- ✅ CPU推論も高速化

---

## 📦 インストール

### 基本パッケージ

```powershell
pip install onnx onnxruntime
```

### GPU対応（Windows - DirectML）

```powershell
pip install onnxruntime-directml
```

**注意:** `onnxruntime`と`onnxruntime-directml`は同時にインストールできません。DirectMLを使う場合は`onnxruntime`をアンインストールしてください。

```powershell
pip uninstall onnxruntime
pip install onnxruntime-directml
```

### GPU対応（Linux/Windows - CUDA）

```powershell
pip install onnxruntime-gpu
```

---

## 🚀 使い方

### ステップ1: PyTorchモデルをONNXに変換

```powershell
cd segmentation
python export_onnx.py --model models/best_unet.pth
```

**出力:** `models/best_unet.onnx`

**オプション:**
```powershell
python export_onnx.py \
  --model models/best_unet.pth \
  --output models/custom_name.onnx \
  --height 512 \
  --width 512 \
  --opset 14
```

---

### ステップ2: ONNX Runtimeで推論

#### CPU推論

```powershell
python inference_onnx.py \
  --images <画像フォルダ> \
  --output <出力フォルダ> \
  --device cpu
```

#### GPU推論（DirectML - Windows）

```powershell
python inference_onnx.py \
  --images <画像フォルダ> \
  --output <出力フォルダ> \
  --device directml
```

または

```powershell
python inference_onnx.py \
  --images <画像フォルダ> \
  --output <出力フォルダ> \
  --directml
```

#### GPU推論（CUDA - Linux/Windows）

```powershell
python inference_onnx.py \
  --images <画像フォルダ> \
  --output <出力フォルダ> \
  --device cuda
```

---

## ⚡ パフォーマンス比較

### 予想されるパフォーマンス

| 環境 | 相対速度 | 備考 |
|-----|---------|------|
| PyTorch CPU | 1x | ベースライン |
| ONNX Runtime CPU | 1.5-2x | 最適化により高速化 |
| ONNX Runtime DirectML (RTX 5070 Ti) | 10-30x | GPU加速 |
| PyTorch CUDA (対応GPU) | 10-50x | 参考値 |

**注意:** 実際のパフォーマンスは画像サイズ、バッチサイズ、GPU性能に依存します。

---

## 🔧 トラブルシューティング

### エラー: onnxruntimeがインストールされていません

```powershell
pip install onnxruntime
```

GPU推論の場合:
```powershell
# Windows
pip install onnxruntime-directml

# Linux/Windows (CUDA)
pip install onnxruntime-gpu
```

### エラー: ONNXモデルが見つかりません

```powershell
cd segmentation
python export_onnx.py --model models/best_unet.pth
```

### DirectMLが動作しない

**確認事項:**
1. Windows 10/11であることを確認
2. 最新のGPUドライバーをインストール
3. `onnxruntime`と`onnxruntime-directml`が競合していないか確認

```powershell
# アンインストールして再インストール
pip uninstall onnxruntime onnxruntime-directml -y
pip install onnxruntime-directml
```

### CUDAが動作しない

**確認事項:**
1. NVIDIA GPUがあることを確認
2. CUDA Toolkit 11.x以上がインストールされているか確認
3. CUDAに対応した`onnxruntime-gpu`バージョンを使用

---

## 📊 ベンチマーク方法

### ベンチマークスクリプトの作成

```python
import time
import numpy as np

# PyTorch推論
from inference import inference

start = time.time()
inference(images_dir='test_images', output_dir='output_pytorch', create_video=False)
pytorch_time = time.time() - start

# ONNX Runtime推論
from inference_onnx import inference_onnx

start = time.time()
inference_onnx(images_dir='test_images', output_dir='output_onnx', 
               device='directml', create_video=False)
onnx_time = time.time() - start

print(f"PyTorch: {pytorch_time:.2f}秒")
print(f"ONNX Runtime: {onnx_time:.2f}秒")
print(f"高速化: {pytorch_time / onnx_time:.2f}x")
```

---

## 🔄 PyTorchとの切り替え

### 推論エンジンの選択

#### PyTorch（デフォルト）

```powershell
python run_inference_analysis.py --images <dir> --output <dir>
```

#### ONNX Runtime

現在は`inference_onnx.py`を直接使用：

```powershell
python inference_onnx.py --images <dir> --output <dir> --directml
```

**TODO:** GUIに推論エンジン選択機能を追加（v2.1予定）

---

## 📝 技術詳細

### ONNX形式の利点

1. **フレームワーク非依存:** PyTorch以外の環境でも動作
2. **最適化:** グラフ最適化により高速化
3. **GPU互換性:** DirectML, CUDA, OpenVINOなど多様なバックエンド
4. **展開性:** エッジデバイスやモバイルでも使用可能

### DirectMLとは

Microsoftが開発したGPU加速ライブラリ。DirectX 12ベースで動作し、
NVIDIA, AMD, Intel等のGPUで動作します。

**対応GPU:**
- NVIDIA GeForce（全シリーズ）
- AMD Radeon
- Intel Arc/Iris

### ONNX Runtimeのバックエンド

| バックエンド | 対応環境 | GPU | 備考 |
|------------|---------|-----|------|
| CPUExecutionProvider | All | ❌ | デフォルト |
| CUDAExecutionProvider | Linux/Win | ✅ | NVIDIA GPU |
| DmlExecutionProvider | Windows | ✅ | DirectML（推奨） |
| TensorrtExecutionProvider | Linux/Win | ✅ | NVIDIA GPU（最速） |
| OpenVINOExecutionProvider | All | ✅ | Intel GPU/CPU |

---

## 🎓 次のステップ

1. **モデル変換を試す**
   ```powershell
   python export_onnx.py
   ```

2. **CPU推論でテスト**
   ```powershell
   python inference_onnx.py --images test_images --output output_test --device cpu
   ```

3. **GPU推論を試す**
   ```powershell
   python inference_onnx.py --images test_images --output output_test --directml
   ```

4. **パフォーマンスを比較**
   - PyTorch vs ONNX Runtime
   - CPU vs GPU

5. **本番環境で使用**
   - 大量の画像で推論
   - 行動解析に統合

---

## 📚 参考リンク

- **ONNX Runtime**: https://onnxruntime.ai/
- **DirectML**: https://github.com/microsoft/DirectML
- **ONNX**: https://onnx.ai/

---

**最終更新: 2026年2月7日**
