# PyTorch GPU インストールガイド - RTX 5070 Ti

## 現在の状況（2026年2月6日）

RTX 5070 Ti は **CUDA Compute Capability sm_120 (12.0)** を持つ最新のBlackwell世代GPUです。

### 問題
- PyTorch 2.6.0 および 2.7.0 (nightly) は **sm_120 に未対応**
- 現在サポートされているのは: sm_50, sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_90 まで
- そのため、CUDAでモデルを実行すると `RuntimeError: CUDA error: no kernel image is available for execution on the device` エラーが発生

### 解決策

#### オプション1: CPUで実行（現在の設定）
```python
# config.py
DEVICE = 'cpu'
BATCH_SIZE = 4  # CPUモード用に削減
NUM_WORKERS = 2
```

**メリット**: 確実に動作する
**デメリット**: 学習が非常に遅い（GPU比で10-50倍遅い）

#### オプション2: PyTorch 2.8以降を待つ（推奨）
PyTorch 2.8 で sm_120 サポートが追加される予定です。

リリースされたら、以下のコマンドでインストール:
```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

その後、`config.py` を編集:
```python
DEVICE = 'cuda'
BATCH_SIZE = 8
NUM_WORKERS = 4
```

#### オプション3: PyTorchをソースからビルド（上級者向け）
CUDA 12.4 ツールキットを使用して、sm_120 サポート付きでPyTorchをコンパイル。

## 現在インストールされているバージョン

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda)"
```

## 動作確認

```powershell
python check_gpu.py
```

## トレーニングの実行

```powershell
cd segmentation
python train.py
```

現在の設定では自動的にCPUモードで実行されます。

## 参考リンク

- PyTorch インストール: https://pytorch.org/get-started/locally/
- CUDA互換性マトリックス: https://developer.nvidia.com/cuda-gpus
- PyTorch GitHub Issues: sm_120サポートの追跡
