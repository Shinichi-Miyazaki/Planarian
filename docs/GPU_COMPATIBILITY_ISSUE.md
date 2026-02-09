# GPU互換性の問題と解決策

## 問題の概要

**エラーメッセージ:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**原因:**
RTX 5070 Ti は CUDA Compute Capability **sm_120 (12.0)** を持つ最新のBlackwell世代GPUです。
しかし、現在のPyTorch（2.6.0および2.7.0 nightly）は sm_120 に対応していません。

サポートされているCompute Capability:
- sm_50, sm_60, sm_61, sm_70, sm_75, sm_80, sm_86, sm_90

## 実施した対応

### 1. Google Colabでのトレーニング対応（推奨）

✅ **最適な解決策**:
- `segmentation/train_colab.ipynb`: Google Colab用ノートブック
- `segmentation/COLAB_TRAINING_GUIDE.md`: 詳細な手順書
- `segmentation/create_data_zip.py`: データZIP作成スクリプト

**メリット**:
- 無料でGPU使用可能（T4、V100など）
- CUDA互換性問題を完全回避
- ローカルPCのリソースを消費しない
- トレーニングのみColabで実行、推論はローカル

**ワークフロー**:
1. ラベリング: ローカルで実行
2. トレーニング: Google Colabで実行（GPU）
3. 推論: ローカルで実行

### 3. config.pyの修正（ローカルCPUモード用）
```python
# 一時的にCPUモードに変更
DEVICE = 'cpu'
BATCH_SIZE = 4  # CPUモード用に削減
NUM_WORKERS = 2
```

### 4. train.pyの修正
- CUDA互換性の自動テスト機能を追加
- エラー発生時に自動的にCPUモードにフォールバック
- ユーザーに分かりやすいエラーメッセージを表示

### 5. ドキュメントの更新
- `install_pytorch_gpu.md`: GPU設定ガイドを作成
- `segmentation/README.md`: 注意事項を追加

## 現在の動作状況

### ✅ 推奨: Google Colabで学習
- 無料でGPU使用可能
- 互換性問題なし
- 学習速度が非常に速い（GPU比）
- 詳細は `segmentation/COLAB_TRAINING_GUIDE.md` を参照

### ✅ ローカルCPUモードで動作可能
- 学習は実行できます（ただし遅い）
- すべての機能が利用可能

### ❌ ローカルGPUモードは未対応
- PyTorch 2.8以降を待つ必要があります

## 今後の対応

### オプション1: Google Colabを使用（推奨）

**最も実用的な解決策**

1. データをZIP圧縮:
```powershell
cd segmentation
python create_data_zip.py
```

2. `train_colab.ipynb` をGoogle Colabで開く

3. GPUランタイムを選択（T4推奨）

4. すべてのセルを実行

5. 学習済みモデルをダウンロード

詳細は `segmentation/COLAB_TRAINING_GUIDE.md` を参照。

---

### オプション2: PyTorch 2.8リリース後（将来）

1. PyTorchを更新:
```powershell
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

2. `config.py` を編集:
```python
DEVICE = 'cuda'
BATCH_SIZE = 8
NUM_WORKERS = 4
```

3. 動作確認:
```powershell
python check_gpu.py
cd segmentation
python train.py
```

### 代替案: PyTorchをソースからビルド（上級者向け）

CUDA 12.4 Toolkitを使用して、sm_120サポート付きでPyTorchをコンパイルする方法もありますが、
時間がかかり、複雑なため推奨しません。

## パフォーマンスの比較

| モード | 学習速度 | 推論速度 | コスト | セットアップ |
|--------|----------|----------|--------|------------|
| **Google Colab (GPU)** | **20-50x** | N/A | 無料※ | やや複雑 |
| ローカル CPU | 1x | 1x | 無料 | 簡単 |
| ローカル GPU (未対応) | - | - | - | - |

※ Google Colab Proは月額1,179円でより高速なGPU使用可能

**推奨**: トレーニングはGoogle Colab、推論はローカルで実行するハイブリッド方式

## 参考情報

- PyTorch公式サイト: https://pytorch.org/
- CUDA Compute Capability: https://developer.nvidia.com/cuda-gpus
- PyTorch GitHub: https://github.com/pytorch/pytorch

## 変更履歴

- 2026-02-06: RTX 5070 Ti (sm_120) の互換性問題を確認、Google Colab対応とCPUモードへ切り替え
  - `train_colab.ipynb`: Google Colab用ノートブック作成
  - `COLAB_TRAINING_GUIDE.md`: 詳細な手順書作成
  - `create_data_zip.py`: データZIP作成スクリプト作成
  - ローカルトレーニングはCPUモードにフォールバック
