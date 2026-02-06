# Legacy Scripts

このフォルダには、旧バージョンや特定用途のスクリプトが含まれています。

## 内容

### Colab関連
- **train_colab.ipynb**: Google Colabでの学習用ノートブック
- **train_colab_single.py**: 単一セル実行版（最もシンプル）
- **create_data_zip.py**: データをZIP圧縮してColab用に準備

### 旧検出スクリプト
- **animal_detector_gui.py**: 従来型画像処理GUI（二値化・輪郭検出）
- **check_cuda.py**: CUDA動作確認
- **check_gpu.py**: GPU情報確認

## 使用方法

### Google Colabでの学習

詳細は`../docs/COLAB_TRAINING_GUIDE.md`を参照してください。

基本的な手順:
1. `create_data_zip.py`でデータをZIP圧縮
2. `train_colab.ipynb`をGoogle Colabで開く
3. GPUランタイムを選択して実行
4. 学習済みモデルをダウンロード

### 旧検出GUI

```powershell
python animal_detector_gui.py
```

**注意:** ディープラーニング版（`run_inference_analysis.py`）の使用を推奨します。
