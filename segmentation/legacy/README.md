# Segmentation Legacy Scripts

Google Colab用および旧バージョンのスクリプトです。

## Colab学習スクリプト

### train_colab.ipynb
Google Colabでの学習用ノートブック。GPU環境で高速学習が可能です。

**使い方:**
1. 親ディレクトリで`create_data_zip.py`を実行してデータをZIP化
2. このノートブックをGoogle Colabで開く
3. GPUランタイムを選択
4. すべてのセルを実行
5. `best_unet.pth`をダウンロード

**詳細:** `../docs/COLAB_TRAINING_GUIDE.md`

### train_colab_single.py
最もシンプルな単一セル実行版。Colabで新規ノートブックを作成し、このファイルの内容をコピー&ペーストして実行できます。

**詳細:** `../docs/COLAB_SINGLE_CELL_GUIDE.md`

### create_data_zip.py
Colab用にデータをZIP圧縮するユーティリティ。

```powershell
python create_data_zip.py
```

`data.zip`が生成され、Colabにアップロードできます。

## 使用推奨

通常の使用には、親ディレクトリの以下のスクリプトを推奨します：
- **run_inference_analysis.py**: 統合スクリプト（推論+解析）
- **inference_analysis_gui.py**: GUIランチャー
- **train.py**: ローカル学習（CPUモード）
