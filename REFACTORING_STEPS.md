# リファクタリング実行コミットメッセージ

## 📋 推奨コミットメッセージ

```bash
git add README.md segmentation/unet_model.py segmentation/inference.py
git commit -m "Refactor: プロジェクトを整理し包括的なREADMEを作成

主な変更:
- README.mdを完全書き換え（包括的な内容に更新）
- 使用中のファイル（labeling_gui.py, inference_analysis_gui.py）を明確化
- クイックスタートガイドを追加
- GPU互換性（RTX 5070 Ti）の説明を統合
- ONNX Runtime対応の方針を記載（v2.1予定）
- トラブルシューティングセクションを強化
- コマンドラインリファレンスを追加

次のステップ:
- 不要MDファイルの削除（手動実行が必要）
- train.py, dataset.pyをlegacyに移動（手動実行が必要）
- ONNX Runtime推論エンジンの実装（次期バージョン）

Version: 2.0.0"
```

---

## 🔄 次のPhase: 手動実行が必要

以下をPyCharmのターミナルで実行してください：

```powershell
cd C:\Users\Shinichi\PycharmProjects\Planarian

# Phase 1: 不要なMDファイルを削除
Remove-Item "QUICK_INSTALL.md" -Force -ErrorAction SilentlyContinue
Remove-Item "ALBUMENTATIONS_REMOVED.md" -Force -ErrorAction SilentlyContinue
Remove-Item "INSTALL_ALBUMENTATIONS.md" -Force -ErrorAction SilentlyContinue

cd segmentation

# Phase 2: 学習用ファイルをlegacyに移動
Move-Item "train.py" "legacy\" -Force
Move-Item "dataset.py" "legacy\" -Force

# Phase 3: segmentation内の不要MDを削除
Remove-Item "GUI_ERROR_FIX.md" -Force -ErrorAction SilentlyContinue
Remove-Item "MODEL_LOADING_FIX.md" -Force -ErrorAction SilentlyContinue

Write-Host "整理完了！"
```

実行後、以下のコミットを追加：

```bash
git add .
git commit -m "chore: 不要ファイルを削除し学習用ファイルをlegacyに移動

- train.py, dataset.py → segmentation/legacy/
- 不要なMDファイルを削除
  - QUICK_INSTALL.md
  - ALBUMENTATIONS_REMOVED.md
  - INSTALL_ALBUMENTATIONS.md
  - GUI_ERROR_FIX.md
  - MODEL_LOADING_FIX.md

整理後のプロジェクト:
- メインGUI: labeling_gui.py, inference_analysis_gui.py
- 学習: legacy/train.py (ローカル), legacy/train_colab.ipynb (Colab)
- ドキュメント: README.md (包括的), segmentation/docs/ (詳細)"
```

---

## 🚀 Phase 3: ONNX Runtime対応（次期実装）

RTX 5070 Ti対応のため、ONNX Runtime推論エンジンを実装します。

### 実装計画

1. **モデル変換スクリプト作成**
   - `segmentation/export_onnx.py`
   - PyTorch → ONNX変換

2. **ONNX推論エンジン作成**
   - `segmentation/inference_onnx.py`
   - DirectMLバックエンド対応

3. **GUIに統合**
   - `inference_analysis_gui.py`に推論エンジン選択機能を追加
   - PyTorch / ONNX Runtime を切り替え可能に

4. **パフォーマンス比較**
   - CPU: PyTorch vs ONNX Runtime
   - GPU: ONNX Runtime (DirectML) vs PyTorch

### 必要なパッケージ

```powershell
pip install onnx onnxruntime onnxruntime-directml
```

### 期待される効果

- RTX 5070 TiでGPU推論が可能に
- 推論速度の向上（最適化）
- より広範なGPU互換性

---

これでPhase 1（README統合）が完了しました。
上記のPowerShellコマンドを実行してPhase 2を完了させてください。
