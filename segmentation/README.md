# セマンティックセグメンテーションシステム
U-Netベースのディープラーニングによるプラナリア検出
## 使い方
### 1. 環境構築
```powershell
pip install -r ../requirements.txt
```
### 2. ラベリング
```powershell
python labeling_gui.py
```
- 画像フォルダとラベルフォルダを選択
- マウスでプラナリアをペイント
- N=次、P=前、S=保存
### 3. 学習
```powershell
python train.py
```
- 100枚以上推奨
- Early Stopping実装済み
- models/best_unet.pth に保存
### 4. 推論
```powershell
python inference.py --images <入力> --output <出力>
```
- analysis_results.csv と動画を出力
- 既存のbehavior_analysis.pyで解析可能
## 設定
config.py でパラメータを調整
## 出力
- models/best_unet.pth: 学習済みモデル
- outputs/training_history.png: 学習曲線
- outputs/analysis_results.csv: 検出結果
- outputs/segmentation_video.avi: 推論動画
