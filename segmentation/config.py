"""
セグメンテーションシステムの設定ファイル

すべてのパラメータをここで一元管理
"""

import os

# =============================================================================
# パス設定
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
LABELS_DIR = os.path.join(DATA_DIR, 'labels')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# モデル保存パス
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_unet.pth')
CHECKPOINT_PATH = os.path.join(MODELS_DIR, 'checkpoint.pth')

# =============================================================================
# モデル設定
# =============================================================================
# 入力画像サイズ（高解像度対応）
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

# U-Netアーキテクチャ
ENCODER_NAME = 'resnet34'  # ResNet34バックボーン（ImageNet事前学習済み）
ENCODER_WEIGHTS = 'imagenet'  # 事前学習済み重み
IN_CHANNELS = 3  # RGB画像
OUT_CHANNELS = 1  # バイナリセグメンテーション（プラナリア vs 背景）

# =============================================================================
# 学習設定
# =============================================================================
# デバイス設定（GPU優先）
# 注意: RTX 5070 Ti (sm_120) は PyTorch 2.6/2.7 では未対応のため、一時的にCPUを使用
# PyTorch 2.8以降でsm_120サポートが追加される予定
DEVICE = 'cpu'  # 'cuda' or 'cpu' - 現在はCPUモード

# バッチサイズ（CPUモード用に削減）
BATCH_SIZE = 4  # GPU使用時は8推奨
NUM_WORKERS = 2  # データローダーのワーカー数（CPUモード用に削減）

# 学習率とオプティマイザー
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# エポック数
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15  # Validation lossが改善しない場合のエポック数

# データ分割
TRAIN_VAL_SPLIT = 0.8  # 80% Train, 20% Validation
RANDOM_SEED = 42

# 損失関数の重み
DICE_WEIGHT = 0.5
BCE_WEIGHT = 0.5

# =============================================================================
# データ拡張設定
# =============================================================================
# 拡張の確率
AUGMENTATION_PROB = 0.5

# 明度変動（夜間画像対策）
BRIGHTNESS_LIMIT = 0.2  # ±20%
CONTRAST_LIMIT = 0.2    # ±20%

# 回転範囲
ROTATE_LIMIT = 15  # ±15度

# ガウシアンノイズ
GAUSSIAN_NOISE_VAR = (10.0, 50.0)

# =============================================================================
# 推論設定
# =============================================================================
# 信頼度閾値（セグメンテーション結果の二値化）
CONFIDENCE_THRESHOLD = 0.5

# 最小面積フィルタ（ノイズ除去）
MIN_CONTOUR_AREA = 100
MAX_CONTOUR_AREA = 50000

# 動画出力設定
VIDEO_FPS = 10
VIDEO_CODEC = 'XVID'  # AVIコーデック
VIDEO_SCALE = 1.0  # スケール（1.0 = オリジナルサイズ）

# =============================================================================
# ラベリングGUI設定
# =============================================================================
# ブラシサイズ
DEFAULT_BRUSH_SIZE = 15
MIN_BRUSH_SIZE = 5
MAX_BRUSH_SIZE = 100

# キャンバスサイズ
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600

# オートセーブ間隔（秒）
AUTOSAVE_INTERVAL = 60

# =============================================================================
# 正規化パラメータ（ImageNet統計）
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
