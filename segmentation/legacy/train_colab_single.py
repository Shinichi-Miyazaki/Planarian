"""
U-Net プラナリアセグメンテーション - Google Colab版（1セル実行）

使用方法:
1. Google Colabで新規ノートブックを作成
2. ランタイム > ランタイムのタイプを変更 > GPU (T4推奨) を選択
3. このスクリプト全体を1つのセルにコピー&ペースト
4. セルを実行
5. 学習完了後、best_unet.pth をダウンロード
"""

# ============================================================================
# 🔧 設定: ここを編集してください
# ============================================================================

# ============================================================================
# パス設定
# ============================================================================

# ベースディレクトリ（すべてのデータとモデルの保存場所）
BASE_DIR = '/content/planarian'  # ここを変更すれば全体の保存先が変わります

# データディレクトリ（ベースディレクトリからの相対パス）
DATA_DIR_NAME = 'data'           # データフォルダ名
MODELS_DIR_NAME = 'models'       # モデル保存フォルダ名
OUTPUTS_DIR_NAME = 'outputs'     # 出力フォルダ名

# 自動生成されるパス（通常は変更不要）
DATA_DIR = f'{BASE_DIR}/{DATA_DIR_NAME}'
MODELS_DIR = f'{BASE_DIR}/{MODELS_DIR_NAME}'
OUTPUTS_DIR = f'{BASE_DIR}/{OUTPUTS_DIR_NAME}'

# ============================================================================
# データソースの設定（以下のいずれかを選択）
# ============================================================================

# 方法1: ZIPファイルをアップロード（推奨）
USE_ZIP = True  # True: ZIP使用, False: Google Drive使用
ZIP_FILENAME = 'data.zip'  # アップロードするZIPファイル名

# ZIP解凍先（通常は変更不要）
# ZIPは BASE_DIR/DATA_DIR_NAME/ に解凍されます
# ZIP内に images/ と labels/ フォルダが必要

# 方法2: Google Driveを使用
# Google Drive内のデータパスを指定（USE_ZIP = False の場合に使用）
GOOGLE_DRIVE_IMAGES_DIR = '/content/drive/MyDrive/Planarian/segmentation/data/images'
GOOGLE_DRIVE_LABELS_DIR = '/content/drive/MyDrive/Planarian/segmentation/data/labels'

# ============================================================================
# 学習設定
# ============================================================================
MAX_EPOCHS = 100
BATCH_SIZE = 8  # T4 GPU用（メモリ不足の場合は4に減らす）
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 15
IMAGE_SIZE = 512

# ============================================================================
# 📦 ライブラリのインストールとインポート
# ============================================================================

print("=" * 70)
print("  U-Net プラナリアセグメンテーション - Google Colab版")
print("=" * 70)
print("\n[1/6] ライブラリをインストール中...\n")

import subprocess
import sys

# 必要なライブラリをインストール
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                      "segmentation-models-pytorch", "albumentations"])

print("✓ インストール完了\n")

# ============================================================================
# インポート
# ============================================================================

import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# GPU確認
print("[2/6] 環境確認中...\n")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ GPUが利用できません。CPUモードで実行されます（遅い）")
print()

# ============================================================================
# 📂 データの準備
# ============================================================================

print("[3/6] データを準備中...\n")

# ベースディレクトリを作成
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

print(f"📁 ベースディレクトリ: {BASE_DIR}")
print(f"   ├─ データ: {DATA_DIR}")
print(f"   ├─ モデル: {MODELS_DIR}")
print(f"   └─ 出力: {OUTPUTS_DIR}\n")

if USE_ZIP:
    # ZIPファイルをアップロード
    from google.colab import files
    print(f"'{ZIP_FILENAME}' をアップロードしてください...")
    uploaded = files.upload()

    # 解凍
    for filename in uploaded.keys():
        if filename.endswith('.zip'):
            print(f"\n{filename} を解凍中...")
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                # DATA_DIR に解凍
                zip_ref.extractall(DATA_DIR)
            print(f"✓ 解凍完了: {DATA_DIR}")

    # パス設定（ZIP内の構造: data/images/, data/labels/）
    IMAGES_DIR = os.path.join(DATA_DIR, 'images')
    LABELS_DIR = os.path.join(DATA_DIR, 'labels')

else:
    # Google Driveをマウント
    from google.colab import drive
    print("Google Driveをマウント中...")
    drive.mount('/content/drive')
    print("✓ マウント完了")

    IMAGES_DIR = GOOGLE_DRIVE_IMAGES_DIR
    LABELS_DIR = GOOGLE_DRIVE_LABELS_DIR

# データ確認
if os.path.exists(IMAGES_DIR) and os.path.exists(LABELS_DIR):
    image_count = len([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    label_count = len([f for f in os.listdir(LABELS_DIR) if f.endswith('.png')])
    print(f"\n✓ データディレクトリを確認:")
    print(f"  画像フォルダ: {IMAGES_DIR}")
    print(f"  ラベルフォルダ: {LABELS_DIR}")
    print(f"  画像数: {image_count} 枚")
    print(f"  ラベル数: {label_count} 枚")

    if image_count == 0 or label_count == 0:
        raise ValueError("画像またはラベルが見つかりません。パスを確認してください。")
else:
    raise ValueError(f"データディレクトリが見つかりません:\n  {IMAGES_DIR}\n  {LABELS_DIR}")

print()


# ============================================================================
# 🔧 データセット定義
# ============================================================================

class PlanarianDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # 画像とラベルのペアを取得
        self.samples = []
        for img_name in os.listdir(images_dir):
            if img_name.endswith(('.jpg', '.png')):
                img_base = os.path.splitext(img_name)[0]
                label_name = img_base + '.png'

                img_path = os.path.join(images_dir, img_name)
                label_path = os.path.join(labels_dir, label_name)

                if os.path.exists(label_path):
                    self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # 画像読み込み
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(label_path).convert('L'))
        mask = (mask > 127).astype(np.float32)

        # 拡張適用
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.unsqueeze(0)
        return image, mask

def get_train_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transform(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# ============================================================================
# 🧠 モデル定義
# ============================================================================

class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss

def dice_coefficient(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            return False

        if val_loss < (self.best_score - self.min_delta):
            self.best_score = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

# ============================================================================
# 🚀 データローダー作成
# ============================================================================

print("[4/6] データローダーを作成中...\n")

# データセット作成
dataset = PlanarianDataset(IMAGES_DIR, LABELS_DIR, transform=None)
print(f"✓ 総サンプル数: {len(dataset)}")

# Train/Val分割
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

# Transformを適用
train_dataset.dataset.transform = get_train_transform(IMAGE_SIZE)
val_dataset.dataset.transform = get_val_transform(IMAGE_SIZE)

# DataLoader作成
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✓ 学習データ: {train_size} サンプル")
print(f"✓ 検証データ: {val_size} サンプル")
print()

# ============================================================================
# 🏗️ モデル構築
# ============================================================================

print("[5/6] モデルを構築中...\n")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"デバイス: {device}\n")

# U-Netモデル
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=3,
    classes=1
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✓ モデル: U-Net (ResNet34)")
print(f"✓ 総パラメータ数: {total_params:,}")
print(f"✓ 学習可能パラメータ数: {trainable_params:,}")
print()

# 損失関数・オプティマイザー
criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)

# ============================================================================
# 🎯 トレーニングループ
# ============================================================================

print("[6/6] トレーニング開始\n")
print("=" * 70)
print(f"設定:")
print(f"  エポック数: {MAX_EPOCHS}")
print(f"  バッチサイズ: {BATCH_SIZE}")
print(f"  学習率: {LEARNING_RATE}")
print(f"  Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
print("=" * 70)
print()

history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}
best_val_loss = float('inf')
best_model_path = os.path.join(MODELS_DIR, 'best_unet.pth')

for epoch in range(MAX_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{MAX_EPOCHS}")
    print("-" * 70)

    # ============================================================================
    # Training
    # ============================================================================
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    pbar = tqdm(train_loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_masks = torch.sigmoid(outputs)
            dice = dice_coefficient(pred_masks, masks)

        running_loss += loss.item()
        running_dice += dice.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})

    train_loss = running_loss / len(train_loader)
    train_dice = running_dice / len(train_loader)

    # ============================================================================
    # Validation
    # ============================================================================
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            pred_masks = torch.sigmoid(outputs)
            dice = dice_coefficient(pred_masks, masks)

            running_loss += loss.item()
            running_dice += dice.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice.item():.4f}'})

    val_loss = running_loss / len(val_loader)
    val_dice = running_dice / len(val_loader)

    # 履歴保存
    history['train_loss'].append(train_loss)
    history['train_dice'].append(train_dice)
    history['val_loss'].append(val_loss)
    history['val_dice'].append(val_dice)

    # 結果表示
    print(f"\n結果:")
    print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}")

    # ベストモデル保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_dice': val_dice,
            'history': history
        }, best_model_path)
        print(f"  ✓ ベストモデルを保存しました (Val Loss: {val_loss:.4f})")

    # Early Stopping
    if early_stopping(val_loss):
        print(f"\n⚠️ Early Stopping発動 (Epoch {epoch + 1})")
        print(f"   {EARLY_STOPPING_PATIENCE} エポック改善なし")
        break

# ============================================================================
# 📊 学習曲線のプロット
# ============================================================================

print("\n" + "=" * 70)
print("  トレーニング完了！")
print("=" * 70)
print(f"Best Validation Loss: {best_val_loss:.4f}\n")

print("学習曲線を作成中...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss
axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Dice
axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
axes[1].plot(history['val_dice'], label='Val Dice', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Dice Coefficient', fontsize=12)
axes[1].set_title('Training & Validation Dice', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
history_plot_path = os.path.join(OUTPUTS_DIR, 'training_history.png')
plt.savefig(history_plot_path, dpi=150, bbox_inches='tight')
print(f"✓ 学習曲線を保存: {history_plot_path}")
plt.show()

# ============================================================================
# 🔍 テスト推論（1枚の画像で結果を確認）
# ============================================================================

print("\n" + "=" * 70)
print("  テスト推論（セグメンテーション結果の確認）")
print("=" * 70)

# テスト画像を1枚選択（最初の画像を使用）
test_image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]
if len(test_image_files) > 0:
    test_image_name = test_image_files[0]
    test_image_path = os.path.join(IMAGES_DIR, test_image_name)
    test_label_path = os.path.join(LABELS_DIR, os.path.splitext(test_image_name)[0] + '.png')

    print(f"\nテスト画像: {test_image_name}")

    # 画像読み込み
    test_image = np.array(Image.open(test_image_path).convert('RGB'))
    original_size = test_image.shape[:2]

    # ラベルも読み込み（存在する場合）
    test_label = None
    if os.path.exists(test_label_path):
        test_label = np.array(Image.open(test_label_path).convert('L'))
        test_label = (test_label > 127).astype(np.uint8) * 255

    # 推論用に前処理
    transform = get_val_transform(IMAGE_SIZE)
    augmented = transform(image=test_image)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    # 推論実行
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]

    # 元のサイズにリサイズ
    pred_mask_resized = np.array(Image.fromarray((pred_mask * 255).astype(np.uint8)).resize(
        (original_size[1], original_size[0]), Image.Resampling.BILINEAR
    ))

    # 二値化
    pred_mask_binary = (pred_mask_resized > 127).astype(np.uint8) * 255

    # 重ね合わせ画像作成（緑色で予測マスクを重ねる）
    overlay = test_image.copy()
    overlay[pred_mask_binary > 0] = overlay[pred_mask_binary > 0] * 0.5 + np.array([0, 255, 0]) * 0.5

    # 可視化
    if test_label is not None:
        # ラベルがある場合: 元画像・正解ラベル・予測マスク・重ね合わせ
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        axes[0, 0].imshow(test_image)
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(test_label, cmap='gray')
        axes[0, 1].set_title('Ground Truth Label', fontsize=14, fontweight='bold')
        axes[0, 1].axis('off')

        axes[1, 0].imshow(pred_mask_binary, cmap='gray')
        axes[1, 0].set_title('Predicted Mask', fontsize=14, fontweight='bold')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Overlay (Green = Prediction)', fontsize=14, fontweight='bold')
        axes[1, 1].axis('off')
    else:
        # ラベルがない場合: 元画像・予測マスク・重ね合わせ
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        axes[0].imshow(test_image)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(pred_mask_binary, cmap='gray')
        axes[1].set_title('Predicted Mask', fontsize=14, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Green = Prediction)', fontsize=14, fontweight='bold')
        axes[2].axis('off')

    plt.tight_layout()
    test_result_path = os.path.join(OUTPUTS_DIR, 'test_inference_result.png')
    plt.savefig(test_result_path, dpi=150, bbox_inches='tight')
    print(f"✓ テスト推論結果を保存: {test_result_path}")
    plt.show()

    # 統計情報
    pred_area = np.sum(pred_mask_binary > 0)
    total_area = pred_mask_binary.shape[0] * pred_mask_binary.shape[1]
    pred_ratio = (pred_area / total_area) * 100

    print(f"\n推論結果の統計:")
    print(f"  - 画像サイズ: {original_size[1]} x {original_size[0]}")
    print(f"  - 検出面積: {pred_area} ピクセル")
    print(f"  - 検出割合: {pred_ratio:.2f}%")

    if test_label is not None:
        label_area = np.sum(test_label > 0)
        label_ratio = (label_area / total_area) * 100

        # IoU計算（ゼロ除算を防ぐ）
        intersection = np.sum((pred_mask_binary > 0) & (test_label > 0))
        union = np.sum((pred_mask_binary > 0) | (test_label > 0))
        iou = intersection / union if union > 0 else 0.0

        print(f"  - 正解面積: {label_area} ピクセル")
        print(f"  - 正解割合: {label_ratio:.2f}%")
        print(f"  - IoU (Intersection over Union): {iou:.4f}")
else:
    print("\n⚠️ テスト画像が見つかりませんでした")
    test_result_path = None

# ============================================================================
# 💾 モデルのダウンロード
# ============================================================================

print("\n" + "=" * 70)
print("  モデルと学習履歴をダウンロード")
print("=" * 70)

from google.colab import files

print("\nダウンロード中...")
files.download(best_model_path)
files.download(history_plot_path)
if test_result_path:
    files.download(test_result_path)

print("\n✓ ダウンロード完了！")
print(f"  - {os.path.basename(best_model_path)}")
print(f"  - {os.path.basename(history_plot_path)}")
if test_result_path:
    print(f"  - {os.path.basename(test_result_path)}")

print("\n📊 ダウンロードしたファイル:")
print("  ✓ best_unet.pth - 学習済みモデル")
print("  ✓ training_history.png - 学習曲線（Loss & Dice）")
if test_result_path:
    print("  ✓ test_inference_result.png - テスト推論結果（セグメンテーション確認）")

print("\n次のステップ:")
print("  1. ダウンロードした best_unet.pth をローカルの segmentation/models/ に配置")
print("  2. test_inference_result.png でモデルの性能を確認")
print("  3. ローカルで推論を実行:")
print("     cd segmentation")
print("     python inference.py --images <入力> --output <出力>")
print("\n" + "=" * 70)
