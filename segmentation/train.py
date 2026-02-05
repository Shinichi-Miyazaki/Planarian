"""
学習スクリプト

U-Netモデルの学習
Early Stopping、学習曲線のプロット、ベストモデルの保存
"""

import os
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np

import config
from utils import create_dirs_if_not_exist, plot_training_history, dice_coefficient, timestamp
from unet_model import build_model, DiceBCELoss
from dataset import create_dataloaders


class EarlyStopping:
    """Early Stopping実装"""

    def __init__(self, patience=15, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """1エポック学習"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0

    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        # 順伝播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        # 逆伝播
        loss.backward()
        optimizer.step()

        # 評価指標
        with torch.no_grad():
            pred_masks = torch.sigmoid(outputs)
            dice = dice_coefficient(pred_masks, masks)

        running_loss += loss.item()
        running_dice += dice.item()

        pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)

    return epoch_loss, epoch_dice


def validate(model, dataloader, criterion, device):
    """検証"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            pred_masks = torch.sigmoid(outputs)
            dice = dice_coefficient(pred_masks, masks)

            running_loss += loss.item()
            running_dice += dice.item()

            pbar.set_postfix({'loss': loss.item(), 'dice': dice.item()})

    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)

    return epoch_loss, epoch_dice


def train(images_dir=None, labels_dir=None):
    """メイン学習関数"""
    print(f"\n{'='*60}")
    print(f"  U-Net学習開始")
    print(f"{'='*60}\n")
    print(f"[{timestamp()}] 初期化中...")

    # ディレクトリ作成
    create_dirs_if_not_exist([config.MODELS_DIR, config.OUTPUTS_DIR])

    # デフォルトパス
    if images_dir is None:
        images_dir = config.IMAGES_DIR
    if labels_dir is None:
        labels_dir = config.LABELS_DIR

    # デバイス確認
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"デバイス: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    # データローダー作成
    print(f"[{timestamp()}] データローダー作成中...")
    train_loader, val_loader = create_dataloaders(
        images_dir=images_dir,
        labels_dir=labels_dir,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        train_val_split=config.TRAIN_VAL_SPLIT,
        random_seed=config.RANDOM_SEED
    )

    # モデル構築
    print(f"\n[{timestamp()}] モデル構築中...")
    model = build_model(
        encoder_name=config.ENCODER_NAME,
        encoder_weights=config.ENCODER_WEIGHTS,
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        device=device
    )

    # 損失関数・オプティマイザー
    criterion = DiceBCELoss(
        dice_weight=config.DICE_WEIGHT,
        bce_weight=config.BCE_WEIGHT
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # Early Stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='min'
    )

    # 学習履歴
    history = {
        'train_loss': [],
        'train_dice': [],
        'val_loss': [],
        'val_dice': []
    }

    best_val_loss = float('inf')

    print(f"\n[{timestamp()}] 学習開始\n")
    print(f"{'='*60}")
    print(f"設定:")
    print(f"  - エポック数: {config.MAX_EPOCHS}")
    print(f"  - バッチサイズ: {config.BATCH_SIZE}")
    print(f"  - 学習率: {config.LEARNING_RATE}")
    print(f"  - Early Stopping Patience: {config.EARLY_STOPPING_PATIENCE}")
    print(f"{'='*60}\n")

    # 学習ループ
    for epoch in range(config.MAX_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{config.MAX_EPOCHS}")
        print(f"{'-'*60}")

        # 学習
        train_loss, train_dice = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 検証
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # 履歴に記録
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

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
            }, config.BEST_MODEL_PATH)
            print(f"  ✓ ベストモデルを保存しました (Val Loss: {val_loss:.4f})")

        # Early Stopping判定
        if early_stopping(val_loss):
            print(f"\n[{timestamp()}] Early Stopping発動 (Epoch {epoch + 1})")
            break

    print(f"\n{'='*60}")
    print(f"  学習完了")
    print(f"{'='*60}\n")
    print(f"ベストモデル: {config.BEST_MODEL_PATH}")
    print(f"Best Val Loss: {best_val_loss:.4f}\n")

    # 学習曲線をプロット
    plot_path = os.path.join(config.OUTPUTS_DIR, 'training_history.png')
    plot_training_history(history, plot_path)

    return model, history


if __name__ == "__main__":
    train()
