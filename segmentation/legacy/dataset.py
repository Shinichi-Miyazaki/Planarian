"""
データセットとデータローダー

画像とマスクを読み込み、前処理・データ拡張を適用
"""

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config
from utils import get_image_files


class PlanarianDataset(Dataset):
    """
    プラナリアセグメンテーション用データセット
    """

    def __init__(self, images_dir, labels_dir, image_size=(512, 512),
                 augmentation=None, preprocessing=None):
        """
        Args:
            images_dir: 画像ディレクトリのパス
            labels_dir: ラベル（マスク）ディレクトリのパス
            image_size: リサイズ後の画像サイズ (height, width)
            augmentation: データ拡張のTransform
            preprocessing: 前処理のTransform
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        # 画像ファイルのリストを取得
        self.image_files = get_image_files(images_dir)

        # 対応するラベルファイルが存在するもののみをフィルタ
        self.valid_files = []
        for img_file in self.image_files:
            # ラベルファイル名（拡張子を.pngに変更）
            label_file = os.path.splitext(img_file)[0] + '.png'
            label_path = os.path.join(labels_dir, label_file)

            if os.path.exists(label_path):
                self.valid_files.append(img_file)

        print(f"データセットを構築しました:")
        print(f"  - 画像ディレクトリ: {images_dir}")
        print(f"  - ラベルディレクトリ: {labels_dir}")
        print(f"  - 有効なサンプル数: {len(self.valid_files)}")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        # 画像ファイル名
        img_file = self.valid_files[idx]

        # 画像を読み込み
        img_path = os.path.join(self.images_dir, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ラベルを読み込み
        label_file = os.path.splitext(img_file)[0] + '.png'
        label_path = os.path.join(self.labels_dir, label_file)
        mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        # マスクを0-1に正規化
        mask = (mask > 127).astype(np.float32)

        # リサイズ
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))
        mask = cv2.resize(mask, (self.image_size[1], self.image_size[0]))

        # データ拡張
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # 前処理
        if self.preprocessing:
            preprocessed = self.preprocessing(image=image, mask=mask)
            image = preprocessed['image']
            mask = preprocessed['mask']

        # マスクの次元を追加 (H, W) -> (1, H, W)
        mask = np.expand_dims(mask, axis=0)

        return image, mask


def get_training_augmentation():
    """
    学習用のデータ拡張

    夜間画像の検出率向上のため、明度変動を含む
    """
    transform = [
        # 水平・垂直反転
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # 回転
        A.Rotate(limit=config.ROTATE_LIMIT, p=0.5),

        # 明度・コントラスト調整（夜間画像対策）
        A.RandomBrightnessContrast(
            brightness_limit=config.BRIGHTNESS_LIMIT,
            contrast_limit=config.CONTRAST_LIMIT,
            p=0.5
        ),

        # ガウシアンノイズ
        A.GaussNoise(var_limit=config.GAUSSIAN_NOISE_VAR, p=0.3),

        # シフト・スケール・回転
        A.ShiftScaleRotate(
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=0,
            p=0.3
        ),
    ]
    return A.Compose(transform)


def get_validation_augmentation():
    """
    検証用の変換（データ拡張なし）
    """
    return None


def get_preprocessing():
    """
    前処理（正規化とテンソル変換）

    ImageNetの統計で正規化
    """
    transform = [
        A.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2(),
    ]
    return A.Compose(transform)


def create_dataloaders(images_dir, labels_dir, batch_size=8, num_workers=4,
                       train_val_split=0.8, random_seed=42):
    """
    学習用・検証用データローダーを作成

    Args:
        images_dir: 画像ディレクトリ
        labels_dir: ラベルディレクトリ
        batch_size: バッチサイズ
        num_workers: データローダーのワーカー数
        train_val_split: 学習データの割合（0-1）
        random_seed: ランダムシード

    Returns:
        train_loader: 学習用データローダー
        val_loader: 検証用データローダー
    """
    # 全データセットを作成
    full_dataset = PlanarianDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        augmentation=None,
        preprocessing=None
    )

    # データ数を取得
    dataset_size = len(full_dataset)
    train_size = int(dataset_size * train_val_split)
    val_size = dataset_size - train_size

    print(f"\nデータ分割:")
    print(f"  - 学習データ: {train_size} サンプル")
    print(f"  - 検証データ: {val_size} サンプル")

    # データセットを分割
    from torch.utils.data import random_split

    generator = torch.Generator().manual_seed(random_seed)
    train_indices, val_indices = random_split(
        range(dataset_size),
        [train_size, val_size],
        generator=generator
    )

    # 学習用データセット
    train_dataset = PlanarianDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing()
    )

    # 検証用データセット
    val_dataset = PlanarianDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_size=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH),
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing()
    )

    # インデックスでサブセットを作成
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, train_indices.indices)
    val_dataset = Subset(val_dataset, val_indices.indices)

    # データローダーを作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
