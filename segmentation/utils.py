"""
ユーティリティ関数群

学習・推論で共通して使用する関数
"""

import os
import json
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from datetime import datetime


def create_dirs_if_not_exist(dirs):
    """ディレクトリが存在しない場合は作成"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def save_json(data, path):
    """JSON形式でデータを保存"""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    """JSON形式でデータを読み込み"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_image_files(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
    """ディレクトリから画像ファイルのリストを取得（自然順ソート）"""
    import re

    def natural_key(s):
        """自然順ソート用のキー生成"""
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    files = []
    for f in os.listdir(directory):
        if f.lower().endswith(extensions):
            files.append(f)

    # 自然順でソート
    files.sort(key=natural_key)
    return files


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Dice係数を計算（評価指標）

    Args:
        pred: 予測マスク (0-1の範囲)
        target: 正解マスク (0 or 1)
        smooth: ゼロ除算回避用の微小値

    Returns:
        dice: Dice係数 (0-1の範囲、1が完全一致)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice


def iou_score(pred, target, smooth=1e-6):
    """
    IoU (Intersection over Union) を計算

    Args:
        pred: 予測マスク (0-1の範囲)
        target: 正解マスク (0 or 1)
        smooth: ゼロ除算回避用の微小値

    Returns:
        iou: IoUスコア (0-1の範囲、1が完全一致)
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou


def plot_training_history(history, save_path):
    """
    学習履歴をプロット

    Args:
        history: 学習履歴の辞書 {'train_loss': [...], 'val_loss': [...], ...}
        save_path: 保存先パス
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Dice Coefficient
    axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
    axes[1].plot(history['val_dice'], label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[1].set_title('Training and Validation Dice', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"学習履歴を保存しました: {save_path}")


def visualize_prediction(image, mask_true, mask_pred, save_path=None):
    """
    画像・正解マスク・予測マスクを可視化

    Args:
        image: 入力画像 (H, W, 3) または (3, H, W)
        mask_true: 正解マスク (H, W)
        mask_pred: 予測マスク (H, W)
        save_path: 保存先パス（Noneの場合は表示のみ）
    """
    # 画像の形状を統一 (H, W, 3)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # 正規化されている場合は元に戻す
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 元画像
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # 正解マスク
    axes[1].imshow(mask_true, cmap='gray')
    axes[1].set_title('Ground Truth Mask', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    # 予測マスク
    axes[2].imshow(mask_pred, cmap='gray')
    axes[2].set_title('Predicted Mask', fontsize=12, fontweight='bold')
    axes[2].axis('off')

    # オーバーレイ
    overlay = image.copy()
    mask_colored = np.zeros_like(overlay)
    mask_colored[:, :, 1] = mask_pred * 255  # 緑色
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Overlay', fontsize=12, fontweight='bold')
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def mask_to_contours(mask, min_area=100, max_area=50000):
    """
    マスクから輪郭を抽出し、形状特徴を計算

    Args:
        mask: バイナリマスク (H, W)
        min_area: 最小面積（これ以下は除外）
        max_area: 最大面積（これ以上は除外）

    Returns:
        results: 検出結果のリスト（辞書形式）
    """
    # マスクを uint8 に変換
    mask_uint8 = (mask * 255).astype(np.uint8)

    # 輪郭検出
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # 面積フィルタ
        if area < min_area or area > max_area:
            continue

        # モーメント計算
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue

        # 重心座標
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']

        # 長軸・短軸
        if len(cnt) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        else:
            MA, ma = 0, 0

        # 真円度
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        results.append({
            'centroid_x': cx,
            'centroid_y': cy,
            'major_axis': MA,
            'minor_axis': ma,
            'circularity': circularity,
            'area': area,
            'contour': cnt
        })

    return results


def timestamp():
    """現在時刻の文字列を返す（ログ用）"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
