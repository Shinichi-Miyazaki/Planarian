import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import re
import pandas as pd
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from datetime import datetime, time
import json

"""
プラナリア（または他の小動物）の自動検出・追跡システム

主な機能:
- 画像からの個体検出（二値化・輪郭検出）
- 形状測定（重心、面積、長軸・短軸、真円度）
- 時系列グラフの自動生成
- ROI（関心領域）設定による検出範囲の制限
- 夜間誤検出対策（相対閾値、ノイズフィルタ、モルフォロジー処理）
- インタラクティブなターゲット選択による半教師あり検出
"""

# =============================================================================
# 定数定義（マジックナンバーを避けるため、ここで一元管理）
# =============================================================================

# フレーム間隔（秒）- 画像は1枚あたり10秒で取得
FRAME_INTERVAL_SECONDS = 10

# デフォルトパラメータ
DEFAULT_MIN_AREA = 100
DEFAULT_MAX_AREA = 10000
DEFAULT_RELATIVE_THRESH = 0.15
DEFAULT_BLOCK_SIZE = 15
DEFAULT_C_VALUE = 8
DEFAULT_BG_KERNEL_SIZE = 31
DEFAULT_TEMPORAL_FRAMES = 100
DEFAULT_VIDEO_FPS = 10
DEFAULT_VIDEO_SCALE = 0.5

# GUI関連
FONT_SIZE_SMALL = 8
FONT_SIZE_NORMAL = 9
FONT_SIZE_LARGE = 10
FONT_SIZE_TITLE = 11
CANVAS_PREVIEW_WIDTH = 350
CANVAS_PREVIEW_HEIGHT = 250

# 検出パラメータ
DEFAULT_MIN_CIRCULARITY = 0.3
DEFAULT_CONTRAST_ALPHA = 1.5
DEFAULT_CLAHE_CLIP_LIMIT = 3.0
DEFAULT_EDGE_WEIGHT = 0.3

# 自動背景補正パラメータ
AUTO_BG_CORRECTION_RADIUS = 15  # 軽量化のため小さく（元は50）
AUTO_BG_GAUSSIAN_KERNEL = 51

# 内部均一性フィルタパラメータ（プラナリアとノイズの区別用）
# プラナリアは内部が均一な低輝度、ノイズは内部が不均一
DEFAULT_MIN_UNIFORMITY = 0.5  # 最小均一性（0.0-1.0、高いほど均一）
UNIFORMITY_SAMPLE_RATIO = 0.3  # 内部サンプリング比率

def create_temporal_median_background(image_paths, num_frames=100):
    """
    Temporal median filterを使って背景を作成

    Args:
        image_paths: 画像ファイルパスのリスト
        num_frames: 背景作成に使用するフレーム数

    Returns:
        background: 作成された背景画像（グレースケール）
    """
    if len(image_paths) < num_frames:
        num_frames = len(image_paths)

    # 等間隔にフレームを選択
    indices = np.linspace(0, len(image_paths) - 1, num_frames, dtype=int)
    selected_paths = [image_paths[i] for i in indices]

    images = []
    for path in selected_paths:
        img = cv2.imread(path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(gray)

    if not images:
        return None

    # すべての画像を3次元配列に変換
    image_stack = np.stack(images, axis=-1)

    # temporal medianを計算
    background = np.median(image_stack, axis=-1).astype(np.uint8)

    return background

def apply_temporal_median_subtraction(image, background):
    """
    画像から背景を引き算して前景を抽出

    Args:
        image: 元画像
        background: 背景画像

    Returns:
        foreground: 前景画像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 背景を引き算
def apply_temporal_median_subtraction(image, background):
    """
    画像から背景を引き算して前景を抽出

    Args:
        image: 元画像
        background: 背景画像

    Returns:
        foreground: 前景画像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 背景を引き算
    foreground = cv2.subtract(gray, background)

    # コントラストを強調
    foreground = cv2.addWeighted(foreground, 1.5, np.zeros_like(foreground), 0, 0)

    return foreground

def remove_background_simple(image, kernel_size=50):
    """
    軽量なバックグラウンド除去（ガウシアンブラー使用）

    Args:
        image: 入力画像（BGR形式）
        kernel_size: カーネルサイズ

    Returns:
        foreground: 前景画像（グレースケール）
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # カーネルサイズを奇数に調整
        if kernel_size % 2 == 0:
            kernel_size += 1

        # 最小値を3に設定
        kernel_size = max(3, kernel_size)

        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        foreground = cv2.subtract(gray, background)
        return foreground
    except Exception as e:
        print(f"remove_background_simple error: {e}")
        # エラー時はグレースケール画像をそのまま返す
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image


def auto_background_correction(image, method='adaptive'):
    """
    自動背景補正機能（プラナリア検出に最適化・高速化版）

    プラナリアの特性：
    - 背景に対して暗い物体
    - 辺縁が整っており、凸凹していない
    - 背景は不均一

    Args:
        image: 入力画像（BGR形式）
        method: 補正方法
            - 'adaptive': 適応的な背景補正（推奨・最速）
            - 'morphological': モルフォロジー法（高速）
            - 'rolling_ball': Rolling Ball法（やや重い）

    Returns:
        corrected_image: 背景補正後の画像（グレースケール、コントラスト強調済み）
        None: エラー時
    """
    try:
        if image is None or image.size == 0:
            return None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        if method == 'adaptive':
            # 適応的背景補正：ガウシアンブラーで高速に背景を推定
            # 大きなカーネルで背景を推定
            bg_large = cv2.GaussianBlur(gray, (AUTO_BG_GAUSSIAN_KERNEL, AUTO_BG_GAUSSIAN_KERNEL), 0)

            # 背景引き算（暗い物体を検出するため、背景から元画像を引く）
            # プラナリアは暗いので、背景 - 元画像 で明るくなる
            corrected = cv2.subtract(bg_large, gray)

            # コントラストを正規化
            corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)

        elif method == 'rolling_ball':
            # Rolling Ball法（OpenCVのモルフォロジー処理で軽量化）
            # Black-hatを使用（暗い物体検出用）
            kernel_size = AUTO_BG_CORRECTION_RADIUS * 2 + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            # プラナリアは暗いので、black_tophat（= closing - 元画像）を使用
            corrected = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        elif method == 'morphological':
            # モルフォロジー法：Opening操作で背景を推定（高速版）
            kernel_size = AUTO_BG_CORRECTION_RADIUS * 2 + 1
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            bg_morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            corrected = cv2.subtract(bg_morph, gray)
            corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
        else:
            corrected = gray

        return corrected.astype(np.uint8)

    except Exception as e:
        print(f"auto_background_correction error: {e}")
        return None

def enhance_contrast(image, use_contrast=False, alpha=1.5, beta=0, use_clahe=False, clip_limit=3.0):
    """
    コントラスト調整関数

    Args:
        image: 入力画像（BGR形式）
        use_contrast: 基本的なコントラスト調整を使用するかどうか
        alpha: コントラスト倍率（1.0=変化なし、>1.0で強化）
        beta: 明度調整（-100～100程度）
        use_clahe: CLAHE（局所適応ヒストグラム平坦化）を使用するかどうか
        clip_limit: CLAHEのクリップ制限

    Returns:
        enhanced_image: コントラスト調整後の画像
    """
    if not use_contrast and not use_clahe:
        return image

    # グレースケールに変換
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    enhanced = gray.copy()

    # 基本的なコントラスト・明度調整
    if use_contrast:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)

    # CLAHE（局所適応ヒストグラム平坦化）
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(enhanced)

    # 3チャンネルに戻す
    if len(image.shape) == 3:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return enhanced

def apply_roi_mask(image, roi_coordinates):
    """
    ROI外の領域をマスクして黒くする（円形マスクを作成）

    Args:
        image: 入力画像
        roi_coordinates: (center_x, center_y, radius) のタプル

    Returns:
        masked_image: マスク適用後の画像
    """
    if roi_coordinates is None:
        return image

    center_x, center_y, radius = roi_coordinates

    # マスクを作成（円形）
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (int(center_x), int(center_y)), int(radius), 255, -1)

    # マスクを適用
    if len(image.shape) == 3:
        # カラー画像の場合
        masked_image = cv2.bitwise_and(image, image, mask=mask)
    else:
        # グレースケール画像の場合
        masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

def is_nighttime_by_brightness(image, brightness_threshold=50):
    """
    画像の明度から夜間かどうかを判定

    Args:
        image: 入力画像
        brightness_threshold: 夜間判定の明度閾値（この値以下なら夜間）

    Returns:
        bool: 夜間の場合True
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    mean_brightness = np.mean(gray)
    return mean_brightness <= brightness_threshold

# --- 画像解析関数 ---

def calculate_internal_uniformity(gray_image, contour, sample_ratio=UNIFORMITY_SAMPLE_RATIO):
    """
    輪郭内部の均一性を計算（プラナリアとノイズの区別用）

    プラナリアの特性：
    - 内部が均一な低輝度
    - 辺縁が整っている

    ノイズの特性：
    - 内部が不均一
    - 辺縁が不規則

    Args:
        gray_image: グレースケール画像
        contour: 輪郭
        sample_ratio: 内部サンプリング比率（0.0-1.0）

    Returns:
        uniformity: 均一性スコア（0.0-1.0、高いほど均一）
                   -1.0: 計算不可
    """
    try:
        # 輪郭のバウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(contour)

        if w < 3 or h < 3:
            return -1.0

        # マスクを作成
        mask = np.zeros(gray_image.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # 内部のピクセル値を取得
        roi = gray_image[y:y+h, x:x+w]
        roi_mask = mask[y:y+h, x:x+w]

        internal_pixels = roi[roi_mask > 0]

        if len(internal_pixels) < 5:
            return -1.0

        # 均一性の計算：標準偏差が小さいほど均一
        mean_val = np.mean(internal_pixels)
        std_val = np.std(internal_pixels)

        # 変動係数（CV）を計算：std/mean
        # CVが小さいほど均一
        if mean_val > 0:
            cv_value = std_val / mean_val
        else:
            cv_value = 1.0

        # 均一性スコアに変換（0-1、高いほど均一）
        # CV=0で均一性=1、CV>=1で均一性=0
        uniformity = max(0.0, 1.0 - cv_value)

        return uniformity

    except Exception as e:
        print(f"calculate_internal_uniformity error: {e}")
        return -1.0


def is_daytime(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > threshold


def calc_threshold_by_histogram(image, std_factor=2.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    std = np.std(gray)
    thresh = int(mean + std_factor * std)
    return thresh

# binarize関数のthresh引数を自動計算に変更
# std_factorは昼夜で別々に指定可能にする

def enhance_detection_with_edges(gray_image, use_edge_enhancement=False, edge_weight=0.3):
    """
    エッジ検出を用いて個体の輪郭を強調

    Args:
        gray_image: グレースケール画像
        use_edge_enhancement: エッジ強調を使用するか
        edge_weight: エッジの重み（0.0-1.0）

    Returns:
        enhanced_image: エッジ強調後の画像
    """
    if not use_edge_enhancement:
        return gray_image

    # Sobelフィルタでエッジ検出
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    # エッジの強度を計算
    edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
    edge_magnitude = np.uint8(edge_magnitude / edge_magnitude.max() * 255)

    # 元画像とエッジを合成
    enhanced = cv2.addWeighted(gray_image, 1.0, edge_magnitude, edge_weight, 0)

    return enhanced

def adaptive_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    適応的ヒストグラム平坦化（CLAHE）を適用

    Args:
        image: 入力画像
        clip_limit: クリップ制限
        tile_grid_size: タイルグリッドサイズ

    Returns:
        equalized_image: 平坦化後の画像
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized = clahe.apply(gray)

    if len(image.shape) == 3:
        equalized = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    return equalized

def binarize(image, method='adaptive', relative_thresh=0.1, block_size=11, c_value=2, use_bg_removal=False, bg_kernel_size=50, roi_coordinates=None, auto_night_contrast=False, night_brightness_threshold=50, use_edge_enhancement=False, edge_weight=0.3):
    """
    統一的な二値化関数（ROIマスクと夜間自動コントラスト調整対応）

    Args:
        method: 'adaptive' (適応的), 'relative' (相対閾値), 'fixed' (固定閾値)
        relative_thresh: 相対閾値方式での閾値（平均輝度からの割合, 0.0-1.0）
        block_size: 適応的二値化のブロックサイズ（奇数）
        c_value: 適応的二値化の定数C（平均から引く値）
        use_bg_removal: バックグラウンド除去を使用するか
        bg_kernel_size: バックグラウンド除去のカーネルサイズ
        roi_coordinates: ROI座標 (center_x, center_y, radius)
        auto_night_contrast: 夜間自動コントラスト調整を使用するか
        night_brightness_threshold: 夜間判定の明度閾値

    詳細説明:
        - block_size: 適応的二値化で使用する近傍領域のサイズ。大きくすると広い範囲の平均を使用
        - c_value: 適応的二値化の定数C。この値が大きいほど暗い部分が白になりにくい
        - relative_thresh: 相対閾値での閾値係数。0.1なら平均輝度の90%を閾値とする
    """
    # ROIマスクを適用（ROI座標が指定されている場合）
    if roi_coordinates is not None:
        image = apply_roi_mask(image, roi_coordinates)

    # 夜間自動判定とコントラスト調整
    if auto_night_contrast and is_nighttime_by_brightness(image, night_brightness_threshold):
        # 夜間と判定された場合のみコントラスト調整を適用
        image = enhance_contrast(image, use_contrast=True, alpha=1.5, beta=10, use_clahe=True, clip_limit=3.0)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ノイズ軽減のためにガウシアンブラーを適用（夜間の誤検出対策）
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # エッジ強調（オプション）
    if use_edge_enhancement:
        gray = enhance_detection_with_edges(gray, use_edge_enhancement=True, edge_weight=edge_weight)

    # バックグラウンド除去（オプション）
    if use_bg_removal:
        gray = remove_background_simple(image, bg_kernel_size)

    if method == 'adaptive':
        # 適応的二値化（動物が暗い場合はTHRESH_BINARY_INVを使用）
        # block_sizeは奇数である必要がある
        if block_size % 2 == 0:
            block_size += 1
        block_size = max(3, block_size)  # 最小値は3

        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block_size, c_value)
    elif method == 'relative':
        # 相対閾値方式（動物が暗い場合は反転）
        mean_val = np.mean(gray)
        thresh = int(mean_val * (1 - relative_thresh))  # 平均より下を閾値に
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    else:  # 'fixed'
        # 従来の固定閾値方式（反転）
        thresh = int(relative_thresh * 255) if relative_thresh <= 1.0 else int(relative_thresh)
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)

    # モルフォロジー処理でノイズを除去（夜間の誤検出対策）
    # Opening: 小さなノイズを除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    # Closing: 小さな穴を埋める
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 二値化後にも ROI マスクを適用して、ROI 外を確実に 0 にする
    if roi_coordinates is not None:
        binary = apply_roi_mask(binary, roi_coordinates)

    return binary

def analyze(binary, min_area=100, max_area=10000, select_center_only=True, select_largest=False, scale_factor=1.0, roi_offset_x=0, roi_offset_y=0, min_circularity=0.3, original_gray=None, min_uniformity=0.0):
    """
    個体検出・解析関数（改良版・内部均一性フィルタ対応）

    Args:
        select_largest: Trueの場合、最大面積かつ最も中心に近い個体を選択
        select_center_only: Trueの場合、中心に最も近い個体を選択（select_largestがFalseの時のみ）
        min_circularity: 最小真円度（0.0-1.0）、これ以下のオブジェクトは除外（ノイズフィルタ）
        original_gray: 元のグレースケール画像（内部均一性計算用、Noneの場合はスキップ）
        min_uniformity: 最小内部均一性（0.0-1.0）、これ以下のオブジェクトは除外
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    valid_contours = []

    # 画像の中心座標を取得
    image_height, image_width = binary.shape
    center_x, center_y = image_width // 2, image_height // 2

    # スケール比を考慮した面積閾値を計算
    scaled_min_area = min_area * (scale_factor ** 2)
    scaled_max_area = max_area * (scale_factor ** 2)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < scaled_min_area or area > scaled_max_area:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        # 中心からの距離を計算
        distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)

        if len(cnt) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
        else:
            MA, ma = 0, 0

        # 真円度を計算（ノイズフィルタ用）
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            circularity = 0

        # 真円度フィルタ: 細長いノイズや不規則な形状を除外
        if circularity < min_circularity:
            continue

        # 内部均一性フィルタ: プラナリアは内部が均一、ノイズは不均一
        uniformity = 1.0  # デフォルト（フィルタなし）
        if original_gray is not None and min_uniformity > 0:
            uniformity = calculate_internal_uniformity(original_gray, cnt)
            if uniformity >= 0 and uniformity < min_uniformity:
                continue  # 均一性が低い（ノイズ）場合はスキップ

        # 実際の画像サイズに換算した値を保存（ROIオフセットを加算）
        actual_area = area / (scale_factor ** 2)
        actual_major_axis = ma / scale_factor if ma > 0 else 0
        actual_minor_axis = MA / scale_factor if MA > 0 else 0
        actual_cx = (cx / scale_factor) + roi_offset_x
        actual_cy = (cy / scale_factor) + roi_offset_y

        candidate = {
            'centroid_x': actual_cx,
            'centroid_y': actual_cy,
            'major_axis': actual_major_axis,
            'minor_axis': actual_minor_axis,
            'circularity': circularity,
            'uniformity': uniformity if uniformity >= 0 else 1.0,  # 内部均一性
            'area': actual_area,
            'distance_from_center': distance_from_center,
            'contour': cnt
        }
        candidates.append(candidate)

    if candidates:
        if select_largest:
            # 最大面積かつ最も中心に近い個体を選択
            # まず面積でソートして上位20%の候補を取得
            candidates_sorted_by_area = sorted(candidates, key=lambda x: x['area'], reverse=True)
            top_area_count = max(1, len(candidates_sorted_by_area) // 5)  # 上位20%、最低1個
            top_area_candidates = candidates_sorted_by_area[:top_area_count]

            # 上位面積の候補の中から最も中心に近い個体を選択
            best_candidate = min(top_area_candidates, key=lambda x: x['distance_from_center'])
            results.append({k: v for k, v in best_candidate.items() if k != 'contour'})
            valid_contours.append(best_candidate['contour'])
        elif select_center_only:
            # 中心に最も近い個体を選択
            closest_candidate = min(candidates, key=lambda x: x['distance_from_center'])
            results.append({k: v for k, v in closest_candidate.items() if k != 'contour'})
            valid_contours.append(closest_candidate['contour'])
        else:
            # すべての候補を返す
            for candidate in candidates:
                results.append({k: v for k, v in candidate.items() if k != 'contour'})
                valid_contours.append(candidate['contour'])

    return results, valid_contours

class AnimalDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('動物検出GUI')
        self.root.geometry('1200x500')  # サイズをコンパクトに調整

        # ウィンドウを最大化可能にする（OS別に安全に処理）
        try:
            ws = self.root.tk.call('tk', 'windowingsystem')
            if ws == 'win32':
                self.root.state('zoomed')
            elif ws == 'x11':
                self.root.attributes('-zoomed', True)
            else:
                # macOS (aqua) は -zoomed 未対応のため、画面サイズに合わせる
                self.root.update_idletasks()
                w = self.root.winfo_screenwidth()
                h = self.root.winfo_screenheight()
                self.root.geometry(f"{w}x{h}+0+0")
        except Exception:
            # 失敗しても通常サイズで続行
            pass

        # 変数
        self.folder_path = tk.StringVar()
        self.day_img_path = tk.StringVar()
        self.night_img_path = tk.StringVar()
        self.min_area = tk.StringVar(value=str(DEFAULT_MIN_AREA))
        self.max_area = tk.StringVar(value=str(DEFAULT_MAX_AREA))

        # 新しい統一的な二値化パラメータ（暗い動物検出に最適化）
        self.binarize_method = tk.StringVar(value='adaptive')  # 'adaptive', 'relative', 'fixed'
        self.relative_thresh = tk.DoubleVar(value=DEFAULT_RELATIVE_THRESH)
        self.block_size = tk.IntVar(value=DEFAULT_BLOCK_SIZE)
        self.c_value = tk.IntVar(value=DEFAULT_C_VALUE)
        self.use_bg_removal = tk.BooleanVar(value=False)
        self.bg_kernel_size = tk.IntVar(value=DEFAULT_BG_KERNEL_SIZE)

        # 自動背景補正設定（新機能）
        self.use_auto_bg_correction = tk.BooleanVar(value=False)
        self.auto_bg_method = tk.StringVar(value='adaptive')  # 'adaptive', 'rolling_ball', 'morphological'

        # Temporal Median Filter設定
        self.use_temporal_median = tk.BooleanVar(value=False)
        self.temporal_frames = tk.IntVar(value=DEFAULT_TEMPORAL_FRAMES)
        self.temporal_background = None

        # 動画保存設定
        self.save_video = tk.BooleanVar(value=False)
        self.video_fps = tk.IntVar(value=DEFAULT_VIDEO_FPS)
        self.video_scale = tk.DoubleVar(value=DEFAULT_VIDEO_SCALE)

        # 時間設定
        self.day_start_time = tk.StringVar(value='07:00')
        self.night_start_time = tk.StringVar(value='19:00')

        # 測定開始時刻設定
        self.measurement_start_time = tk.StringVar(value='09:00:00')

        # フレーム間隔設定（秒）- 設定可能に
        self.frame_interval = tk.IntVar(value=FRAME_INTERVAL_SECONDS)

        # 測定開始日付設定
        self.measurement_date = tk.StringVar(value='2025-01-01')

        # ROI関連
        self.roi_coordinates = None
        self.roi_active = tk.BooleanVar(value=False)
        self.drawing_roi = False
        self.roi_start_x = 0
        self.roi_start_y = 0

        # ターゲット選択機能（インタラクティブ学習用）
        self.target_selecting = False
        self.target_info = None  # 選択されたターゲットの情報を保存
        self.learned_brightness = None  # 学習した輝度値
        self.learned_area_range = None  # 学習した面積範囲
        self.target_canvas = None  # ターゲット選択用のキャンバス
        self.target_image_path = None  # ターゲット選択用の画像パス

        # 個体選択方法設定を追加
        self.select_largest = tk.BooleanVar(value=True)
        self.constant_darkness = tk.BooleanVar(value=False)

        # 内部均一性フィルタ設定（プラナリアとノイズの区別用）
        self.use_uniformity_filter = tk.BooleanVar(value=False)
        self.min_uniformity = tk.DoubleVar(value=DEFAULT_MIN_UNIFORMITY)

        # コントラスト調整設定を追加
        self.use_contrast_enhancement = tk.BooleanVar(value=False)
        self.contrast_alpha = tk.DoubleVar(value=DEFAULT_CONTRAST_ALPHA)
        self.brightness_beta = tk.IntVar(value=0)  # 明度調整（-100～100程度）
        self.use_clahe = tk.BooleanVar(value=False)  # CLAHE（局所適応ヒストグラム平坦化）の使用
        self.clahe_clip_limit = tk.DoubleVar(value=3.0)  # CLAHEのクリップ制限

        # エッジ強調設定を追加
        self.use_edge_enhancement = tk.BooleanVar(value=False)  # エッジ強調の使用
        self.edge_weight = tk.DoubleVar(value=0.3)  # エッジの重み（0.0-1.0）

        self.create_widgets()

    def create_widgets(self):
        # メインフレーム
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左側：設定パネル
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # フォルダ選択（1行に統合）
        folder_frame = tk.Frame(left_frame)
        folder_frame.grid(row=0, column=0, columnspan=2, pady=1, sticky='ew')
        tk.Label(folder_frame, text='フォルダ:', font=('Arial', 7, 'bold')).pack(side=tk.LEFT, padx=(0,2))
        tk.Entry(folder_frame, textvariable=self.folder_path, width=28, font=('Arial', 7)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(folder_frame, text='選択', command=self.select_folder, font=('Arial', 7), width=5).pack(side=tk.RIGHT, padx=(2,0))

        # 昼・夜の代表画像（1行に統合）
        images_frame = tk.Frame(left_frame)
        images_frame.grid(row=1, column=0, columnspan=2, pady=1, sticky='ew')

        # 昼の代表画像
        tk.Label(images_frame, text='昼:', font=('Arial', 7, 'bold')).pack(side=tk.LEFT, padx=(0,1))
        tk.Entry(images_frame, textvariable=self.day_img_path, width=13, font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Button(images_frame, text='選択', command=self.select_day_image, font=('Arial', 7), width=4).pack(side=tk.LEFT, padx=(1,3))

        # 夜の代表画像
        tk.Label(images_frame, text='夜:', font=('Arial', 7, 'bold')).pack(side=tk.LEFT, padx=(0,1))
        tk.Entry(images_frame, textvariable=self.night_img_path, width=13, font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Button(images_frame, text='選択', command=self.select_night_image, font=('Arial', 7), width=4).pack(side=tk.LEFT, padx=1)

        # 個体選択方法設定
        selection_frame = tk.LabelFrame(left_frame, text='検出設定', font=('Arial', 7, 'bold'))
        selection_frame.grid(row=2, column=0, columnspan=2, pady=2, padx=5, sticky='ew')

        selection_row = tk.Frame(selection_frame)
        selection_row.pack(pady=1)
        tk.Checkbutton(selection_row, text='Largest', variable=self.select_largest, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Checkbutton(selection_row, text='Const.Dark', variable=self.constant_darkness,
                      command=self.toggle_constant_darkness, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=(5,1))

        # 内部均一性フィルタ（プラナリアとノイズの区別用）
        selection_row2 = tk.Frame(selection_frame)
        selection_row2.pack(pady=1)
        tk.Checkbutton(selection_row2, text='均一性', variable=self.use_uniformity_filter,
                      command=self.update_preview, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Entry(selection_row2, textvariable=self.min_uniformity, width=4, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Label(selection_row2, text='(0-1)', font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)

        # パラメータ設定
        param_frame = tk.LabelFrame(left_frame, text='面積範囲', font=('Arial', FONT_SIZE_SMALL, 'bold'))
        param_frame.grid(row=3, column=0, columnspan=2, pady=2, padx=5, sticky='ew')

        size_frame = tk.Frame(param_frame)
        size_frame.pack(pady=1)
        tk.Label(size_frame, text='最小:', font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Entry(size_frame, textvariable=self.min_area, width=6, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT)
        tk.Label(size_frame, text='最大:', font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=(5,1))
        tk.Entry(size_frame, textvariable=self.max_area, width=6, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT)
        tk.Button(size_frame, text='更新', command=self.update_preview,
                 bg='lightblue', font=('Arial', FONT_SIZE_SMALL), width=6).pack(side=tk.LEFT, padx=(5,0))

        # 時間設定
        time_frame = tk.LabelFrame(left_frame, text='時間', font=('Arial', FONT_SIZE_SMALL, 'bold'))
        time_frame.grid(row=4, column=0, columnspan=2, pady=2, padx=5, sticky='ew')

        time_entry_frame = tk.Frame(time_frame)
        time_entry_frame.pack(pady=1)
        tk.Label(time_entry_frame, text='昼:', font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Entry(time_entry_frame, textvariable=self.day_start_time, width=5, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Label(time_entry_frame, text='夜:', font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=(3,1))
        tk.Entry(time_entry_frame, textvariable=self.night_start_time, width=5, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)

        time_entry_frame2 = tk.Frame(time_frame)
        time_entry_frame2.pack(pady=1)
        tk.Label(time_entry_frame2, text='測定開始:', font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Entry(time_entry_frame2, textvariable=self.measurement_start_time, width=7, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Label(time_entry_frame2, text='間隔(秒):', font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=(3,1))
        tk.Entry(time_entry_frame2, textvariable=self.frame_interval, width=4, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)

        # Temporal Median Filter設定
        temporal_frame = tk.LabelFrame(left_frame, text='背景除去', font=('Arial', FONT_SIZE_SMALL, 'bold'))
        temporal_frame.grid(row=5, column=0, columnspan=2, pady=2, padx=5, sticky='ew')

        temporal_row = tk.Frame(temporal_frame)
        temporal_row.pack(pady=1)
        tk.Checkbutton(temporal_row, text='Temporal', variable=self.use_temporal_median, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Label(temporal_row, text='F:', font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=(3,1))
        tk.Entry(temporal_row, textvariable=self.temporal_frames, width=4, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Button(temporal_row, text='作成', command=self.create_background,
                 bg='lightyellow', font=('Arial', FONT_SIZE_SMALL), width=5).pack(side=tk.LEFT, padx=(3,0))
        self.temporal_status_label = tk.Label(temporal_row, text='未作成', font=('Arial', FONT_SIZE_SMALL))
        self.temporal_status_label.pack(side=tk.LEFT, padx=(3,0))

        # 自動背景補正設定（新機能）
        auto_bg_row = tk.Frame(temporal_frame)
        auto_bg_row.pack(pady=1)
        tk.Checkbutton(auto_bg_row, text='自動補正', variable=self.use_auto_bg_correction,
                      command=self.update_preview, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        ttk.Combobox(auto_bg_row, textvariable=self.auto_bg_method,
                    values=['adaptive', 'rolling_ball', 'morphological'],
                    state='readonly', width=10, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)

        # 動画保存設定
        video_frame = tk.LabelFrame(left_frame, text='動画', font=('Arial', FONT_SIZE_SMALL, 'bold'))
        video_frame.grid(row=6, column=0, columnspan=2, pady=2, padx=5, sticky='ew')

        video_row = tk.Frame(video_frame)
        video_row.pack(pady=1)
        tk.Checkbutton(video_row, text='保存', variable=self.save_video, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Label(video_row, text='FPS:', font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=(3,1))
        tk.Entry(video_row, textvariable=self.video_fps, width=3, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Label(video_row, text='Scale:', font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=(3,1))
        tk.Entry(video_row, textvariable=self.video_scale, width=4, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)

        # 二値化設定
        binarize_frame = tk.LabelFrame(left_frame, text='二値化', font=('Arial', FONT_SIZE_SMALL, 'bold'))
        binarize_frame.grid(row=7, column=0, columnspan=2, pady=2, padx=5, sticky='ew')

        # 2行に分けてコンパクトに
        tk.Label(binarize_frame, text='方式:', font=('Arial', FONT_SIZE_SMALL)).grid(row=0, column=0, padx=1, pady=1, sticky='w')
        ttk.Combobox(binarize_frame, textvariable=self.binarize_method, values=['adaptive', 'relative', 'fixed'], state='readonly', width=7, font=('Arial', FONT_SIZE_SMALL)).grid(row=0, column=1, padx=1, pady=1)
        tk.Label(binarize_frame, text='C:', font=('Arial', FONT_SIZE_SMALL)).grid(row=0, column=2, padx=(3,1), pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.c_value, width=4, font=('Arial', FONT_SIZE_SMALL)).grid(row=0, column=3, padx=1, pady=1)
        tk.Label(binarize_frame, text='BG:', font=('Arial', FONT_SIZE_SMALL)).grid(row=0, column=4, padx=(3,1), pady=1, sticky='w')
        ttk.Combobox(binarize_frame, textvariable=self.use_bg_removal, values=[True, False], state='readonly', width=5, font=('Arial', FONT_SIZE_SMALL)).grid(row=0, column=5, padx=1, pady=1)

        tk.Label(binarize_frame, text='閾値:', font=('Arial', FONT_SIZE_SMALL)).grid(row=1, column=0, padx=1, pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.relative_thresh, width=5, font=('Arial', FONT_SIZE_SMALL)).grid(row=1, column=1, padx=1, pady=1)
        tk.Label(binarize_frame, text='Block:', font=('Arial', FONT_SIZE_SMALL)).grid(row=1, column=2, padx=(3,1), pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.block_size, width=4, font=('Arial', FONT_SIZE_SMALL)).grid(row=1, column=3, padx=1, pady=1)
        tk.Label(binarize_frame, text='Kernel:', font=('Arial', FONT_SIZE_SMALL)).grid(row=1, column=4, padx=(3,1), pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.bg_kernel_size, width=4, font=('Arial', FONT_SIZE_SMALL)).grid(row=1, column=5, padx=1, pady=1)

        # ROI設定
        roi_frame = tk.LabelFrame(left_frame, text='ROI', font=('Arial', FONT_SIZE_SMALL, 'bold'))
        roi_frame.grid(row=8, column=0, columnspan=2, pady=2, padx=5, sticky='ew')

        roi_control_frame = tk.Frame(roi_frame)
        roi_control_frame.pack(pady=1)
        tk.Checkbutton(roi_control_frame, text='使用', variable=self.roi_active,
                      command=self.toggle_roi, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Button(roi_control_frame, text='設定', command=self.set_roi_mode,
                 bg='orange', font=('Arial', FONT_SIZE_SMALL), width=5).pack(side=tk.LEFT, padx=2)
        tk.Button(roi_control_frame, text='クリア', command=self.clear_roi,
                 bg='lightgray', font=('Arial', FONT_SIZE_SMALL), width=5).pack(side=tk.LEFT, padx=1)
        self.roi_info_label = tk.Label(roi_control_frame, text='未設定', font=('Arial', FONT_SIZE_SMALL))
        self.roi_info_label.pack(side=tk.LEFT, padx=(3,0))

        # ターゲット選択機能（インタラクティブ学習）
        target_frame = tk.LabelFrame(left_frame, text='ターゲット選択', font=('Arial', FONT_SIZE_SMALL, 'bold'))
        target_frame.grid(row=9, column=0, columnspan=2, pady=2, padx=5, sticky='ew')

        target_control_frame = tk.Frame(target_frame)
        target_control_frame.pack(pady=1)
        tk.Button(target_control_frame, text='選択', command=self.start_target_selection,
                 bg='#90EE90', font=('Arial', FONT_SIZE_SMALL), width=5).pack(side=tk.LEFT, padx=1)
        tk.Button(target_control_frame, text='適用', command=self.apply_learned_params,
                 bg='#87CEEB', font=('Arial', FONT_SIZE_SMALL), width=5).pack(side=tk.LEFT, padx=1)
        tk.Button(target_control_frame, text='クリア', command=self.clear_target,
                 bg='lightgray', font=('Arial', FONT_SIZE_SMALL), width=5).pack(side=tk.LEFT, padx=1)
        self.target_info_label = tk.Label(target_control_frame, text='未選択', font=('Arial', FONT_SIZE_SMALL))
        self.target_info_label.pack(side=tk.LEFT, padx=(3,0))

        # コントラスト調整設定
        contrast_frame = tk.LabelFrame(left_frame, text='画像強調', font=('Arial', FONT_SIZE_SMALL, 'bold'))
        contrast_frame.grid(row=10, column=0, columnspan=2, pady=2, padx=5, sticky='ew')

        # 1行目
        contrast_row1 = tk.Frame(contrast_frame)
        contrast_row1.pack(pady=1)
        tk.Checkbutton(contrast_row1, text='Cont', variable=self.use_contrast_enhancement,
                      command=self.update_preview, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Entry(contrast_row1, textvariable=self.contrast_alpha, width=3, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Checkbutton(contrast_row1, text='CLAHE', variable=self.use_clahe,
                      command=self.update_preview, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=(3,1))
        tk.Entry(contrast_row1, textvariable=self.clahe_clip_limit, width=3, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)
        tk.Checkbutton(contrast_row1, text='Edge', variable=self.use_edge_enhancement,
                      command=self.update_preview, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=(3,1))
        tk.Entry(contrast_row1, textvariable=self.edge_weight, width=3, font=('Arial', FONT_SIZE_SMALL)).pack(side=tk.LEFT, padx=1)

        # メインボタン
        button_frame = tk.Frame(left_frame)
        button_frame.grid(row=11, column=0, columnspan=2, pady=2)

        # 設定保存・ロードボタン
        config_frame = tk.Frame(button_frame)
        config_frame.pack(pady=1)
        tk.Button(config_frame, text='保存', command=self.save_config,
                 bg='lightblue', font=('Arial', FONT_SIZE_SMALL), width=7).pack(side=tk.LEFT, padx=1)
        tk.Button(config_frame, text='ロード', command=self.load_config,
                 bg='lightyellow', font=('Arial', FONT_SIZE_SMALL), width=7).pack(side=tk.LEFT, padx=1)
        tk.Button(config_frame, text='更新', command=self.update_preview,
                 bg='lightcyan', font=('Arial', FONT_SIZE_SMALL), width=7).pack(side=tk.LEFT, padx=1)

        tk.Button(button_frame, text='解析開始', command=self.start_analysis,
                 bg='lightgreen', font=('Arial', FONT_SIZE_NORMAL, 'bold'), width=22).pack(pady=2)
        tk.Button(button_frame, text='終了', command=self.root.quit,
                 bg='lightcoral', font=('Arial', FONT_SIZE_SMALL), width=22).pack(pady=1)

        # 右側：プレビューパネル
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # プレビューエリア
        preview_frame = tk.LabelFrame(right_frame, text='検出プレビュー', font=('Arial', 9, 'bold'))
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        # タブ
        self.notebook = ttk.Notebook(preview_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.day_frame = tk.Frame(self.notebook)
        self.night_frame = tk.Frame(self.notebook)
        self.bg_preview_frame = tk.Frame(self.notebook)  # 背景プレビュー用タブを追加
        self.notebook.add(self.day_frame, text='昼画像プレビュー')
        self.notebook.add(self.night_frame, text='夜画像プレビュー')
        self.notebook.add(self.bg_preview_frame, text='背景プレビュー')

        # 昼画像プレビュー
        self.day_canvas = tk.Canvas(self.day_frame, bg='white', width=300, height=200)
        self.day_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 夜画像プレビュー
        self.night_canvas = tk.Canvas(self.night_frame, bg='white', width=300, height=200)
        self.night_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 背景プレビュー（Temporal Median Filter効果確認用）
        bg_preview_control_frame = tk.Frame(self.bg_preview_frame)
        bg_preview_control_frame.pack(pady=2)
        tk.Button(bg_preview_control_frame, text='元画像', command=lambda: self.show_bg_preview('original'),
                 bg='lightblue', font=('Arial', 8), width=8).pack(side=tk.LEFT, padx=1)
        tk.Button(bg_preview_control_frame, text='背景画像', command=lambda: self.show_bg_preview('background'),
                 bg='lightyellow', font=('Arial', 8), width=8).pack(side=tk.LEFT, padx=1)
        tk.Button(bg_preview_control_frame, text='減算結果', command=lambda: self.show_bg_preview('subtracted'),
                 bg='lightgreen', font=('Arial', 8), width=8).pack(side=tk.LEFT, padx=1)

        self.bg_canvas = tk.Canvas(self.bg_preview_frame, bg='white', width=300, height=200)
        self.bg_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 出力テキストエリア
        output_frame = tk.LabelFrame(right_frame, text='解析結果', font=('Arial', 9, 'bold'))
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(output_frame, width=50, height=8, font=('Arial', 8))
        self.output_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def select_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path.set(folder)

    def select_day_image(self):
        filename = filedialog.askopenfilename(
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp')]
        )
        if filename:
            self.day_img_path.set(filename)
            self.update_preview()

    def select_night_image(self):
        filename = filedialog.askopenfilename(
            filetypes=[('Image files', '*.jpg *.jpeg *.png *.bmp')]
        )
        if filename:
            self.night_img_path.set(filename)
            self.update_preview()

    def resize_image_for_canvas(self, image, max_width=300, max_height=200):
        h, w = image.shape[:2]
        scale = min(max_width/w, max_height/h)
        new_w, new_h = int(w*scale), int(h*scale)
        return cv2.resize(image, (new_w, new_h))

    def update_preview(self):
        try:
            min_area = int(self.min_area.get())
            max_area = int(self.max_area.get())
        except ValueError:
            return

        # 昼画像プレビュー
        if self.day_img_path.get() and os.path.exists(self.day_img_path.get()):
            day_img = cv2.imread(self.day_img_path.get())
            if day_img is not None:
                self.show_preview(day_img, min_area, max_area, self.day_canvas, "昼")

        # 夜画像プレビュー
        if self.night_img_path.get() and os.path.exists(self.night_img_path.get()):
            night_img = cv2.imread(self.night_img_path.get())
            if night_img is not None:
                self.show_preview(night_img, min_area, max_area, self.night_canvas, "夜")

    def show_preview(self, image, min_area, max_area, canvas, time_type):
        try:
            # 入力画像のバリデーション
            if image is None or image.size == 0:
                print(f"show_preview: Invalid image for {time_type}")
                return

            # 元画像のサイズを取得
            original_h, original_w = image.shape[:2]

            # ROIが有効な場合は対象領域を切り出し
            roi_img = image.copy()
            roi_offset_x, roi_offset_y = 0, 0
            if self.roi_active.get() and self.roi_coordinates:
                try:
                    # 円形ROI座標（center_x, center_y, radius）を矩形座標に変換
                    center_x, center_y, radius = self.roi_coordinates
                    x1 = max(0, int(center_x - radius))
                    y1 = max(0, int(center_y - radius))
                    x2 = min(image.shape[1], int(center_x + radius))
                    y2 = min(image.shape[0], int(center_y + radius))
                    if x2 > x1 and y2 > y1:  # 有効なROIかチェック
                        roi_img = image[y1:y2, x1:x2].copy()
                        roi_offset_x, roi_offset_y = x1, y1
                except Exception as e:
                    print(f"ROI extraction error: {e}")
                    roi_img = image.copy()

            # 自動背景補正を適用（新機能）
            if self.use_auto_bg_correction.get():
                try:
                    corrected = auto_background_correction(roi_img, method=self.auto_bg_method.get())
                    # Noneチェック
                    if corrected is not None:
                        roi_img = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
                except Exception as e:
                    print(f"Background correction error: {e}")

            # コントラスト調整を適用
            if self.use_contrast_enhancement.get() or self.use_clahe.get():
                try:
                    alpha = float(self.contrast_alpha.get())
                    beta = int(self.brightness_beta.get())
                    clip_limit = float(self.clahe_clip_limit.get())
                    roi_img = enhance_contrast(roi_img,
                                             use_contrast=self.use_contrast_enhancement.get(),
                                             alpha=alpha, beta=beta,
                                             use_clahe=self.use_clahe.get(),
                                             clip_limit=clip_limit)
                except (ValueError, Exception) as e:
                    # パラメータエラーの場合はコントラスト調整をスキップ
                    print(f"Contrast adjustment error: {e}")

            # 画像を適切なサイズにリサイズ
            resized_img = self.resize_image_for_canvas(roi_img)
            if resized_img is None or resized_img.size == 0:
                print(f"Resize failed for {time_type}")
                return

            resized_h, resized_w = resized_img.shape[:2]

            # スケール比を計算
            scale_factor = min(resized_w / roi_img.shape[1], resized_h / roi_img.shape[0])

            # キャンバスにスケール情報を保存（ROI設定で使用）
            canvas.scale_factor = scale_factor
            canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else CANVAS_PREVIEW_WIDTH
            canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else CANVAS_PREVIEW_HEIGHT
            canvas.image_offset_x = (canvas_width - resized_w) // 2
            canvas.image_offset_y = (canvas_height - resized_h) // 2

            # 二値化（ROI座標を渡して完全なマスクを適用）
            roi_coords_for_binarize = None
            if self.roi_active.get() and self.roi_coordinates:
                # プレビュー用の座標をリサイズ後の座標に変換
                center_x, center_y, radius = self.roi_coordinates
                # ROI座標をリサイズ後の画像座標に変換
                roi_center_x = (center_x - roi_offset_x) * scale_factor
                roi_center_y = (center_y - roi_offset_y) * scale_factor
                roi_radius = radius * scale_factor
                roi_coords_for_binarize = (roi_center_x, roi_center_y, roi_radius)

            binary = binarize(resized_img,
                             method=self.binarize_method.get(),
                             relative_thresh=self.relative_thresh.get(),
                             block_size=self.block_size.get(),
                             c_value=self.c_value.get(),
                             use_bg_removal=self.use_bg_removal.get(),
                             bg_kernel_size=self.bg_kernel_size.get(),
                             roi_coordinates=roi_coords_for_binarize,
                             use_edge_enhancement=self.use_edge_enhancement.get(),
                             edge_weight=self.edge_weight.get())

            # 内部均一性フィルタ用のグレースケール画像を準備
            original_gray_for_uniformity = None
            min_uniformity_val = 0.0
            if self.use_uniformity_filter.get():
                original_gray_for_uniformity = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                min_uniformity_val = self.min_uniformity.get()

            # 検出（スケール比を考慮）
            results, contours = analyze(
                binary,
                min_area=min_area,
                max_area=max_area,
                select_center_only=(not self.select_largest.get()),
                select_largest=self.select_largest.get(),
                scale_factor=scale_factor,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
                original_gray=original_gray_for_uniformity,
                min_uniformity=min_uniformity_val
            )

            # 検出結果を描画
            preview_img = resized_img.copy()
            cv2.drawContours(preview_img, contours, -1, (0, 255, 0), 2)
            for result in results:
                # ROI座標を考慮してプレビュー座標に変換
                roi_rel_x = result['centroid_x'] - roi_offset_x
                roi_rel_y = result['centroid_y'] - roi_offset_y
                cv2.circle(preview_img, (int(roi_rel_x * scale_factor), int(roi_rel_y * scale_factor)), 3, (0, 0, 255), -1)

            # ROI矩形を描画（ROI有効時）
            if self.roi_active.get() and self.roi_coordinates:
                # ROI全体を表示する場合の元画像プレビュー
                full_resized = self.resize_image_for_canvas(image)
                full_scale = min(300 / original_w, 200 / original_h)

                # 円形ROI座標（center_x, center_y, radius）を矩形座標に変換
                center_x, center_y, radius = self.roi_coordinates
                roi_x1 = int((center_x - radius) * full_scale)
                roi_y1 = int((center_y - radius) * full_scale)
                roi_x2 = int((center_x + radius) * full_scale)
                roi_y2 = int((center_y + radius) * full_scale)

                # ROI領域のみ表示
                preview_img_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)
            else:
                preview_img_rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB)

            # PIL Imageに変換してtkinterで表示
            pil_img = Image.fromarray(preview_img_rgb)
            photo = ImageTk.PhotoImage(pil_img)

            # キャンバスに描画
            canvas.delete("all")

            x = canvas.image_offset_x
            y = canvas.image_offset_y
            canvas.create_image(x, y, anchor=tk.NW, image=photo)
            canvas.image = photo  # 参照を保持

            # 検出情報をキャンバスに表示
            roi_status = " (ROI)" if self.roi_active.get() and self.roi_coordinates else ""
            contrast_status = " +Contrast" if self.use_contrast_enhancement.get() or self.use_clahe.get() else ""
            if results:
                info_text = f"{time_type}{roi_status}{contrast_status}: {len(results)}個検出 (面積: {results[0]['area']:.0f})"
            else:
                info_text = f"{time_type}{roi_status}{contrast_status}: {len(results)}個検出"
            canvas.create_text(10, 10, anchor=tk.NW, text=info_text,
                              fill="blue", font=("Arial", 10, "bold"))
        except Exception as e:
            # プレビュー更新中のエラーは静かに無視（クラッシュ防止）
            print(f"Preview error ({time_type}): {e}")

    def log_output(self, text):
        self.output_text.insert(tk.END, text + '\n')
        self.output_text.see(tk.END)
        self.root.update()

    def create_background(self):
        """Temporal median filterで背景を作成"""
        if not self.folder_path.get():
            messagebox.showerror('エラー', '画像フォルダを選択してください')
            return

        try:
            num_frames = self.temporal_frames.get()
            if num_frames <= 0:
                messagebox.showerror('エラー', 'フレーム数は正の整数で入力してください')
                return
        except ValueError:
            messagebox.showerror('エラー', 'フレーム数は整数で入力してください')
            return

        self.log_output('--- 背景作成開始 ---')

        # 画像ファイルのパスリストを作成
        image_files = sorted([f for f in os.listdir(self.folder_path.get())
                             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        image_paths = [os.path.join(self.folder_path.get(), f) for f in image_files]

        if not image_paths:
            messagebox.showerror('エラー', '画像ファイルが見つかりません')
            return

        self.log_output(f'全{len(image_paths)}枚の画像から{num_frames}フレームを選択して背景作成中...')

        # 別スレッドで背景作成
        def create_bg_thread():
            try:
                background = create_temporal_median_background(image_paths, num_frames)
                if background is not None:
                    self.temporal_background = background
                    self.root.after(0, lambda: self.temporal_status_label.config(text='背景作成完了'))
                    self.root.after(0, lambda: self.log_output('背景作成が完了しました'))
                else:
                    self.root.after(0, lambda: self.temporal_status_label.config(text='背景作成失敗'))
                    self.root.after(0, lambda: self.log_output('背景作成に失敗しました'))
            except Exception as e:
                self.root.after(0, lambda: self.temporal_status_label.config(text='背景作成エラー'))
                self.root.after(0, lambda: self.log_output(f'背景作成中にエラーが発生しました: {e}'))

        threading.Thread(target=create_bg_thread, daemon=True).start()
        self.temporal_status_label.config(text='背景作成中...')

    def run_analysis(self, folder_path, min_area, max_area, binarize_method, relative_thresh, block_size, c_value, use_bg_removal, bg_kernel_size, roi_active, roi_coordinates):
        self.log_output('--- 解析開始 ---')

        # Temporal median filterが有効で背景が作成されていない場合は確認
        if self.use_temporal_median.get() and self.temporal_background is None:
            self.log_output('警告: Temporal median filterが有効ですが、背景が作成されていません。')
            self.log_output('先に「背景作成」ボタンで背景を作成するか、Temporal median filterを無効にしてください。')
            return

        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        results_list = []

        # 動画保存の準備
        video_writer = None
        if self.save_video.get():
            try:
                fps = self.video_fps.get()
                scale = self.video_scale.get()
                video_path = os.path.join(folder_path, 'detection_video.avi')

                # 最初の画像から動画のサイズを決定
                if image_files:
                    first_img = cv2.imread(os.path.join(folder_path, image_files[0]))
                    if first_img is not None:
                        # ROI適用後のサイズを取得
                        if roi_active and roi_coordinates:
                            # 円形ROI(center_x, cy, r) から矩形を算出
                            cx, cy, r = roi_coordinates
                            x1 = max(0, cx - r)
                            y1 = max(0, cy - r)
                            x2 = min(first_img.shape[1], cx + r)
                            y2 = min(first_img.shape[0], cy + r)
                            first_img = first_img[y1:y2, x1:x2]

                        # スケール適用
                        original_h, original_w = first_img.shape[:2]
                        new_w = int(original_w * scale)
                        new_h = int(original_h * scale)

                        # 動画ライターを初期化
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (new_w, new_h))
                        self.log_output(f'動画保存を開始します: {video_path} ({new_w}x{new_h}, {fps}fps)')

            except Exception as e:
                self.log_output(f'動画保存の初期化に失敗しました: {e}')
                video_writer = None

        total_files = len(image_files)
        for i, filename in enumerate(image_files):
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)
            if image is None:
                self.log_output(f'画像を読み込めません: {filename}')
                continue

            # ROIが有効な場合は対象領域を切り出し
            roi_img = image
            roi_offset_x, roi_offset_y = 0, 0
            roi_rel_coords = None  # binarize 用のROI（ROI矩形内の相対座標）
            if roi_active and roi_coordinates:
                # 円形ROI座標（center_x, center_y, radius）を矩形座標に変換
                center_x, center_y, radius = roi_coordinates
                x1 = max(0, center_x - radius)
                y1 = max(0, center_y - radius)
                x2 = min(image.shape[1], center_x + radius)
                y2 = min(image.shape[0], center_y + radius)
                roi_img = image[y1:y2, x1:x2]
                roi_offset_x, roi_offset_y = x1, y1
                # binarize に渡すためのROI円（切り出し矩形に対する相対座標）
                roi_rel_coords = (
                    int(center_x - roi_offset_x),
                    int(center_y - roi_offset_y),
                    int(radius)
                )

            # Temporal median filterを適用（有効な場合）
            if self.use_temporal_median.get() and self.temporal_background is not None:
                # ROIが適用されている場合は、背景もROIでクリッピング
                bg_roi = self.temporal_background
                if roi_active and roi_coordinates:
                    cx, cy, r = roi_coordinates
                    x1 = max(0, cx - r)
                    y1 = max(0, cy - r)
                    x2 = min(self.temporal_background.shape[1], cx + r)
                    y2 = min(self.temporal_background.shape[0], cy + r)
                    bg_roi = self.temporal_background[y1:y2, x1:x2]

                # 背景引き算を適用
                processed_img = apply_temporal_median_subtraction(roi_img, bg_roi)
                # 3チャンネルに変換して既存の処理と互換性を保つ
                roi_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

            # 自動背景補正を適用（新機能）
            if self.use_auto_bg_correction.get():
                corrected = auto_background_correction(roi_img, method=self.auto_bg_method.get())
                if corrected is not None:
                    roi_img = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)

            # コントラスト調整を適用（有効な場合）
            if self.use_contrast_enhancement.get() or self.use_clahe.get():
                try:
                    alpha = self.contrast_alpha.get()
                    beta = self.brightness_beta.get()
                    clip_limit = self.clahe_clip_limit.get()
                    roi_img = enhance_contrast(roi_img, 
                                             use_contrast=self.use_contrast_enhancement.get(),
                                             alpha=alpha, beta=beta,
                                             use_clahe=self.use_clahe.get(),
                                             clip_limit=clip_limit)
                except ValueError:
                    # パラメータエラーの場合はコントラスト調整をスキップ
                    pass

            # 二値化
            binary = binarize(roi_img,
                             method=binarize_method,
                             relative_thresh=relative_thresh,
                             block_size=block_size,
                             c_value=c_value,
                             use_bg_removal=use_bg_removal,
                             bg_kernel_size=bg_kernel_size,
                             roi_coordinates=roi_rel_coords,
                             use_edge_enhancement=self.use_edge_enhancement.get(),
                             edge_weight=self.edge_weight.get())

            # 内部均一性フィルタ用のグレースケール画像を準備
            original_gray_for_uniformity = None
            min_uniformity_val = 0.0
            if self.use_uniformity_filter.get():
                original_gray_for_uniformity = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                min_uniformity_val = self.min_uniformity.get()

            # 解析
            results, contours = analyze(
                binary,
                min_area=min_area,
                max_area=max_area,
                select_center_only=(not self.select_largest.get()),
                select_largest=self.select_largest.get(),
                scale_factor=1.0,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y,
                original_gray=original_gray_for_uniformity,
                min_uniformity=min_uniformity_val
            )

            # 動画フレーム作成（動画保存が有効な場合）
            if video_writer is not None:
                try:
                    # 検出結果を描画した画像を作成
                    video_frame = roi_img.copy()

                    # 輪郭を描画
                    cv2.drawContours(video_frame, contours, -1, (0, 255, 0), 2)

                    # 重心を描画
                    for result in results:
                        # 座標をROI相対座標に変換
                        cx = int(result['centroid_x'] - roi_offset_x)
                        cy = int(result['centroid_y'] - roi_offset_y)
                        cv2.circle(video_frame, (cx, cy), 5, (0, 0, 255), -1)

                        # 面積情報をテキストで表示
                        text = f"Area: {result['area']:.0f}"
                        cv2.putText(video_frame, text, (cx + 10, cy - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    # フレーム番号を表示
                    cv2.putText(video_frame, f"Frame: {i+1}/{total_files}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # スケール適用
                    scale = self.video_scale.get()
                    if scale != 1.0:
                        h, w = video_frame.shape[:2]
                        new_w, new_h = int(w * scale), int(h * scale)
                        video_frame = cv2.resize(video_frame, (new_w, new_h))

                    # 動画に書き込み
                    video_writer.write(video_frame)

                except Exception as e:
                    self.log_output(f'動画フレーム作成エラー（フレーム{i+1}）: {e}')

            if results:
                result = results[0]
                result['filename'] = filename
                results_list.append(result)
            else:
                # 検出されなかった場合も記録
                results_list.append({'filename': filename, 'centroid_x': np.nan, 'centroid_y': np.nan,
                                     'major_axis': np.nan, 'minor_axis': np.nan, 'circularity': np.nan, 'area': 0})

            # 進捗をログに出力
            if (i + 1) % 10 == 0 or (i + 1) == total_files:
                self.log_output(f'進捗: {i + 1}/{total_files} ファイル処理完了')

        # 動画ライターを閉じる
        if video_writer is not None:
            video_writer.release()
            self.log_output('動画保存が完了しました')

        df = pd.DataFrame(results_list)
        output_path = os.path.join(folder_path, 'analysis_results.csv')
        df.to_csv(output_path, index=False)

        # 時間設定を保存（測定開始時間も含める）
        time_config = {
            'day_start_time': self.day_start_time.get(),
            'night_start_time': self.night_start_time.get(),
            'measurement_start_time': self.measurement_start_time.get()
        }
        config_path = os.path.join(folder_path, 'time_config.json')
        import json
        with open(config_path, 'w') as f:
            json.dump(time_config, f)

        self.log_output(f'--- 解析完了 ---')
        self.log_output(f'結果を {output_path} に保存しました。')
        self.log_output(f'時間設定を {config_path} に保存しました。')

        # メインスレッドでグラフ描画を呼び出し
        self.root.after(0, self.plot_results, df)

    def plot_results(self, df):
        if df.empty:
            self.log_output("Analysis results are empty. No graph will be created.")
            return

        try:
            # --- タイムスタンプ生成ロジック ---
            # GUIの測定開始時刻とフレーム間隔でタイムスタンプを生成
            from datetime import datetime, timedelta, date

            measurement_start_str = self.measurement_start_time.get()
            start_time = datetime.strptime(measurement_start_str, '%H:%M:%S').time()

            # フレーム間隔を取得（GUIで設定可能）
            frame_interval = self.frame_interval.get()

            # 画像ファイルの日付を特定する（ここでは今日の日付を基準とする）
            today = date.today()
            start_datetime = datetime.combine(today, start_time)

            # 解析結果のDataFrameに'filename'列が存在することを確認
            if 'filename' not in df.columns:
                self.log_output("Error: 'filename' column not found in analysis results.")
                messagebox.showerror("Error", "'filename' column not found. Cannot generate time axis.")
                return

            # ファイル名でソートされたリストを取得
            # 自然順ソート（数字部分を数値としてソートする）
            def natural_key(s):
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]

            sorted_filenames = sorted(df['filename'].unique(), key=natural_key)

            timestamps = []
            current_datetime = start_datetime
            for _ in sorted_filenames:
                timestamps.append(current_datetime)
                current_datetime += timedelta(seconds=frame_interval)

            if len(sorted_filenames) != len(timestamps):
                 self.log_output(f"Warning: Mismatch between number of files ({len(sorted_filenames)}) and timestamps ({len(timestamps)}).")

            time_df = pd.DataFrame({
                'filename': sorted_filenames,
                'datetime': timestamps
            })

            merged_df = pd.merge(df, time_df, on='filename', how='left')
            # --- 変更ここまで ---

            if 'datetime' not in merged_df.columns or merged_df['datetime'].isnull().all():
                self.log_output("Failed to merge datetime data. Please check if filenames match.")
                return

            merged_df.sort_values('datetime', inplace=True)
            merged_df.dropna(subset=['datetime'], inplace=True)  # Remove data without datetime

            # 測定開始時間でデータをフィルタリング（オプション）
            measurement_start_str = self.measurement_start_time.get()
            try:
                measurement_start = datetime.strptime(measurement_start_str, '%H:%M:%S').time()
                # 現在はフィルタリングせず、マーカーのみ表示
                self.log_output(f'測定開始時間: {measurement_start_str}, フレーム間隔: {frame_interval}秒')
            except ValueError:
                measurement_start = None
                self.log_output(f'測定開始時間の形式が無効です: {measurement_start_str}')

            # Create graph window
            graph_window = tk.Toplevel(self.root)
            graph_window.title("Analysis Results - Animal Activity")
            graph_window.geometry("1200x700")

            # Create matplotlib figure with English labels
            fig, ax = plt.subplots(figsize=(12, 7))

            # Handle different conditions
            if self.constant_darkness.get():
                # Constant darkness condition - no day/night shading
                ax.plot(merged_df['datetime'], merged_df['area'],
                       marker='.', linestyle='-', markersize=3,
                       color='blue', linewidth=1, label='Animal Area')
                ax.set_title("Animal Activity under Constant Darkness", fontsize=14, fontweight='bold')
                # Add text annotation for constant darkness
                ax.text(0.02, 0.98, 'Constant Darkness', transform=ax.transAxes,
                       fontsize=12, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'),
                       color='white', fontweight='bold')
            else:
                # Normal light/dark cycle
                day_start_time = datetime.strptime(self.day_start_time.get(), '%H:%M').time()
                night_start_time = datetime.strptime(self.night_start_time.get(), '%H:%M').time()

                # Add day/night shading
                if not merged_df.empty:
                    unique_dates = merged_df['datetime'].dt.date.unique()
                    for d in unique_dates:
                        night_start = datetime.combine(d, night_start_time)
                        # Handle different day/night patterns
                        if night_start_time > day_start_time:
                            day_end = datetime.combine(d + pd.Timedelta(days=1), day_start_time)
                        else:
                            day_end = datetime.combine(d, day_start_time)

                        ax.axvspan(night_start, day_end, facecolor='gray', alpha=0.2, label='Dark Period' if d == unique_dates[0] else "")

                ax.plot(merged_df['datetime'], merged_df['area'],
                       marker='.', linestyle='-', markersize=3,
                       color='blue', linewidth=1, label='Animal Area')
                ax.set_title("Animal Activity over Time", fontsize=14, fontweight='bold')

            # 測定開始時間のマーカーを表示
            if measurement_start is not None and not merged_df.empty:
                unique_dates = merged_df['datetime'].dt.date.unique()
                for d in unique_dates:
                    measurement_time = datetime.combine(d, measurement_start)
                    # データの時間範囲内にある場合のみマーカーを表示
                    if merged_df['datetime'].min() <= measurement_time <= merged_df['datetime'].max():
                        ax.axvline(x=measurement_time, color='red', linestyle='--', linewidth=2,
                                  label='Measurement Start' if d == unique_dates[0] else "", alpha=0.7)

            # Set English axis labels and formatting
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Area (pixels²)", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

            # Add statistics text box with measurement start time info
            valid_areas = merged_df['area'][merged_df['area'] > 0]
            if not valid_areas.empty:
                stats_text = f'Statistics:\nMean: {valid_areas.mean():.1f}\nStd: {valid_areas.std():.1f}\nMax: {valid_areas.max():.1f}\nMin: {valid_areas.min():.1f}'
                if measurement_start is not None:
                    stats_text += f'\nMeasurement Start: {measurement_start_str}'
                ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            # Embed plot in tkinter window
            canvas = FigureCanvasTkAgg(fig, master=graph_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            # Add toolbar for zooming/panning
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, graph_window)
            toolbar.update()

            plt.tight_layout()

            # Save graph as PNG
            graph_save_path = os.path.join(self.folder_path.get(), 'activity_graph.png')
            fig.savefig(graph_save_path, dpi=300, bbox_inches='tight')
            self.log_output(f'Graph saved as: {graph_save_path}')

        except Exception as e:
            self.log_output(f"Error creating graph: {e}")
            messagebox.showerror("Error", f"Error creating graph:\n{e}")

    # =============================================================================
    # ターゲット選択機能（インタラクティブ学習）
    # プラナリアをクリックして、その特徴を学習し、検出パラメータを自動推定
    # =============================================================================

    def start_target_selection(self):
        """ターゲット選択モードを開始（昼・夜両方対応）"""
        # 昼か夜のどちらかの画像が必要
        day_path = self.day_img_path.get()
        night_path = self.night_img_path.get()

        if not day_path and not night_path:
            messagebox.showwarning('警告', '先に昼または夜の代表画像を選択してください')
            return

        # 現在表示中のタブを確認して、対象画像とキャンバスを決定
        current_tab = self.notebook.index(self.notebook.select())

        if current_tab == 0:  # 昼画像タブ
            if not day_path or not os.path.exists(day_path):
                messagebox.showwarning('警告', '昼の代表画像が見つかりません')
                return
            target_canvas = self.day_canvas
            target_path = day_path
            target_type = '昼'
        elif current_tab == 1:  # 夜画像タブ
            if not night_path or not os.path.exists(night_path):
                messagebox.showwarning('警告', '夜の代表画像が見つかりません')
                return
            target_canvas = self.night_canvas
            target_path = night_path
            target_type = '夜'
        else:
            # 背景プレビュータブの場合は昼画像を使用
            if day_path and os.path.exists(day_path):
                target_canvas = self.day_canvas
                target_path = day_path
                target_type = '昼'
            elif night_path and os.path.exists(night_path):
                target_canvas = self.night_canvas
                target_path = night_path
                target_type = '夜'
            else:
                messagebox.showwarning('警告', '代表画像が見つかりません')
                return

        # プレビューを更新してスケール情報を確実に設定
        self.update_preview()

        # スケール情報が設定されているか確認
        if not hasattr(target_canvas, 'scale_factor'):
            messagebox.showwarning('警告', 'プレビューの準備ができていません。\n再度お試しください。')
            return

        # ターゲット選択用の情報を保存
        self.target_canvas = target_canvas
        self.target_image_path = target_path

        messagebox.showinfo('ターゲット選択',
                           f'{target_type}画像プレビューでプラナリアをクリックしてください。\n'
                           'クリックした位置の輝度・面積情報を基に検出パラメータを自動推定します。')

        self.target_selecting = True
        target_canvas.bind("<Button-1>", self.on_target_click)
        self.log_output(f'ターゲット選択モードを開始しました（{target_type}画像）。プラナリアをクリックしてください。')

    def on_target_click(self, event):
        """ターゲットクリック時の処理（昼・夜両方対応・座標変換修正版）"""
        try:
            if not self.target_selecting:
                return

            self.target_selecting = False

            # 使用するキャンバスと画像パスを取得
            target_canvas = getattr(self, 'target_canvas', self.day_canvas)
            target_path = getattr(self, 'target_image_path', self.day_img_path.get())

            target_canvas.unbind("<Button-1>")

            # クリック座標を元画像座標に変換
            if not hasattr(target_canvas, 'scale_factor'):
                self.log_output('エラー: キャンバスのスケール情報がありません')
                return

            offset_x = getattr(target_canvas, 'image_offset_x', 0)
            offset_y = getattr(target_canvas, 'image_offset_y', 0)
            scale_factor = getattr(target_canvas, 'scale_factor', 1.0)

            # ゼロ除算防止
            if scale_factor <= 0:
                scale_factor = 1.0

            # キャンバス座標をプレビュー画像座標に変換
            preview_x = max(0, event.x - offset_x)
            preview_y = max(0, event.y - offset_y)

            # プレビュー画像座標を元画像座標に変換
            # ROIが有効な場合はROIオフセットを加算
            roi_offset_x, roi_offset_y = 0, 0
            if self.roi_active.get() and self.roi_coordinates:
                center_x, center_y, radius = self.roi_coordinates
                roi_offset_x = max(0, int(center_x - radius))
                roi_offset_y = max(0, int(center_y - radius))

            orig_x = int(preview_x / scale_factor) + roi_offset_x
            orig_y = int(preview_y / scale_factor) + roi_offset_y

            self.log_output(f'クリック位置: キャンバス({event.x},{event.y}) → 元画像({orig_x},{orig_y})')

            # 元画像を読み込んでターゲット情報を取得
            self.analyze_target_at_position(orig_x, orig_y, target_path)

        except Exception as e:
            self.log_output(f'ターゲットクリック処理エラー: {e}')
            self.target_selecting = False
            if hasattr(self, 'target_canvas'):
                self.target_canvas.unbind("<Button-1>")

    def analyze_target_at_position(self, x, y, image_path=None):
        """指定位置のターゲット情報を解析（昼・夜両方対応・検出ロジック改善版）"""
        try:
            # 画像パスが指定されていない場合は昼画像を使用
            if image_path is None:
                image_path = self.day_img_path.get()

            if not image_path or not os.path.exists(image_path):
                self.log_output('画像が見つかりません')
                return

            image = cv2.imread(image_path)
            if image is None:
                self.log_output('画像を読み込めませんでした')
                return

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 座標が画像範囲内か確認
            if x < 0 or x >= gray.shape[1] or y < 0 or y >= gray.shape[0]:
                self.log_output(f'クリック位置({x},{y})が画像範囲外です（画像サイズ: {gray.shape[1]}x{gray.shape[0]}）')
                return

            # クリック位置周辺の輝度を取得（5x5ピクセルの平均）
            sample_size = 5
            x1 = max(0, x - sample_size)
            y1 = max(0, y - sample_size)
            x2 = min(gray.shape[1], x + sample_size)
            y2 = min(gray.shape[0], y + sample_size)

            target_brightness = float(np.mean(gray[y1:y2, x1:x2]))
            background_brightness = float(np.mean(gray))

            # 輝度差を計算
            brightness_diff = float(background_brightness - target_brightness)

            self.log_output(f'ターゲット輝度: {target_brightness:.1f}')
            self.log_output(f'背景輝度: {background_brightness:.1f}')
            self.log_output(f'輝度差: {brightness_diff:.1f}')

            # 自動背景補正を適用してターゲットを検出
            corrected = auto_background_correction(image, method='adaptive')

            if corrected is None:
                self.log_output('背景補正に失敗しました。元画像で検出を試行します。')
                corrected = gray.copy()

            # 補正後の画像で閾値を自動推定
            corrected_target_val = int(corrected[y, x]) if 0 <= y < corrected.shape[0] and 0 <= x < corrected.shape[1] else 0

            # 複数の方法でターゲット検出を試行（高速化・精度向上版）
            target_contour = None
            target_area = 0
            detection_method = "未検出"

            # 方法1: クリック位置中心の局所的な閾値で検出（最優先・最速）
            search_radius = 150
            sx1 = max(0, x - search_radius)
            sy1 = max(0, y - search_radius)
            sx2 = min(gray.shape[1], x + search_radius)
            sy2 = min(gray.shape[0], y + search_radius)
            local_gray = gray[sy1:sy2, sx1:sx2]

            # 局所領域でOtsu（暗い物体検出なのでINV）
            _, local_binary = cv2.threshold(local_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # ノイズ除去
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            local_binary = cv2.morphologyEx(local_binary, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(local_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 局所座標に変換
            local_x = x - sx1
            local_y = y - sy1

            for cnt in contours:
                if cv2.pointPolygonTest(cnt, (local_x, local_y), False) >= 0:
                    target_contour = cnt
                    target_area = float(cv2.contourArea(cnt))
                    detection_method = "局所Otsu二値化"
                    break

            # 方法2: 局所で失敗した場合、補正画像でOtsu
            if target_contour is None:
                _, binary_otsu = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.pointPolygonTest(cnt, (x, y), False) >= 0:
                        target_contour = cnt
                        target_area = float(cv2.contourArea(cnt))
                        detection_method = "Otsu二値化（背景補正後）"
                        break

            # 方法3: 最近傍輪郭を選択（最後の手段）
            if target_contour is None and len(contours) > 0:
                min_dist = float('inf')
                for cnt in contours:
                    M = cv2.moments(cnt)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # 局所座標の場合は元座標に変換
                        if detection_method == "未検出":
                            cx += sx1
                            cy += sy1
                        dist = np.sqrt((cx - x)**2 + (cy - y)**2)
                        if dist < min_dist and dist < 100:  # 100px以内
                            min_dist = dist
                            target_contour = cnt
                            target_area = float(cv2.contourArea(cnt))
                            detection_method = f"最近傍輪郭（距離: {min_dist:.1f}px）"

            if target_contour is not None and target_area > 0:
                # ターゲット情報を保存（JSON保存可能なPython型に変換）
                self.target_info = {
                    'x': int(x),
                    'y': int(y),
                    'brightness': target_brightness,
                    'background_brightness': background_brightness,
                    'brightness_diff': brightness_diff,
                    'area': target_area,
                    'corrected_target_val': corrected_target_val
                }

                self.learned_brightness = target_brightness
                self.learned_area_range = (max(10, int(target_area * 0.3)), int(target_area * 3.0))

                self.target_info_label.config(text=f'面積:{target_area:.0f}')
                self.log_output(f'ターゲットを検出: {detection_method}')
                self.log_output(f'面積={target_area:.0f}px²')
                self.log_output(f'推定面積範囲: {self.learned_area_range[0]} - {self.learned_area_range[1]}')

                # 検出結果をプレビューに反映
                self.update_preview()
            else:
                self.log_output('ターゲットを検出できませんでした。別の位置をクリックしてください。')
                self.target_info_label.config(text='検出失敗')
                messagebox.showwarning('警告', 'ターゲットを検出できませんでした。\nプラナリアの中心付近を正確にクリックしてください。')

        except Exception as e:
            self.log_output(f'ターゲット解析エラー: {e}')
            import traceback
            traceback.print_exc()
            self.target_info_label.config(text='エラー')
            messagebox.showerror('エラー', f'ターゲット解析中にエラーが発生しました:\n{e}')

    def apply_learned_params(self):
        """学習したパラメータを検出設定に適用"""
        try:
            if self.target_info is None:
                messagebox.showwarning('警告', '先にターゲットを選択してください')
                return

            # 学習した面積範囲を適用
            if self.learned_area_range:
                self.min_area.set(str(self.learned_area_range[0]))
                self.max_area.set(str(self.learned_area_range[1]))
                self.log_output(f'面積範囲を更新しました: {self.learned_area_range[0]} - {self.learned_area_range[1]}')

            # 自動背景補正を有効化
            self.use_auto_bg_correction.set(True)
            self.log_output('自動背景補正を有効にしました')

            # 相対閾値を調整（輝度差に基づく）
            brightness_diff = self.target_info.get('brightness_diff', 0)
            background_brightness = self.target_info.get('background_brightness', 1)  # ゼロ除算防止
            if brightness_diff > 0 and background_brightness > 0:
                # 背景より暗いターゲット（プラナリア）
                suggested_thresh = min(0.3, brightness_diff / background_brightness)
                self.relative_thresh.set(round(suggested_thresh, 2))
                self.log_output(f'相対閾値を {suggested_thresh:.2f} に設定しました')

            self.update_preview()
            messagebox.showinfo('適用完了', '学習したパラメータを適用しました。\nプレビューを確認してください。')
        except Exception as e:
            self.log_output(f'パラメータ適用エラー: {e}')
            messagebox.showerror('エラー', f'パラメータ適用中にエラーが発生しました:\n{e}')

    def clear_target(self):
        """ターゲット選択をクリア"""
        self.target_info = None
        self.learned_brightness = None
        self.learned_area_range = None
        self.target_selecting = False
        self.target_info_label.config(text='未選択')
        self.day_canvas.unbind("<Button-1>")
        self.log_output('ターゲット選択をクリアしました')

    # =============================================================================
    # ROI関連メソッド
    # =============================================================================
    def toggle_roi(self):
        if self.roi_active.get():
            self.log_output('ROI機能を有効にしました')
        else:
            self.log_output('ROI機能を無効にしました')
        self.update_preview()

    def toggle_constant_darkness(self):
        if self.constant_darkness.get():
            self.log_output('Constant Darkness mode enabled')
        else:
            self.log_output('Constant Darkness mode disabled')
        self.update_preview()

    def set_roi_mode(self):
        if not self.day_img_path.get():
            messagebox.showwarning('警告', '先に昼の代表画像を選択してください')
            return

        messagebox.showinfo('ROI設定',
                           '昼画像プレビューでマウスをドラッグしてROIを設定してください。\n'
                           '左上から右下にドラッグしてください。')

        # マウスイベントをバインド
        self.day_canvas.bind("<Button-1>", self.on_roi_start)
        self.day_canvas.bind("<B1-Motion>", self.on_roi_drag)
        self.day_canvas.bind("<ButtonRelease-1>", self.on_roi_end)

    def clear_roi(self):
        self.roi_coordinates = None
        self.roi_info_label.config(text='ROI未設定')
        self.log_output('ROIをクリアしました')
        self.update_preview()

    def on_roi_start(self, event):
        self.drawing_roi = True
        self.roi_start_x = event.x
        self.roi_start_y = event.y

    def on_roi_drag(self, event):
        if self.drawing_roi:
            # 既存のROI円を削除
            self.day_canvas.delete("roi_circle")
            # 円の半径を計算
            radius = int(((event.x - self.roi_start_x)**2 + (event.y - self.roi_start_y)**2)**0.5)
            # 新しいROI円を描画
            self.day_canvas.create_oval(
                self.roi_start_x - radius, self.roi_start_y - radius,
                self.roi_start_x + radius, self.roi_start_y + radius,
                outline="red", width=2, tags="roi_circle"
            )

    def on_roi_end(self, event):
        if not self.drawing_roi:
            return

        self.drawing_roi = False

        # キャンバス座標を元画像座標に変換
        if hasattr(self.day_canvas, 'image_offset_x') and hasattr(self.day_canvas, 'scale_factor'):
            # プレビュー画像のオフセットとスケールを取得
            offset_x = getattr(self.day_canvas, 'image_offset_x', 0)
            offset_y = getattr(self.day_canvas, 'image_offset_y', 0)
            scale_factor = getattr(self.day_canvas, 'scale_factor', 1.0)

            # ゼロ除算防止
            if scale_factor <= 0:
                scale_factor = 1.0

            # キャンバス座標をプレビュー画像座標に変換
            preview_center_x = max(0, self.roi_start_x - offset_x)
            preview_center_y = max(0, self.roi_start_y - offset_y)
            preview_end_x = max(0, event.x - offset_x)
            preview_end_y = max(0, event.y - offset_y)

            # 円の半径を計算
            preview_radius = int(((preview_end_x - preview_center_x)**2 + (preview_end_y - preview_center_y)**2)**0.5)

            # プレビュー画像座標を元画像座標に変換
            orig_center_x = int(preview_center_x / scale_factor)
            orig_center_y = int(preview_center_y / scale_factor)
            orig_radius = int(preview_radius / scale_factor)

            # 有効なROIかチェック
            if orig_radius <= 0:
                self.log_output('ROIが小さすぎます。再度ドラッグしてください。')
                return

            # 円形ROI座標として保存（中心座標と半径）
            self.roi_coordinates = (orig_center_x, orig_center_y, orig_radius)
            self.roi_info_label.config(text=f'ROI円: 中心({orig_center_x},{orig_center_y}) 半径{orig_radius}')
            self.log_output(f'円形ROIを設定しました: 中心({orig_center_x},{orig_center_y}) 半径{orig_radius}')

            # ROIチェックボックスを自動で有効にする
            self.roi_active.set(True)
            self.update_preview()

        # マウスイベントのバインドを解除
        self.day_canvas.unbind("<Button-1>")
        self.day_canvas.unbind("<B1-Motion>")
        self.day_canvas.unbind("<ButtonRelease-1>")

    def start_analysis(self):
        # バリデーション
        if not self.folder_path.get():
            messagebox.showerror('エラー', '画像フォルダを選択してください')
            return
        if not self.day_img_path.get():
            messagebox.showerror('エラー', '昼の代表画像を選択してください')
            return
        if not self.night_img_path.get():
            messagebox.showerror('エラー', '夜の代表画像を選択してください')
            return

        try:
            min_area = int(self.min_area.get())
            max_area = int(self.max_area.get())
        except ValueError:
            messagebox.showerror('エラー', 'パラメータは整数で入力してください')
            return

        # 別スレッドで解析実行
        threading.Thread(target=self.run_analysis, args=(
            self.folder_path.get(),
            min_area,
            max_area,
            self.binarize_method.get(),
            self.relative_thresh.get(),
            self.block_size.get(),
            self.c_value.get(),
            self.use_bg_removal.get(),
            self.bg_kernel_size.get(),
            self.roi_active.get(),
            self.roi_coordinates
        ), daemon=True).start()

    def show_bg_preview(self, mode):
        """背景プレビューを表示（Temporal Median Filter効果確認用）"""
        if not self.day_img_path.get() or not os.path.exists(self.day_img_path.get()):
            messagebox.showwarning('警告', '昼の代表画像を選択してください')
            return

        if mode in ['background', 'subtracted'] and self.temporal_background is None:
            messagebox.showwarning('警告', '先に背景を作成してください')
            return

        # 昼の代表画像を読み込み
        image = cv2.imread(self.day_img_path.get())
        if image is None:
            return

        # ROI適用
        roi_img = image
        if self.roi_active.get() and self.roi_coordinates:
            # 円形ROI座標（center_x, center_y, radius）を矩形座標に変換
            center_x, center_y, radius = self.roi_coordinates
            x1 = max(0, center_x - radius)
            y1 = max(0, center_y - radius)
            x2 = min(image.shape[1], center_x + radius)
            y2 = min(image.shape[0], center_y + radius)
            roi_img = image[y1:y2, x1:x2]

        try:
            display_img = None
            title = ""

            if mode == 'original':
                # 元画像を表示
                display_img = roi_img.copy()
                title = "元画像"
            elif mode == 'background':
                # 背景画像を表示
                bg_roi = self.temporal_background
                if self.roi_active.get() and self.roi_coordinates:
                    center_x, center_y, radius = self.roi_coordinates
                    x1 = max(0, center_x - radius)
                    y1 = max(0, center_y - radius)
                    x2 = min(self.temporal_background.shape[1], center_x + radius)
                    y2 = min(self.temporal_background.shape[0], center_y + radius)
                    bg_roi = self.temporal_background[y1:y2, x1:x2]

                # グレースケールをBGRに変換
                display_img = cv2.cvtColor(bg_roi, cv2.COLOR_GRAY2BGR)
                title = "背景画像（Temporal Median）"
            elif mode == 'subtracted':
                # 減算結果を表示
                bg_roi = self.temporal_background
                if self.roi_active.get() and self.roi_coordinates:
                    center_x, center_y, radius = self.roi_coordinates
                    x1 = max(0, center_x - radius)
                    y1 = max(0, center_y - radius)
                    x2 = min(self.temporal_background.shape[1], center_x + radius)
                    y2 = min(self.temporal_background.shape[0], center_y + radius)
                    bg_roi = self.temporal_background[y1:y2, x1:x2]

                # 背景減算を適用
                subtracted = apply_temporal_median_subtraction(roi_img, bg_roi)
                display_img = cv2.cvtColor(subtracted, cv2.COLOR_GRAY2BGR)
                title = "背景減算結果"

            # display_imgがNoneの場合はエラー
            if display_img is None:
                self.log_output('背景プレビューエラー: 表示画像が作成されませんでした')
                return

            # プレビューサイズに調整
            resized_img = self.resize_image_for_canvas(display_img)

            # RGB変換
            preview_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

            # PIL Imageに変換
            pil_img = Image.fromarray(preview_img_rgb)
            photo = ImageTk.PhotoImage(pil_img)

            # キャンバスに描画
            self.bg_canvas.delete("all")

            # 中央配置
            canvas_width = self.bg_canvas.winfo_width() if self.bg_canvas.winfo_width() > 1 else 300
            canvas_height = self.bg_canvas.winfo_height() if self.bg_canvas.winfo_height() > 1 else 200
            img_w, img_h = resized_img.shape[1], resized_img.shape[0]
            x = (canvas_width - img_w) // 2
            y = (canvas_height - img_h) // 2

            self.bg_canvas.create_image(x, y, anchor=tk.NW, image=photo)
            self.bg_canvas.image = photo  # 参照を保持

            # タイトルを表示
            self.bg_canvas.create_text(10, 10, anchor=tk.NW, text=title,
                                      fill="blue", font=("Arial", 10, "bold"))

            self.log_output(f'背景プレビュー: {title}を表示しました')

        except Exception as e:
            self.log_output(f'背景プレビューエラー: {e}')
            messagebox.showerror('エラー', f'背景プレビューでエラーが発生しました: {e}')

    def save_config(self):
        """設定をファイルに保存"""
        config = {
            'folder_path': self.folder_path.get(),
            'day_img_path': self.day_img_path.get(),
            'night_img_path': self.night_img_path.get(),
            'min_area': self.min_area.get(),
            'max_area': self.max_area.get(),
            'binarize_method': self.binarize_method.get(),
            'relative_thresh': self.relative_thresh.get(),
            'block_size': self.block_size.get(),
            'c_value': self.c_value.get(),
            'use_bg_removal': self.use_bg_removal.get(),
            'bg_kernel_size': self.bg_kernel_size.get(),
            'use_temporal_median': self.use_temporal_median.get(),
            'temporal_frames': self.temporal_frames.get(),
            'save_video': self.save_video.get(),
            'video_fps': self.video_fps.get(),
            'video_scale': self.video_scale.get(),
            'day_start_time': self.day_start_time.get(),
            'night_start_time': self.night_start_time.get(),
            'measurement_start_time': self.measurement_start_time.get(),
            'frame_interval': self.frame_interval.get(),  # フレーム間隔
            'roi_coordinates': self.roi_coordinates,
            # 自動背景補正設定
            'use_auto_bg_correction': self.use_auto_bg_correction.get(),
            'auto_bg_method': self.auto_bg_method.get(),
            # 内部均一性フィルタ設定
            'use_uniformity_filter': self.use_uniformity_filter.get(),
            'min_uniformity': self.min_uniformity.get(),
            # コントラスト調整設定
            'use_contrast_enhancement': self.use_contrast_enhancement.get(),
            'contrast_alpha': self.contrast_alpha.get(),
            'brightness_beta': self.brightness_beta.get(),
            'use_clahe': self.use_clahe.get(),
            'clahe_clip_limit': self.clahe_clip_limit.get(),
            # エッジ強調設定
            'use_edge_enhancement': self.use_edge_enhancement.get(),
            'edge_weight': self.edge_weight.get(),
            # ターゲット選択情報
            'target_info': self.target_info,
            'learned_area_range': self.learned_area_range
        }

        # JSONファイルに保存
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
            self.log_output(f'設定を保存しました: {file_path}')

    def load_config(self):
        """設定をファイルからロード"""
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not file_path:
            return

        with open(file_path, 'r') as f:
            config = json.load(f)

        # 設定を反映
        self.folder_path.set(config.get('folder_path', ''))
        self.day_img_path.set(config.get('day_img_path', ''))
        self.night_img_path.set(config.get('night_img_path', ''))
        self.min_area.set(config.get('min_area', str(DEFAULT_MIN_AREA)))
        self.max_area.set(config.get('max_area', str(DEFAULT_MAX_AREA)))
        self.binarize_method.set(config.get('binarize_method', 'adaptive'))
        self.relative_thresh.set(config.get('relative_thresh', DEFAULT_RELATIVE_THRESH))
        self.block_size.set(config.get('block_size', DEFAULT_BLOCK_SIZE))
        self.c_value.set(config.get('c_value', DEFAULT_C_VALUE))
        self.use_bg_removal.set(config.get('use_bg_removal', False))
        self.bg_kernel_size.set(config.get('bg_kernel_size', DEFAULT_BG_KERNEL_SIZE))
        self.use_temporal_median.set(config.get('use_temporal_median', False))
        self.temporal_frames.set(config.get('temporal_frames', DEFAULT_TEMPORAL_FRAMES))
        self.save_video.set(config.get('save_video', False))
        self.video_fps.set(config.get('video_fps', DEFAULT_VIDEO_FPS))
        self.video_scale.set(config.get('video_scale', DEFAULT_VIDEO_SCALE))
        self.day_start_time.set(config.get('day_start_time', '07:00'))
        self.night_start_time.set(config.get('night_start_time', '19:00'))
        self.measurement_start_time.set(config.get('measurement_start_time', '09:00:00'))
        self.frame_interval.set(config.get('frame_interval', FRAME_INTERVAL_SECONDS))
        self.roi_coordinates = config.get('roi_coordinates', None)

        # 自動背景補正設定をロード
        self.use_auto_bg_correction.set(config.get('use_auto_bg_correction', False))
        self.auto_bg_method.set(config.get('auto_bg_method', 'adaptive'))

        # 内部均一性フィルタ設定をロード
        self.use_uniformity_filter.set(config.get('use_uniformity_filter', False))
        self.min_uniformity.set(config.get('min_uniformity', DEFAULT_MIN_UNIFORMITY))

        # コントラスト調整設定をロード
        self.use_contrast_enhancement.set(config.get('use_contrast_enhancement', False))
        self.contrast_alpha.set(config.get('contrast_alpha', DEFAULT_CONTRAST_ALPHA))
        self.brightness_beta.set(config.get('brightness_beta', 0))
        self.use_clahe.set(config.get('use_clahe', False))
        self.clahe_clip_limit.set(config.get('clahe_clip_limit', DEFAULT_CLAHE_CLIP_LIMIT))

        # エッジ強調設定をロード
        self.use_edge_enhancement.set(config.get('use_edge_enhancement', False))
        self.edge_weight.set(config.get('edge_weight', DEFAULT_EDGE_WEIGHT))

        # ターゲット選択情報をロード
        self.target_info = config.get('target_info', None)
        self.learned_area_range = config.get('learned_area_range', None)
        if self.target_info:
            self.target_info_label.config(text=f"面積:{self.target_info.get('area', 0):.0f}")
        else:
            self.target_info_label.config(text='未選択')

        self.log_output(f'設定をロードしました: {file_path}')

        # 自動背景補正設定の反映をログに出力
        if self.use_auto_bg_correction.get():
            self.log_output(f'自動背景補正: 有効 (方式={self.auto_bg_method.get()})')

        # 内部均一性フィルタ設定の反映をログに出力
        if self.use_uniformity_filter.get():
            self.log_output(f'内部均一性フィルタ: 有効 (閾値={self.min_uniformity.get()})')

        # コントラスト調整設定の反映をログに出力
        if self.use_contrast_enhancement.get():
            self.log_output(f'コントラスト調整: 有効 (倍率={self.contrast_alpha.get()}, 明度={self.brightness_beta.get()})')

        if self.use_clahe.get():
            self.log_output(f'CLAHE: 有効 (制限={self.clahe_clip_limit.get()})')

        # プレビューを更新
        self.update_preview()

        # ROI情報ラベルの更新
        if self.roi_coordinates:
            try:
                # 円形ROI (cx, cy, r)
                if len(self.roi_coordinates) == 3:
                    cx, cy, r = self.roi_coordinates
                    self.roi_info_label.config(text=f'ROI円: 中心({cx},{cy}) 半径{r}')
                # 旧形式の矩形ROI (x1, y1, x2, y2)
                elif len(self.roi_coordinates) == 4:
                    x1, y1, x2, y2 = self.roi_coordinates
                    self.roi_info_label.config(text=f'ROI矩形: ({x1},{y1})-({x2},{y2})')
                else:
                    self.roi_info_label.config(text=f'ROI: {self.roi_coordinates}')
            except Exception:
                self.roi_info_label.config(text=f'ROI: {self.roi_coordinates}')
        else:
            self.roi_info_label.config(text='ROI未設定')

        # Temporal Median Filterのステータス更新
        if self.use_temporal_median.get() and self.temporal_background is None:
            self.temporal_status_label.config(text='背景未作成')
        elif self.temporal_background is not None:
            self.temporal_status_label.config(text='背景作成済み')

        # 主要設定のサマリーログ
        self.log_output(f'時間設定: 昼={self.day_start_time.get()}, 夜={self.night_start_time.get()}, '
                       f'測定開始={self.measurement_start_time.get()}, 間隔={self.frame_interval.get()}秒')

        # すぐにプレビューを更新
        self.root.after(100, self.update_preview)


# =============================================================================
# エントリーポイント
# =============================================================================
def main():
    root = tk.Tk()
    app = AnimalDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
