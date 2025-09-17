import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from skimage.morphology import white_tophat, disk
import os
import pandas as pd
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from datetime import datetime, time
import json

"""
todo filtering後に最大のparticleを個体とした方がよさそう。 (たいていのオブジェクトはfilter outできる。) 
おそらく現状は中央に最も近いものが選択されている。
rolling ballがうまくいっていない感じなので、Image Jと何が違うのかを調べて改良

Readmeの書き直し
"""

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
    foreground = cv2.subtract(gray, background)

    # コントラストを強調
    foreground = cv2.addWeighted(foreground, 1.5, np.zeros_like(foreground), 0, 0)

    return foreground

def remove_background_rolling_ball(image, radius=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    selem = disk(radius)
    bg_removed = white_tophat(gray, selem)
    return bg_removed

def remove_background_rolling_ball_improved(image, radius=50):
    """
    ImageJのRolling Ball Background Subtractionにより近い実装
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # より大きなカーネルサイズでモルフォロジー処理
    kernel_size = max(radius * 2 + 1, 3)  # radiusの2倍+1、最小3
    if kernel_size % 2 == 0:
        kernel_size += 1  # 奇数にする

    # Top-hat変換による背景除去
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # ガウシアンフィルタでノイズを軽減
    tophat = cv2.GaussianBlur(tophat, (3, 3), 0)

    return tophat

def remove_background_simple(image, kernel_size=50):
    """軽量なバックグラウンド除去（ガウシアンブラー使用）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # カーネルサイズを奇数に調整
    if kernel_size % 2 == 0:
        kernel_size += 1

    # 最小値を3に設定
    kernel_size = max(3, kernel_size)

    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    foreground = cv2.subtract(gray, background)
    return foreground

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

# --- 画像解析関数 ---
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

def binarize(image, method='adaptive', relative_thresh=0.1, block_size=11, c_value=2, use_bg_removal=False, bg_kernel_size=50):
    """
    統一的な二値化関数
    method: 'adaptive' (適応的), 'relative' (相対閾値), 'fixed' (固定閾値)
    relative_thresh: 相対閾値方式での閾値（平均輝度からの割合, 0.0-1.0）
    block_size: 適応的二値化のブロックサイズ
    c_value: 適応的二値化の定数
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # バックグラウンド除去（オプション）
    if use_bg_removal:
        gray = remove_background_simple(image, bg_kernel_size)

    if method == 'adaptive':
        # 適応的二値化（動物が暗い場合はTHRESH_BINARY_INVを使用）
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

    return binary

def analyze(binary, min_area=100, max_area=10000, select_center_only=True, select_largest=False, scale_factor=1.0, roi_offset_x=0, roi_offset_y=0):
    """
    個体検出・解析関数（改良版）

    Args:
        select_largest: Trueの場合、最大面積の個体を選択
        select_center_only: Trueの場合、中心に最も近い個体を選択（select_largestがFalseの時のみ）
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
        circularity = 4 * np.pi * area / (cv2.arcLength(cnt, True) ** 2)

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
            'area': actual_area,
            'distance_from_center': distance_from_center,
            'contour': cnt
        }
        candidates.append(candidate)

    if candidates:
        if select_largest:
            # 最大面積の個体を選択
            largest_candidate = max(candidates, key=lambda x: x['area'])
            results.append({k: v for k, v in largest_candidate.items() if k != 'contour'})
            valid_contours.append(largest_candidate['contour'])
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
        self.root.geometry('1200x800')  # サイズを少し大きく調整

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
        self.min_area = tk.StringVar(value='100')
        self.max_area = tk.StringVar(value='10000')

        # 新しい統一的な二値化パラメータ（暗い動物検出に最適化）
        self.binarize_method = tk.StringVar(value='adaptive')  # 'adaptive', 'relative', 'fixed'
        self.relative_thresh = tk.DoubleVar(value=0.15)  # 相対閾値を上げて明るい部分を除外
        self.block_size = tk.IntVar(value=15)  # 適応的二値化のブロックサイズ
        self.c_value = tk.IntVar(value=8)  # 定数Cを上げて明るい部分を除外
        self.use_bg_removal = tk.BooleanVar(value=False)  # バックグラウンド除去の使用
        self.bg_kernel_size = tk.IntVar(value=31)  # バックグラウンド除去のカーネルサイズ

        # Temporal Median Filter設定
        self.use_temporal_median = tk.BooleanVar(value=False)  # Temporal median filterの使用
        self.temporal_frames = tk.IntVar(value=100)  # 背景作成に使用するフレーム数
        self.temporal_background = None  # 作成された背景画像を保存

        # 動画保存設定
        self.save_video = tk.BooleanVar(value=False)  # 動画保存の使用
        self.video_fps = tk.IntVar(value=10)  # 動画のFPS
        self.video_scale = tk.DoubleVar(value=0.5)  # 動画のスケール（低画質化）

        # 時間設定
        self.day_start_time = tk.StringVar(value='07:00')
        self.night_start_time = tk.StringVar(value='19:00')

        # 測定開始時刻設定
        self.measurement_start_time = tk.StringVar(value='09:00:00')  # デフォルト9時開始

        # ROI関連
        self.roi_coordinates = None  # (x1, y1, x2, y2) in original image coordinates
        self.roi_active = tk.BooleanVar(value=False)
        self.drawing_roi = False
        self.roi_start_x = 0
        self.roi_start_y = 0

        # 個体選択方法設定を追加
        self.select_largest = tk.BooleanVar(value=True)  # 最大面積の個体を選択
        self.constant_darkness = tk.BooleanVar(value=False)  # Constant darkness条件

        # コントラスト調整設定を追加
        self.use_contrast_enhancement = tk.BooleanVar(value=False)  # コントラスト強化の使用
        self.contrast_alpha = tk.DoubleVar(value=1.5)  # コントラスト倍率（1.0=変化なし、>1.0で強化）
        self.brightness_beta = tk.IntVar(value=0)  # 明度調整（-100～100程度）
        self.use_clahe = tk.BooleanVar(value=False)  # CLAHE（局所適応ヒストグラム平坦化）の使用
        self.clahe_clip_limit = tk.DoubleVar(value=3.0)  # CLAHEのクリップ制限

        self.create_widgets()

    def create_widgets(self):
        # メインフレーム
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左側：設定パネル
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # フォルダ選択（コンパクト化）
        tk.Label(left_frame, text='画像フォルダ:', font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky='w', pady=(2,0))
        folder_frame = tk.Frame(left_frame)
        folder_frame.grid(row=1, column=0, columnspan=2, pady=1, sticky='ew')
        tk.Entry(folder_frame, textvariable=self.folder_path, width=35, font=('Arial', 8)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(folder_frame, text='選択', command=self.select_folder, font=('Arial', 8), width=6).pack(side=tk.RIGHT, padx=(2,0))

        # 昼の代表画像（コンパクト化）
        tk.Label(left_frame, text='昼の代表画像:', font=('Arial', 9, 'bold')).grid(row=2, column=0, sticky='w', pady=(3,0))
        day_frame = tk.Frame(left_frame)
        day_frame.grid(row=3, column=0, columnspan=2, pady=1, sticky='ew')
        tk.Entry(day_frame, textvariable=self.day_img_path, width=35, font=('Arial', 8)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(day_frame, text='選択', command=self.select_day_image, font=('Arial', 8), width=6).pack(side=tk.RIGHT, padx=(2,0))

        # 夜の代表画像（コンパクト化）
        tk.Label(left_frame, text='夜の代表画像:', font=('Arial', 9, 'bold')).grid(row=4, column=0, sticky='w', pady=(3,0))
        night_frame = tk.Frame(left_frame)
        night_frame.grid(row=5, column=0, columnspan=2, pady=1, sticky='ew')
        tk.Entry(night_frame, textvariable=self.night_img_path, width=35, font=('Arial', 8)).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(night_frame, text='選択', command=self.select_night_image, font=('Arial', 8), width=6).pack(side=tk.RIGHT, padx=(2,0))

        # 個体選択方法設定を追加
        selection_frame = tk.LabelFrame(left_frame, text='Individual Selection Method', font=('Arial', 8, 'bold'))
        selection_frame.grid(row=6, column=0, columnspan=2, pady=3, padx=5, sticky='ew')

        selection_row = tk.Frame(selection_frame)
        selection_row.pack(pady=2)
        tk.Checkbutton(selection_row, text='Select Largest Particle', variable=self.select_largest, font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Checkbutton(selection_row, text='Constant Darkness', variable=self.constant_darkness,
                      command=self.toggle_constant_darkness, font=('Arial', 7)).pack(side=tk.LEFT, padx=(10,1))

        # パラメータ設定（さらにコンパクト化）
        param_frame = tk.LabelFrame(left_frame, text='パラメータ設定', font=('Arial', 8, 'bold'))
        param_frame.grid(row=7, column=0, columnspan=2, pady=3, padx=5, sticky='ew')

        size_frame = tk.Frame(param_frame)
        size_frame.pack(pady=1)
        tk.Label(size_frame, text='最小面積:', font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Entry(size_frame, textvariable=self.min_area, width=6, font=('Arial', 8)).pack(side=tk.LEFT)
        tk.Label(size_frame, text='最大面積:', font=('Arial', 7)).pack(side=tk.LEFT, padx=(5,1))
        tk.Entry(size_frame, textvariable=self.max_area, width=6, font=('Arial', 8)).pack(side=tk.LEFT)
        tk.Button(size_frame, text='更新', command=self.update_preview,
                 bg='lightblue', font=('Arial', 7), width=6).pack(side=tk.LEFT, padx=(5,0))

        # 時間設定（コンパクト化）
        time_frame = tk.LabelFrame(left_frame, text='時間設定', font=('Arial', 8, 'bold'))
        time_frame.grid(row=8, column=0, columnspan=2, pady=3, padx=5, sticky='ew')

        # 1行目：昼夜の開始時間
        time_entry_frame1 = tk.Frame(time_frame)
        time_entry_frame1.pack(pady=1)
        tk.Label(time_entry_frame1, text='昼開始:', font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Entry(time_entry_frame1, textvariable=self.day_start_time, width=6, font=('Arial', 8)).pack(side=tk.LEFT, padx=1)
        tk.Label(time_entry_frame1, text='夜開始:', font=('Arial', 7)).pack(side=tk.LEFT, padx=(5,1))
        tk.Entry(time_entry_frame1, textvariable=self.night_start_time, width=6, font=('Arial', 8)).pack(side=tk.LEFT, padx=1)

        # 2行目：測定開始時刻
        time_entry_frame2 = tk.Frame(time_frame)
        time_entry_frame2.pack(pady=1)
        tk.Label(time_entry_frame2, text='測定開始時刻:', font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Entry(time_entry_frame2, textvariable=self.measurement_start_time, width=10, font=('Arial', 8)).pack(side=tk.LEFT, padx=1)

        # Temporal Median Filter設定（コンパクト化）
        temporal_frame = tk.LabelFrame(left_frame, text='Temporal Median Filter', font=('Arial', 8, 'bold'))
        temporal_frame.grid(row=9, column=0, columnspan=2, pady=3, padx=5, sticky='ew')

        # 1行にまとめる
        temporal_row = tk.Frame(temporal_frame)
        temporal_row.pack(pady=2)
        tk.Checkbutton(temporal_row, text='使用', variable=self.use_temporal_median, font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Label(temporal_row, text='フレーム数:', font=('Arial', 7)).pack(side=tk.LEFT, padx=(5,1))
        tk.Entry(temporal_row, textvariable=self.temporal_frames, width=5, font=('Arial', 8)).pack(side=tk.LEFT, padx=1)
        tk.Button(temporal_row, text='背景作成', command=self.create_background,
                 bg='lightyellow', font=('Arial', 7), width=8).pack(side=tk.LEFT, padx=(5,0))

        # ステータス表示を次の行に
        self.temporal_status_label = tk.Label(temporal_frame, text='背景未作成', font=('Arial', 6))
        self.temporal_status_label.pack(pady=(0,2))

        # 動画保存設定（コンパクト化）
        video_frame = tk.LabelFrame(left_frame, text='動画保存設定', font=('Arial', 8, 'bold'))
        video_frame.grid(row=10, column=0, columnspan=2, pady=3, padx=5, sticky='ew')

        # 1行にまとめる
        video_row = tk.Frame(video_frame)
        video_row.pack(pady=2)
        tk.Checkbutton(video_row, text='保存', variable=self.save_video, font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Label(video_row, text='FPS:', font=('Arial', 7)).pack(side=tk.LEFT, padx=(5,1))
        tk.Entry(video_row, textvariable=self.video_fps, width=3, font=('Arial', 8)).pack(side=tk.LEFT, padx=1)
        tk.Label(video_row, text='スケール:', font=('Arial', 7)).pack(side=tk.LEFT, padx=(5,1))
        tk.Entry(video_row, textvariable=self.video_scale, width=4, font=('Arial', 8)).pack(side=tk.LEFT, padx=1)

        # 二値化設定（コンパクト化）
        binarize_frame = tk.LabelFrame(left_frame, text='二値化設定', font=('Arial', 8, 'bold'))
        binarize_frame.grid(row=11, column=0, columnspan=2, pady=3, padx=5, sticky='ew')

        # 3列に分けてよりコンパクトに
        tk.Label(binarize_frame, text='メソッド:', font=('Arial', 7)).grid(row=0, column=0, padx=1, pady=1, sticky='w')
        ttk.Combobox(binarize_frame, textvariable=self.binarize_method, values=['adaptive', 'relative', 'fixed'], state='readonly', width=6, font=('Arial', 7)).grid(row=0, column=1, padx=1, pady=1)
        tk.Label(binarize_frame, text='定数C:', font=('Arial', 7)).grid(row=0, column=2, padx=(5,1), pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.c_value, width=5, font=('Arial', 8)).grid(row=0, column=3, padx=1, pady=1)
        tk.Label(binarize_frame, text='BG除去:', font=('Arial', 7)).grid(row=0, column=4, padx=(5,1), pady=1, sticky='w')
        ttk.Combobox(binarize_frame, textvariable=self.use_bg_removal, values=[True, False], state='readonly', width=4, font=('Arial', 7)).grid(row=0, column=5, padx=1, pady=1)

        tk.Label(binarize_frame, text='相対閾値:', font=('Arial', 7)).grid(row=1, column=0, padx=1, pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.relative_thresh, width=6, font=('Arial', 8)).grid(row=1, column=1, padx=1, pady=1)
        tk.Label(binarize_frame, text='ブロック:', font=('Arial', 7)).grid(row=1, column=2, padx=(5,1), pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.block_size, width=5, font=('Arial', 8)).grid(row=1, column=3, padx=1, pady=1)
        tk.Label(binarize_frame, text='BGカーネル:', font=('Arial', 7)).grid(row=1, column=4, padx=(5,1), pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.bg_kernel_size, width=4, font=('Arial', 8)).grid(row=1, column=5, padx=1, pady=1)

        # ROI設定（さらにコンパクト化）
        roi_frame = tk.LabelFrame(left_frame, text='ROI設定', font=('Arial', 8, 'bold'))
        roi_frame.grid(row=12, column=0, columnspan=2, pady=3, padx=5, sticky='ew')

        roi_control_frame = tk.Frame(roi_frame)
        roi_control_frame.pack(pady=2)
        tk.Checkbutton(roi_control_frame, text='ROI使用', variable=self.roi_active,
                      command=self.toggle_roi, font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Button(roi_control_frame, text='設定', command=self.set_roi_mode,
                 bg='orange', font=('Arial', 7), width=6).pack(side=tk.LEFT, padx=2)
        tk.Button(roi_control_frame, text='クリア', command=self.clear_roi,
                 bg='lightgray', font=('Arial', 7), width=6).pack(side=tk.LEFT, padx=1)

        self.roi_info_label = tk.Label(roi_frame, text='ROI未設定', font=('Arial', 6))
        self.roi_info_label.pack(pady=(0,2))

        # コントラスト調整設定を追加
        contrast_frame = tk.LabelFrame(left_frame, text='コントラスト調整', font=('Arial', 8, 'bold'))
        contrast_frame.grid(row=13, column=0, columnspan=2, pady=3, padx=5, sticky='ew')

        # 1行目：基本設定
        contrast_row1 = tk.Frame(contrast_frame)
        contrast_row1.pack(pady=2)
        tk.Checkbutton(contrast_row1, text='使用', variable=self.use_contrast_enhancement,
                      command=self.update_preview, font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Label(contrast_row1, text='倍率:', font=('Arial', 7)).pack(side=tk.LEFT, padx=(5,1))
        tk.Entry(contrast_row1, textvariable=self.contrast_alpha, width=4, font=('Arial', 8)).pack(side=tk.LEFT, padx=1)
        tk.Label(contrast_row1, text='明度:', font=('Arial', 7)).pack(side=tk.LEFT, padx=(5,1))
        tk.Entry(contrast_row1, textvariable=self.brightness_beta, width=4, font=('Arial', 8)).pack(side=tk.LEFT, padx=1)

        # 2行目：CLAHE設定
        contrast_row2 = tk.Frame(contrast_frame)
        contrast_row2.pack(pady=1)
        tk.Checkbutton(contrast_row2, text='CLAHE', variable=self.use_clahe,
                      command=self.update_preview, font=('Arial', 7)).pack(side=tk.LEFT, padx=1)
        tk.Label(contrast_row2, text='制限:', font=('Arial', 7)).pack(side=tk.LEFT, padx=(5,1))
        tk.Entry(contrast_row2, textvariable=self.clahe_clip_limit, width=4, font=('Arial', 8)).pack(side=tk.LEFT, padx=1)
        tk.Button(contrast_row2, text='プレビュー更新', command=self.update_preview,
                 bg='lightcyan', font=('Arial', 7), width=10).pack(side=tk.LEFT, padx=(5,0))

        # メインボタン（コンパクト化）
        button_frame = tk.Frame(left_frame)
        button_frame.grid(row=14, column=0, columnspan=2, pady=3)

        # 設定保存・ロードボタンを追加
        config_frame = tk.Frame(button_frame)
        config_frame.pack(pady=1)
        tk.Button(config_frame, text='設定保存', command=self.save_config,
                 bg='lightblue', font=('Arial', 8), width=8).pack(side=tk.LEFT, padx=1)
        tk.Button(config_frame, text='設定ロード', command=self.load_config,
                 bg='lightyellow', font=('Arial', 8), width=8).pack(side=tk.LEFT, padx=1)

        tk.Button(button_frame, text='解析開始', command=self.start_analysis,
                 bg='lightgreen', font=('Arial', 10, 'bold'), width=10, height=2).pack(pady=1)
        tk.Button(button_frame, text='終了', command=self.root.quit,
                 bg='lightcoral', font=('Arial', 9), width=10).pack(pady=1)

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
        # 元画像のサイズを取得
        original_h, original_w = image.shape[:2]

        # ROIが有効な場合は対象領域を切り出し
        roi_img = image
        roi_offset_x, roi_offset_y = 0, 0
        if self.roi_active.get() and self.roi_coordinates:
            x1, y1, x2, y2 = self.roi_coordinates
            roi_img = image[y1:y2, x1:x2]
            roi_offset_x, roi_offset_y = x1, y1

        # コントラスト調整を適用
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

        # 画像を適切なサイズにリサイズ
        resized_img = self.resize_image_for_canvas(roi_img)
        resized_h, resized_w = resized_img.shape[:2]

        # スケール比を計算
        scale_factor = min(resized_w / roi_img.shape[1], resized_h / roi_img.shape[0])

        # キャンバスにスケール情報を保存（ROI設定で使用）
        canvas.scale_factor = scale_factor
        canvas_width = canvas.winfo_width() if canvas.winfo_width() > 1 else 300
        canvas_height = canvas.winfo_height() if canvas.winfo_height() > 1 else 200
        canvas.image_offset_x = (canvas_width - resized_w) // 2
        canvas.image_offset_y = (canvas_height - resized_h) // 2

        # 二値化（新しい統一的なパラメータを使用）
        binary = binarize(resized_img,
                         method=self.binarize_method.get(),
                         relative_thresh=self.relative_thresh.get(),
                         block_size=self.block_size.get(),
                         c_value=self.c_value.get(),
                         use_bg_removal=self.use_bg_removal.get(),
                         bg_kernel_size=self.bg_kernel_size.get())

        # 検出（スケール比を考慮）
        results, contours = analyze(
            binary,
            min_area=min_area,
            max_area=max_area,
            select_center_only=(not self.select_largest.get()),
            select_largest=self.select_largest.get(),
            scale_factor=scale_factor,
            roi_offset_x=roi_offset_x,
            roi_offset_y=roi_offset_y
        )

        # 検出結果を描画
        preview_img = resized_img.copy()
        cv2.drawContours(preview_img, contours, -1, (0, 255, 0), 2)
        for result in results:
            # ROI座標を考慮してプレビュー座標に変換
            roi_rel_x = result['centroid_x'] - roi_offset_x
            roi_rel_y = result['centroid_y'] - roi_offset_y
            preview_x = int(roi_rel_x * scale_factor)
            preview_y = int(roi_rel_y * scale_factor)
            cv2.circle(preview_img, (preview_x, preview_y), 3, (0, 0, 255), -1)

        # ROI矩形を描画（ROI有効時）
        if self.roi_active.get() and self.roi_coordinates:
            # ROI全体を表示する場合の元画像プレビュー
            full_resized = self.resize_image_for_canvas(image)
            full_scale = min(300 / original_w, 200 / original_h)
            roi_x1 = int(self.roi_coordinates[0] * full_scale)
            roi_y1 = int(self.roi_coordinates[1] * full_scale)
            roi_x2 = int(self.roi_coordinates[2] * full_scale)
            roi_y2 = int(self.roi_coordinates[3] * full_scale)

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

        # ROI矩形を表示（設定済みの場合）
        if self.roi_active.get() and self.roi_coordinates and not self.roi_active.get():
            # 元画像全体を表示している場合のROI矩形
            pass

        # 検出情報をキャンバスに表示
        roi_status = " (ROI)" if self.roi_active.get() and self.roi_coordinates else ""
        contrast_status = " +Contrast" if self.use_contrast_enhancement.get() or self.use_clahe.get() else ""
        if results:
            info_text = f"{time_type}{roi_status}{contrast_status}: {len(results)}個検出 (面積: {results[0]['area']:.0f})"
        else:
            info_text = f"{time_type}{roi_status}{contrast_status}: {len(results)}個検出"
        canvas.create_text(10, 10, anchor=tk.NW, text=info_text,
                          fill="blue", font=("Arial", 10, "bold"))

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
                            x1, y1, x2, y2 = roi_coordinates
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
            if roi_active and roi_coordinates:
                x1, y1, x2, y2 = roi_coordinates
                roi_img = image[y1:y2, x1:x2]
                roi_offset_x, roi_offset_y = x1, y1

            # Temporal median filterを適用（有効な場合）
            if self.use_temporal_median.get() and self.temporal_background is not None:
                # ROIが適用されている場合は、背景もROIでクリッピング
                bg_roi = self.temporal_background
                if roi_active and roi_coordinates:
                    x1, y1, x2, y2 = roi_coordinates
                    bg_roi = self.temporal_background[y1:y2, x1:x2]

                # 背景引き算を適用
                processed_img = apply_temporal_median_subtraction(roi_img, bg_roi)
                # 3チャンネルに変換して既存の処理と互換性を保つ
                roi_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)

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
                             bg_kernel_size=bg_kernel_size)

            # 解析
            results, contours = analyze(
                binary,
                min_area=min_area,
                max_area=max_area,
                select_center_only=(not self.select_largest.get()),
                select_largest=self.select_largest.get(),
                scale_factor=1.0,
                roi_offset_x=roi_offset_x,
                roi_offset_y=roi_offset_y
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

        # 時間設定を保存
        time_config = {
            'day_start_time': self.day_start_time.get(),
            'night_start_time': self.night_start_time.get()
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

        timelog_path = os.path.join(self.folder_path.get(), 'timelog.txt')
        if not os.path.exists(timelog_path):
            self.log_output(f"timelog.txt not found: {timelog_path}")
            messagebox.showwarning("Warning", "timelog.txt not found. No graph will be created.")
            return

        try:
            # Read timelog.txt (supporting time-only format)
            with open(timelog_path, 'r') as f:
                time_lines = f.readlines()

            # Process time data
            timestamps = []
            filenames = []

            # Get sorted image filenames
            image_files = sorted([f for f in os.listdir(self.folder_path.get())
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

            for i, time_line in enumerate(time_lines):
                time_str = time_line.strip()
                if time_str and i < len(image_files):
                    # Combine today's date with time to create complete datetime
                    from datetime import date
                    today = date.today()
                    full_datetime_str = f"{today} {time_str}"
                    try:
                        timestamp = pd.to_datetime(full_datetime_str)
                        timestamps.append(timestamp)
                        filenames.append(image_files[i])
                    except:
                        continue

            # Create DataFrame
            time_df = pd.DataFrame({
                'filename': filenames,
                'datetime': timestamps
            })

            merged_df = pd.merge(time_df, df, on='filename', how='right')
            if 'datetime' not in merged_df.columns or merged_df['datetime'].isnull().all():
                self.log_output("Failed to merge datetime data. Please check if filenames match.")
                return

            merged_df.sort_values('datetime', inplace=True)
            merged_df.dropna(subset=['datetime'], inplace=True)  # Remove data without datetime

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

            # Set English axis labels and formatting
            ax.set_xlabel("Time", fontsize=12)
            ax.set_ylabel("Area (pixels²)", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

            # Add statistics text box
            valid_areas = merged_df['area'][merged_df['area'] > 0]
            if not valid_areas.empty:
                stats_text = f'Statistics:\nMean: {valid_areas.mean():.1f}\nStd: {valid_areas.std():.1f}\nMax: {valid_areas.max():.1f}\nMin: {valid_areas.min():.1f}'
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

    # ROI関連メソッド
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
            # 既存のROI矩形を削除
            self.day_canvas.delete("roi_rect")
            # 新しいROI矩形を描画
            self.day_canvas.create_rectangle(
                self.roi_start_x, self.roi_start_y, event.x, event.y,
                outline="red", width=2, tags="roi_rect"
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

            # キャンバス座標をプレビュー画像座標に変換
            preview_x1 = max(0, self.roi_start_x - offset_x)
            preview_y1 = max(0, self.roi_start_y - offset_y)
            preview_x2 = max(0, event.x - offset_x)
            preview_y2 = max(0, event.y - offset_y)

            # プレビュー画像座標を元画像座標に変換
            orig_x1 = int(preview_x1 / scale_factor)
            orig_y1 = int(preview_y1 / scale_factor)
            orig_x2 = int(preview_x2 / scale_factor)
            orig_y2 = int(preview_y2 / scale_factor)

            # 座標を正規化（左上が小さい値になるように）
            x1, x2 = min(orig_x1, orig_x2), max(orig_x1, orig_x2)
            y1, y2 = min(orig_y1, orig_y2), max(orig_y1, orig_y2)

            self.roi_coordinates = (x1, y1, x2, y2)
            self.roi_info_label.config(text=f'ROI: ({x1},{y1})-({x2},{y2})')
            self.log_output(f'ROIを設定しました: ({x1},{y1})-({x2},{y2})')

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
            x1, y1, x2, y2 = self.roi_coordinates
            roi_img = image[y1:y2, x1:x2]

        try:
            if mode == 'original':
                # 元画像を表示
                display_img = roi_img.copy()
                title = "元画像"
            elif mode == 'background':
                # 背景画像を表示
                bg_roi = self.temporal_background
                if self.roi_active.get() and self.roi_coordinates:
                    x1, y1, x2, y2 = self.roi_coordinates
                    bg_roi = self.temporal_background[y1:y2, x1:x2]

                # グレースケールをBGRに変換
                display_img = cv2.cvtColor(bg_roi, cv2.COLOR_GRAY2BGR)
                title = "背景画像（Temporal Median）"
            elif mode == 'subtracted':
                # 減算結果を表示
                bg_roi = self.temporal_background
                if self.roi_active.get() and self.roi_coordinates:
                    x1, y1, x2, y2 = self.roi_coordinates
                    bg_roi = self.temporal_background[y1:y2, x1:x2]

                # 背景減算を適用
                subtracted = apply_temporal_median_subtraction(roi_img, bg_roi)
                display_img = cv2.cvtColor(subtracted, cv2.COLOR_GRAY2BGR)
                title = "背景減算結果"

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
            'roi_coordinates': self.roi_coordinates,
            # コントラスト調整設定を追加
            'use_contrast_enhancement': self.use_contrast_enhancement.get(),
            'contrast_alpha': self.contrast_alpha.get(),
            'brightness_beta': self.brightness_beta.get(),
            'use_clahe': self.use_clahe.get(),
            'clahe_clip_limit': self.clahe_clip_limit.get()
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
        self.min_area.set(config.get('min_area', '100'))
        self.max_area.set(config.get('max_area', '10000'))
        self.binarize_method.set(config.get('binarize_method', 'adaptive'))
        self.relative_thresh.set(config.get('relative_thresh', 0.15))
        self.block_size.set(config.get('block_size', 15))
        self.c_value.set(config.get('c_value', 8))
        self.use_bg_removal.set(config.get('use_bg_removal', False))
        self.bg_kernel_size.set(config.get('bg_kernel_size', 31))
        self.use_temporal_median.set(config.get('use_temporal_median', False))
        self.temporal_frames.set(config.get('temporal_frames', 100))
        self.save_video.set(config.get('save_video', False))
        self.video_fps.set(config.get('video_fps', 10))
        self.video_scale.set(config.get('video_scale', 0.5))
        self.day_start_time.set(config.get('day_start_time', '07:00'))
        self.night_start_time.set(config.get('night_start_time', '19:00'))
        self.measurement_start_time.set(config.get('measurement_start_time', '09:00:00'))
        self.roi_coordinates = config.get('roi_coordinates', None)

        # コントラスト調整設定をロード
        self.use_contrast_enhancement.set(config.get('use_contrast_enhancement', False))
        self.contrast_alpha.set(config.get('contrast_alpha', 1.5))
        self.brightness_beta.set(config.get('brightness_beta', 0))
        self.use_clahe.set(config.get('use_clahe', False))
        self.clahe_clip_limit.set(config.get('clahe_clip_limit', 3.0))

        self.log_output(f'設定をロードしました: {file_path}')

        # コントラスト調整設定の反映をログに出力
        if self.use_contrast_enhancement.get():
            self.log_output(f'コントラスト調整: 有効 (倍率={self.contrast_alpha.get()}, 明度={self.brightness_beta.get()})')
        else:
            self.log_output('コントラスト調整: 無効')
        
        if self.use_clahe.get():
            self.log_output(f'CLAHE: 有効 (制限={self.clahe_clip_limit.get()})')
        else:
            self.log_output('CLAHE: 無効')

        # プレビューを更新
        self.update_preview()

        # ROI情報ラベルの更新
        if self.roi_coordinates:
            x1, y1, x2, y2 = self.roi_coordinates
            self.roi_info_label.config(text=f'ROI: ({x1},{y1})-({x2},{y2})')
        else:
            self.roi_info_label.config(text='ROI未設定')

        # Temporal Median Filterのステータス更新
        if self.use_temporal_median.get() and self.temporal_background is None:
            self.temporal_status_label.config(text='背景未作成')
        else:
            self.temporal_status_label.config(text='背景作成済み')

        # 動画保存設定の反映
        if self.save_video.get():
            self.log_output('動画保存: 有効')
        else:
            self.log_output('動画保存: 無効')

        # バックグラウンド除去設定の反映
        if self.use_bg_removal.get():
            self.log_output('バックグラウンド除去: 有効')
        else:
            self.log_output('バックグラウンド除去: 無効')

        # ROI設定の反映
        if self.roi_active.get():
            self.log_output('ROI機能: 有効')
        else:
            self.log_output('ROI機能: 無効')

        # 解析結果ファイルのパスを更新
        analysis_results_path = os.path.join(self.folder_path.get(), 'analysis_results.csv')
        if os.path.exists(analysis_results_path):
            self.log_output(f'解析結果ファイル: {analysis_results_path}')
        else:
            self.log_output('解析結果ファイル: 未作成')

        # 時間設定の反映
        self.log_output(f'昼開始時刻: {self.day_start_time.get()}')
        self.log_output(f'夜開始時刻: {self.night_start_time.get()}')
        self.log_output(f'測定開始時刻: {self.measurement_start_time.get()}')

        # グラフ描画のための時刻データの確認
        timelog_path = os.path.join(self.folder_path.get(), 'timelog.txt')
        if os.path.exists(timelog_path):
            self.log_output(f'timelog.txt: {timelog_path} （存在します）')
        else:
            self.log_output(f'timelog.txt: {timelog_path} （存在しません）')

        # すぐに解析結果グラフを表示（オプション）
        # self.root.after(100, self.plot_results, pd.read_csv(analysis_results_path))

        # Temporal Median Filterの背景画像をプレビューに反映
        if self.temporal_background is not None:
            bg_preview = cv2.cvtColor(self.temporal_background, cv2.COLOR_GRAY2BGR)
            bg_preview = self.resize_image_for_canvas(bg_preview, 300, 200)
            bg_preview_rgb = cv2.cvtColor(bg_preview, cv2.COLOR_BGR2RGB)
            pil_bg_img = Image.fromarray(bg_preview_rgb)
            photo_bg = ImageTk.PhotoImage(pil_bg_img)

            # 背景画像キャンバスに描画
            self.day_canvas.delete("all")
            self.day_canvas.create_image(0, 0, anchor=tk.NW, image=photo_bg)
            self.day_canvas.image = photo_bg  # 参照を保持

            self.log_output('Temporal Median Filterの背景画像をプレビューに反映しました')

        # すぐに プレビューを更新
        self.root.after(100, self.update_preview)

# エントリーポイント
def main():
    root = tk.Tk()
    app = AnimalDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
