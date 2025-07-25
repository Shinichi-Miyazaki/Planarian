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

def remove_background_rolling_ball(image, radius=50):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    selem = disk(radius)
    bg_removed = white_tophat(gray, selem)
    return bg_removed

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

def analyze(binary, min_area=100, max_area=10000, select_center_only=True, scale_factor=1.0, roi_offset_x=0, roi_offset_y=0):
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

    if select_center_only and candidates:
        # 中心に最も近い1つだけを選択
        closest_candidate = min(candidates, key=lambda x: x['distance_from_center'])
        results.append({k: v for k, v in closest_candidate.items() if k != 'contour'})
        valid_contours.append(closest_candidate['contour'])
    else:
        # すべての候補を返す（従来の動作）
        for candidate in candidates:
            results.append({k: v for k, v in candidate.items() if k != 'contour'})
            valid_contours.append(candidate['contour'])

    return results, valid_contours

class AnimalDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('動物検出GUI')
        self.root.geometry('1200x800')  # サイズを少し大きく調整

        # ウィンドウを最大化可能にする
        self.root.state('zoomed') if self.root.tk.call('tk', 'windowingsystem') == 'win32' else self.root.attributes('-zoomed', True)

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

        # 時間設定
        self.day_start_time = tk.StringVar(value='07:00')
        self.night_start_time = tk.StringVar(value='19:00')

        # ROI関連
        self.roi_coordinates = None  # (x1, y1, x2, y2) in original image coordinates
        self.roi_active = tk.BooleanVar(value=False)
        self.drawing_roi = False
        self.roi_start_x = 0
        self.roi_start_y = 0

        self.create_widgets()

    def create_widgets(self):
        # メインフレーム
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左側：設定パネル
        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # フォルダ選択
        tk.Label(left_frame, text='画像フォルダ:', font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=(5,0))
        tk.Entry(left_frame, textvariable=self.folder_path, width=40).grid(row=1, column=0, padx=5, pady=2)
        tk.Button(left_frame, text='選択', command=self.select_folder).grid(row=1, column=1, padx=5, pady=2)

        # 昼の代表画像
        tk.Label(left_frame, text='昼の代表画像:', font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=(5,0))
        tk.Entry(left_frame, textvariable=self.day_img_path, width=40).grid(row=3, column=0, padx=5, pady=2)
        tk.Button(left_frame, text='選択', command=self.select_day_image).grid(row=3, column=1, padx=5, pady=2)

        # 夜の代表画像
        tk.Label(left_frame, text='夜の代表画像:', font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky='w', pady=(5,0))
        tk.Entry(left_frame, textvariable=self.night_img_path, width=40).grid(row=5, column=0, padx=5, pady=2)
        tk.Button(left_frame, text='選択', command=self.select_night_image).grid(row=5, column=1, padx=5, pady=2)

        # パラメータ設定（コンパクト化）
        param_frame = tk.LabelFrame(left_frame, text='パラメータ設定', font=('Arial', 9, 'bold'))
        param_frame.grid(row=6, column=0, columnspan=2, pady=10, padx=5, sticky='ew')

        size_frame = tk.Frame(param_frame)
        size_frame.pack(pady=2)
        tk.Label(size_frame, text='最小面積:', font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        tk.Entry(size_frame, textvariable=self.min_area, width=8).pack(side=tk.LEFT)
        tk.Label(size_frame, text='最大面積:', font=('Arial', 8)).pack(side=tk.LEFT, padx=2)
        tk.Entry(size_frame, textvariable=self.max_area, width=8).pack(side=tk.LEFT)

        # プレビューボタンをパラメータフレーム内から移動
        tk.Button(param_frame, text='プレビュー更新', command=self.update_preview,
                 bg='lightblue', font=('Arial', 9)).pack(pady=5)

        # 時間設定
        time_frame = tk.LabelFrame(left_frame, text='時間設定', font=('Arial', 9, 'bold'))
        time_frame.grid(row=7, column=0, columnspan=2, pady=5, padx=5, sticky='ew')
        time_entry_frame = tk.Frame(time_frame)
        time_entry_frame.pack(pady=5)
        tk.Label(time_entry_frame, text='昼開始:', font=('Arial', 8)).pack(side=tk.LEFT, padx=3)
        tk.Entry(time_entry_frame, textvariable=self.day_start_time, width=8).pack(side=tk.LEFT, padx=3)
        tk.Label(time_entry_frame, text='夜開始:', font=('Arial', 8)).pack(side=tk.LEFT, padx=3)
        tk.Entry(time_entry_frame, textvariable=self.night_start_time, width=8).pack(side=tk.LEFT, padx=3)

        # 二値化設定（コンパクト化）
        binarize_frame = tk.LabelFrame(left_frame, text='二値化設定', font=('Arial', 9, 'bold'))
        binarize_frame.grid(row=8, column=0, columnspan=2, pady=5, padx=5, sticky='ew')

        # 1列目
        tk.Label(binarize_frame, text='メソッド:', font=('Arial', 8)).grid(row=0, column=0, padx=3, pady=1, sticky='w')
        ttk.Combobox(binarize_frame, textvariable=self.binarize_method, values=['adaptive', 'relative', 'fixed'], state='readonly', width=8).grid(row=0, column=1, padx=3, pady=1)
        tk.Label(binarize_frame, text='相対閾値:', font=('Arial', 8)).grid(row=1, column=0, padx=3, pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.relative_thresh, width=8).grid(row=1, column=1, padx=3, pady=1)
        tk.Label(binarize_frame, text='ブロックサイズ:', font=('Arial', 8)).grid(row=2, column=0, padx=3, pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.block_size, width=8).grid(row=2, column=1, padx=3, pady=1)

        # 2列目
        tk.Label(binarize_frame, text='定数C:', font=('Arial', 8)).grid(row=0, column=2, padx=(10,3), pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.c_value, width=8).grid(row=0, column=3, padx=3, pady=1)
        tk.Label(binarize_frame, text='BG除去:', font=('Arial', 8)).grid(row=1, column=2, padx=(10,3), pady=1, sticky='w')
        ttk.Combobox(binarize_frame, textvariable=self.use_bg_removal, values=[True, False], state='readonly', width=6).grid(row=1, column=3, padx=3, pady=1)
        tk.Label(binarize_frame, text='BGカーネル:', font=('Arial', 8)).grid(row=2, column=2, padx=(10,3), pady=1, sticky='w')
        tk.Entry(binarize_frame, textvariable=self.bg_kernel_size, width=8).grid(row=2, column=3, padx=3, pady=1)

        # ROI設定（コンパクト化）
        roi_frame = tk.LabelFrame(left_frame, text='ROI設定', font=('Arial', 9, 'bold'))
        roi_frame.grid(row=9, column=0, columnspan=2, pady=5, padx=5, sticky='ew')

        tk.Checkbutton(roi_frame, text='ROIを使用', variable=self.roi_active,
                      command=self.toggle_roi, font=('Arial', 8)).grid(row=0, column=0, columnspan=2, pady=2)

        roi_button_frame = tk.Frame(roi_frame)
        roi_button_frame.grid(row=1, column=0, columnspan=2, pady=2)

        tk.Button(roi_button_frame, text='ROI設定', command=self.set_roi_mode,
                 bg='orange', font=('Arial', 8), width=8).pack(side=tk.LEFT, padx=1)
        tk.Button(roi_button_frame, text='ROIクリア', command=self.clear_roi,
                 bg='lightgray', font=('Arial', 8), width=8).pack(side=tk.LEFT, padx=1)

        self.roi_info_label = tk.Label(roi_frame, text='ROI未設定', font=('Arial', 7))
        self.roi_info_label.grid(row=2, column=0, columnspan=2, pady=1)

        # メインボタン（コンパクト化）
        button_frame = tk.Frame(left_frame)
        button_frame.grid(row=10, column=0, columnspan=2, pady=5)

        tk.Button(button_frame, text='解析開始', command=self.start_analysis,
                 bg='lightgreen', font=('Arial', 10, 'bold'), width=12).pack(pady=2)
        tk.Button(button_frame, text='終了', command=self.root.quit,
                 bg='lightcoral', font=('Arial', 10), width=12).pack(pady=2)

        # 右側：プレビューパネル
        right_frame = tk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # プレビューエリア
        preview_frame = tk.LabelFrame(right_frame, text='検出プレビュー', font=('Arial', 10, 'bold'))
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # タブ
        self.notebook = ttk.Notebook(preview_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.day_frame = tk.Frame(self.notebook)
        self.night_frame = tk.Frame(self.notebook)
        self.notebook.add(self.day_frame, text='昼画像プレビュー')
        self.notebook.add(self.night_frame, text='夜画像プレビュー')

        # 昼画像プレビュー
        self.day_canvas = tk.Canvas(self.day_frame, bg='white', width=300, height=200)
        self.day_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 夜画像プレビュー
        self.night_canvas = tk.Canvas(self.night_frame, bg='white', width=300, height=200)
        self.night_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 出力テキストエリア
        output_frame = tk.LabelFrame(right_frame, text='解析結果', font=('Arial', 10, 'bold'))
        output_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = scrolledtext.ScrolledText(output_frame, width=50, height=10)
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
        results, contours = analyze(binary, min_area, max_area, True, scale_factor, roi_offset_x, roi_offset_y)

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
        if results:
            info_text = f"{time_type}{roi_status}: {len(results)}個検出 (面積: {results[0]['area']:.0f})"
        else:
            info_text = f"{time_type}{roi_status}: {len(results)}個検出"
        canvas.create_text(10, 10, anchor=tk.NW, text=info_text,
                          fill="blue", font=("Arial", 10, "bold"))

    def log_output(self, text):
        self.output_text.insert(tk.END, text + '\n')
        self.output_text.see(tk.END)
        self.root.update()

    def run_analysis(self, folder_path, min_area, max_area, binarize_method, relative_thresh, block_size, c_value, use_bg_removal, bg_kernel_size, roi_active, roi_coordinates):
        self.log_output('--- 解析開始 ---')
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        results_list = []

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

            # 二値化
            binary = binarize(roi_img,
                             method=binarize_method,
                             relative_thresh=relative_thresh,
                             block_size=block_size,
                             c_value=c_value,
                             use_bg_removal=use_bg_removal,
                             bg_kernel_size=bg_kernel_size)

            # 解析
            results, _ = analyze(binary, min_area, max_area, True, 1.0, roi_offset_x, roi_offset_y)

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
            self.log_output("解析結果が空のため、グラフは作成されません。")
            return

        timelog_path = os.path.join(self.folder_path.get(), 'timelog.txt')
        if not os.path.exists(timelog_path):
            self.log_output(f"timelog.txtが見つかりません: {timelog_path}")
            messagebox.showwarning("警告", "timelog.txt が見つかりません。グラフは作成されません。")
            return

        try:
            # timelog.txtを読み込み（時刻のみの形式に対応）
            with open(timelog_path, 'r') as f:
                time_lines = f.readlines()

            # 時刻データを処理
            timestamps = []
            filenames = []

            # ソートされた画像ファイル名を取得
            image_files = sorted([f for f in os.listdir(self.folder_path.get())
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

            for i, time_line in enumerate(time_lines):
                time_str = time_line.strip()
                if time_str and i < len(image_files):
                    # 今日の日付と時刻を組み合わせて完全なdatetimeを作成
                    from datetime import date
                    today = date.today()
                    full_datetime_str = f"{today} {time_str}"
                    try:
                        timestamp = pd.to_datetime(full_datetime_str)
                        timestamps.append(timestamp)
                        filenames.append(image_files[i])
                    except:
                        continue

            # DataFrameを作成
            time_df = pd.DataFrame({
                'filename': filenames,
                'datetime': timestamps
            })

            merged_df = pd.merge(df, time_df[['filename', 'datetime']], on='filename', how='left')
            if 'datetime' not in merged_df.columns or merged_df['datetime'].isnull().all():
                self.log_output("日時の結合に失敗しました。ファイル名が一致しているか確認してください。")
                return

            merged_df.sort_values('datetime', inplace=True)
            merged_df.dropna(subset=['datetime'], inplace=True) # 日時がないデータは除外

            day_start_time = datetime.strptime(self.day_start_time.get(), '%H:%M').time()
            night_start_time = datetime.strptime(self.night_start_time.get(), '%H:%M').time()

            graph_window = tk.Toplevel(self.root)
            graph_window.title("解析結果グラフ")
            graph_window.geometry("1000x600")

            fig, ax = plt.subplots(figsize=(10, 6))

            # 夜の時間帯をグレー表示
            if not merged_df.empty:
                unique_dates = merged_df['datetime'].dt.date.unique()
                for d in unique_dates:
                    night_start = datetime.combine(d, night_start_time)
                    # 夜の開始時間が昼の開始時間より遅い場合（通常の昼夜）
                    if night_start_time > day_start_time:
                        day_end = datetime.combine(d + pd.Timedelta(days=1), day_start_time)
                    # 日をまたぐ場合（例：夜19:00～朝7:00）
                    else:
                        day_end = datetime.combine(d, day_start_time)

                    ax.axvspan(night_start, day_end, facecolor='gray', alpha=0.2)


            ax.plot(merged_df['datetime'], merged_df['area'], marker='.', linestyle='-', markersize=4, label='面積 (Area)')
            ax.set_xlabel("日時")
            ax.set_ylabel("面積")
            ax.set_title("動物の面積の時系列変化")
            ax.legend()
            ax.grid(True)

            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

            canvas = FigureCanvasTkAgg(fig, master=graph_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

            plt.tight_layout()

        except Exception as e:
            self.log_output(f"グラフ作成中にエラーが発生しました: {e}")
            messagebox.showerror("エラー", f"グラフ作成中にエラーが発生しました:\n{e}")

    # ROI関連メソッド
    def toggle_roi(self):
        if self.roi_active.get():
            self.log_output('ROI機能を有効にしました')
        else:
            self.log_output('ROI機能を無効にしました')
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

if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalDetectorGUI(root)
    root.mainloop()
