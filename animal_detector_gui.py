import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import pandas as pd
import threading

# --- 画像解析関数 ---
def is_daytime(image, threshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > threshold

def binarize(image, thresh):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        self.root.geometry('1000x700')

        # 変数
        self.folder_path = tk.StringVar()
        self.day_img_path = tk.StringVar()
        self.night_img_path = tk.StringVar()
        self.day_thresh = tk.StringVar(value='120')
        self.night_thresh = tk.StringVar(value='40')
        self.min_area = tk.StringVar(value='100')
        self.max_area = tk.StringVar(value='10000')

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
        tk.Label(left_frame, text='画像フォルダ:', font=('Arial', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=5)
        tk.Entry(left_frame, textvariable=self.folder_path, width=40).grid(row=1, column=0, padx=5, pady=2)
        tk.Button(left_frame, text='選択', command=self.select_folder).grid(row=1, column=1, padx=5, pady=2)

        # 昼の代表画像
        tk.Label(left_frame, text='昼の代表画像:', font=('Arial', 10, 'bold')).grid(row=2, column=0, sticky='w', pady=(15,5))
        tk.Entry(left_frame, textvariable=self.day_img_path, width=40).grid(row=3, column=0, padx=5, pady=2)
        tk.Button(left_frame, text='選択', command=self.select_day_image).grid(row=3, column=1, padx=5, pady=2)

        # 夜の代表画像
        tk.Label(left_frame, text='夜の代表画像:', font=('Arial', 10, 'bold')).grid(row=4, column=0, sticky='w', pady=(15,5))
        tk.Entry(left_frame, textvariable=self.night_img_path, width=40).grid(row=5, column=0, padx=5, pady=2)
        tk.Button(left_frame, text='選択', command=self.select_night_image).grid(row=5, column=1, padx=5, pady=2)

        # パラメータ設定
        param_frame = tk.LabelFrame(left_frame, text='パラメータ設定', font=('Arial', 10, 'bold'))
        param_frame.grid(row=6, column=0, columnspan=2, pady=15, padx=5, sticky='ew')

        tk.Label(param_frame, text='昼の閾値:').grid(row=0, column=0, padx=5, pady=5, sticky='w')
        tk.Entry(param_frame, textvariable=self.day_thresh, width=10).grid(row=0, column=1, padx=5, pady=5)

        tk.Label(param_frame, text='夜の閾値:').grid(row=1, column=0, padx=5, pady=5, sticky='w')
        tk.Entry(param_frame, textvariable=self.night_thresh, width=10).grid(row=1, column=1, padx=5, pady=5)

        tk.Label(param_frame, text='最小面積:').grid(row=2, column=0, padx=5, pady=5, sticky='w')
        tk.Entry(param_frame, textvariable=self.min_area, width=10).grid(row=2, column=1, padx=5, pady=5)

        tk.Label(param_frame, text='最大面積:').grid(row=3, column=0, padx=5, pady=5, sticky='w')
        tk.Entry(param_frame, textvariable=self.max_area, width=10).grid(row=3, column=1, padx=5, pady=5)

        # ROI設定
        roi_frame = tk.LabelFrame(left_frame, text='ROI設定', font=('Arial', 10, 'bold'))
        roi_frame.grid(row=7, column=0, columnspan=2, pady=15, padx=5, sticky='ew')

        tk.Checkbutton(roi_frame, text='ROIを使用', variable=self.roi_active,
                      command=self.toggle_roi).grid(row=0, column=0, columnspan=2, pady=5)

        roi_button_frame = tk.Frame(roi_frame)
        roi_button_frame.grid(row=1, column=0, columnspan=2, pady=5)

        tk.Button(roi_button_frame, text='ROI設定', command=self.set_roi_mode,
                 bg='orange', font=('Arial', 9), width=10).pack(side=tk.LEFT, padx=2)
        tk.Button(roi_button_frame, text='ROIクリア', command=self.clear_roi,
                 bg='lightgray', font=('Arial', 9), width=10).pack(side=tk.LEFT, padx=2)

        self.roi_info_label = tk.Label(roi_frame, text='ROI未設定', font=('Arial', 8))
        self.roi_info_label.grid(row=2, column=0, columnspan=2, pady=2)

        # プレビューボタン
        tk.Button(param_frame, text='プレビュー更新', command=self.update_preview,
                 bg='lightblue', font=('Arial', 10)).grid(row=4, column=0, columnspan=2, pady=10)

        # メインボタン
        button_frame = tk.Frame(left_frame)
        button_frame.grid(row=8, column=0, columnspan=2, pady=15)

        tk.Button(button_frame, text='解析開始', command=self.start_analysis,
                 bg='lightgreen', font=('Arial', 12, 'bold'), width=12).pack(pady=5)
        tk.Button(button_frame, text='終了', command=self.root.quit,
                 bg='lightcoral', font=('Arial', 12), width=12).pack(pady=5)

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
            day_thresh = int(self.day_thresh.get())
            night_thresh = int(self.night_thresh.get())
            min_area = int(self.min_area.get())
            max_area = int(self.max_area.get())
        except ValueError:
            return

        # 昼画像プレビュー
        if self.day_img_path.get() and os.path.exists(self.day_img_path.get()):
            day_img = cv2.imread(self.day_img_path.get())
            if day_img is not None:
                self.show_preview(day_img, day_thresh, min_area, max_area, self.day_canvas, "昼")

        # 夜画像プレビュー
        if self.night_img_path.get() and os.path.exists(self.night_img_path.get()):
            night_img = cv2.imread(self.night_img_path.get())
            if night_img is not None:
                self.show_preview(night_img, night_thresh, min_area, max_area, self.night_canvas, "夜")

    def show_preview(self, image, thresh, min_area, max_area, canvas, time_type):
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

        # 二値化
        binary = binarize(resized_img, thresh)

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
            dayth = int(self.day_thresh.get())
            nightth = int(self.night_thresh.get())
            minarea = int(self.min_area.get())
            maxarea = int(self.max_area.get())
        except ValueError:
            messagebox.showerror('エラー', 'パラメータは整数で入力してください')
            return

        # 別スレッドで解析実行
        threading.Thread(target=self.run_analysis, args=(dayth, nightth, minarea, maxarea), daemon=True).start()

    def run_analysis(self, dayth, nightth, minarea, maxarea):
        try:
            self.output_text.delete(1.0, tk.END)
            self.log_output('解析を開始します...')

            folder = self.folder_path.get()
            dayimg_path = self.day_img_path.get()
            nightimg_path = self.night_img_path.get()

            # 昼夜判定用閾値を自動設定
            dayimg = cv2.imread(dayimg_path)
            nightimg = cv2.imread(nightimg_path)
            day_brightness = np.mean(cv2.cvtColor(dayimg, cv2.COLOR_BGR2GRAY))
            night_brightness = np.mean(cv2.cvtColor(nightimg, cv2.COLOR_BGR2GRAY))
            judge_th = (day_brightness + night_brightness) / 2
            self.log_output(f'昼夜判定閾値: {judge_th:.1f}')

            # ROI情報の表示
            roi_info = ""
            if self.roi_active.get() and self.roi_coordinates:
                roi_info = f', ROI: {self.roi_coordinates}'
                self.log_output(f'ROI適用: {self.roi_coordinates}')

            self.log_output(f'使用パラメータ - 昼閾値:{dayth}, 夜閾値:{nightth}, 面積範囲:{minarea}-{maxarea}{roi_info}')

            # 出力フォルダ
            outdir = os.path.join(folder, 'output')
            os.makedirs(outdir, exist_ok=True)

            results = []
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
            self.log_output(f'処理対象ファイル数: {len(files)}')

            for i, fname in enumerate(files):
                fpath = os.path.join(folder, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue

                # ROI適用
                roi_img = img
                roi_offset_x, roi_offset_y = 0, 0
                if self.roi_active.get() and self.roi_coordinates:
                    x1, y1, x2, y2 = self.roi_coordinates
                    roi_img = img[y1:y2, x1:x2]
                    roi_offset_x, roi_offset_y = x1, y1

                is_day = is_daytime(roi_img, judge_th)
                th = dayth if is_day else nightth
                binary = binarize(roi_img, th)

                # 解析実行（ROIオフセットを考慮）
                feats, contours = analyze(binary, minarea, maxarea, True, 1.0, roi_offset_x, roi_offset_y)
                for feat in feats:
                    feat['filename'] = fname
                    feat['time_type'] = '昼' if is_day else '夜'
                    results.append(feat)

                # 検出結果画像保存（元画像全体に結果を描画）
                outimg = img.copy()

                # ROI矩形を描画（ROI使用時）
                if self.roi_active.get() and self.roi_coordinates:
                    x1, y1, x2, y2 = self.roi_coordinates
                    cv2.rectangle(outimg, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 青色でROI矩形

                # ROI領域内での輪郭描画
                if contours:
                    # 輪郭座標をROIオフセット分調整
                    adjusted_contours = []
                    for cnt in contours:
                        adjusted_cnt = cnt.copy()
                        adjusted_cnt[:, :, 0] += roi_offset_x
                        adjusted_cnt[:, :, 1] += roi_offset_y
                        adjusted_contours.append(adjusted_cnt)
                    cv2.drawContours(outimg, adjusted_contours, -1, (0,255,0), 2)

                for feat in feats:
                    cv2.circle(outimg, (int(feat['centroid_x']), int(feat['centroid_y'])), 5, (0,0,255), -1)

                cv2.imwrite(os.path.join(outdir, fname), outimg)

                if (i + 1) % 10 == 0:
                    self.log_output(f'処理済み: {i + 1}/{len(files)}')

            if results:
                df = pd.DataFrame(results)
                df.to_csv(os.path.join(folder, 'results.csv'), index=False)
                self.log_output(f'解析完了！検出された動物数: {len(results)}')
                self.log_output('results.csvとoutputフォルダを確認してください。')
                messagebox.showinfo('完了', f'解析が完了しました。\n検出された動物数: {len(results)}')
            else:
                self.log_output('動物が検出されませんでした。パラメータを調整してください。')
                messagebox.showwarning('結果', '動物が検出されませんでした。\nパラメータを調整してください。')

        except Exception as e:
            self.log_output(f'エラーが発生しました: {str(e)}')
            messagebox.showerror('エラー', f'解析中にエラーが発生しました:\n{str(e)}')

if __name__ == "__main__":
    root = tk.Tk()
    app = AnimalDetectorGUI(root)
    root.mainloop()
