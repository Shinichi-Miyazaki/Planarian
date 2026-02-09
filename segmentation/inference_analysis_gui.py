"""
セグメンテーション推論 + 行動解析 GUIランチャー

注: albumentationsは不要です（推論時）
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import date, datetime
import threading

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import config
from run_inference_analysis import run_inference_and_analysis


class InferenceAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("セグメンテーション推論 + 行動解析")
        self.root.geometry("900x700")

        # 単一フォルダモード用の変数
        self.images_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_path = tk.StringVar(value=config.BEST_MODEL_PATH)
        self.day_start = tk.StringVar(value='12:00')
        self.night_start = tk.StringVar(value='23:00')
        self.measurement_start = tk.StringVar(value='09:00:00')
        self.measurement_date = tk.StringVar(value=date.today().strftime('%Y-%m-%d'))
        self.time_interval = tk.IntVar(value=10)
        self.create_video = tk.BooleanVar(value=False)
        self.use_onnx = tk.BooleanVar(value=False)
        self.inference_device = tk.StringVar(value='cpu')

        # バッチモード用の変数
        self.parent_dir = tk.StringVar()
        self.excel_file = tk.StringVar()
        self.image_threshold = tk.IntVar(value=1000)
        self.batch_use_onnx = tk.BooleanVar(value=False)
        self.batch_inference_device = tk.StringVar(value='cpu')
        self.detected_folders = []  # [(folder_path, image_count, measurement_date, measurement_start_time), ...]

        self.create_widgets()

    def create_widgets(self):
        # ノートブック（タブ）を作成
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # タブ1: 単一フォルダモード
        single_frame = ttk.Frame(notebook, padding="10")
        notebook.add(single_frame, text="単一フォルダ解析")
        self.create_single_mode_widgets(single_frame)

        # タブ2: バッチモード
        batch_frame = ttk.Frame(notebook, padding="10")
        notebook.add(batch_frame, text="バッチフォルダ解析")
        self.create_batch_mode_widgets(batch_frame)

    def create_single_mode_widgets(self, parent):
        """単一フォルダモードのウィジェットを作成"""
        row = 0

        ttk.Label(parent, text="単一フォルダ解析",
                 font=('Arial', 14, 'bold')).grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=5)
        row += 1

        ttk.Label(parent, text="画像フォルダ:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.images_dir, width=50).grid(row=row, column=1, pady=5)
        ttk.Button(parent, text="参照", command=self.browse_images).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        ttk.Label(parent, text="モデル:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.model_path, width=50).grid(row=row, column=1, pady=5)
        ttk.Button(parent, text="参照", command=self.browse_model).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        ttk.Label(parent, text="出力フォルダ:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.output_dir, width=50).grid(row=row, column=1, pady=5)
        ttk.Button(parent, text="参照", command=self.browse_output).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=10)
        row += 1

        ttk.Label(parent, text="時間設定", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(parent, text="測定日付 (YYYY-MM-DD):").grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(parent, textvariable=self.measurement_date, width=20).grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1

        ttk.Label(parent, text="測定開始 (HH:MM:SS):").grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(parent, textvariable=self.measurement_start, width=20).grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1

        ttk.Label(parent, text="昼開始 (HH:MM):").grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(parent, textvariable=self.day_start, width=20).grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1

        ttk.Label(parent, text="夜開始 (HH:MM):").grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(parent, textvariable=self.night_start, width=20).grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1

        ttk.Label(parent, text="時間間隔（分）:").grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(parent, textvariable=self.time_interval, width=20).grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1

        ttk.Checkbutton(parent, text="動画を作成",
                       variable=self.create_video).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=10)
        row += 1

        # ONNX / GPU設定を1行に
        ttk.Label(parent, text="推論設定", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1

        inference_frame = ttk.Frame(parent)
        inference_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=3)

        ttk.Checkbutton(inference_frame, text="ONNX Runtime (GPU対応)",
                       variable=self.use_onnx).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(inference_frame, text="デバイス:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Radiobutton(inference_frame, text="CPU", variable=self.inference_device, value='cpu').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(inference_frame, text="CUDA (NVIDIA GPU)", variable=self.inference_device, value='cuda').pack(side=tk.LEFT, padx=5)
        row += 1

        ttk.Label(parent, text="※ ONNX使用時: .pthモデルは自動的に.onnxに変換 | RTX 5070 TiはCUDA推奨",
                 font=('Arial', 8), foreground='gray').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=10)
        row += 1

        self.run_button = ttk.Button(parent, text="実行", command=self.run_single_analysis)
        self.run_button.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        self.status_label = ttk.Label(parent, text="準備完了", foreground="green")
        self.status_label.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1

        self.progress = ttk.Progressbar(parent, mode='indeterminate', length=400)
        self.progress.grid(row=row, column=0, columnspan=3, pady=5)

    def create_batch_mode_widgets(self, parent):
        """バッチモードのウィジェットを作成"""
        row = 0

        ttk.Label(parent, text="バッチフォルダ解析",
                 font=('Arial', 14, 'bold')).grid(row=row, column=0, columnspan=6, pady=10)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=6, sticky='ew', pady=5)
        row += 1

        # 親フォルダとExcelファイルを同じ行に
        ttk.Label(parent, text="親フォルダ:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.parent_dir, width=30).grid(row=row, column=1, pady=5, sticky='ew')
        ttk.Button(parent, text="参照", command=self.browse_parent_dir).grid(row=row, column=2, padx=5, pady=5)

        ttk.Label(parent, text="Excel:").grid(row=row, column=3, sticky=tk.W, pady=5, padx=(10,0))
        ttk.Entry(parent, textvariable=self.excel_file, width=30).grid(row=row, column=4, pady=5, sticky='ew')
        ttk.Button(parent, text="参照", command=self.browse_excel_file).grid(row=row, column=5, padx=5, pady=5)
        row += 1

        ttk.Label(parent, text="※ Excel: dir_name, start_date, start_time列が必要",
                 font=('Arial', 8), foreground='gray').grid(row=row, column=0, columnspan=6, sticky=tk.W, pady=(0, 5))
        row += 1

        # 画像数閾値とモデルを同じ行に
        ttk.Label(parent, text="画像数閾値:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(parent, textvariable=self.image_threshold, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)

        ttk.Label(parent, text="モデル:").grid(row=row, column=2, sticky=tk.W, pady=5, padx=(10,0))
        ttk.Entry(parent, textvariable=self.model_path, width=30).grid(row=row, column=3, columnspan=2, pady=5, sticky='ew')
        ttk.Button(parent, text="参照", command=self.browse_model).grid(row=row, column=5, padx=5, pady=5)
        row += 1

        # フォルダ検索ボタン
        ttk.Button(parent, text="フォルダを検索", command=self.search_folders).grid(
            row=row, column=0, columnspan=6, pady=10)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=6, sticky='ew', pady=5)
        row += 1

        # 共通時間設定を1行に
        ttk.Label(parent, text="時間設定", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=6, sticky=tk.W, pady=5)
        row += 1

        time_frame = ttk.Frame(parent)
        time_frame.grid(row=row, column=0, columnspan=6, sticky='ew', pady=5)

        ttk.Label(time_frame, text="昼開始:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(time_frame, textvariable=self.day_start, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(time_frame, text="夜開始:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(time_frame, textvariable=self.night_start, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Label(time_frame, text="間隔（分）:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Entry(time_frame, textvariable=self.time_interval, width=8).pack(side=tk.LEFT, padx=5)

        ttk.Checkbutton(time_frame, text="動画を作成", variable=self.create_video).pack(side=tk.LEFT, padx=(20, 5))
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=6, sticky='ew', pady=5)
        row += 1

        # ONNX / GPU設定を1行に
        ttk.Label(parent, text="推論設定", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=6, sticky=tk.W, pady=5)
        row += 1

        inference_frame = ttk.Frame(parent)
        inference_frame.grid(row=row, column=0, columnspan=6, sticky='ew', pady=5)

        ttk.Checkbutton(inference_frame, text="ONNX Runtime (GPU対応)",
                       variable=self.batch_use_onnx).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(inference_frame, text="デバイス:").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Radiobutton(inference_frame, text="CPU", variable=self.batch_inference_device, value='cpu').pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(inference_frame, text="CUDA (NVIDIA GPU)", variable=self.batch_inference_device, value='cuda').pack(side=tk.LEFT, padx=5)
        row += 1

        ttk.Label(parent, text="※ RTX 5070 TiはCUDAを推奨",
                 font=('Arial', 8), foreground='orange').grid(row=row, column=0, columnspan=6, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=6, sticky='ew', pady=5)
        row += 1

        # 検出されたフォルダリスト
        ttk.Label(parent, text="検出されたフォルダ", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=6, sticky=tk.W, pady=5)
        row += 1

        # Treeviewでフォルダリストを表示
        tree_frame = ttk.Frame(parent)
        tree_frame.grid(row=row, column=0, columnspan=6, sticky='nsew', pady=5)
        parent.rowconfigure(row, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(4, weight=1)

        # スクロールバー
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Treeview作成
        self.folder_tree = ttk.Treeview(
            tree_frame,
            columns=('path', 'count', 'date', 'time'),
            show='headings',
            yscrollcommand=tree_scroll.set,
            height=10
        )
        self.folder_tree.heading('path', text='フォルダパス')
        self.folder_tree.heading('count', text='画像数')
        self.folder_tree.heading('date', text='測定日付')
        self.folder_tree.heading('time', text='測定開始時刻')

        self.folder_tree.column('path', width=400)
        self.folder_tree.column('count', width=80)
        self.folder_tree.column('date', width=100)
        self.folder_tree.column('time', width=100)

        self.folder_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.folder_tree.yview)

        # ダブルクリックで編集
        self.folder_tree.bind('<Double-1>', self.edit_folder_time)
        row += 1

        # ボタンフレーム
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, columnspan=6, pady=10)

        ttk.Button(button_frame, text="選択フォルダを削除", command=self.remove_selected_folder).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="全て削除", command=self.clear_all_folders).pack(side=tk.LEFT, padx=5)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=6, sticky='ew', pady=10)
        row += 1

        # バッチ実行ボタン
        self.batch_run_button = ttk.Button(parent, text="バッチ実行", command=self.run_batch_analysis)
        self.batch_run_button.grid(row=row, column=0, columnspan=6, pady=10)
        row += 1

        # ステータス
        self.batch_status_label = ttk.Label(parent, text="準備完了", foreground="green")
        self.batch_status_label.grid(row=row, column=0, columnspan=6, pady=5)
        row += 1

        self.batch_progress = ttk.Progressbar(parent, mode='determinate', length=400)
        self.batch_progress.grid(row=row, column=0, columnspan=6, pady=5)

    def browse_images(self):
        directory = filedialog.askdirectory(title="画像フォルダを選択")
        if directory:
            self.images_dir.set(directory)

    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="モデルファイルを選択",
            filetypes=[("PyTorchモデル", "*.pth"), ("すべて", "*.*")])
        if filename:
            self.model_path.set(filename)

    def browse_output(self):
        directory = filedialog.askdirectory(title="出力フォルダを選択")
        if directory:
            self.output_dir.set(directory)

    def browse_parent_dir(self):
        """親フォルダを選択"""
        directory = filedialog.askdirectory(title="親フォルダを選択")
        if directory:
            self.parent_dir.set(directory)

    def browse_excel_file(self):
        """Excelファイルを選択"""
        filename = filedialog.askopenfilename(
            title="Excelファイルを選択",
            filetypes=[("Excelファイル", "*.xlsx *.xls"), ("すべて", "*.*")])
        if filename:
            self.excel_file.set(filename)

    def search_folders(self):
        """親フォルダ配下を走査して画像フォルダを検出（Excel指定時はExcelから読み込み）"""
        parent = self.parent_dir.get()
        if not parent:
            messagebox.showerror("エラー", "親フォルダを指定してください")
            return

        if not os.path.exists(parent):
            messagebox.showerror("エラー", f"親フォルダが存在しません:\n{parent}")
            return

        threshold = self.image_threshold.get()
        excel_file = self.excel_file.get()

        self.batch_status_label.config(text="フォルダを検索中...", foreground="blue")
        self.batch_progress.config(mode='indeterminate')
        self.batch_progress.start()

        # 別スレッドで検索
        thread = threading.Thread(target=self._search_folders_thread, args=(parent, threshold, excel_file))
        thread.daemon = True
        thread.start()

    def _search_folders_thread(self, parent_path, threshold, excel_file=None):
        """フォルダ検索のスレッド処理（Excel指定時はExcelから読み込み）"""
        try:
            found_folders = []
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

            # Excelファイルが指定されている場合
            if excel_file and os.path.exists(excel_file):
                import pandas as pd

                print(f"Excelファイルを読み込み中: {excel_file}")

                # Excelファイルを読み込み（1枚目のシート）
                df = pd.read_excel(excel_file, sheet_name=0)

                # 必要なカラムの確認
                required_columns = ['dir_name', 'start_date', 'start_time']
                missing_columns = [col for col in required_columns if col not in df.columns]

                if missing_columns:
                    error_msg = f"Excelファイルに必要な列がありません: {', '.join(missing_columns)}\n\n必要な列: dir_name, start_date, start_time"
                    self.root.after(0, lambda: self._on_search_error(error_msg))
                    return

                print(f"Excelから{len(df)}行読み込み")

                # 各行を処理
                for _, row in df.iterrows():
                    dir_name = str(row['dir_name']).strip()
                    start_date = str(row['start_date']).strip()
                    start_time = str(row['start_time']).strip()

                    # dir_nameに一致するフォルダを親フォルダ配下で検索
                    folder_path = None
                    for root, dirs, files in os.walk(parent_path):
                        if os.path.basename(root) == dir_name:
                            folder_path = root
                            break

                    if not folder_path:
                        print(f"⚠️ フォルダが見つかりません: {dir_name}")
                        continue

                    # 画像ファイル数をカウント
                    image_count = sum(1 for f in os.listdir(folder_path)
                                    if f.lower().endswith(image_extensions))

                    if image_count < threshold:
                        print(f"⚠️ 画像数が閾値未満: {dir_name} ({image_count}枚 < {threshold}枚)")
                        continue

                    # start_dateをYYYY-MM-DD形式に変換（yyyyMMdd → YYYY-MM-DD）
                    if len(start_date) == 8 and start_date.isdigit():
                        formatted_date = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:8]}"
                    else:
                        # すでにYYYY-MM-DD形式の場合やその他の形式
                        formatted_date = start_date

                    # start_timeがHH:mm:ss形式かチェック
                    if len(start_time.split(':')) == 3:
                        formatted_time = start_time
                    else:
                        print(f"⚠️ 時刻フォーマットが不正: {dir_name} - {start_time}")
                        continue

                    found_folders.append((folder_path, image_count, formatted_date, formatted_time))
                    print(f"✓ 検出: {dir_name} ({image_count}枚, {formatted_date} {formatted_time})")

            else:
                # Excelファイルが指定されていない場合は従来の検索方法
                print(f"親フォルダを走査中: {parent_path}")

                for root, dirs, files in os.walk(parent_path):
                    # 画像ファイル数をカウント
                    image_count = sum(1 for f in files if f.lower().endswith(image_extensions))

                    # 閾値以上の画像があればリストに追加
                    if image_count >= threshold:
                        # デフォルトの測定日付と時刻を設定
                        default_date = date.today().strftime('%Y-%m-%d')
                        default_time = self.measurement_start.get()
                        found_folders.append((root, image_count, default_date, default_time))
                        print(f"✓ 検出: {os.path.basename(root)} ({image_count}枚)")

            # UIスレッドで結果を表示
            self.root.after(0, lambda: self._update_folder_list(found_folders))

        except Exception as e:
            import traceback
            error_msg = f"フォルダ検索中にエラーが発生しました:\n{str(e)}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._on_search_error(error_msg))

    def _update_folder_list(self, folders):
        """検出されたフォルダをTreeviewに表示"""
        self.batch_progress.stop()
        self.batch_progress.config(mode='determinate')

        # 既存のアイテムをクリア
        for item in self.folder_tree.get_children():
            self.folder_tree.delete(item)

        # 新しいフォルダを追加
        self.detected_folders = folders
        for folder_path, image_count, meas_date, meas_time in folders:
            self.folder_tree.insert('', 'end', values=(folder_path, image_count, meas_date, meas_time))

        self.batch_status_label.config(
            text=f"{len(folders)}個のフォルダを検出しました",
            foreground="green"
        )

        if len(folders) == 0:
            messagebox.showinfo("結果", f"画像数が{self.image_threshold.get()}枚以上のフォルダが見つかりませんでした")

    def _on_search_error(self, error_msg):
        """検索エラー時の処理"""
        self.batch_progress.stop()
        self.batch_status_label.config(text="エラー", foreground="red")
        messagebox.showerror("エラー", error_msg)

    def edit_folder_time(self, event):
        """フォルダの測定日時を編集"""
        selection = self.folder_tree.selection()
        if not selection:
            return

        item = selection[0]
        values = self.folder_tree.item(item, 'values')

        # 編集ダイアログを表示
        dialog = tk.Toplevel(self.root)
        dialog.title("測定日時の編集")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text=f"フォルダ: {os.path.basename(values[0])}",
                 font=('Arial', 10, 'bold')).pack(pady=10)

        # 日付入力
        date_frame = ttk.Frame(dialog)
        date_frame.pack(pady=5)
        ttk.Label(date_frame, text="測定日付 (YYYY-MM-DD):").pack(side=tk.LEFT, padx=5)
        date_var = tk.StringVar(value=values[2])
        ttk.Entry(date_frame, textvariable=date_var, width=15).pack(side=tk.LEFT, padx=5)

        # 時刻入力
        time_frame = ttk.Frame(dialog)
        time_frame.pack(pady=5)
        ttk.Label(time_frame, text="測定開始時刻 (HH:MM:SS):").pack(side=tk.LEFT, padx=5)
        time_var = tk.StringVar(value=values[3])
        ttk.Entry(time_frame, textvariable=time_var, width=15).pack(side=tk.LEFT, padx=5)

        # 保存ボタン
        def save_changes():
            try:
                # 日付フォーマットの検証
                datetime.strptime(date_var.get(), '%Y-%m-%d')
                # 時刻フォーマットの検証
                datetime.strptime(time_var.get(), '%H:%M:%S')

                # Treeviewを更新
                self.folder_tree.item(item, values=(values[0], values[1], date_var.get(), time_var.get()))

                # detected_foldersも更新
                idx = self.folder_tree.index(item)
                folder_list = list(self.detected_folders)
                folder_list[idx] = (values[0], int(values[1]), date_var.get(), time_var.get())
                self.detected_folders = folder_list

                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("エラー", f"日付または時刻の形式が正しくありません:\n{str(e)}")

        ttk.Button(dialog, text="保存", command=save_changes).pack(pady=10)

    def remove_selected_folder(self):
        """選択されたフォルダをリストから削除"""
        selection = self.folder_tree.selection()
        if not selection:
            messagebox.showinfo("情報", "削除するフォルダを選択してください")
            return

        for item in selection:
            self.folder_tree.delete(item)

        # detected_foldersを更新
        self.detected_folders = [
            (self.folder_tree.item(item, 'values')[0],
             int(self.folder_tree.item(item, 'values')[1]),
             self.folder_tree.item(item, 'values')[2],
             self.folder_tree.item(item, 'values')[3])
            for item in self.folder_tree.get_children()
        ]

    def clear_all_folders(self):
        """全フォルダをリストから削除"""
        if not self.detected_folders:
            return

        if messagebox.askyesno("確認", "全てのフォルダをリストから削除しますか?"):
            for item in self.folder_tree.get_children():
                self.folder_tree.delete(item)
            self.detected_folders = []
            self.batch_status_label.config(text="準備完了", foreground="green")

    def validate_inputs(self):
        if not self.images_dir.get():
            messagebox.showerror("エラー", "画像フォルダを指定してください")
            return False
        if not os.path.exists(self.images_dir.get()):
            messagebox.showerror("エラー", f"画像フォルダが存在しません:\n{self.images_dir.get()}")
            return False

        # 画像ファイルの確認
        image_files = [f for f in os.listdir(self.images_dir.get())
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
        if len(image_files) == 0:
            messagebox.showerror("エラー", f"画像フォルダに画像ファイルが見つかりません:\n{self.images_dir.get()}\n\n対応形式: .png, .jpg, .jpeg, .bmp, .tif, .tiff")
            return False

        if not self.model_path.get():
            messagebox.showerror("エラー", "モデルファイルを指定してください")
            return False
        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("エラー", f"モデルファイルが存在しません:\n{self.model_path.get()}\n\nGoogle Colabで学習済みモデルを生成し、\nmodels/best_unet.pthに配置してください。")
            return False
        if not self.output_dir.get():
            messagebox.showerror("エラー", "出力フォルダを指定してください")
            return False

        # 出力フォルダの作成確認
        try:
            os.makedirs(self.output_dir.get(), exist_ok=True)
        except Exception as e:
            messagebox.showerror("エラー", f"出力フォルダを作成できません:\n{self.output_dir.get()}\n\n{str(e)}")
            return False

        return True

    def run_single_analysis(self):
        """単一フォルダモードの実行"""
        if not self.validate_inputs():
            return

        self.run_button.config(state='disabled')
        self.status_label.config(text="実行中...", foreground="blue")
        self.progress.start()

        thread = threading.Thread(target=self._run_single_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_single_analysis_thread(self):
        """単一フォルダモードのスレッド処理"""
        try:
            measurement_date = datetime.strptime(self.measurement_date.get(), '%Y-%m-%d').date()

            if self.use_onnx.get():
                # ONNX Runtime推論
                self._run_onnx_inference_and_analysis(measurement_date)
            else:
                # PyTorch推論
                analysis_info = run_inference_and_analysis(
                    images_dir=self.images_dir.get(),
                    output_dir=self.output_dir.get(),
                    model_path=self.model_path.get(),
                    create_video=self.create_video.get(),
                    time_interval_minutes=self.time_interval.get(),
                    day_start_time=self.day_start.get(),
                    night_start_time=self.night_start.get(),
                    measurement_start_time=self.measurement_start.get(),
                    measurement_date=measurement_date)

            self.root.after(0, lambda: self._on_single_completion(True, "処理が完了しました"))
        except Exception as e:
            import traceback
            error_msg = f"エラーが発生しました:\n\n{type(e).__name__}: {str(e)}\n\n詳細:\n{traceback.format_exc()}"
            print(error_msg)
            self.root.after(0, lambda: self._on_single_completion(False, error_msg))

    def _run_onnx_inference_and_analysis(self, measurement_date):
        """ONNX Runtime推論 + 行動解析"""
        import json
        from inference_onnx import inference_onnx
        from behavior_analysis import BehaviorAnalyzer

        images_dir = self.images_dir.get()
        output_dir = self.output_dir.get()
        model_path = self.model_path.get()
        device = self.inference_device.get()

        # 時間設定を保存
        time_config = {
            'day_start_time': self.day_start.get(),
            'night_start_time': self.night_start.get(),
            'measurement_start_time': self.measurement_start.get(),
            'measurement_date': measurement_date.strftime('%Y-%m-%d')
        }
        config_path = os.path.join(output_dir, 'time_config.json')
        os.makedirs(output_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(time_config, f, indent=2)

        # .pthモデルを.onnxに変換（必要な場合）
        if model_path.endswith('.pth'):
            onnx_model_path = model_path.replace('.pth', '.onnx')
            if not os.path.exists(onnx_model_path):
                print(f"ONNXモデルが見つかりません。変換中: {model_path} -> {onnx_model_path}")
                from export_onnx import export_to_onnx
                export_to_onnx(model_path, onnx_model_path)
            model_path = onnx_model_path

        # ONNX推論
        print(f"ONNX Runtime推論を開始... (device: {device})")
        use_directml = (device == 'directml')

        inference_onnx(
            images_dir=images_dir,
            output_dir=output_dir,
            model_path=model_path,
            create_video=self.create_video.get(),
            device=device,
            use_directml=use_directml
        )

        csv_path = os.path.join(output_dir, 'analysis_results.csv')

        # 行動解析
        print("行動解析を開始...")
        analyzer = BehaviorAnalyzer(
            csv_path=csv_path,
            time_interval_minutes=self.time_interval.get(),
            day_start_time=self.day_start.get(),
            night_start_time=self.night_start.get(),
            measurement_start_time=self.measurement_start.get(),
            measurement_date=measurement_date
        )

        analyzer.load_data()
        analyzer.calculate_movement()
        analyzer.calculate_immobility_ratio()
        analyzer.aggregate_by_time()
        analyzer.apply_moving_average(window=1)
        analyzer.create_plots(output_dir)
        analyzer.save_detailed_csv(output_dir)
        analyzer.generate_summary_report(output_dir)

    def _on_single_completion(self, success, message):
        """単一フォルダモード完了時の処理"""
        self.progress.stop()
        self.run_button.config(state='normal')

        if success:
            self.status_label.config(text="完了", foreground="green")
            messagebox.showinfo("完了", message)
        else:
            self.status_label.config(text="エラー", foreground="red")
            if len(message) > 200:
                self._show_error_detail(message)
            else:
                messagebox.showerror("エラー", message)

    def run_batch_analysis(self):
        """バッチモードの実行"""
        if not self.detected_folders:
            messagebox.showerror("エラー", "実行するフォルダがありません。\n「フォルダを検索」ボタンでフォルダを検出してください。")
            return

        if not self.model_path.get():
            messagebox.showerror("エラー", "モデルファイルを指定してください")
            return

        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("エラー", f"モデルファイルが存在しません:\n{self.model_path.get()}")
            return

        # 確認ダイアログ
        if not messagebox.askyesno("確認", f"{len(self.detected_folders)}個のフォルダに対して解析を実行しますか?"):
            return

        self.batch_run_button.config(state='disabled')
        self.batch_status_label.config(text="バッチ処理中...", foreground="blue")
        self.batch_progress['value'] = 0
        self.batch_progress['maximum'] = len(self.detected_folders)

        # 別スレッドでバッチ実行
        thread = threading.Thread(target=self._run_batch_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_batch_analysis_thread(self):
        """バッチ解析のスレッド処理"""
        results = []  # (folder_path, success, message)
        analysis_results = []  # 統合解析用のデータ収集

        # ONNX使用時の事前準備
        use_onnx = self.batch_use_onnx.get()
        device = self.batch_inference_device.get()
        onnx_model_path = None

        if use_onnx:
            model_path = self.model_path.get()
            if model_path.endswith('.pth'):
                onnx_model_path = model_path.replace('.pth', '.onnx')
                if not os.path.exists(onnx_model_path):
                    print(f"ONNXモデルが見つかりません。変換中: {model_path} -> {onnx_model_path}")
                    try:
                        from export_onnx import export_to_onnx
                        export_to_onnx(model_path, onnx_model_path)
                        print(f"✓ ONNX変換完了: {onnx_model_path}")
                    except Exception as e:
                        error_msg = f"ONNX変換エラー: {str(e)}"
                        print(error_msg)
                        self.root.after(0, lambda: self._on_batch_completion([(None, False, error_msg)]))
                        return
            else:
                onnx_model_path = model_path
            print(f"バッチ処理: ONNX Runtime使用 (device: {device})")

        for idx, (folder_path, image_count, meas_date, meas_time) in enumerate(self.detected_folders):
            try:
                # ステータス更新
                self.root.after(0, lambda i=idx, f=folder_path: self._update_batch_status(i, f))

                # 出力フォルダを設定（画像フォルダ内にsegmentation_analysisを作成）
                output_dir = os.path.join(folder_path, 'segmentation_analysis')

                # 測定日付を変換
                measurement_date = datetime.strptime(meas_date, '%Y-%m-%d').date()

                # ONNX使用時とPyTorch使用時で処理を分岐
                if use_onnx:
                    # ONNX推論 + 行動解析
                    analysis_info = self._run_batch_onnx_analysis(
                        folder_path, output_dir, onnx_model_path,
                        device, meas_time, measurement_date
                    )
                else:
                    # PyTorch推論 + 行動解析
                    analysis_info = run_inference_and_analysis(
                        images_dir=folder_path,
                        output_dir=output_dir,
                        model_path=self.model_path.get(),
                        create_video=self.create_video.get(),
                        time_interval_minutes=self.time_interval.get(),
                        day_start_time=self.day_start.get(),
                        night_start_time=self.night_start.get(),
                        measurement_start_time=meas_time,
                        measurement_date=measurement_date
                    )

                results.append((folder_path, True, "成功"))

                # 統合解析用の情報を収集
                if analysis_info:
                    dir_name = os.path.basename(folder_path)
                    start_date_str = meas_date.replace('-', '')  # YYYY-MM-DD → yyyyMMdd
                    analysis_results.append({
                        'dir_name': dir_name,
                        'start_date': start_date_str,
                        'start_time': meas_time,
                        'output_dir': analysis_info['output_dir'],
                        'aggregated_csv': analysis_info['aggregated_csv'],
                        'time_interval': analysis_info['time_interval']
                    })

            except Exception as e:
                import traceback
                error_msg = f"{type(e).__name__}: {str(e)}"
                print(f"エラー [{folder_path}]:\n{traceback.format_exc()}")
                results.append((folder_path, False, error_msg))

            # 進捗バーを更新
            self.root.after(0, lambda v=idx+1: self._update_batch_progress(v))

        # 統合解析を実行（2つ以上の実験がある場合）
        if len(analysis_results) >= 2:
            self.root.after(0, lambda: self._update_batch_status(len(results), "統合解析中..."))
            try:
                from batch_summary_analysis import run_batch_summary_analysis

                # 親フォルダ直下にbatch_summary/を作成
                parent_path = self.parent_dir.get()
                summary_output_dir = os.path.join(parent_path, 'batch_summary')

                print("\n" + "="*70)
                print("  バッチ統合解析を開始します")
                print("="*70 + "\n")

                run_batch_summary_analysis(analysis_results, summary_output_dir)

                print("\n✓ バッチ統合解析が完了しました")
                print(f"  出力: {summary_output_dir}\n")

            except Exception as e:
                import traceback
                print(f"統合解析エラー:\n{traceback.format_exc()}")
                # 統合解析のエラーは警告として扱い、個別解析の結果は保持

        # 完了処理
        self.root.after(0, lambda: self._on_batch_completion(results))

    def _run_batch_onnx_analysis(self, images_dir, output_dir, model_path, device, meas_time, measurement_date):
        """バッチ処理用のONNX推論 + 行動解析"""
        import json
        from inference_onnx import inference_onnx
        from behavior_analysis import BehaviorAnalyzer

        # 時間設定を保存
        time_config = {
            'day_start_time': self.day_start.get(),
            'night_start_time': self.night_start.get(),
            'measurement_start_time': meas_time,
            'measurement_date': measurement_date.strftime('%Y-%m-%d')
        }
        config_path = os.path.join(output_dir, 'time_config.json')
        os.makedirs(output_dir, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(time_config, f, indent=2)

        # ONNX推論
        use_directml = (device == 'directml')
        inference_onnx(
            images_dir=images_dir,
            output_dir=output_dir,
            model_path=model_path,
            create_video=self.create_video.get(),
            device=device,
            use_directml=use_directml
        )

        csv_path = os.path.join(output_dir, 'analysis_results.csv')

        # 行動解析
        analyzer = BehaviorAnalyzer(
            csv_path=csv_path,
            time_interval_minutes=self.time_interval.get(),
            day_start_time=self.day_start.get(),
            night_start_time=self.night_start.get(),
            measurement_start_time=meas_time,
            measurement_date=measurement_date
        )

        analyzer.load_data()
        analyzer.calculate_movement()
        analyzer.calculate_immobility_ratio()
        analyzer.aggregate_by_time()
        analyzer.apply_moving_average(window=1)
        analyzer.create_plots(output_dir)
        analyzer.save_detailed_csv(output_dir)
        analyzer.generate_summary_report(output_dir)

        # 解析結果情報を返す
        return {
            'output_dir': output_dir,
            'aggregated_csv': os.path.join(output_dir, 'aggregated_immobility_analysis.csv'),
            'detailed_csv': os.path.join(output_dir, 'detailed_immobility_analysis.csv'),
            'summary_csv': os.path.join(output_dir, 'day_night_summary.csv'),
            'time_interval': self.time_interval.get()
        }


    def _update_batch_status(self, index, folder_path):
        """バッチ処理のステータスを更新"""
        folder_name = os.path.basename(folder_path)
        total = len(self.detected_folders)
        self.batch_status_label.config(
            text=f"処理中 ({index+1}/{total}): {folder_name}...",
            foreground="blue"
        )

    def _update_batch_progress(self, value):
        """バッチ処理の進捗バーを更新"""
        self.batch_progress['value'] = value

    def _on_batch_completion(self, results):
        """バッチ処理完了時の処理"""
        self.batch_run_button.config(state='normal')

        # 成功と失敗をカウント
        success_count = sum(1 for _, success, _ in results if success)
        failure_count = len(results) - success_count

        # サマリーメッセージを作成
        summary = f"バッチ処理が完了しました\n\n"
        summary += f"成功: {success_count}/{len(results)}\n"
        summary += f"失敗: {failure_count}/{len(results)}\n\n"

        if failure_count > 0:
            summary += "失敗したフォルダ:\n"
            for folder_path, success, msg in results:
                if not success:
                    summary += f"  - {os.path.basename(folder_path)}: {msg}\n"

        self.batch_status_label.config(
            text=f"完了 (成功: {success_count}, 失敗: {failure_count})",
            foreground="green" if failure_count == 0 else "orange"
        )

        # 詳細を表示
        self._show_batch_result_detail(results)

    def _show_error_detail(self, error_message):
        """詳細なエラーメッセージを別ウィンドウで表示"""
        error_window = tk.Toplevel(self.root)
        error_window.title("エラー詳細")
        error_window.geometry("800x600")

        frame = ttk.Frame(error_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        text_widget.insert(1.0, error_message)
        text_widget.config(state=tk.DISABLED)

        ttk.Button(error_window, text="閉じる", command=error_window.destroy).pack(pady=10)

    def _show_batch_result_detail(self, results):
        """バッチ処理結果の詳細を表示"""
        result_window = tk.Toplevel(self.root)
        result_window.title("バッチ処理結果")
        result_window.geometry("900x600")

        frame = ttk.Frame(result_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        # 成功/失敗の統計
        success_count = sum(1 for _, success, _ in results if success)
        failure_count = len(results) - success_count

        stats_label = ttk.Label(
            frame,
            text=f"完了: {len(results)}フォルダ  |  成功: {success_count}  |  失敗: {failure_count}",
            font=('Arial', 11, 'bold')
        )
        stats_label.pack(pady=10)

        # Treeviewで結果を表示
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        result_tree = ttk.Treeview(
            tree_frame,
            columns=('path', 'status', 'message'),
            show='headings',
            yscrollcommand=tree_scroll.set
        )
        result_tree.heading('path', text='フォルダパス')
        result_tree.heading('status', text='ステータス')
        result_tree.heading('message', text='メッセージ')

        result_tree.column('path', width=500)
        result_tree.column('status', width=80)
        result_tree.column('message', width=300)

        result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.config(command=result_tree.yview)

        # 結果を追加
        for folder_path, success, message in results:
            status_text = "成功" if success else "失敗"
            tag = 'success' if success else 'failure'
            result_tree.insert('', 'end', values=(folder_path, status_text, message), tags=(tag,))

        # タグの色設定
        result_tree.tag_configure('success', foreground='green')
        result_tree.tag_configure('failure', foreground='red')

        ttk.Button(result_window, text="閉じる", command=result_window.destroy).pack(pady=10)


def main():
    root = tk.Tk()
    app = InferenceAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
