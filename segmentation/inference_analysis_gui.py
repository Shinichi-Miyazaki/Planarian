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
        self.root.geometry("950x600")

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
        self.batch_output_dir = tk.StringVar()  # 統合解析結果の出力先
        self.image_threshold = tk.IntVar(value=1000)
        self.batch_use_onnx = tk.BooleanVar(value=False)
        self.batch_inference_device = tk.StringVar(value='cpu')
        self.force_reanalysis = tk.BooleanVar(value=False)  # 強制再解析オプション
        self.detected_folders = []  # [(folder_path, image_count, measurement_date, measurement_start_time, is_analyzed), ...]

        # スムージングパラメータ
        self.smoothing_window = tk.IntVar(value=10)  # 個別データの移動平均ウィンドウ（データポイント数）
        self.time_bin_size = tk.DoubleVar(value=1.0)  # 時間ビンサイズ（時間単位）
        self.mean_smooth_size = tk.IntVar(value=5)  # Mean/SEMの追加スムージング（データポイント数）

        self.create_widgets()

    def create_widgets(self):
        # ノートブック（タブ）を作成
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # タブ1: 単一フォルダモード
        single_frame = ttk.Frame(notebook, padding="5")
        notebook.add(single_frame, text="単一フォルダ解析")
        self.create_single_mode_widgets(single_frame)

        # タブ2: バッチモード
        batch_frame = ttk.Frame(notebook, padding="5")
        notebook.add(batch_frame, text="バッチフォルダ解析")
        self.create_batch_mode_widgets(batch_frame)

    def create_single_mode_widgets(self, parent):
        """単一フォルダモードのウィジェットを作成"""
        row = 0

        ttk.Label(parent, text="単一フォルダ解析",
                 font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=3, pady=2)
        row += 1

        ttk.Label(parent, text="画像フォルダ:").grid(row=row, column=0, sticky=tk.W, pady=1)
        ttk.Entry(parent, textvariable=self.images_dir, width=50).grid(row=row, column=1, pady=1)
        ttk.Button(parent, text="参照", command=self.browse_images).grid(row=row, column=2, padx=3, pady=1)
        row += 1

        ttk.Label(parent, text="モデル:").grid(row=row, column=0, sticky=tk.W, pady=1)
        ttk.Entry(parent, textvariable=self.model_path, width=50).grid(row=row, column=1, pady=1)
        ttk.Button(parent, text="参照", command=self.browse_model).grid(row=row, column=2, padx=3, pady=1)
        row += 1

        ttk.Label(parent, text="出力フォルダ:").grid(row=row, column=0, sticky=tk.W, pady=1)
        ttk.Entry(parent, textvariable=self.output_dir, width=50).grid(row=row, column=1, pady=1)
        ttk.Button(parent, text="参照", command=self.browse_output).grid(row=row, column=2, padx=3, pady=1)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=2)
        row += 1

        # 時間設定を2列レイアウトに
        ttk.Label(parent, text="時間設定", font=('Arial', 9, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=1)
        row += 1

        # 測定日付と測定開始を1行に
        ttk.Label(parent, text="測定日付:").grid(row=row, column=0, sticky=tk.W, pady=1)
        date_entry = ttk.Entry(parent, textvariable=self.measurement_date, width=15)
        date_entry.grid(row=row, column=1, sticky=tk.W, pady=1)
        ttk.Label(parent, text="開始時刻:").grid(row=row, column=1, sticky=tk.W, pady=1, padx=(120, 0))
        time_entry = ttk.Entry(parent, textvariable=self.measurement_start, width=15)
        time_entry.grid(row=row, column=1, sticky=tk.W, pady=1, padx=(180, 0))
        row += 1

        # 昼開始と夜開始を1行に
        ttk.Label(parent, text="昼開始:").grid(row=row, column=0, sticky=tk.W, pady=1)
        ttk.Entry(parent, textvariable=self.day_start, width=10).grid(row=row, column=1, sticky=tk.W, pady=1)
        ttk.Label(parent, text="夜開始:").grid(row=row, column=1, sticky=tk.W, pady=1, padx=(90, 0))
        ttk.Entry(parent, textvariable=self.night_start, width=10).grid(row=row, column=1, sticky=tk.W, pady=1, padx=(145, 0))
        ttk.Label(parent, text="間隔(分):").grid(row=row, column=1, sticky=tk.W, pady=1, padx=(240, 0))
        ttk.Entry(parent, textvariable=self.time_interval, width=7).grid(row=row, column=1, sticky=tk.W, pady=1, padx=(300, 0))
        row += 1

        ttk.Checkbutton(parent, text="動画を作成",
                       variable=self.create_video).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=1)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=2)
        row += 1

        # ONNX / GPU設定を1行に
        ttk.Label(parent, text="推論設定", font=('Arial', 9, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=1)
        row += 1

        inference_frame = ttk.Frame(parent)
        inference_frame.grid(row=row, column=0, columnspan=3, sticky='ew', pady=1)

        ttk.Checkbutton(inference_frame, text="ONNX Runtime",
                       variable=self.use_onnx).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(inference_frame, text="デバイス:").pack(side=tk.LEFT, padx=(8, 3))
        ttk.Radiobutton(inference_frame, text="CPU", variable=self.inference_device, value='cpu').pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(inference_frame, text="CUDA", variable=self.inference_device, value='cuda').pack(side=tk.LEFT, padx=3)
        row += 1

        ttk.Label(parent, text="※ RTX 5070 TiはCUDA推奨",
                 font=('Arial', 7), foreground='gray').grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(0, 1))
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=2)
        row += 1

        self.run_button = ttk.Button(parent, text="実行", command=self.run_single_analysis)
        self.run_button.grid(row=row, column=0, columnspan=3, pady=2)
        row += 1

        self.status_label = ttk.Label(parent, text="準備完了", foreground="green")
        self.status_label.grid(row=row, column=0, columnspan=3, pady=1)
        row += 1

        self.progress = ttk.Progressbar(parent, mode='indeterminate', length=400)
        self.progress.grid(row=row, column=0, columnspan=3, pady=1)

    def create_batch_mode_widgets(self, parent):
        """バッチモードのウィジェットを作成"""
        row = 0

        ttk.Label(parent, text="バッチフォルダ解析",
                 font=('Arial', 11, 'bold')).grid(row=row, column=0, columnspan=6, pady=2)
        row += 1

        # 親フォルダとExcelファイルを同じ行に
        ttk.Label(parent, text="親フォルダ:").grid(row=row, column=0, sticky=tk.W, pady=1)
        ttk.Entry(parent, textvariable=self.parent_dir, width=28).grid(row=row, column=1, pady=1, sticky='ew')
        ttk.Button(parent, text="参照", command=self.browse_parent_dir).grid(row=row, column=2, padx=3, pady=1)

        ttk.Label(parent, text="Excel:").grid(row=row, column=3, sticky=tk.W, pady=1, padx=(8,0))
        ttk.Entry(parent, textvariable=self.excel_file, width=28).grid(row=row, column=4, pady=1, sticky='ew')
        ttk.Button(parent, text="参照", command=self.browse_excel_file).grid(row=row, column=5, padx=3, pady=1)
        row += 1

        ttk.Label(parent, text="※ Excel: dir_name, start_date, start_time列が必要",
                 font=('Arial', 7), foreground='gray').grid(row=row, column=0, columnspan=6, sticky=tk.W, pady=(0, 1))
        row += 1

        # 統合解析結果の出力先
        ttk.Label(parent, text="統合結果:").grid(row=row, column=0, sticky=tk.W, pady=1)
        ttk.Entry(parent, textvariable=self.batch_output_dir, width=28).grid(row=row, column=1, pady=1, sticky='ew')
        ttk.Button(parent, text="参照", command=self.browse_batch_output).grid(row=row, column=2, padx=3, pady=1)

        ttk.Label(parent, text="※ 未指定時は親フォルダ/batch_summary",
                 font=('Arial', 7), foreground='gray').grid(row=row, column=3, columnspan=3, sticky=tk.W, pady=1, padx=(8,0))
        row += 1

        # 画像数閾値とモデルを同じ行に
        ttk.Label(parent, text="画像数閾値:").grid(row=row, column=0, sticky=tk.W, pady=1)
        ttk.Entry(parent, textvariable=self.image_threshold, width=10).grid(row=row, column=1, sticky=tk.W, pady=1)

        ttk.Label(parent, text="モデル:").grid(row=row, column=2, sticky=tk.W, pady=1, padx=(8,0))
        ttk.Entry(parent, textvariable=self.model_path, width=28).grid(row=row, column=3, columnspan=2, pady=1, sticky='ew')
        ttk.Button(parent, text="参照", command=self.browse_model).grid(row=row, column=5, padx=3, pady=1)
        row += 1

        # フォルダ検索ボタンと強制再解析オプション
        search_frame = ttk.Frame(parent)
        search_frame.grid(row=row, column=0, columnspan=6, pady=2)

        ttk.Button(search_frame, text="フォルダを検索", command=self.search_folders).pack(side=tk.LEFT, padx=3)
        ttk.Checkbutton(search_frame, text="解析済みフォルダも強制再解析",
                       variable=self.force_reanalysis).pack(side=tk.LEFT, padx=15)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=6, sticky='ew', pady=2)
        row += 1

        # 共通時間設定を1行に
        ttk.Label(parent, text="時間設定", font=('Arial', 9, 'bold')).grid(
            row=row, column=0, columnspan=6, sticky=tk.W, pady=1)
        row += 1

        time_frame = ttk.Frame(parent)
        time_frame.grid(row=row, column=0, columnspan=6, sticky='ew', pady=1)

        ttk.Label(time_frame, text="昼開始:").pack(side=tk.LEFT, padx=(0, 3))
        ttk.Entry(time_frame, textvariable=self.day_start, width=7).pack(side=tk.LEFT, padx=2)

        ttk.Label(time_frame, text="夜開始:").pack(side=tk.LEFT, padx=(8, 3))
        ttk.Entry(time_frame, textvariable=self.night_start, width=7).pack(side=tk.LEFT, padx=2)

        ttk.Label(time_frame, text="間隔(分):").pack(side=tk.LEFT, padx=(8, 3))
        ttk.Entry(time_frame, textvariable=self.time_interval, width=7).pack(side=tk.LEFT, padx=2)

        ttk.Checkbutton(time_frame, text="動画", variable=self.create_video).pack(side=tk.LEFT, padx=(15, 3))
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=6, sticky='ew', pady=2)
        row += 1

        # ONNX / GPU設定を1行に
        ttk.Label(parent, text="推論設定", font=('Arial', 9, 'bold')).grid(
            row=row, column=0, columnspan=6, sticky=tk.W, pady=1)
        row += 1

        inference_frame = ttk.Frame(parent)
        inference_frame.grid(row=row, column=0, columnspan=6, sticky='ew', pady=1)

        ttk.Checkbutton(inference_frame, text="ONNX Runtime",
                       variable=self.batch_use_onnx).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(inference_frame, text="デバイス:").pack(side=tk.LEFT, padx=(8, 3))
        ttk.Radiobutton(inference_frame, text="CPU", variable=self.batch_inference_device, value='cpu').pack(side=tk.LEFT, padx=3)
        ttk.Radiobutton(inference_frame, text="CUDA", variable=self.batch_inference_device, value='cuda').pack(side=tk.LEFT, padx=3)
        row += 1

        ttk.Label(parent, text="※ RTX 5070 TiはCUDAを推奨",
                 font=('Arial', 7), foreground='orange').grid(row=row, column=0, columnspan=6, sticky=tk.W, pady=(0, 1))
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=6, sticky='ew', pady=2)
        row += 1

        # スムージングパラメータ設定
        ttk.Label(parent, text="スムージング設定", font=('Arial', 9, 'bold')).grid(
            row=row, column=0, columnspan=6, sticky=tk.W, pady=1)
        row += 1

        smooth_frame = ttk.Frame(parent)
        smooth_frame.grid(row=row, column=0, columnspan=6, sticky='ew', pady=1)

        ttk.Label(smooth_frame, text="個別:").pack(side=tk.LEFT, padx=(0, 2))
        ttk.Entry(smooth_frame, textvariable=self.smoothing_window, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Label(smooth_frame, text="pt").pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(smooth_frame, text="時間ビン:").pack(side=tk.LEFT, padx=(3, 2))
        ttk.Entry(smooth_frame, textvariable=self.time_bin_size, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Label(smooth_frame, text="hr").pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(smooth_frame, text="Mean:").pack(side=tk.LEFT, padx=(3, 2))
        ttk.Entry(smooth_frame, textvariable=self.mean_smooth_size, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Label(smooth_frame, text="pt").pack(side=tk.LEFT)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=6, sticky='ew', pady=2)
        row += 1

        # 検出されたフォルダリスト
        ttk.Label(parent, text="検出されたフォルダ", font=('Arial', 9, 'bold')).grid(
            row=row, column=0, columnspan=6, sticky=tk.W, pady=1)
        row += 1

        # Treeviewでフォルダリストを表示（高さを4行に削減）
        tree_frame = ttk.Frame(parent)
        tree_frame.grid(row=row, column=0, columnspan=6, sticky='ew', pady=2)

        # スクロールバー
        tree_scroll = ttk.Scrollbar(tree_frame)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Treeview作成（高さを4行に制限）
        self.folder_tree = ttk.Treeview(
            tree_frame,
            columns=('path', 'count', 'date', 'time', 'status'),
            show='headings',
            yscrollcommand=tree_scroll.set,
            height=4
        )
        self.folder_tree.heading('path', text='フォルダパス')
        self.folder_tree.heading('count', text='画像数')
        self.folder_tree.heading('date', text='測定日付')
        self.folder_tree.heading('time', text='測定開始時刻')
        self.folder_tree.heading('status', text='解析状態')

        self.folder_tree.column('path', width=320)
        self.folder_tree.column('count', width=60)
        self.folder_tree.column('date', width=80)
        self.folder_tree.column('time', width=70)
        self.folder_tree.column('status', width=70)

        self.folder_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.config(command=self.folder_tree.yview)

        # ダブルクリックで編集
        self.folder_tree.bind('<Double-1>', self.edit_folder_time)
        row += 1

        # ボタンフレーム
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=row, column=0, columnspan=6, pady=3)

        ttk.Button(button_frame, text="選択フォルダを削除", command=self.remove_selected_folder).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="全て削除", command=self.clear_all_folders).pack(side=tk.LEFT, padx=3)
        row += 1

        ttk.Separator(parent, orient='horizontal').grid(
            row=row, column=0, columnspan=6, sticky='ew', pady=3)
        row += 1

        # バッチ実行ボタン
        self.batch_run_button = ttk.Button(parent, text="バッチ実行", command=self.run_batch_analysis)
        self.batch_run_button.grid(row=row, column=0, columnspan=6, pady=3)
        row += 1

        # ステータス
        self.batch_status_label = ttk.Label(parent, text="準備完了", foreground="green")
        self.batch_status_label.grid(row=row, column=0, columnspan=6, pady=2)
        row += 1

        self.batch_progress = ttk.Progressbar(parent, mode='determinate', length=400)
        self.batch_progress.grid(row=row, column=0, columnspan=6, pady=2)

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

    def browse_batch_output(self):
        """統合解析結果の出力先フォルダを選択"""
        directory = filedialog.askdirectory(title="統合解析結果の出力先を選択")
        if directory:
            self.batch_output_dir.set(directory)

    def check_folder_analyzed(self, folder_path):
        """
        フォルダが既に解析済みかチェック

        Parameters:
        -----------
        folder_path : str
            チェック対象のフォルダパス

        Returns:
        --------
        tuple : (is_analyzed, is_complete)
            is_analyzed : bool - segmentation_analysisフォルダが存在するか
            is_complete : bool - 必要なCSVファイルが全て存在するか
        """
        analysis_dir = os.path.join(folder_path, 'segmentation_analysis')

        if not os.path.exists(analysis_dir):
            return False, False

        # 必須ファイルのリスト
        required_files = [
            'analysis_results.csv',
            'detailed_immobility_analysis.csv',
            'aggregated_immobility_analysis.csv',
            'day_night_summary.csv',
            'time_config.json'
        ]

        is_complete = all(os.path.exists(os.path.join(analysis_dir, f)) for f in required_files)

        return True, is_complete

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

                    # 解析済みかチェック
                    is_analyzed, is_complete = self.check_folder_analyzed(folder_path)

                    if is_analyzed and not is_complete:
                        print(f"⚠️ 部分的な解析結果を検出: {dir_name} - 再解析が必要です")

                    status = "解析済み" if (is_analyzed and is_complete) else "未解析"

                    found_folders.append((folder_path, image_count, formatted_date, formatted_time, is_analyzed and is_complete))
                    print(f"✓ 検出: {dir_name} ({image_count}枚, {formatted_date} {formatted_time}, {status})")

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

                        # 解析済みかチェック
                        is_analyzed, is_complete = self.check_folder_analyzed(root)

                        if is_analyzed and not is_complete:
                            print(f"⚠️ 部分的な解析結果を検出: {os.path.basename(root)} - 再解析が必要です")

                        status = "解析済み" if (is_analyzed and is_complete) else "未解析"

                        found_folders.append((root, image_count, default_date, default_time, is_analyzed and is_complete))
                        print(f"✓ 検出: {os.path.basename(root)} ({image_count}枚, {status})")

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

        # タグの色設定
        self.folder_tree.tag_configure('analyzed', foreground='#228B22')  # 緑色
        self.folder_tree.tag_configure('not_analyzed', foreground='#DC143C')  # 赤色

        # 新しいフォルダを追加
        self.detected_folders = folders
        for folder_path, image_count, meas_date, meas_time, is_analyzed in folders:
            status_text = "✓ 解析済み" if is_analyzed else "未解析"
            tag = 'analyzed' if is_analyzed else 'not_analyzed'
            self.folder_tree.insert('', 'end',
                                   values=(folder_path, image_count, meas_date, meas_time, status_text),
                                   tags=(tag,))

        # 統計を表示
        analyzed_count = sum(1 for _, _, _, _, is_analyzed in folders if is_analyzed)
        not_analyzed_count = len(folders) - analyzed_count

        self.batch_status_label.config(
            text=f"{len(folders)}個のフォルダを検出 (解析済み: {analyzed_count}, 未解析: {not_analyzed_count})",
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
                self.folder_tree.item(item, values=(values[0], values[1], date_var.get(), time_var.get(), values[4]))

                # detected_foldersも更新
                idx = self.folder_tree.index(item)
                folder_list = list(self.detected_folders)
                folder_list[idx] = (values[0], int(values[1]), date_var.get(), time_var.get(), folder_list[idx][4])
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
             self.folder_tree.item(item, 'values')[3],
             self.folder_tree.item(item, 'values')[4] == "✓ 解析済み")
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
            'measurement_date': measurement_date.strftime('%Y-%m-%d'),
            'time_interval': self.time_interval.get()
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

        for idx, (folder_path, image_count, meas_date, meas_time, is_analyzed) in enumerate(self.detected_folders):
            try:
                # ステータス更新
                self.root.after(0, lambda i=idx, f=folder_path: self._update_batch_status(i, f))

                # 出力フォルダを設定（画像フォルダ内にsegmentation_analysisを作成）
                output_dir = os.path.join(folder_path, 'segmentation_analysis')

                # 測定日付を変換
                measurement_date = datetime.strptime(meas_date, '%Y-%m-%d').date()

                # 解析済みフォルダをチェック
                should_skip = is_analyzed and not self.force_reanalysis.get()

                if should_skip:
                    # 解析済みフォルダをスキップして既存データを読み込む
                    print(f"\n{'='*70}")
                    print(f"  スキップ: {os.path.basename(folder_path)} (既に解析済み)")
                    print(f"{'='*70}")

                    # 既存の解析結果を統合解析用に収集
                    analysis_info = self._load_existing_analysis_info(output_dir, meas_time, measurement_date)

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
                        results.append((folder_path, True, "スキップ (既存データ使用)"))
                    else:
                        results.append((folder_path, False, "既存データの読み込みエラー"))

                    # 進捗バーを更新
                    self.root.after(0, lambda v=idx+1: self._update_batch_progress(v))
                    continue

                # 新規解析を実行
                print(f"\n{'='*70}")
                print(f"  解析開始: {os.path.basename(folder_path)}")
                print(f"{'='*70}")

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

                # 統合解析結果の出力先を決定
                if self.batch_output_dir.get():
                    summary_output_dir = self.batch_output_dir.get()
                else:
                    # デフォルト: 親フォルダ直下にbatch_summary/を作成
                    parent_path = self.parent_dir.get()
                    summary_output_dir = os.path.join(parent_path, 'batch_summary')

                print("\n" + "="*70)
                print("  バッチ統合解析を開始します")
                print(f"  出力先: {summary_output_dir}")
                print("="*70 + "\n")

                run_batch_summary_analysis(
                    analysis_results,
                    summary_output_dir,
                    smoothing_window=self.smoothing_window.get(),
                    time_bin_size=self.time_bin_size.get(),
                    mean_smooth_size=self.mean_smooth_size.get()
                )

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
            'measurement_date': measurement_date.strftime('%Y-%m-%d'),
            'time_interval': self.time_interval.get()
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

    def _load_existing_analysis_info(self, output_dir, meas_time, measurement_date):
        """
        既存の解析結果から情報を読み込む

        Parameters:
        -----------
        output_dir : str
            解析結果が保存されているディレクトリ
        meas_time : str
            測定開始時刻
        measurement_date : date
            測定開始日付

        Returns:
        --------
        dict or None : 解析情報の辞書、エラーの場合はNone
        """
        import json

        try:
            # 必須ファイルの存在確認
            aggregated_csv = os.path.join(output_dir, 'aggregated_immobility_analysis.csv')
            time_config_path = os.path.join(output_dir, 'time_config.json')

            if not os.path.exists(aggregated_csv):
                print(f"  ⚠️ 警告: aggregated_immobility_analysis.csv が見つかりません")
                return None

            # time_interval を time_config.json から読み込む
            time_interval = self.time_interval.get()  # デフォルト値
            if os.path.exists(time_config_path):
                with open(time_config_path, 'r') as f:
                    time_config = json.load(f)
                    # time_intervalがあれば使用（古いバージョンでは無いかもしれない）
                    if 'time_interval' in time_config:
                        time_interval = time_config['time_interval']

            return {
                'output_dir': output_dir,
                'aggregated_csv': aggregated_csv,
                'detailed_csv': os.path.join(output_dir, 'detailed_immobility_analysis.csv'),
                'summary_csv': os.path.join(output_dir, 'day_night_summary.csv'),
                'time_interval': time_interval
            }

        except Exception as e:
            print(f"  ⚠️ エラー: 既存データの読み込みに失敗 - {str(e)}")
            return None


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

        # 成功、スキップ、失敗をカウント
        success_count = sum(1 for _, success, msg in results if success and msg == "成功")
        skipped_count = sum(1 for _, success, msg in results if success and "スキップ" in msg)
        failure_count = sum(1 for _, success, _ in results if not success)

        # サマリーメッセージを作成
        summary = f"バッチ処理が完了しました\n\n"
        summary += f"成功: {success_count}/{len(results)}\n"
        summary += f"スキップ: {skipped_count}/{len(results)}\n"
        summary += f"失敗: {failure_count}/{len(results)}\n\n"

        if failure_count > 0:
            summary += "失敗したフォルダ:\n"
            for folder_path, success, msg in results:
                if not success:
                    summary += f"  - {os.path.basename(folder_path)}: {msg}\n"

        self.batch_status_label.config(
            text=f"完了 (成功: {success_count}, スキップ: {skipped_count}, 失敗: {failure_count})",
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

        # 成功/スキップ/失敗の統計
        success_count = sum(1 for _, success, msg in results if success and msg == "成功")
        skipped_count = sum(1 for _, success, msg in results if success and "スキップ" in msg)
        failure_count = sum(1 for _, success, _ in results if not success)

        stats_label = ttk.Label(
            frame,
            text=f"完了: {len(results)}フォルダ  |  成功: {success_count}  |  スキップ: {skipped_count}  |  失敗: {failure_count}",
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
            if not success:
                status_text = "失敗"
                tag = 'failure'
            elif "スキップ" in message:
                status_text = "スキップ"
                tag = 'skipped'
            else:
                status_text = "成功"
                tag = 'success'

            result_tree.insert('', 'end', values=(folder_path, status_text, message), tags=(tag,))

        # タグの色設定
        result_tree.tag_configure('success', foreground='green')
        result_tree.tag_configure('skipped', foreground='blue')
        result_tree.tag_configure('failure', foreground='red')

        ttk.Button(result_window, text="閉じる", command=result_window.destroy).pack(pady=10)


def main():
    root = tk.Tk()
    app = InferenceAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
