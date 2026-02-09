"""
セグメンテーション推論 + 行動解析 GUIランチャー

注: albumentationsは不要です（推論時）
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import date
import threading

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import config
from run_inference_analysis import run_inference_and_analysis


class InferenceAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("セグメンテーション推論 + 行動解析")
        self.root.geometry("700x500")

        self.images_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_path = tk.StringVar(value=config.BEST_MODEL_PATH)
        self.day_start = tk.StringVar(value='07:00')
        self.night_start = tk.StringVar(value='19:00')
        self.measurement_start = tk.StringVar(value='09:00:00')
        self.measurement_date = tk.StringVar(value=date.today().strftime('%Y-%m-%d'))
        self.time_interval = tk.IntVar(value=10)
        self.create_video = tk.BooleanVar(value=False)

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0

        ttk.Label(main_frame, text="セグメンテーション推論 + 行動解析",
                 font=('Arial', 14, 'bold')).grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        ttk.Separator(main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=5)
        row += 1

        ttk.Label(main_frame, text="画像フォルダ:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.images_dir, width=50).grid(row=row, column=1, pady=5)
        ttk.Button(main_frame, text="参照", command=self.browse_images).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        ttk.Label(main_frame, text="モデル:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.model_path, width=50).grid(row=row, column=1, pady=5)
        ttk.Button(main_frame, text="参照", command=self.browse_model).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        ttk.Label(main_frame, text="出力フォルダ:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_dir, width=50).grid(row=row, column=1, pady=5)
        ttk.Button(main_frame, text="参照", command=self.browse_output).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=10)
        row += 1

        ttk.Label(main_frame, text="時間設定", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(main_frame, text="測定日付 (YYYY-MM-DD):").grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(main_frame, textvariable=self.measurement_date, width=20).grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1

        ttk.Label(main_frame, text="測定開始 (HH:MM:SS):").grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(main_frame, textvariable=self.measurement_start, width=20).grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1

        ttk.Label(main_frame, text="昼開始 (HH:MM):").grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(main_frame, textvariable=self.day_start, width=20).grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1

        ttk.Label(main_frame, text="夜開始 (HH:MM):").grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(main_frame, textvariable=self.night_start, width=20).grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1

        ttk.Label(main_frame, text="時間間隔（分）:").grid(row=row, column=0, sticky=tk.W, pady=3)
        ttk.Entry(main_frame, textvariable=self.time_interval, width=20).grid(row=row, column=1, sticky=tk.W, pady=3)
        row += 1

        ttk.Checkbutton(main_frame, text="動画を作成",
                       variable=self.create_video).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1

        ttk.Separator(main_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky='ew', pady=10)
        row += 1

        self.run_button = ttk.Button(main_frame, text="実行", command=self.run_analysis)
        self.run_button.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        self.status_label = ttk.Label(main_frame, text="準備完了", foreground="green")
        self.status_label.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1

        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=400)
        self.progress.grid(row=row, column=0, columnspan=3, pady=5)

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

    def run_analysis(self):
        if not self.validate_inputs():
            return

        self.run_button.config(state='disabled')
        self.status_label.config(text="実行中...", foreground="blue")
        self.progress.start()

        thread = threading.Thread(target=self._run_analysis_thread)
        thread.daemon = True
        thread.start()

    def _run_analysis_thread(self):
        try:
            from datetime import datetime
            measurement_date = datetime.strptime(self.measurement_date.get(), '%Y-%m-%d').date()

            if self.use_onnx.get():
                # ONNX Runtime推論
                self._run_onnx_inference_and_analysis(measurement_date)
            else:
                # PyTorch推論
                run_inference_and_analysis(
                    images_dir=self.images_dir.get(),
                    output_dir=self.output_dir.get(),
                    model_path=self.model_path.get(),
                    create_video=self.create_video.get(),
                    time_interval_minutes=self.time_interval.get(),
                    day_start_time=self.day_start.get(),
                    night_start_time=self.night_start.get(),
                    measurement_start_time=self.measurement_start.get(),
                    measurement_date=measurement_date)

            self.root.after(0, lambda: self._on_completion(True, "処理が完了しました"))
        except Exception as e:
            import traceback
            error_msg = f"エラーが発生しました:\n\n{type(e).__name__}: {str(e)}\n\n詳細:\n{traceback.format_exc()}"
            print(error_msg)  # コンソールにも出力
            self.root.after(0, lambda: self._on_completion(False, error_msg))

    def _run_onnx_inference_and_analysis(self, measurement_date):
        """ONNX Runtime推論 + 行動解析"""
        import json
        from inference_onnx import inference_onnx
        from behavior_analysis import BehaviorAnalyzer

        images_dir = self.images_dir.get()
        output_dir = self.output_dir.get()
        model_path = self.model_path.get()

        # 時間設定を保存
        time_config = {
            'day_start_time': self.day_start.get(),
            'night_start_time': self.night_start.get(),
            'measurement_start_time': self.measurement_start.get(),
            'measurement_date': measurement_date.strftime('%Y-%m-%d')
        }
        config_path = os.path.join(output_dir, 'time_config.json')
        with open(config_path, 'w') as f:
            json.dump(time_config, f, indent=2)

        # ONNX推論
        print("ONNX Runtime推論を開始...")
        device = self.onnx_device.get()
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

    def _on_completion(self, success, message):
        self.progress.stop()
        self.run_button.config(state='normal')

        if success:
            self.status_label.config(text="完了", foreground="green")
            messagebox.showinfo("完了", message)
        else:
            self.status_label.config(text="エラー", foreground="red")
            # エラーメッセージが長い場合は詳細ウィンドウを表示
            if len(message) > 200:
                self._show_error_detail(message)
            else:
                messagebox.showerror("エラー", message)

    def _show_error_detail(self, error_message):
        """詳細なエラーメッセージを別ウィンドウで表示"""
        error_window = tk.Toplevel(self.root)
        error_window.title("エラー詳細")
        error_window.geometry("800x600")

        # スクロールバー付きテキストウィジェット
        frame = ttk.Frame(error_window, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        text_widget.insert(1.0, error_message)
        text_widget.config(state=tk.DISABLED)

        # 閉じるボタン
        ttk.Button(error_window, text="閉じる", command=error_window.destroy).pack(pady=10)


def main():
    root = tk.Tk()
    app = InferenceAnalysisGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
