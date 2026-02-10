"""
PyTorchモデルをONNX形式に変換するGUI

RTX 5070 Ti等の新しいGPUでの推論を可能にするため、
PyTorchモデルをONNX形式に変換します。
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import threading
import sys

import config
from export_onnx import export_to_onnx


class ExportONNXGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PyTorch → ONNX 変換")
        self.root.geometry("700x500")

        self.model_path = tk.StringVar(value=config.BEST_MODEL_PATH)
        self.output_path = tk.StringVar()
        self.image_height = tk.IntVar(value=512)
        self.image_width = tk.IntVar(value=512)
        self.opset_version = tk.IntVar(value=14)

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        row = 0

        ttk.Label(main_frame, text="PyTorch → ONNX モデル変換",
                 font=('Arial', 14, 'bold')).grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        ttk.Label(main_frame, text="新しいGPU (RTX 5070 Ti等) での推論を可能にします",
                 foreground="gray").grid(row=row, column=0, columnspan=3, pady=(0, 10))
        row += 1

        ttk.Separator(main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=5)
        row += 1

        # 入力モデル
        ttk.Label(main_frame, text="入力モデル (.pth):").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.model_path, width=50).grid(row=row, column=1, pady=5)
        ttk.Button(main_frame, text="参照", command=self.browse_model).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        # 出力モデル
        ttk.Label(main_frame, text="出力モデル (.onnx):").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.output_path, width=50).grid(row=row, column=1, pady=5)
        ttk.Button(main_frame, text="参照", command=self.browse_output).grid(row=row, column=2, padx=5, pady=5)
        row += 1

        ttk.Label(main_frame, text="※出力パスを空にすると自動設定されます",
                 foreground="gray", font=('Arial', 8)).grid(row=row, column=1, sticky=tk.W, pady=(0, 5))
        row += 1

        ttk.Separator(main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=10)
        row += 1

        # 画像サイズ設定
        ttk.Label(main_frame, text="画像サイズ設定", font=('Arial', 11, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=5)
        row += 1

        size_frame = ttk.Frame(main_frame)
        size_frame.grid(row=row, column=1, sticky=tk.W, pady=5)

        ttk.Label(size_frame, text="高さ:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(size_frame, textvariable=self.image_height, width=10).pack(side=tk.LEFT, padx=(0, 15))
        ttk.Label(size_frame, text="幅:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(size_frame, textvariable=self.image_width, width=10).pack(side=tk.LEFT)
        row += 1

        # ONNX Opset バージョン
        ttk.Label(main_frame, text="ONNX Opset:").grid(row=row, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.opset_version, width=10).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Separator(main_frame, orient='horizontal').grid(
            row=row, column=0, columnspan=3, sticky='ew', pady=10)
        row += 1

        # 変換ボタン
        self.convert_button = ttk.Button(main_frame, text="変換開始", command=self.start_conversion)
        self.convert_button.grid(row=row, column=0, columnspan=3, pady=10)
        row += 1

        # ステータス
        self.status_label = ttk.Label(main_frame, text="準備完了", foreground="green")
        self.status_label.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1

        # プログレスバー
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate', length=400)
        self.progress.grid(row=row, column=0, columnspan=3, pady=5)
        row += 1

        # ログ表示
        ttk.Label(main_frame, text="変換ログ:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, columnspan=3, sticky=tk.W, pady=(10, 5))
        row += 1

        self.log_text = scrolledtext.ScrolledText(main_frame, height=10, width=80, wrap=tk.WORD)
        self.log_text.grid(row=row, column=0, columnspan=3, pady=5)
        self.log_text.config(state=tk.DISABLED)

    def browse_model(self):
        filename = filedialog.askopenfilename(
            title="入力PyTorchモデルを選択",
            filetypes=[("PyTorchモデル", "*.pth"), ("すべて", "*.*")])
        if filename:
            self.model_path.set(filename)
            # 出力パスを自動設定
            if not self.output_path.get():
                self.output_path.set(filename.replace('.pth', '.onnx'))

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            title="出力ONNXモデルを指定",
            defaultextension=".onnx",
            filetypes=[("ONNXモデル", "*.onnx"), ("すべて", "*.*")])
        if filename:
            self.output_path.set(filename)

    def log(self, message):
        """ログテキストに追加"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.root.update()

    def validate_inputs(self):
        if not self.model_path.get():
            messagebox.showerror("エラー", "入力モデルファイルを指定してください")
            return False

        if not os.path.exists(self.model_path.get()):
            messagebox.showerror("エラー", f"入力モデルファイルが存在しません:\n{self.model_path.get()}")
            return False

        # 出力パスが空の場合は自動設定
        if not self.output_path.get():
            self.output_path.set(self.model_path.get().replace('.pth', '.onnx'))
            self.log(f"出力パスを自動設定: {self.output_path.get()}")

        return True

    def start_conversion(self):
        if not self.validate_inputs():
            return

        self.convert_button.config(state='disabled')
        self.status_label.config(text="変換中...", foreground="blue")
        self.progress.start()
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)

        # 標準出力をログにリダイレクト
        sys.stdout = TextRedirector(self.log_text, "stdout")
        sys.stderr = TextRedirector(self.log_text, "stderr")

        thread = threading.Thread(target=self._conversion_thread)
        thread.daemon = True
        thread.start()

    def _conversion_thread(self):
        try:
            export_to_onnx(
                model_path=self.model_path.get(),
                output_path=self.output_path.get(),
                image_size=(self.image_height.get(), self.image_width.get()),
                opset_version=self.opset_version.get()
            )
            self.root.after(0, lambda: self._on_completion(True, "変換が完了しました"))
        except Exception as e:
            import traceback
            error_msg = f"エラーが発生しました:\n\n{type(e).__name__}: {str(e)}\n\n詳細:\n{traceback.format_exc()}"
            print(error_msg)
            self.root.after(0, lambda: self._on_completion(False, error_msg))
        finally:
            # 標準出力を元に戻す
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__

    def _on_completion(self, success, message):
        self.progress.stop()
        self.convert_button.config(state='normal')

        if success:
            self.status_label.config(text="変換完了", foreground="green")
            messagebox.showinfo("完了", f"ONNXモデルへの変換が完了しました!\n\n出力: {self.output_path.get()}")
        else:
            self.status_label.config(text="エラー", foreground="red")
            messagebox.showerror("エラー", "変換中にエラーが発生しました。\n詳細はログを確認してください。")


class TextRedirector:
    """標準出力をテキストウィジェットにリダイレクト"""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, message):
        self.widget.config(state=tk.NORMAL)
        self.widget.insert(tk.END, message, (self.tag,))
        self.widget.see(tk.END)
        self.widget.config(state=tk.DISABLED)

    def flush(self):
        pass


def main():
    root = tk.Tk()
    app = ExportONNXGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
