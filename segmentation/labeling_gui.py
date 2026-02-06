import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import re

def get_image_files(directory, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
    """ディレクトリから画像ファイルのリストを取得（自然順ソート）"""
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

class LabelingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Planarian Labeling Tool")
        self.images_dir = None
        self.labels_dir = None
        self.files = []
        self.idx = 0
        self.img = None
        self.mask = None
        self.brush = 15  # ブラシサイズ
        self.mode = 'draw'  # 'draw' or 'erase'
        self.painting = False

        # GUI構築
        self._build_gui()

    def _build_gui(self):
        """GUIを構築"""
        # トップフレーム：ファイル選択
        f_top = tk.Frame(self.root)
        f_top.pack(pady=5)
        tk.Button(f_top, text="画像フォルダ選択", command=self.sel_img, width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(f_top, text="ラベルフォルダ選択", command=self.sel_lbl, width=15).pack(side=tk.LEFT, padx=5)

        # 進捗表示ラベル
        self.progress_label = tk.Label(f_top, text="画像: 0 / 0", font=('Arial', 10))
        self.progress_label.pack(side=tk.LEFT, padx=20)

        # ナビゲーションフレーム
        f_nav = tk.Frame(self.root)
        f_nav.pack(pady=5)
        tk.Button(f_nav, text="← 前へ (P)", command=self.prev, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(f_nav, text="次へ (N) →", command=self.next, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(f_nav, text="保存 (S)", command=self.save, width=12, bg='lightgreen').pack(side=tk.LEFT, padx=5)
        tk.Button(f_nav, text="クリア (C)", command=self.clear_mask, width=12, bg='lightyellow').pack(side=tk.LEFT, padx=5)

        # ツールフレーム：ブラシとモード
        f_tool = tk.Frame(self.root)
        f_tool.pack(pady=5)

        tk.Label(f_tool, text="ブラシサイズ:").pack(side=tk.LEFT, padx=5)
        self.brush_scale = tk.Scale(f_tool, from_=5, to=50, orient=tk.HORIZONTAL,
                                     command=self.update_brush, length=200)
        self.brush_scale.set(self.brush)
        self.brush_scale.pack(side=tk.LEFT, padx=5)

        self.mode_label = tk.Label(f_tool, text="モード: 描画", font=('Arial', 10, 'bold'), fg='blue')
        self.mode_label.pack(side=tk.LEFT, padx=20)

        tk.Button(f_tool, text="描画モード (D)", command=self.set_draw_mode, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(f_tool, text="消しゴムモード (E)", command=self.set_erase_mode, width=15).pack(side=tk.LEFT, padx=5)

        # キャンバス
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg='gray', cursor='crosshair')
        self.canvas.pack(pady=10)

        # マウスイベント
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.release)

        # キーボードショートカット
        self.root.bind('n', lambda e: self.next())
        self.root.bind('p', lambda e: self.prev())
        self.root.bind('s', lambda e: self.save())
        self.root.bind('c', lambda e: self.clear_mask())
        self.root.bind('d', lambda e: self.set_draw_mode())
        self.root.bind('e', lambda e: self.set_erase_mode())

        # ヘルプ
        f_help = tk.Frame(self.root)
        f_help.pack(pady=5)
        help_text = "ショートカット: N=次へ | P=前へ | S=保存 | C=クリア | D=描画 | E=消しゴム"
        tk.Label(f_help, text=help_text, font=('Arial', 9), fg='gray').pack()

    def update_brush(self, value):
        """ブラシサイズを更新"""
        self.brush = int(value)

    def set_draw_mode(self):
        """描画モードに設定"""
        self.mode = 'draw'
        self.mode_label.config(text="モード: 描画", fg='blue')
        self.canvas.config(cursor='crosshair')

    def set_erase_mode(self):
        """消しゴムモードに設定"""
        self.mode = 'erase'
        self.mode_label.config(text="モード: 消しゴム", fg='red')
        self.canvas.config(cursor='circle')

    def clear_mask(self):
        """現在のマスクをクリア"""
        if self.mask is not None:
            if messagebox.askyesno("確認", "現在のマスクをクリアしますか？"):
                self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
                self.show()

    def update_progress(self):
        """進捗表示を更新"""
        if len(self.files) > 0:
            self.progress_label.config(text=f"画像: {self.idx + 1} / {len(self.files)} ({self.files[self.idx]})")
        else:
            self.progress_label.config(text="画像: 0 / 0")
    def sel_img(self):
        """画像フォルダを選択"""
        d = filedialog.askdirectory(title="画像フォルダを選択")
        if d:
            self.images_dir = d
            self.files = get_image_files(d)
            print(f"画像フォルダ: {d}")
            print(f"画像数: {len(self.files)}枚")
            if self.labels_dir and len(self.files) > 0:
                self.load(0)
            self.update_progress()

    def sel_lbl(self):
        """ラベルフォルダを選択"""
        d = filedialog.askdirectory(title="ラベルフォルダを選択")
        if d:
            self.labels_dir = d
            print(f"ラベルフォルダ: {d}")
            if self.images_dir and len(self.files) > 0:
                self.load(0)

    def load(self, i):
        """指定インデックスの画像とマスクを読み込み"""
        if i < 0 or i >= len(self.files):
            return

        self.idx = i
        f = self.files[i]

        # 画像読み込み
        img_path = os.path.join(self.images_dir, f)
        self.img = cv2.imread(img_path)
        if self.img is None:
            messagebox.showerror("エラー", f"画像を読み込めません: {f}")
            return
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # マスク読み込み（または新規作成）
        lf = os.path.splitext(f)[0] + '.png'
        lp = os.path.join(self.labels_dir, lf)
        if os.path.exists(lp):
            self.mask = cv2.imread(lp, 0)
            print(f"マスク読み込み: {lf}")
        else:
            self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
            print(f"新規マスク作成: {lf}")

        self.show()
        self.update_progress()
    def show(self):
        """画像とマスクを表示"""
        if self.img is None:
            return

        # スケール計算
        h, w = self.img.shape[:2]
        s = min(800/w, 600/h)
        nw, nh = int(w*s), int(h*s)
        self.scale = s

        # リサイズ
        disp = cv2.resize(self.img, (nw, nh))
        m = cv2.resize(self.mask, (nw, nh))

        # マスクをオーバーレイ（緑色）
        disp[:,:,1] = np.clip(disp[:,:,1] + m*0.5, 0, 255).astype(np.uint8)

        # Tkinterで表示
        pil = Image.fromarray(disp)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete('all')
        self.canvas.create_image(400, 300, image=self.photo)

    def press(self, e):
        """マウス押下時の処理"""
        self.painting = True
        self.paint(e.x, e.y)
        self.lx, self.ly = e.x, e.y

    def drag(self, e):
        """マウスドラッグ時の処理"""
        if self.painting:
            self.line(self.lx, self.ly, e.x, e.y)
            self.lx, self.ly = e.x, e.y

    def release(self, e):
        """マウスリリース時の処理"""
        self.painting = False

    def paint(self, x, y):
        """指定座標にペイント"""
        if self.mask is None:
            return

        h, w = self.mask.shape
        ox = int((x-400)/self.scale + w/2)
        oy = int((y-300)/self.scale + h/2)

        if 0 <= ox < w and 0 <= oy < h:
            color = 255 if self.mode == 'draw' else 0
            cv2.circle(self.mask, (ox, oy), self.brush, color, -1)
            self.show()

    def line(self, x1, y1, x2, y2):
        """2点間に線を描画"""
        if self.mask is None:
            return

        h, w = self.mask.shape
        ox1 = int((x1-400)/self.scale + w/2)
        oy1 = int((y1-300)/self.scale + h/2)
        ox2 = int((x2-400)/self.scale + w/2)
        oy2 = int((y2-300)/self.scale + h/2)

        color = 255 if self.mode == 'draw' else 0
        cv2.line(self.mask, (ox1, oy1), (ox2, oy2), color, self.brush*2)
        self.show()

    def save(self):
        """現在のマスクを保存"""
        if self.mask is None:
            messagebox.showwarning("警告", "保存するマスクがありません")
            return

        f = self.files[self.idx]
        lf = os.path.splitext(f)[0] + '.png'
        lp = os.path.join(self.labels_dir, lf)
        cv2.imwrite(lp, self.mask)
        print(f'保存完了: {lf}')
        messagebox.showinfo("保存完了", f"{lf} を保存しました")

    def next(self):
        """次の画像へ"""
        if self.idx < len(self.files) - 1:
            self.load(self.idx + 1)
        else:
            messagebox.showinfo("情報", "最後の画像です")

    def prev(self):
        """前の画像へ"""
        if self.idx > 0:
            self.load(self.idx - 1)
        else:
            messagebox.showinfo("情報", "最初の画像です")
if __name__ == '__main__':
    root = tk.Tk()
    app = LabelingGUI(root)
    root.mainloop()
