import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import config
from utils import get_image_files
class LabelingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Labeling Tool")
        self.images_dir = None
        self.labels_dir = None
        self.files = []
        self.idx = 0
        self.img = None
        self.mask = None
        self.brush = 15
        self.mode = 'fg'
        self.painting = False
        # GUI
        f1 = tk.Frame(root)
        f1.pack()
        tk.Button(f1, text="Images", command=self.sel_img).pack(side=tk.LEFT)
        tk.Button(f1, text="Labels", command=self.sel_lbl).pack(side=tk.LEFT)
        tk.Button(f1, text="Prev(P)", command=self.prev).pack(side=tk.LEFT)
        tk.Button(f1, text="Next(N)", command=self.next).pack(side=tk.LEFT)
        tk.Button(f1, text="Save(S)", command=self.save).pack(side=tk.LEFT)
        self.canvas = tk.Canvas(root, width=800, height=600, bg='gray')
        self.canvas.pack()
        self.canvas.bind('<ButtonPress-1>', self.press)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', self.release)
        root.bind('n', lambda e: self.next())
        root.bind('p', lambda e: self.prev())
        root.bind('s', lambda e: self.save())
    def sel_img(self):
        d = filedialog.askdirectory()
        if d:
            self.images_dir = d
            self.files = get_image_files(d)
            if self.labels_dir:
                self.load(0)
    def sel_lbl(self):
        d = filedialog.askdirectory()
        if d:
            self.labels_dir = d
            if self.images_dir:
                self.load(0)
    def load(self, i):
        if i < 0 or i >= len(self.files):
            return
        self.idx = i
        f = self.files[i]
        self.img = cv2.imread(os.path.join(self.images_dir, f))
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        lf = os.path.splitext(f)[0] + '.png'
        lp = os.path.join(self.labels_dir, lf)
        if os.path.exists(lp):
            self.mask = cv2.imread(lp, 0)
        else:
            self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.show()
    def show(self):
        if self.img is None:
            return
        h, w = self.img.shape[:2]
        s = min(800/w, 600/h)
        nw, nh = int(w*s), int(h*s)
        self.scale = s
        disp = cv2.resize(self.img, (nw, nh))
        m = cv2.resize(self.mask, (nw, nh))
        disp[:,:,1] = np.clip(disp[:,:,1] + m*0.5, 0, 255).astype(np.uint8)
        pil = Image.fromarray(disp)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete('all')
        self.canvas.create_image(400, 300, image=self.photo)
    def press(self, e):
        self.painting = True
        self.paint(e.x, e.y)
        self.lx, self.ly = e.x, e.y
    def drag(self, e):
        if self.painting:
            self.line(self.lx, self.ly, e.x, e.y)
            self.lx, self.ly = e.x, e.y
    def release(self, e):
        self.painting = False
    def paint(self, x, y):
        if self.mask is None:
            return
        h, w = self.mask.shape
        ox = int((x-400)/self.scale + w/2)
        oy = int((y-300)/self.scale + h/2)
        if 0 <= ox < w and 0 <= oy < h:
            cv2.circle(self.mask, (ox, oy), self.brush, 255, -1)
            self.show()
    def line(self, x1, y1, x2, y2):
        if self.mask is None:
            return
        h, w = self.mask.shape
        ox1 = int((x1-400)/self.scale + w/2)
        oy1 = int((y1-300)/self.scale + h/2)
        ox2 = int((x2-400)/self.scale + w/2)
        oy2 = int((y2-300)/self.scale + h/2)
        cv2.line(self.mask, (ox1, oy1), (ox2, oy2), 255, self.brush*2)
        self.show()
    def save(self):
        if self.mask is None:
            return
        f = self.files[self.idx]
        lf = os.path.splitext(f)[0] + '.png'
        cv2.imwrite(os.path.join(self.labels_dir, lf), self.mask)
        print(f'Saved: {lf}')
    def next(self):
        if self.idx < len(self.files) - 1:
            self.load(self.idx + 1)
    def prev(self):
        if self.idx > 0:
            self.load(self.idx - 1)
if __name__ == '__main__':
    root = tk.Tk()
    app = LabelingGUI(root)
    root.mainloop()
