"""商品区域框选标注工具 — 简化版

操作：
  - 鼠标左键拖拽角点/边/框内部 调整框
  - 鼠标滚轮 缩放
  - 中键/右键拖拽 平移

  S / Enter    保存并跳到下一张
  R            重置为 rembg bbox
  F            适应窗口
  1 / 2 / 3    缩放 100% / 200% / 300%
  Space / →    下一张
  ←            上一张
  Escape       退出
"""

import json
import tkinter as tk
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, ImageTk


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
HANDLE_SIZE = 8
BOX_COLOR = "#00FF00"
BOX_SAVED = "#FFD700"
HANDLE_COLOR = "#FF4444"
REMBG_COLOR = "#666666"


class Annotator:

    def __init__(self, source_dir: str, output_json: str):
        self.src = Path(source_dir)
        self.out = Path(output_json)

        self.files = sorted(
            [f.name for f in self.src.iterdir()
             if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        )
        if not self.files:
            raise SystemExit(f"{source_dir} 中没有图片")

        self.annotations: dict[str, dict] = {}
        if self.out.exists():
            d = json.loads(self.out.read_text("utf-8"))
            for a in d.get("annotations", []):
                self.annotations[a["file"]] = a

        self.idx = 0
        self.pil_img: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self.scale = 0.25
        self.ox = 0
        self.oy = 0
        self.bbox = [0, 0, 0, 0]
        self.rembg_bbox = None
        self.saved_bbox = None
        self._drag = None
        self._drag_sx = 0
        self._drag_sy = 0
        self._drag_box = None
        self._pan_ox = 0
        self._pan_oy = 0
        self._pan_mx = 0
        self._pan_my = 0
        self._rsess = None
        self._canvas_w = 1200
        self._canvas_h = 800

        # === UI ===
        self.root = tk.Tk()
        self.root.title(f"框选标注 — {self.src.name}")
        self.root.geometry("1400x900")
        self.root.configure(bg="#2B2B2B")

        bar = tk.Frame(self.root, bg="#3C3C3C")
        bar.pack(fill=tk.X)
        btn = {"bg": "#555", "fg": "white", "relief": tk.FLAT,
               "font": ("Microsoft YaHei UI", 9), "padx": 8, "pady": 2}
        tk.Button(bar, text="◀ 上一张", command=self.prev, **btn).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(bar, text="下一张 ▶", command=self.next, **btn).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Frame(bar, width=12, bg="#3C3C3C").pack(side=tk.LEFT)
        tk.Button(bar, text="保存 (S)", command=self.save, **btn).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(bar, text="重置 (R)", command=self.reset, **btn).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(bar, text="适应 (F)", command=self.fit, **btn).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(bar, text="1:1", command=lambda: self.set_zoom(1.0), **btn).pack(side=tk.LEFT, padx=2, pady=4)
        tk.Button(bar, text="2:1", command=lambda: self.set_zoom(2.0), **btn).pack(side=tk.LEFT, padx=2, pady=4)
        self._info = tk.Label(bar, text="", bg="#3C3C3C", fg="#CCC", font=("Consolas", 9))
        self._info.pack(side=tk.RIGHT, padx=10)

        self.canvas = tk.Canvas(self.root, bg="#1E1E1E", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self._status = tk.StringVar()
        tk.Label(self.root, textvariable=self._status, anchor=tk.W,
                 relief=tk.SUNKEN, font=("Consolas", 9), bg="#3C3C3C", fg="#CCC").pack(fill=tk.X)

        # 事件
        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_down)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_down)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)

        # 键盘
        self.root.bind("<Right>", lambda e: self.next())
        self.root.bind("<space>", lambda e: self.next())
        self.root.bind("<Left>", lambda e: self.prev())
        self.root.bind("<s>", lambda e: self.save())
        self.root.bind("<S>", lambda e: self.save())
        self.root.bind("<Return>", lambda e: self.save())
        self.root.bind("<r>", lambda e: self.reset())
        self.root.bind("<R>", lambda e: self.reset())
        self.root.bind("<f>", lambda e: self.fit())
        self.root.bind("<F>", lambda e: self.fit())
        self.root.bind("<1>", lambda e: self.set_zoom(1.0))
        self.root.bind("<2>", lambda e: self.set_zoom(2.0))
        self.root.bind("<3>", lambda e: self.set_zoom(3.0))
        self.root.bind("<Escape>", lambda e: self.root.destroy())

        self.root.after(300, self._load)
        self.root.mainloop()

    # ============================================================
    # 图像加载
    # ============================================================

    def _rembg_sess(self):
        if self._rsess is None:
            from rembg import new_session
            self._rsess = new_session()
        return self._rsess

    def _compute_rembg_bbox(self):
        from rembg import remove
        try:
            mask = remove(self.pil_img, session=self._rembg_sess(), only_mask=True)
            w, h = self.pil_img.size
            if mask.size != (w, h):
                mask = mask.resize((w, h), Image.LANCZOS)
            arr = np.array(mask)
            rows, cols = np.where(arr > 30)
            if len(rows) >= 100:
                return [int(cols.min()), int(rows.min()),
                        int(cols.max()), int(rows.max())]
        except Exception:
            pass
        return [0, 0, self.pil_img.width, self.pil_img.height]

    def _load(self):
        if self.idx < 0 or self.idx >= len(self.files):
            return
        fname = self.files[self.idx]

        self.pil_img = ImageOps.exif_transpose(
            Image.open(self.src / fname)).convert("RGB")

        if fname in self.annotations:
            ann = self.annotations[fname]
            self.bbox = list(ann["bbox"])
            self.rembg_bbox = ann.get("rembg_bbox")
            self.saved_bbox = list(ann["bbox"])
        else:
            self.rembg_bbox = self._compute_rembg_bbox()
            self.bbox = list(self.rembg_bbox)
            self.saved_bbox = None

        self.fit()
        self._update_info()

    def _update_info(self):
        fname = self.files[self.idx]
        saved = "✓" if fname in self.annotations else "✗"
        x1, y1, x2, y2 = self.bbox
        self._info.config(text=f"[{self.idx+1}/{len(self.files)}] {fname}")
        self._status.set(
            f"  [{self.idx+1}/{len(self.files)}] {fname}  "
            f"图像: {self.pil_img.width}×{self.pil_img.height}  "
            f"框: ({x1},{y1})-({x2},{y2}) {x2-x1}×{y2-y1}  "
            f"zoom: {self.scale:.0%}  {saved}"
        )

    # ============================================================
    # 视图
    # ============================================================

    def fit(self):
        if self.pil_img is None:
            return
        self.root.update_idletasks()
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        iw, ih = self.pil_img.size
        s = min(cw / iw, ch / ih, 1.0) * 0.90
        self.scale = max(0.02, s)
        self.ox = (cw - iw * self.scale) / 2
        self.oy = (ch - ih * self.scale) / 2
        self._draw()

    def set_zoom(self, s):
        self.scale = max(0.02, float(s))
        self.root.update_idletasks()
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        iw, ih = self.pil_img.size
        self.ox = (cw - iw * self.scale) / 2
        self.oy = (ch - ih * self.scale) / 2
        self._draw()

    # ============================================================
    # 坐标
    # ============================================================

    def _to_canvas(self, ix, iy):
        return (ix * self.scale + self.ox, iy * self.scale + self.oy)

    def _to_image(self, cx, cy):
        return (int((cx - self.ox) / self.scale), int((cy - self.oy) / self.scale))

    # ============================================================
    # 绘制
    # ============================================================

    def _draw(self):
        if self.pil_img is None:
            return
        self.canvas.delete("all")

        iw, ih = self.pil_img.size
        dw = max(1, int(iw * self.scale))
        dh = max(1, int(ih * self.scale))
        disp = self.pil_img.resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.create_image(self.ox, self.oy, anchor=tk.NW, image=self._photo)

        # rembg 框（灰色虚线）
        if self.rembg_bbox and not self._is_saved():
            self._draw_rect(self.rembg_bbox, REMBG_COLOR, 1, dash=(4, 8))

        # 当前框
        color = BOX_SAVED if self._is_saved() else BOX_COLOR
        self._draw_rect(self.bbox, color, 2)

        # 控制点
        if not self._is_saved():
            self._draw_handles()

    def _draw_rect(self, b, color, w, dash=()):
        x1, y1 = self._to_canvas(b[0], b[1])
        x2, y2 = self._to_canvas(b[2], b[3])
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=w, dash=dash)

    def _draw_handles(self):
        b = self.bbox
        x1, y1 = self._to_canvas(b[0], b[1])
        x2, y2 = self._to_canvas(b[2], b[3])
        hs = HANDLE_SIZE
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        for px, py in [(x1, y1), (x2, y1), (x1, y2), (x2, y2),
                       (mx, y1), (mx, y2), (x1, my), (x2, my)]:
            self.canvas.create_rectangle(px - hs, py - hs, px + hs, py + hs,
                                         fill=HANDLE_COLOR, outline="white", width=1)

    def _is_saved(self):
        return self.saved_bbox is not None and self.bbox == self.saved_bbox

    # ============================================================
    # 命中检测
    # ============================================================

    def _hit_handle(self, cx, cy):
        b = self.bbox
        hx1, hy1 = self._to_canvas(b[0], b[1])
        hx2, hy2 = self._to_canvas(b[2], b[3])
        mx, my = (hx1 + hx2) // 2, (hy1 + hy2) // 2
        dist = HANDLE_SIZE + 5
        pts = {"nw": (hx1, hy1), "ne": (hx2, hy1), "sw": (hx1, hy2), "se": (hx2, hy2),
               "n": (mx, hy1), "s": (mx, hy2), "w": (hx1, my), "e": (hx2, my)}
        best, best_d = None, dist
        for tag, (px, py) in pts.items():
            d = ((cx - px)**2 + (cy - py)**2)**0.5
            if d < best_d:
                best, best_d = tag, d
        return best

    def _in_box(self, cx, cy):
        b = self.bbox
        x1, y1 = self._to_canvas(b[0], b[1])
        x2, y2 = self._to_canvas(b[2], b[3])
        return x1 <= cx <= x2 and y1 <= cy <= y2

    # ============================================================
    # 鼠标
    # ============================================================

    def _on_down(self, event):
        if self._is_saved():
            return
        h = self._hit_handle(event.x, event.y)
        if h:
            self._drag = h
            self._drag_sx, self._drag_sy = event.x, event.y
            self._drag_box = list(self.bbox)
            return
        if self._in_box(event.x, event.y):
            self._drag = "move"
            self._drag_sx, self._drag_sy = event.x, event.y
            self._drag_box = list(self.bbox)

    def _on_move(self, event):
        if not self._drag or not self._drag_box:
            return
        dx = (event.x - self._drag_sx) / self.scale
        dy = (event.y - self._drag_sy) / self.scale
        iw, ih = self.pil_img.width, self.pil_img.height
        x1, y1, x2, y2 = self._drag_box

        moves = {
            "move": lambda: (
                max(0, min(iw - (x2 - x1), x1 + dx)),
                max(0, min(ih - (y2 - y1), y1 + dy)),
                max(0, min(iw - (x2 - x1), x1 + dx)) + (x2 - x1),
                max(0, min(ih - (y2 - y1), y1 + dy)) + (y2 - y1),
            ),
            "nw": lambda: (min(max(0, x1 + dx), x2 - 10), min(max(0, y1 + dy), y2 - 10), x2, y2),
            "ne": lambda: (x1, min(max(0, y1 + dy), y2 - 10), max(x1 + 10, min(iw, x2 + dx)), y2),
            "sw": lambda: (min(max(0, x1 + dx), x2 - 10), y1, x2, max(y1 + 10, min(ih, y2 + dy))),
            "se": lambda: (x1, y1, max(x1 + 10, min(iw, x2 + dx)), max(y1 + 10, min(ih, y2 + dy))),
            "n": lambda: (x1, min(max(0, y1 + dy), y2 - 10), x2, y2),
            "s": lambda: (x1, y1, x2, max(y1 + 10, min(ih, y2 + dy))),
            "w": lambda: (min(max(0, x1 + dx), x2 - 10), y1, x2, y2),
            "e": lambda: (x1, y1, max(x1 + 10, min(iw, x2 + dx)), y2),
        }
        fn = moves.get(self._drag)
        if fn:
            self.bbox = [int(v) for v in fn()]
            self.saved_bbox = None
            self._draw()
            self._update_info()

    def _on_up(self, event):
        self._drag = None
        self._drag_box = None

    def _on_wheel(self, event):
        if self.pil_img is None:
            return
        mx, my = event.x, event.y
        old_ix, old_iy = self._to_image(mx, my)
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.scale = max(0.02, min(10.0, self.scale * factor))
        self.ox = mx - old_ix * self.scale
        self.oy = my - old_iy * self.scale
        self._draw()

    def _on_pan_down(self, event):
        self._pan_ox = self.ox
        self._pan_oy = self.oy
        self._pan_mx = event.x
        self._pan_my = event.y

    def _on_pan_move(self, event):
        self.ox = self._pan_ox + (event.x - self._pan_mx)
        self.oy = self._pan_oy + (event.y - self._pan_my)
        self._draw()

    # ============================================================
    # 操作
    # ============================================================

    def next(self):
        if self.idx < len(self.files) - 1:
            self.idx += 1
            self._load()

    def prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._load()

    def save(self):
        fname = self.files[self.idx]
        self.annotations[fname] = {
            "file": fname,
            "bbox": list(self.bbox),
            "rembg_bbox": list(self.rembg_bbox) if self.rembg_bbox else None,
        }
        self.saved_bbox = list(self.bbox)
        data = {"source_dir": str(self.src), "annotations": list(self.annotations.values())}
        self.out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._draw()
        self._update_info()
        if self.idx < len(self.files) - 1:
            self.root.after(200, self.next)

    def reset(self):
        if self.rembg_bbox:
            self.bbox = list(self.rembg_bbox)
        self.saved_bbox = None
        self._draw()
        self._update_info()

    def set_zoom(self, s):
        self.root.update_idletasks()
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        iw, ih = self.pil_img.size
        self.ox = (cw - iw * self.scale) / 2
        self.oy = (ch - ih * self.scale) / 2
        self._draw()


def main():
    import sys
    if len(sys.argv) < 2:
        print("用法: python annotator.py <素材目录> [输出JSON]")
        sys.exit(1)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f"{src}/annotations.json"
    Annotator(src, out)


if __name__ == "__main__":
    main()
