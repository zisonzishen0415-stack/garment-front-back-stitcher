"""Mask 标注工具 — 画笔擦除 rembg mask 中的多余区域（人台/杆子/支架）

启动:
    python mask_annotator.py <素材目录>

操作:
    鼠标拖拽      擦除 mask（变透明）
    右键拖拽      恢复 mask
    滚轮          缩放
    Ctrl + 滚轮   笔刷大小 ±5
    [ / ]         笔刷大小 ±5
    中键拖拽      平移
    ← → / ◀ ▶   上一张 / 下一张（自动保存）
    S / Enter     保存当前
    R             重置为原始 rembg mask
    F             适应窗口
    Escape        退出
"""

import sys
import threading
import tkinter as tk
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageOps, ImageTk

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
MASK_ALPHA = 100        # 红色叠加层透明度 (0-255, PIL RGBA)
BRUSH_COLOR = "#00BFFF"  # 笔刷光标颜色
BRUSH_DEFAULT = 40


# ═══════════════════════════════════════════════════════════════
# MaskCanvas — 图片 + mask 显示与编辑画布
# ═══════════════════════════════════════════════════════════════

class MaskCanvas(tk.Canvas):
    """图片显示 + mask 半透明红色叠加 + 画笔擦除/恢复 + 缩放/平移"""

    def __init__(self, parent, on_changed=None, **kw):
        super().__init__(parent, bg="#1E1E1E", highlightthickness=0, **kw)
        self.pil_img: Optional[Image.Image] = None
        self.mask_arr: Optional[np.ndarray] = None   # uint8, 255=保留 0=排除
        self._original_mask: Optional[np.ndarray] = None
        self.scale = 0.2
        self.ox = 0.0
        self.oy = 0.0
        self._photo = None
        self._overlay_photo = None
        self._brush_radius = BRUSH_DEFAULT
        self._cursor_x = 0       # 鼠标画布坐标（光标跟随）
        self._cursor_y = 0
        self._mode = None          # 'erase' | 'restore' | 'pan'
        self._last_brush_x = 0
        self._last_brush_y = 0
        self._pan_data = None      # (ox0, oy0, mx0, my0)
        self._last_fit_w = 0
        self._last_fit_h = 0
        self._on_changed = on_changed

        self.bind("<ButtonPress-1>",  lambda e: self._start_brush(e, 'erase'))
        self.bind("<ButtonPress-3>",  lambda e: self._start_brush(e, 'restore'))
        self.bind("<B1-Motion>",      self._on_brush)
        self.bind("<B3-Motion>",      self._on_brush)
        self.bind("<ButtonRelease-1>", self._on_up)
        self.bind("<ButtonRelease-3>", self._on_up)
        self.bind("<ButtonPress-2>",  self._on_pan_down)
        self.bind("<B2-Motion>",      self._on_pan_move)
        self.bind("<MouseWheel>",     self._on_wheel)
        self.bind("<Configure>",      self._on_configure)
        self.bind("<Motion>",         self._on_cursor_move)

    # ── 坐标转换 ────────────────────────────────────────────────

    def _to_image(self, cx, cy):
        return int((cx - self.ox) / self.scale), int((cy - self.oy) / self.scale)

    def _to_canvas(self, ix, iy):
        return ix * self.scale + self.ox, iy * self.scale + self.oy

    # ── 公共接口 ────────────────────────────────────────────────

    def set_image(self, pil_img, mask_arr):
        self.pil_img = pil_img
        self.mask_arr = mask_arr.copy() if mask_arr is not None else None
        self._original_mask = mask_arr.copy() if mask_arr is not None else None
        self._last_fit_w = 0
        self._fit()

    def set_brush(self, radius):
        self._brush_radius = max(3, min(200, radius))
        self._draw_cursor()

    def reset_mask(self):
        if self._original_mask is not None:
            self.mask_arr = self._original_mask.copy()
            self._redraw()
            if self._on_changed:
                self._on_changed()

    # ── 布局与渲染 ──────────────────────────────────────────────

    def _fit(self):
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw < 30 or ch < 30:
            return
        if not self.pil_img:
            self.delete("all")
            return
        if abs(cw - self._last_fit_w) < 4 and abs(ch - self._last_fit_h) < 4:
            return
        self._last_fit_w = cw
        self._last_fit_h = ch
        iw, ih = self.pil_img.size
        if ih > iw:
            s = ch / ih
        else:
            s = min(cw / iw, ch / ih) * 0.88
        self.scale = max(0.02, min(5.0, s))
        self.ox = (cw - iw * self.scale) / 2
        self.oy = (ch - ih * self.scale) / 2
        self._redraw()

    def _redraw(self):
        """完整重绘：背景图 + mask 叠加层 + 光标（切换图片/缩放/平移时调用）。"""
        self.delete("all")
        if not self.pil_img:
            return
        iw, ih = self.pil_img.size
        dw = max(1, int(iw * self.scale))
        dh = max(1, int(ih * self.scale))

        # 背景图 (BILINEAR)
        disp = self.pil_img.resize((dw, dh), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(disp)
        self.create_image(self.ox, self.oy, anchor=tk.NW, image=self._photo, tags="bg")

        self._redraw_overlay()

    def _redraw_overlay(self):
        """只重绘叠加层 + 光标（涂抹时调用，不重建背景图）。"""
        self.delete("overlay", "cursor")
        if self.mask_arr is None:
            return
        iw, ih = self.pil_img.size
        dw = max(1, int(iw * self.scale))
        dh = max(1, int(ih * self.scale))

        # mask 半透明红色叠加
        h, w = self.mask_arr.shape
        if (dw, dh) != (w, h):
            mask_sm = np.array(
                Image.fromarray(self.mask_arr).resize((dw, dh), Image.BILINEAR))
        else:
            mask_sm = self.mask_arr
        overlay = np.zeros((dh, dw, 4), dtype=np.uint8)
        fg = mask_sm > 128
        overlay[fg, 0] = 255       # R
        overlay[fg, 3] = MASK_ALPHA  # A
        self._overlay_photo = ImageTk.PhotoImage(
            Image.fromarray(overlay))
        self.create_image(self.ox, self.oy, anchor=tk.NW,
                          image=self._overlay_photo, tags="overlay")

        # 笔刷光标圆
        self._draw_cursor()

    def _draw_cursor(self):
        """绘制笔刷光标：圆圈轮廓，显示当前笔刷大小。"""
        self.delete("cursor")
        if self.pil_img is None:
            return
        r = max(3, int(self._brush_radius * self.scale))
        x, y = self._cursor_x, self._cursor_y
        self.create_oval(x - r, y - r, x + r, y + r,
                         outline=BRUSH_COLOR, width=2, tags="cursor")

    def _on_cursor_move(self, event):
        self._cursor_x = event.x
        self._cursor_y = event.y
        if self.pil_img is None:
            return
        self._draw_cursor()

    def _on_configure(self, event):
        self._fit()

    # ── 笔刷涂抹 ────────────────────────────────────────────────

    def _apply_brush(self, cx, cy, erase):
        """在画布坐标 (cx, cy) 处涂抹笔刷圆。erase=True → 设0，False → 设255。"""
        if self.mask_arr is None:
            return
        ix, iy = self._to_image(cx, cy)
        h, w = self.mask_arr.shape
        r = max(1, self._brush_radius)
        y1, y2 = max(0, iy - r), min(h, iy + r + 1)
        x1, x2 = max(0, ix - r), min(w, ix + r + 1)
        if y1 >= y2 or x1 >= x2:
            return
        yy, xx = np.ogrid[y1:y2, x1:x2]
        dist = np.sqrt((xx - ix) ** 2 + (yy - iy) ** 2)
        circle = dist <= r
        self.mask_arr[y1:y2, x1:x2][circle] = 0 if erase else 255

    def _start_brush(self, event, mode):
        self._mode = mode
        self._cursor_x = self._last_brush_x = event.x
        self._cursor_y = self._last_brush_y = event.y
        self._apply_brush(event.x, event.y, mode == 'erase')
        self._redraw_overlay()

    def _on_brush(self, event):
        if self._mode not in ('erase', 'restore'):
            return
        self._cursor_x = event.x
        self._cursor_y = event.y
        # 沿线段插值，避免快速拖拽时空隙
        dx = event.x - self._last_brush_x
        dy = event.y - self._last_brush_y
        dist = max(abs(dx), abs(dy))
        erase = self._mode == 'erase'
        if dist > 0:
            steps = max(1, int(dist / 3))
            for i in range(1, steps + 1):
                t = i / steps
                self._apply_brush(
                    self._last_brush_x + dx * t,
                    self._last_brush_y + dy * t, erase)
        else:
            self._apply_brush(event.x, event.y, erase)
        self._last_brush_x = event.x
        self._last_brush_y = event.y
        self._redraw_overlay()
        if self._on_changed:
            self._on_changed()

    def _on_up(self, event):
        self._mode = None

    # ── 平移 ────────────────────────────────────────────────────

    def _on_pan_down(self, event):
        self._mode = 'pan'
        self._pan_data = (self.ox, self.oy, event.x, event.y)

    def _on_pan_move(self, event):
        if self._mode != 'pan' or not self._pan_data:
            return
        ox0, oy0, mx0, my0 = self._pan_data
        self.ox = ox0 + (event.x - mx0)
        self.oy = oy0 + (event.y - my0)
        self._redraw()

    # ── 缩放 ────────────────────────────────────────────────────

    def _on_wheel(self, event):
        if event.state & 0x4:  # Ctrl
            self._brush_radius = max(3, min(200,
                self._brush_radius + (-5 if event.delta < 0 else 5)))
            return
        if not self.pil_img:
            return
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        mx, my = self._to_image(event.x, event.y)
        self.scale = max(0.02, min(5.0, self.scale * factor))
        self.ox = event.x - mx * self.scale
        self.oy = event.y - my * self.scale
        self._redraw()


# ═══════════════════════════════════════════════════════════════
# MaskAnnotator — 主窗口
# ═══════════════════════════════════════════════════════════════

class MaskAnnotator(tk.Tk):
    """Mask 标注主窗口：逐张编辑 rembg mask，保存为 _mask.png"""

    def __init__(self, source_dir: str):
        super().__init__()
        self.title("PS Mask 标注")
        self.geometry("1400x800")
        self.minsize(600, 400)

        self.src = Path(source_dir).resolve()
        self.files = sorted(
            [f for f in self.src.iterdir()
             if f.is_file() and f.suffix.lower() in IMAGE_EXTS],
            key=lambda f: f.name)
        if not self.files:
            print("未找到支持的图片文件"); self.destroy(); return

        self.idx = 0
        self._masks = {}         # filename → np.ndarray (rembg 缓存)
        self._mask_files = {}    # filename → bool (有 _mask.png?)
        self._dirty = False
        self._session = None
        self._rembg_done = 0
        self._rembg_total = len(self.files)

        # 扫描已有 _mask.png
        for f in self.files:
            mp = self.src / f"{f.stem}_mask.png"
            if mp.exists():
                self._mask_files[f.name] = True

        self._build_ui()
        self._bind_keys()
        self._start_rembg_worker()
        self._load_current()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI 构建 ─────────────────────────────────────────────────

    def _build_ui(self):
        bar = tk.Frame(self)
        bar.pack(fill="x", padx=8, pady=(6, 2))

        tk.Label(bar, text="笔刷:").pack(side="left", padx=(0, 4))
        self.lbl_brush = tk.Label(bar, text=str(BRUSH_DEFAULT), width=4,
                                  font=("TkDefaultFont", 11, "bold"))
        self.lbl_brush.pack(side="left")
        tk.Button(bar, text="-5", width=3,
                  command=lambda: self._adj_brush(-5)).pack(side="left", padx=2)
        tk.Button(bar, text="+5", width=3,
                  command=lambda: self._adj_brush(5)).pack(side="left", padx=2)

        tk.Frame(bar, width=1, height=24, bg="#555").pack(side="left", padx=10)
        tk.Button(bar, text="重置 (R)",
                  command=self._reset).pack(side="left", padx=2)
        tk.Button(bar, text="保存 (S)", command=self._save,
                  bg="#2B8C3C", fg="white").pack(side="left", padx=2)

        tk.Frame(bar, width=1, height=24, bg="#555").pack(side="left", padx=10)
        tk.Button(bar, text="◀", width=3,
                  command=self._prev).pack(side="left", padx=1)
        self.lbl_idx = tk.Label(bar, text="0 / 0",
                                font=("TkDefaultFont", 11, "bold"), width=10)
        self.lbl_idx.pack(side="left")
        tk.Button(bar, text="▶", width=3,
                  command=self._next).pack(side="left", padx=1)

        self.lbl_status = tk.Label(bar, text="", fg="#999")
        self.lbl_status.pack(side="right", padx=8)

        # 画布
        self.canvas = MaskCanvas(self, on_changed=self._on_changed)
        self.canvas.pack(fill="both", expand=True)

        # 底部状态
        bottom = tk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=(2, 6))
        self.lbl_fname = tk.Label(bottom, text="", fg="#AAA")
        self.lbl_fname.pack(side="left")
        self.lbl_info = tk.Label(bottom, text="", fg="#888")
        self.lbl_info.pack(side="right")

    # ── 快捷键 ──────────────────────────────────────────────────

    def _bind_keys(self):
        self.bind("<s>", lambda e: self._save())
        self.bind("<S>", lambda e: self._save())
        self.bind("<Return>", lambda e: self._save())
        self.bind("<r>", lambda e: self._reset())
        self.bind("<R>", lambda e: self._reset())
        self.bind("<f>", lambda e: self.canvas._fit())
        self.bind("<F>", lambda e: self.canvas._fit())
        self.bind("<Right>", lambda e: self._next())
        self.bind("<Left>", lambda e: self._prev())
        self.bind("<bracketleft>", lambda e: self._adj_brush(-5))
        self.bind("<bracketright>", lambda e: self._adj_brush(5))
        self.bind("<Escape>", lambda e: self._on_close())

    # ── rembg 后台 ──────────────────────────────────────────────

    def _get_session(self):
        if self._session is None:
            from rembg import new_session
            self._session = new_session()
        return self._session

    def _start_rembg_worker(self):
        def worker():
            for i, f in enumerate(self.files):
                if f.name in self._mask_files:
                    self._rembg_done = i + 1
                    continue
                try:
                    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
                    from rembg import remove
                    mask = remove(img, session=self._get_session(), only_mask=True)
                    if mask.size != img.size:
                        mask = mask.resize(img.size, Image.LANCZOS)
                    self._masks[f.name] = np.array(mask)
                except Exception:
                    self._masks[f.name] = None
                self._rembg_done = i + 1
                # 如果刚好是当前图片，加载
                if i == self.idx:
                    self.after(0, self._load_current)
            self.after(0, self._update_status)
        threading.Thread(target=worker, daemon=True).start()

    # ── 文件加载/保存 ───────────────────────────────────────────

    def _load_current(self):
        if not self.files:
            return
        f = self.files[self.idx]
        self._dirty = False
        try:
            img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
        except Exception:
            return

        mp = self.src / f"{f.stem}_mask.png"
        if mp.exists():
            mask_arr = np.array(Image.open(mp).convert("L"))
            self._mask_files[f.name] = True
        elif f.name in self._masks and self._masks[f.name] is not None:
            mask_arr = self._masks[f.name]
        else:
            mask_arr = None

        self.canvas.set_image(img, mask_arr)
        self._update_status()

    def _save(self):
        f = self.files[self.idx]
        if self.canvas.mask_arr is None:
            return
        mp = self.src / f"{f.stem}_mask.png"
        Image.fromarray(self.canvas.mask_arr).save(mp, "PNG")
        self._mask_files[f.name] = True
        self._dirty = False
        self._update_status()

    # ── 导航 ────────────────────────────────────────────────────

    def _next(self):
        if self.idx < len(self.files) - 1:
            self._save()
            self.idx += 1
            self._load_current()

    def _prev(self):
        if self.idx > 0:
            self._save()
            self.idx -= 1
            self._load_current()

    def _reset(self):
        self.canvas.reset_mask()
        self._dirty = False
        self._update_status()

    def _adj_brush(self, delta):
        r = self.canvas._brush_radius + delta
        self.canvas.set_brush(r)
        self.lbl_brush.configure(text=str(r))
        self._update_status()

    def _on_changed(self):
        self._dirty = True
        self._update_status()

    def _update_status(self):
        if not self.files:
            return
        f = self.files[self.idx]
        self.lbl_idx.configure(text=f"{self.idx + 1} / {len(self.files)}")
        self.lbl_fname.configure(text=f.stem)
        self.lbl_brush.configure(text=str(self.canvas._brush_radius))
        dirty_mark = " *" if self._dirty else ""
        has_mask = "✓" if f.name in self._mask_files else "○"
        progress = f"rembg: {self._rembg_done}/{self._rembg_total}"
        self.lbl_status.configure(text=f"{progress}  |  已保存: {has_mask}{dirty_mark}")
        r = self.canvas._brush_radius
        mod = "已修改" if self._dirty else "未修改"
        self.lbl_info.configure(text=f"笔刷: {r}px  |  {mod}")

    def _on_close(self):
        self._save()
        self.destroy()


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("用法: python mask_annotator.py <素材目录>")
        sys.exit(1)
    app = MaskAnnotator(sys.argv[1])
    app.mainloop()

if __name__ == "__main__":
    main()
