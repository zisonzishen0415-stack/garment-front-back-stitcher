"""审核编辑工具 — v4
流程：选文件夹 → AI+CV 流式处理（完成一对立即可审）
左预览 + 右编辑 + 角度旋钮
"""
import json
import math
import threading
import tkinter as tk
from pathlib import Path
from typing import Optional
from PIL import Image, ImageOps, ImageTk
import customtkinter as ctk

from processor_v11 import ImageProcessorV11 as ImageProcessor
from liquify import LiquifyTool

MARGIN = 0.12
HANDLE_SIZE = 5
BOX_COLOR = "#00FF00"
HANDLE_COLOR = "#FF4444"
AI_COLOR = "#666666"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ═══════════════════════════════════════════════════════════════
# Bbox 编辑器（无旋转手柄）
# ═══════════════════════════════════════════════════════════════

class BBoxEditor(tk.Canvas):
    """bbox 编辑器：缩放/平移/角点+边中点拖拽"""

    def __init__(self, parent, on_change=None, **kw):
        super().__init__(parent, bg="#1E1E1E", highlightthickness=0, **kw)
        self.pil_img: Optional[Image.Image] = None
        self.bbox = [0, 0, 0, 0]
        self.angle = 0.0
        self.ai_bbox = None
        self.scale = 0.2
        self.ox = 0
        self.oy = 0
        self._photo = None
        self._drag = None
        self._drag_sx = 0
        self._drag_sy = 0
        self._drag_box = None
        self._pan_data = None
        self._on_change = on_change

        self.bind("<ButtonPress-1>", self._on_down)
        self.bind("<B1-Motion>", self._on_move)
        self.bind("<ButtonRelease-1>", self._on_up)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<ButtonPress-2>", self._on_pan_down)
        self.bind("<B2-Motion>", self._on_pan_move)
        self.bind("<ButtonPress-3>", self._on_pan_down)
        self.bind("<B3-Motion>", self._on_pan_move)

    def set_image(self, pil_img, bbox, ai_bbox=None, angle=0.0):
        self.pil_img = pil_img
        self.bbox = list(bbox)
        self.angle = float(angle)
        self.ai_bbox = list(ai_bbox) if ai_bbox else None
        self._fit()

    def _fit(self):
        if not self.pil_img: return
        self.update_idletasks()
        cw = max(self.winfo_width(), 50)
        ch = max(self.winfo_height(), 50)
        iw, ih = self.pil_img.size
        s = min(cw / iw, ch / ih) * 0.85
        self.scale = max(0.02, s)
        self.ox = (cw - iw * self.scale) / 2
        self.oy = (ch - ih * self.scale) / 2
        self._redraw()

    def _to_canvas(self, ix, iy):
        return (ix * self.scale + self.ox, iy * self.scale + self.oy)

    def _to_image(self, cx, cy):
        return (int((cx - self.ox) / self.scale), int((cy - self.oy) / self.scale))

    def _rotated_corners(self, b, angle_deg):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        corners = [(b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])]
        rotated = []
        for x, y in corners:
            dx, dy = x - cx, y - cy
            rotated.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
        return rotated

    def _redraw(self):
        """完整重绘：背景图 + 叠加层"""
        self.delete("all")
        if not self.pil_img: return

        # 背景图
        iw, ih = self.pil_img.size
        dw, dh = max(1, int(iw * self.scale)), max(1, int(ih * self.scale))
        disp = self.pil_img.resize((dw, dh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(disp)
        self.create_image(self.ox, self.oy, anchor=tk.NW, image=self._photo, tags="all")

        # AI bbox（灰虚线）
        if self.ai_bbox:
            self._draw_poly(self.ai_bbox, 0.0, AI_COLOR, 1, (4, 8))

        # 当前 bbox（绿实线）
        self._draw_poly(self.bbox, self.angle, BOX_COLOR, 2)

        # 控制点
        corners = self._rotated_corners(self.bbox, self.angle)
        hs = HANDLE_SIZE
        for ix, iy in corners:
            cx_, cy_ = self._to_canvas(ix, iy)
            self.create_rectangle(cx_ - hs, cy_ - hs, cx_ + hs, cy_ + hs,
                                  fill=HANDLE_COLOR, outline="white", width=1)
        edges = [
            ((corners[0][0]+corners[1][0])/2, (corners[0][1]+corners[1][1])/2),
            ((corners[1][0]+corners[2][0])/2, (corners[1][1]+corners[2][1])/2),
            ((corners[2][0]+corners[3][0])/2, (corners[2][1]+corners[3][1])/2),
            ((corners[3][0]+corners[0][0])/2, (corners[3][1]+corners[0][1])/2),
        ]
        for ix, iy in edges:
            cx_, cy_ = self._to_canvas(ix, iy)
            self.create_rectangle(cx_ - hs, cy_ - hs, cx_ + hs, cy_ + hs,
                                  fill="#FFFFFF", outline=BOX_COLOR, width=1)

    def _draw_poly(self, b, angle_deg, color, width, dash=()):
        corners = self._rotated_corners(b, angle_deg)
        pts = []
        for x, y in corners:
            cx_, cy_ = self._to_canvas(x, y)
            pts.extend([cx_, cy_])
        self.create_polygon(*pts, outline=color, width=width, fill="", dash=dash)

    # ── 命中检测 ──────────────────────────────────────────────

    def _hit_corner(self, cx, cy):
        corners = self._rotated_corners(self.bbox, self.angle)
        tags = ["nw", "ne", "se", "sw"]
        best, best_d = None, HANDLE_SIZE + 5
        for tag, (ix, iy) in zip(tags, corners):
            cix, ciy = self._to_canvas(ix, iy)
            d = ((cx - cix)**2 + (cy - ciy)**2)**0.5
            if d < best_d:
                best, best_d = tag, d
        return best

    def _hit_edge(self, cx, cy):
        corners = self._rotated_corners(self.bbox, self.angle)
        tags = ["n", "e", "s", "w"]
        edge_defs = [
            (corners[0], corners[1]), (corners[1], corners[2]),
            (corners[2], corners[3]), (corners[3], corners[0]),
        ]
        best, best_d = None, HANDLE_SIZE + 5
        for tag, ((x1,y1),(x2,y2)) in zip(tags, edge_defs):
            cx1, cy1 = self._to_canvas(x1, y1)
            cx2, cy2 = self._to_canvas(x2, y2)
            dx, dy = cx2 - cx1, cy2 - cy1
            length2 = dx*dx + dy*dy
            if length2 == 0: continue
            t = max(0, min(1, ((cx - cx1)*dx + (cy - cy1)*dy) / length2))
            px, py = cx1 + t*dx, cy1 + t*dy
            d = ((cx - px)**2 + (cy - py)**2)**0.5
            if d < best_d:
                best, best_d = tag, d
        return best

    def _poly_contains(self, cx, cy):
        corners = self._rotated_corners(self.bbox, self.angle)
        pts = [(self._to_canvas(ix, iy)) for ix, iy in corners]
        n = len(pts)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = pts[i]
            xj, yj = pts[j]
            if ((yi > cy) != (yj > cy)) and (cx < (xj - xi) * (cy - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    # ── 鼠标事件 ──────────────────────────────────────────────

    def _on_down(self, event):
        h = self._hit_corner(event.x, event.y)
        if h:
            self._drag = h
            self._drag_sx, self._drag_sy = event.x, event.y
            self._drag_box = list(self.bbox)
            return
        e = self._hit_edge(event.x, event.y)
        if e:
            self._drag = e
            self._drag_sx, self._drag_sy = event.x, event.y
            self._drag_box = list(self.bbox)
            return
        if self._poly_contains(event.x, event.y):
            self._drag = "move"
            self._drag_sx, self._drag_sy = event.x, event.y
            self._drag_box = list(self.bbox)

    def _on_move(self, event):
        if not self._drag or self._drag_box is None: return
        dx = (event.x - self._drag_sx) / self.scale
        dy = (event.y - self._drag_sy) / self.scale
        # 反旋转到轴对齐
        rad = math.radians(-self.angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rdx = dx * cos_a - dy * sin_a
        rdy = dx * sin_a + dy * cos_a

        iw, ih = self.pil_img.width, self.pil_img.height
        x1, y1, x2, y2 = self._drag_box
        moves = {
            "move": lambda: (int(max(0, x1+rdx)), int(max(0, y1+rdy)),
                             int(min(iw, x1+rdx+(x2-x1))), int(min(ih, y1+rdy+(y2-y1))))
                     if x1+rdx+(x2-x1) <= iw and y1+rdy+(y2-y1) <= ih else None,
            "nw": lambda: (int(min(max(0, x1+rdx), x2-10)), int(min(max(0, y1+rdy), y2-10)), x2, y2),
            "ne": lambda: (x1, int(min(max(0, y1+rdy), y2-10)), int(max(x1+10, min(iw, x2+rdx))), y2),
            "sw": lambda: (int(min(max(0, x1+rdx), x2-10)), y1, x2, int(max(y1+10, min(ih, y2+rdy)))),
            "se": lambda: (x1, y1, int(max(x1+10, min(iw, x2+rdx))), int(max(y1+10, min(ih, y2+rdy)))),
            "n":  lambda: (x1, int(min(max(0, y1+rdy), y2-10)), x2, y2),
            "s":  lambda: (x1, y1, x2, int(max(y1+10, min(ih, y2+rdy)))),
            "w":  lambda: (int(min(max(0, x1+rdx), x2-10)), y1, x2, y2),
            "e":  lambda: (x1, y1, int(max(x1+10, min(iw, x2+rdx))), y2),
        }
        fn = moves.get(self._drag)
        if fn:
            new = fn()
            if new and len(new) == 4:
                self.bbox = list(new)
                self._redraw()
                if self._on_change: self._on_change()

    def _on_up(self, event):
        self._drag = None
        self._drag_box = None

    def _on_wheel(self, event):
        if not self.pil_img: return
        mx, my = event.x, event.y
        old_ix, old_iy = self._to_image(mx, my)
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.scale = max(0.02, min(5.0, self.scale * factor))
        self.ox = mx - old_ix * self.scale
        self.oy = my - old_iy * self.scale
        self._redraw()

    def _on_pan_down(self, event):
        self._pan_data = (self.ox, self.oy, event.x, event.y)

    def _on_pan_move(self, event):
        if self._pan_data:
            ox0, oy0, mx0, my0 = self._pan_data
            self.ox = ox0 + (event.x - mx0)
            self.oy = oy0 + (event.y - my0)
            self._redraw()

    def update_display(self):
        self._fit()


# ═══════════════════════════════════════════════════════════════
# 主应用
# ═══════════════════════════════════════════════════════════════

class ReviewerApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("PS 审核编辑")
        self.geometry("1500x850")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.processor = ImageProcessor()
        self.pairs: list[tuple[Path, Path]] = []
        self.pair_idx = 0
        self.img_a: Optional[Image.Image] = None
        self.img_b: Optional[Image.Image] = None
        self.bbox_a = None
        self.bbox_b = None
        self.angle_a = 0.0
        self.angle_b = 0.0
        self.ai_bbox_a = None
        self.ai_bbox_b = None
        self.annotations: dict[str, dict] = {}
        self.input_dir: Optional[Path] = None

        # 流式处理
        self._proc_done = 0
        self._proc_total = 0
        self._results: list = []
        self._first_loaded = False

        self.btn_debug = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────

    def _build_ui(self):
        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=8, pady=(8, 4))

        ctk.CTkLabel(bar, text="文件夹:").pack(side="left", padx=(8, 4))
        self.entry_dir = ctk.CTkEntry(bar, width=280)
        self.entry_dir.pack(side="left", padx=4)
        ctk.CTkButton(bar, text="浏览", width=50, command=self._pick_dir).pack(side="left", padx=2)
        ctk.CTkButton(bar, text="AI 处理", width=70, command=self._start_process).pack(side="left", padx=(10, 2))
        self.btn_debug = ctk.CTkButton(bar, text="调试", width=50, fg_color="#555",
                                        command=self._start_debug, state="disabled")
        self.btn_debug.pack(side="left", padx=2)

        ctk.CTkFrame(bar, width=1, height=24, fg_color="#555").pack(side="left", padx=10)

        self.btn_prev = ctk.CTkButton(bar, text="◀ 上一对", width=70, command=self._prev_pair, state="disabled")
        self.btn_prev.pack(side="left", padx=2)
        self.lbl_idx = ctk.CTkLabel(bar, text="0 / 0", font=ctk.CTkFont(size=13, weight="bold"), width=70)
        self.lbl_idx.pack(side="left", padx=2)
        self.btn_next = ctk.CTkButton(bar, text="下一对 ▶", width=70, command=self._next_pair, state="disabled")
        self.btn_next.pack(side="left", padx=2)

        ctk.CTkFrame(bar, width=1, height=24, fg_color="#555").pack(side="left", padx=10)

        ctk.CTkButton(bar, text="⇄ 互换", width=60, command=self._swap_fb).pack(side="left", padx=2)
        ctk.CTkButton(bar, text="重置AI", width=60, command=self._reset_ai).pack(side="left", padx=2)
        ctk.CTkButton(bar, text="液化", width=50, fg_color="#6B3FA0", hover_color="#56338A",
                       command=self._liquify).pack(side="left", padx=2)

        ctk.CTkFrame(bar, width=1, height=24, fg_color="#555").pack(side="left", padx=10)

        ctk.CTkButton(bar, text="导出当前", width=70, fg_color="#2B8C3C", hover_color="#236E30",
                       command=self._export_single).pack(side="left", padx=2)
        ctk.CTkButton(bar, text="全部导出", width=70, command=self._export_all).pack(side="left", padx=2)

        self.lbl_fname = ctk.CTkLabel(bar, text="", font=ctk.CTkFont(size=9), text_color="#999")
        self.lbl_fname.pack(side="right", padx=8)

        # 主体
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        # 左：预览面板 — 强制 1:1 正方形（宽 = main 可用高度）
        left = ctk.CTkFrame(main, width=500, height=500)
        left.pack(side="left", fill="y", padx=(4, 2), pady=4)
        left.pack_propagate(False)
        ctk.CTkLabel(left, text="拼接预览", font=ctk.CTkFont(weight="bold")).pack(pady=(6, 2))
        self.preview_canvas = tk.Canvas(left, bg="#1E1E1E", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        main.bind("<Configure>", lambda e: left.configure(width=max(200, e.height - 8)))

        # 右：编辑（左右并排，占据剩余水平空间）
        right = ctk.CTkFrame(main)
        right.pack(side="left", fill="both", expand=True, padx=(2, 4), pady=4)

        # -- 正面（左） --
        frame_a = ctk.CTkFrame(right)
        frame_a.pack(side="left", fill="both", expand=True, padx=(2, 2), pady=4)

        row_a = ctk.CTkFrame(frame_a)
        row_a.pack(fill="x", padx=4, pady=(4, 2))
        ctk.CTkLabel(row_a, text="正面", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(row_a, text="↻顺", width=30, command=lambda: self._adj_angle('a', -0.5)).pack(side="right", padx=1)
        self.lbl_angle_a = ctk.CTkLabel(row_a, text="0.0°", width=40, font=ctk.CTkFont(size=11))
        self.lbl_angle_a.pack(side="right", padx=2)
        ctk.CTkButton(row_a, text="↺逆", width=30, command=lambda: self._adj_angle('a', +0.5)).pack(side="right", padx=1)

        self.editor_a = BBoxEditor(frame_a, on_change=self._on_bbox_changed)
        self.editor_a.pack(fill="both", expand=True, padx=4, pady=(2, 4))

        # -- 反面（右） --
        frame_b = ctk.CTkFrame(right)
        frame_b.pack(side="left", fill="both", expand=True, padx=(2, 2), pady=4)

        row_b = ctk.CTkFrame(frame_b)
        row_b.pack(fill="x", padx=4, pady=(4, 2))
        ctk.CTkLabel(row_b, text="反面", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(row_b, text="↻顺", width=30, command=lambda: self._adj_angle('b', -0.5)).pack(side="right", padx=1)
        self.lbl_angle_b = ctk.CTkLabel(row_b, text="0.0°", width=40, font=ctk.CTkFont(size=11))
        self.lbl_angle_b.pack(side="right", padx=2)
        ctk.CTkButton(row_b, text="↺逆", width=30, command=lambda: self._adj_angle('b', +0.5)).pack(side="right", padx=1)

        self.editor_b = BBoxEditor(frame_b, on_change=self._on_bbox_changed)
        self.editor_b.pack(fill="both", expand=True, padx=4, pady=(2, 4))

        # 状态栏
        self.status = ctk.CTkLabel(self, text="就绪 — 选择文件夹并点击「AI 处理」", anchor="w")
        self.status.pack(fill="x", padx=12, pady=(0, 6))

        # 键盘
        self.bind("<Right>", lambda e: self._next_pair())
        self.bind("<Left>", lambda e: self._prev_pair())
        self.bind("<e>", lambda e: self._export_single()); self.bind("<E>", lambda e: self._export_single())
        self.bind("<s>", lambda e: self._export_single()); self.bind("<S>", lambda e: self._export_single())
        self.bind("<f>", lambda e: self._fit_editors()); self.bind("<F>", lambda e: self._fit_editors())
        self.bind("<x>", lambda e: self._swap_fb()); self.bind("<X>", lambda e: self._swap_fb())
        self.bind("<r>", lambda e: self._reset_rotation()); self.bind("<R>", lambda e: self._reset_rotation())
        # 句号逗号微调角度
        self.bind("<comma>", lambda e: self._adj_angle('a', -0.5))
        self.bind("<period>", lambda e: self._adj_angle('a', +0.5))
        self.bind("<comma>", lambda e: self._adj_angle('b', -0.5), add=True)
        self.bind("<period>", lambda e: self._adj_angle('b', +0.5), add=True)

    # ── 角度控制 ──────────────────────────────────────────────

    def _adj_angle(self, which, delta):
        if not self.pairs: return
        if which == 'a':
            self.angle_a = max(-45, min(45, round((self.angle_a + delta) * 2) / 2))
            self.editor_a.angle = self.angle_a
            self.editor_a._redraw()
            self.lbl_angle_a.configure(text=f"{self.angle_a:+.1f}°")
        else:
            self.angle_b = max(-45, min(45, round((self.angle_b + delta) * 2) / 2))
            self.editor_b.angle = self.angle_b
            self.editor_b._redraw()
            self.lbl_angle_b.configure(text=f"{self.angle_b:+.1f}°")
        self._update_preview()
        self._auto_save_debounce()

    def _reset_rotation(self):
        self.angle_a = 0.0; self.angle_b = 0.0
        self.editor_a.angle = 0.0; self.editor_b.angle = 0.0
        self.lbl_angle_a.configure(text="0.0°"); self.lbl_angle_b.configure(text="0.0°")
        self.editor_a._redraw(); self.editor_b._redraw()
        self._update_preview()
        self._auto_save_debounce()

    # ── 流式处理 ──────────────────────────────────────────────

    def _pick_dir(self):
        from tkinter import filedialog
        path = filedialog.askdirectory(title="选择原图文件夹")
        if path:
            self.entry_dir.delete(0, "end")
            self.entry_dir.insert(0, path)

    def _start_process(self):
        d = self.entry_dir.get().strip()
        if not d or not Path(d).is_dir():
            self.status.configure(text="请先选择有效的文件夹")
            return

        self.input_dir = Path(d)
        files = sorted(
            [f for f in self.input_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS],
            key=lambda f: f.name)
        self.pairs = [(files[i], files[i + 1]) for i in range(0, len(files) - 1, 2)]

        if not self.pairs:
            self.status.configure(text="未找到可配对的图片")
            return

        ann_path = self.input_dir / "annotations.json"
        if ann_path.exists():
            data = json.loads(ann_path.read_text("utf-8"))
            for a in data.get("annotations", []):
                self.annotations[a["file"]] = a

        self._proc_total = len(self.pairs)
        self._proc_done = 0
        self._results = [None] * self._proc_total
        self.pair_idx = 0
        self._first_loaded = False

        self.btn_prev.configure(state="disabled")
        self.btn_next.configure(state="disabled")
        self.lbl_idx.configure(text="0 / 0")
        self.status.configure(text=f"AI+CV 处理中... 0/{self._proc_total}")

        def worker():
            for i, (pa, pb) in enumerate(self.pairs):
                try:
                    img_a = ImageOps.exif_transpose(Image.open(pa)).convert("RGB")
                    img_b = ImageOps.exif_transpose(Image.open(pb)).convert("RGB")
                    bb_a, bb_b = self.processor._joint_detect(img_a, img_b)
                    self._results[i] = (bb_a, bb_b)
                except Exception:
                    self._results[i] = (None, None)
                self.after(0, lambda idx=i: self._on_one_done(idx))
            self.after(0, self._on_all_done)
        threading.Thread(target=worker, daemon=True).start()

    def _on_one_done(self, idx):
        self._proc_done = idx + 1
        remain = self._proc_total - self._proc_done
        self.status.configure(
            text=f"AI+CV: {self._proc_done}/{self._proc_total} 已就绪" +
            (f"，{remain} 处理中..." if remain > 0 else " — 全部完成"))
        if not self._first_loaded:
            self._first_loaded = True
            self.pair_idx = 0
            self._load_current_pair()
            self.lbl_idx.configure(text=f"1 / {self._proc_total}")
            if self.btn_debug:
                self.btn_debug.configure(state="normal")
        self._update_nav_buttons()

    def _on_all_done(self):
        self._proc_done = self._proc_total
        self.status.configure(text=f"全部完成 — {self._proc_total} 对已就绪")
        self._update_nav_buttons()

    def _update_nav_buttons(self):
        if self.pair_idx < self._proc_done - 1:
            self.btn_next.configure(state="normal")
        else:
            self.btn_next.configure(state="disabled")
        if self.pair_idx > 0:
            self.btn_prev.configure(state="normal")
        else:
            self.btn_prev.configure(state="disabled")

    # ── 导航 ──────────────────────────────────────────────────

    def _next_pair(self):
        if self.pair_idx < self._proc_done - 1:
            self.pair_idx += 1
            self._load_current_pair()
            self._update_nav_buttons()

    def _prev_pair(self):
        if self.pair_idx > 0:
            self.pair_idx -= 1
            self._load_current_pair()
            self._update_nav_buttons()

    # ── 加载 ──────────────────────────────────────────────────

    def _load_current_pair(self):
        if not self.pairs or self.pair_idx >= self._proc_done:
            return

        pa, pb = self.pairs[self.pair_idx]
        self.lbl_idx.configure(text=f"{self.pair_idx + 1} / {self._proc_total}")
        self.lbl_fname.configure(text=f"{pa.stem}  +  {pb.stem}")

        self.img_a = ImageOps.exif_transpose(Image.open(pa)).convert("RGB")
        self.img_b = ImageOps.exif_transpose(Image.open(pb)).convert("RGB")

        res = self._results[self.pair_idx]
        self.ai_bbox_a, self.ai_bbox_b = res if res else (None, None)

        ann_a = self.annotations.get(pa.name, {})
        ann_b = self.annotations.get(pb.name, {})

        if "bbox" in ann_a:
            self.bbox_a = list(ann_a["bbox"]); self.angle_a = ann_a.get("angle", 0.0)
        elif self.ai_bbox_a:
            self.bbox_a = list(self.ai_bbox_a); self.angle_a = 0.0
        else:
            self.bbox_a = [self.img_a.width//4, self.img_a.height//4,
                           self.img_a.width*3//4, self.img_a.height*3//4]
            self.angle_a = 0.0

        if "bbox" in ann_b:
            self.bbox_b = list(ann_b["bbox"]); self.angle_b = ann_b.get("angle", 0.0)
        elif self.ai_bbox_b:
            self.bbox_b = list(self.ai_bbox_b); self.angle_b = 0.0
        else:
            self.bbox_b = [self.img_b.width//4, self.img_b.height//4,
                           self.img_b.width*3//4, self.img_b.height*3//4]
            self.angle_b = 0.0

        self.editor_a.set_image(self.img_a, self.bbox_a, self.ai_bbox_a, self.angle_a)
        self.editor_b.set_image(self.img_b, self.bbox_b, self.ai_bbox_b, self.angle_b)
        self.lbl_angle_a.configure(text=f"{self.angle_a:+.1f}°")
        self.lbl_angle_b.configure(text=f"{self.angle_b:+.1f}°")
        self._update_preview()
        self.after(100, self._fit_editors)

    # ── 预览 ──────────────────────────────────────────────────

    @staticmethod
    def _natural_w(bbox):
        bw = bbox[2] - bbox[0]; bh = bbox[3] - bbox[1]
        cw = int(bw * (1 + MARGIN)); ch = cw * 2
        if ch < bh * (1 + MARGIN):
            ch = int(bh * (1 + MARGIN)); ch += ch % 2; cw = ch // 2
        return cw

    @staticmethod
    def _simple_crop(img, bbox, anchor, crop_w, angle=0.0):
        if angle != 0.0:
            cx = (bbox[0] + bbox[2]) / 2; cy = (bbox[1] + bbox[3]) / 2
            img = img.rotate(angle, Image.BICUBIC, center=(cx, cy),
                             expand=False, fillcolor=(255, 255, 255))
        w, h = img.size; x1, y1, x2, y2 = bbox
        bcy = (y1 + y2) / 2; bw = x2 - x1
        crop_h = crop_w * 2
        if anchor == "right":
            right = min(w, int(x2 + bw * MARGIN)); cx = right - crop_w
        else:
            left = max(0, int(x1 - bw * MARGIN)); cx = left
        if cx < 0: cx = 0
        if cx + crop_w > w: cx = w - crop_w
        cy = int(bcy - crop_h / 2)
        if cy < 0: cy = 0
        if cy + crop_h > h: cy = h - crop_h
        return img.crop((cx, cy, cx + crop_w, cy + crop_h))

    def _update_preview(self):
        if not self.img_a or not self.img_b: return
        unified_cw = max(self._natural_w(self.bbox_a), self._natural_w(self.bbox_b))
        crop_a = self._simple_crop(self.img_a, self.bbox_a, "right", unified_cw, self.angle_a)
        crop_b = self._simple_crop(self.img_b, self.bbox_b, "left", unified_cw, self.angle_b)
        if crop_a.width != crop_b.width:
            tw = max(crop_a.width, crop_b.width); th = tw * 2
            for crp, is_a in [(crop_a, True), (crop_b, False)]:
                if crp.width != tw:
                    tmp = Image.new("RGB", (tw, th), (255, 255, 255))
                    tmp.paste(crp, ((tw - crp.width)//2, (th - crp.height)//2))
                    if is_a: crop_a = tmp
                    else:    crop_b = tmp
        th = min(crop_a.height, crop_b.height); th += th % 2; hw = th // 2
        left = crop_a.resize((hw, th), Image.LANCZOS)
        right = crop_b.resize((hw, th), Image.LANCZOS)
        preview = Image.new("RGB", (th, th), (255, 255, 255))
        preview.paste(left, (0, 0)); preview.paste(right, (hw, 0))

        c = self.preview_canvas
        cw_canvas = max(c.winfo_width(), 100); ch_canvas = max(c.winfo_height(), 100)
        display_size = min(cw_canvas, ch_canvas) - 20
        ds = max(display_size, 100)
        if th != ds:
            preview = preview.resize((ds, ds), Image.LANCZOS)

        self._preview_img = ImageTk.PhotoImage(preview)
        c.delete("all")
        px = (cw_canvas - ds) // 2; py = (ch_canvas - ds) // 2
        c.create_image(px, py, anchor=tk.NW, image=self._preview_img)

        # 半透明灰色虚线网格（stipple 实现半透明效果）
        gray = "#666666"; ds_sub = (3, 15)
        for frac in [0.25, 0.5, 0.75]:
            ly = py + int(ds * frac)
            c.create_line(px, ly, px + ds, ly, fill=gray, width=1, dash=ds_sub, stipple="gray50")
        for frac in [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
            lx = px + int(ds * frac)
            c.create_line(lx, py, lx, py + ds, fill=gray, width=1, dash=ds_sub, stipple="gray50")

    def _on_bbox_changed(self):
        self.bbox_a = list(self.editor_a.bbox); self.bbox_b = list(self.editor_b.bbox)
        self._update_preview()
        self._auto_save_debounce()

    # ── 按钮操作 ──────────────────────────────────────────────

    def _auto_save_debounce(self):
        if hasattr(self, '_auto_save_id'): self.after_cancel(self._auto_save_id)
        self._auto_save_id = self.after(800, self._auto_save)

    def _liquify(self):
        """对当前拼接结果打开液化工具"""
        if not self.img_a or not self.img_b: return
        # 生成当前拼接预览（全分辨率）
        unified_cw = max(self._natural_w(self.bbox_a), self._natural_w(self.bbox_b))
        crop_a = self._simple_crop(self.img_a, self.bbox_a, "right", unified_cw, self.angle_a)
        crop_b = self._simple_crop(self.img_b, self.bbox_b, "left", unified_cw, self.angle_b)
        if crop_a.width != crop_b.width:
            tw = max(crop_a.width, crop_b.width); th = tw * 2
            for crp, is_a in [(crop_a, True), (crop_b, False)]:
                if crp.width != tw:
                    tmp = Image.new("RGB", (tw, th), (255, 255, 255))
                    tmp.paste(crp, ((tw - crp.width)//2, (th - crp.height)//2))
                    if is_a: crop_a = tmp
                    else:    crop_b = tmp
        th = min(crop_a.height, crop_b.height); th += th % 2; hw = th // 2
        stitched = Image.new("RGB", (th, th), (255, 255, 255))
        stitched.paste(crop_a.resize((hw, th), Image.LANCZOS), (0, 0))
        stitched.paste(crop_b.resize((hw, th), Image.LANCZOS), (hw, 0))

        pa = self.pairs[self.pair_idx][0]
        tool = LiquifyTool(stitched, f"液化 — {pa.stem}.png",
                           on_apply=lambda result: self._on_liquify_done(result, pa))
        self.wait_window(tool)

    def _on_liquify_done(self, result, pa):
        if result:
            out_dir = self.input_dir / "审核输出" if self.input_dir else Path("审核输出")
            out_dir.mkdir(parents=True, exist_ok=True)
            result.save(out_dir / f"{pa.stem}.png", "PNG")
            self.status.configure(text=f"液化已保存 {pa.stem}.png")

    def _fit_editors(self):
        self.editor_a.update_display(); self.editor_b.update_display()

    def _swap_fb(self):
        if not self.pairs: return
        self.img_a, self.img_b = self.img_b, self.img_a
        self.bbox_a, self.bbox_b = self.bbox_b, self.bbox_a
        self.angle_a, self.angle_b = self.angle_b, self.angle_a
        self.ai_bbox_a, self.ai_bbox_b = self.ai_bbox_b, self.ai_bbox_a
        pa, pb = self.pairs[self.pair_idx]; self.pairs[self.pair_idx] = (pb, pa)
        self.editor_a.set_image(self.img_a, self.bbox_a, self.ai_bbox_a, self.angle_a)
        self.editor_b.set_image(self.img_b, self.bbox_b, self.ai_bbox_b, self.angle_b)
        self.lbl_angle_a.configure(text=f"{self.angle_a:+.1f}°")
        self.lbl_angle_b.configure(text=f"{self.angle_b:+.1f}°")
        self._update_preview()
        self.lbl_fname.configure(text=f"{pb.stem}  +  {pa.stem}")
        self.status.configure(text="已互换正反面")
        self._auto_save()

    def _reset_ai(self):
        if self.ai_bbox_a:
            self.bbox_a = list(self.ai_bbox_a); self.angle_a = 0.0
            self.editor_a.bbox = list(self.ai_bbox_a); self.editor_a.angle = 0.0
            self.lbl_angle_a.configure(text="0.0°")
        if self.ai_bbox_b:
            self.bbox_b = list(self.ai_bbox_b); self.angle_b = 0.0
            self.editor_b.bbox = list(self.ai_bbox_b); self.editor_b.angle = 0.0
            self.lbl_angle_b.configure(text="0.0°")
        self._update_preview(); self.editor_a._redraw(); self.editor_b._redraw()
        self._auto_save()

    def _auto_save(self):
        if not self.pairs or not self.input_dir: return
        pa, pb = self.pairs[self.pair_idx]
        self.annotations[pa.name] = {"file": pa.name, "bbox": list(self.bbox_a), "angle": self.angle_a}
        self.annotations[pb.name] = {"file": pb.name, "bbox": list(self.bbox_b), "angle": self.angle_b}
        data = {"source_dir": str(self.input_dir), "annotations": list(self.annotations.values())}
        (self.input_dir / "annotations.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 导出 ──────────────────────────────────────────────────

    def _export_single(self):
        if not self.pairs: return
        pa, pb = self.pairs[self.pair_idx]
        out_dir = self.input_dir / "审核输出" if self.input_dir else Path("审核输出")
        out_dir.mkdir(parents=True, exist_ok=True)
        unified_cw = max(self._natural_w(self.bbox_a), self._natural_w(self.bbox_b))
        crop_a = self._simple_crop(self.img_a, self.bbox_a, "right", unified_cw, self.angle_a)
        crop_b = self._simple_crop(self.img_b, self.bbox_b, "left", unified_cw, self.angle_b)
        if crop_a.width != crop_b.width:
            tw = max(crop_a.width, crop_b.width); th = tw * 2
            for crp, is_a in [(crop_a, True), (crop_b, False)]:
                if crp.width != tw:
                    tmp = Image.new("RGB", (tw, th), (255, 255, 255))
                    tmp.paste(crp, ((tw - crp.width)//2, (th - crp.height)//2))
                    if is_a: crop_a = tmp
                    else:    crop_b = tmp
        th = min(crop_a.height, crop_b.height); th += th % 2; hw = th // 2
        result = Image.new("RGB", (th, th), (255, 255, 255))
        result.paste(crop_a.resize((hw, th), Image.LANCZOS), (0, 0))
        result.paste(crop_b.resize((hw, th), Image.LANCZOS), (hw, 0))
        result.save(out_dir / f"{pa.stem}.png", "PNG")
        self.status.configure(text=f"已导出 {pa.stem}.png")

    def _export_all(self):
        if not self.input_dir: return
        out_dir = self.input_dir / "审核输出"; out_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for i in range(self._proc_done):
            pa, pb = self.pairs[i]
            if i == self.pair_idx:
                ba, bb = list(self.bbox_a), list(self.bbox_b)
                aa, ab = self.angle_a, self.angle_b
            else:
                ann_a = self.annotations.get(pa.name, {}); ann_b = self.annotations.get(pb.name, {})
                ba = ann_a.get("bbox") if "bbox" in ann_a else (self._results[i][0] if self._results[i] else None)
                bb = ann_b.get("bbox") if "bbox" in ann_b else (self._results[i][1] if self._results[i] else None)
                aa = ann_a.get("angle", 0.0); ab = ann_b.get("angle", 0.0)
            if not ba or not bb: continue
            try:
                ia = ImageOps.exif_transpose(Image.open(pa)).convert("RGB")
                ib = ImageOps.exif_transpose(Image.open(pb)).convert("RGB")
                ucw = max(self._natural_w(ba), self._natural_w(bb))
                ca = self._simple_crop(ia, ba, "right", ucw, aa)
                cb = self._simple_crop(ib, bb, "left", ucw, ab)
                if ca.width != cb.width:
                    tw = max(ca.width, cb.width); th = tw * 2
                    for crp, is_a in [(ca, True), (cb, False)]:
                        if crp.width != tw:
                            tmp = Image.new("RGB", (tw, th), (255, 255, 255))
                            tmp.paste(crp, ((tw - crp.width)//2, (th - crp.height)//2))
                            if is_a: ca = tmp
                            else:    cb = tmp
                th = min(ca.height, cb.height); th += th % 2; hw = th // 2
                r = Image.new("RGB", (th, th), (255, 255, 255))
                r.paste(ca.resize((hw, th), Image.LANCZOS), (0, 0))
                r.paste(cb.resize((hw, th), Image.LANCZOS), (hw, 0))
                r.save(out_dir / f"{pa.stem}.png", "PNG")
                ok += 1
            except Exception:
                pass
        self.status.configure(text=f"全部导出完成: {ok}/{self._proc_done} → {out_dir}")

    def _start_debug(self):
        """对当前第一对运行调试检测并弹出可视化窗口。"""
        if not self.pairs or self._proc_done < 1:
            return
        pa, pb = self.pairs[0]
        try:
            ia = ImageOps.exif_transpose(Image.open(pa)).convert("RGB")
            ib = ImageOps.exif_transpose(Image.open(pb)).convert("RGB")
            _, _, debug_entries = self.processor._joint_detect_debug(ia, ib)
            DebugWindow(self, debug_entries, f"调试 — {pa.stem} + {pb.stem}")
        except Exception as e:
            self.status.configure(text=f"调试失败: {e}")


class DebugWindow(tk.Toplevel):
    """可滚动的调试图像展示窗口。"""

    def __init__(self, parent, entries, title="调试"):
        super().__init__(parent)
        self.title(title)
        self.geometry("760x800")
        self.configure(bg="#1E1E1E")

        info = tk.Label(self, text=f"共 {len(entries)} 步", bg="#1E1E1E", fg="#999",
                        font=("Microsoft YaHei UI", 10))
        info.pack(pady=(8, 4))

        canvas = tk.Canvas(self, bg="#1E1E1E", highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg="#1E1E1E")

        scroll_frame.bind("<Configure>",
                          lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        scrollbar.pack(side="right", fill="y", pady=8)

        self._photos = []  # 防止 GC 回收
        DISP_W = 500  # 统一显示宽度
        for label, img in entries:
            lbl = tk.Label(scroll_frame, text=label, bg="#1E1E1E", fg="#CCC",
                           font=("Microsoft YaHei UI", 11, "bold"))
            lbl.pack(pady=(12, 2))
            # 缩放到统一宽度
            w, h = img.size
            if w != DISP_W:
                img = img.resize((DISP_W, int(h * DISP_W / w)), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._photos.append(photo)
            img_lbl = tk.Label(scroll_frame, image=photo, bg="#1E1E1E")
            img_lbl.pack(pady=(0, 6))

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        # 滚轮滚动（绑定到 Toplevel，随窗口销毁自动清理）
        self.bind("<MouseWheel>", _on_mousewheel)


def main():
    ReviewerApp().mainloop()

if __name__ == "__main__":
    main()
