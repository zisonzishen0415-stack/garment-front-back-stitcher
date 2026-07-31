"""审核编辑工具 — v4
流程：选文件夹 → AI+CV 流式处理（完成一对立即可审）
左预览 + 右编辑 + 角度旋钮
"""
import sys, os, threading
# onedir 安装：模型在安装目录 _internal/models/ 下，设置 U2NET_HOME 让 rembg 直接读取
# 开发模式（python reviewer.py）：默认从 ~/.u2net/ 加载
_EXE_DIR = sys._MEIPASS if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
_BUNDLED_MODEL = os.path.join(_EXE_DIR, 'models', 'u2net.onnx')
if os.path.exists(_BUNDLED_MODEL):
    os.environ['U2NET_HOME'] = os.path.join(_EXE_DIR, 'models')

import json
import math
import time
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
    """bbox 编辑器：缩放/平移/角点+边中点拖拽 + 旋转"""
    _placeholder = None  # 类属性：空编辑器居中显示的品牌 logo

    @classmethod
    def set_placeholder(cls, img):
        cls._placeholder = img

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
        self._drag_angle = 0.0
        self._pan_data = None
        self._on_change = on_change
        self._last_fit_w = 0
        # 性能优化：帧率节流 + 延迟预览
        self._last_redraw = 0.0       # perf_counter for 16ms throttle
        self._defer_preview = False   # drag 期间推迟预览
        self._preview_deferred = False
        self._display_cache = None    # 下采样显示缓存
        self._display_cache_scale = 0.0
        self._display_cache_size = (0, 0)
        self._wheel_id = None         # after() id for wheel throttle
        self._pan_id = None           # after() id for pan throttle

        self.bind("<ButtonPress-1>", self._on_down)
        self.bind("<B1-Motion>", self._on_move)
        self.bind("<ButtonRelease-1>", self._on_up)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<ButtonPress-2>", self._on_pan_down)
        self.bind("<B2-Motion>", self._on_pan_move)
        self.bind("<ButtonPress-3>", self._on_pan_down)
        self.bind("<B3-Motion>", self._on_pan_move)
        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", lambda e: self._clear_hover())
        self.bind("<Configure>", self._on_configure)

    def set_image(self, pil_img, bbox, ai_bbox=None, angle=0.0):
        self.pil_img = pil_img
        self.bbox = list(bbox)
        self.angle = float(angle)
        self.ai_bbox = list(ai_bbox) if ai_bbox else None
        self._display_cache = None   # 清除显示缓存
        self._display_cache_scale = 0.0
        # _on_configure 或 _fit 决定何时布局；set_image 先跑一次
        self._last_fit_w = 0  # 追踪上次 fit 的画布尺寸，避免重复重绘
        self._fit()

    def _fit(self):
        cw = self.winfo_width()
        ch = self.winfo_height()
        if cw < 30 or ch < 30:
            return
        if not self.pil_img:
            self._redraw()  # 显示 placeholder logo
            return
        if abs(cw - self._last_fit_w) < 4 and abs(ch - getattr(self, '_last_fit_h', 0)) < 4:
            return
        self._last_fit_w = cw
        self._last_fit_h = ch
        iw, ih = self.pil_img.size
        if ih > iw:
            s = ch / ih  # 竖向图：撑满高度，上下触边
        else:
            s = min(cw / iw, ch / ih) * 0.85
        self.scale = max(0.02, s)
        self.ox = (cw - iw * self.scale) / 2
        self.oy = (ch - ih * self.scale) / 2
        self._redraw()

    def _on_configure(self, event):
        self._fit()

    def _to_canvas(self, ix, iy):
        return (ix * self.scale + self.ox, iy * self.scale + self.oy)

    def _to_image(self, cx, cy):
        return (int((cx - self.ox) / self.scale), int((cy - self.oy) / self.scale))

    PLACEHOLDER_W = 220  # 统一定宽，三区域尺寸一致

    def _redraw(self):
        """完整重绘：重建 bg + overlay。切图/缩放/平移时调用。"""
        self.delete("all")
        if not self.pil_img:
            ph = self._placeholder
            if ph is not None:
                cw = max(self.winfo_width(), 50)
                ch = max(self.winfo_height(), 50)
                pw, ph_h = ph.size
                dw = min(self.PLACEHOLDER_W, cw - 20)
                dh = int(dw * ph_h / pw)
                self._photo = ImageTk.PhotoImage(ph.resize((dw, dh), Image.LANCZOS))
                self.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self._photo)
            return

        self._update_bg()
        self._draw_overlay()

    def _update_bg(self):
        """仅更新背景图 PhotoImage。旋转拖拽时调用此方法 + _draw_overlay。"""
        iw, ih = self.pil_img.size
        dw, dh = max(1, int(iw * self.scale)), max(1, int(ih * self.scale))
        cache_key = (self.scale, self.angle, dw, dh)
        if (self._display_cache is None
                or getattr(self, '_cache_key', None) != cache_key):
            if abs(self.angle) > 0.005:
                base = self.pil_img.resize((dw, dh), Image.BILINEAR)
                self._display_cache = base.rotate(self.angle, Image.BICUBIC,
                                                  expand=False, fillcolor=(30, 30, 30))
            else:
                self._display_cache = self.pil_img.resize((dw, dh), Image.BILINEAR)
            self._cache_key = cache_key
        self._photo = ImageTk.PhotoImage(self._display_cache)
        self.create_image(self.ox, self.oy, anchor=tk.NW, image=self._photo, tags="bg")

    def _redraw_rotation_frame(self):
        """旋转拖拽帧更新：替换背景图 + 重绘叠加层（不 delete all）。"""
        self.delete("bg", "overlay")
        self._update_bg()
        self._draw_overlay()

    def _draw_overlay(self):
        """轴对齐绿框 + 控制点（背景图已旋转，绿框不再旋转）。"""
        self.delete("overlay")

        if self.ai_bbox:
            self._draw_rect(self.ai_bbox, AI_COLOR, 1, (4, 8))

        x1, y1, x2, y2 = self.bbox
        cx1, cy1 = self._to_canvas(x1, y1)
        cx2, cy2 = self._to_canvas(x2, y2)

        # AI bbox（灰虚线）
        if self.ai_bbox:
            ax1, ay1, ax2, ay2 = self.ai_bbox
            acx1, acy1 = self._to_canvas(ax1, ay1)
            acx2, acy2 = self._to_canvas(ax2, ay2)
            self.create_rectangle(acx1, acy1, acx2, acy2,
                                  outline=AI_COLOR, width=1, dash=(4, 8), tags="overlay")

        # 当前 bbox（绿实线）
        self.create_rectangle(cx1, cy1, cx2, cy2,
                              outline=BOX_COLOR, width=2, tags="overlay")

        # 四角控制点
        hs = HANDLE_SIZE
        self._corner_canvas_pos = [(cx1, cy1), (cx2, cy1), (cx2, cy2), (cx1, cy2)]
        for ccx, ccy in self._corner_canvas_pos:
            self.create_rectangle(ccx - hs, ccy - hs, ccx + hs, ccy + hs,
                                  fill=HANDLE_COLOR, outline="white", width=1, tags="overlay")

        # 四边中点
        mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2
        edge_positions = [(mx, cy1), (cx2, my), (mx, cy2), (cx1, my)]
        for ccx, ccy in edge_positions:
            self.create_rectangle(ccx - hs, ccy - hs, ccx + hs, ccy + hs,
                                  fill="#FFFFFF", outline=BOX_COLOR, width=1, tags="overlay")

    def _draw_rect(self, b, color, width, dash=()):
        """画轴对齐矩形。"""
        x1, y1, x2, y2 = b
        cx1, cy1 = self._to_canvas(x1, y1)
        cx2, cy2 = self._to_canvas(x2, y2)
        self.create_rectangle(cx1, cy1, cx2, cy2,
                              outline=color, width=width, dash=dash, tags="overlay")

    # ── 命中检测 ──────────────────────────────────────────────

    def _hit_corner(self, cx, cy):
        for ccx, ccy in self._corner_canvas_pos:
            if abs(cx - ccx) <= HANDLE_SIZE + 5 and abs(cy - ccy) <= HANDLE_SIZE + 5:
                # Return tag for the corner — needed for resize direction
                # But with axis-aligned bbox, tags are ["nw","ne","se","sw"] for corners
                return True
        return False

    def _corner_name(self, cx, cy):
        """返回最近的角点名（用于 resize 方向）。"""
        names = ["nw", "ne", "se", "sw"]
        best, best_d = None, HANDLE_SIZE + 8
        for name, (ccx, ccy) in zip(names, self._corner_canvas_pos):
            d = ((cx - ccx)**2 + (cy - ccy)**2)**0.5
            if d < best_d:
                best, best_d = name, d
        return best

    def _hit_rotation_zone(self, cx, cy):
        """框外靠近角点 = 旋转区域。"""
        if not hasattr(self, '_corner_canvas_pos'):
            return False
        x1, y1, x2, y2 = self.bbox
        cx1, cy1 = self._to_canvas(x1, y1)
        cx2, cy2 = self._to_canvas(x2, y2)
        # 框内 → 不是旋转
        if cx1 - 2 <= cx <= cx2 + 2 and cy1 - 2 <= cy <= cy2 + 2:
            return False
        # 在角点 handle 上 → resize，不是旋转
        for ccx, ccy in self._corner_canvas_pos:
            if abs(cx - ccx) <= HANDLE_SIZE + 5 and abs(cy - ccy) <= HANDLE_SIZE + 5:
                return False
        # 靠近任意角点外侧 → 旋转
        for ccx, ccy in self._corner_canvas_pos:
            if ((cx - ccx)**2 + (cy - ccy)**2)**0.5 <= 50:
                return True
        return False

    def _hit_edge(self, cx, cy):
        """检测鼠标是否在边中点附近（轴对齐）。"""
        x1, y1, x2, y2 = self.bbox
        cx1, cy1 = self._to_canvas(x1, y1)
        cx2, cy2 = self._to_canvas(x2, y2)
        mx, my = (cx1 + cx2) / 2, (cy1 + cy2) / 2
        edges = [("n", mx, cy1), ("s", mx, cy2), ("w", cx1, my), ("e", cx2, my)]
        for name, ex, ey in edges:
            if abs(cx - ex) <= HANDLE_SIZE + 5 and abs(cy - ey) <= HANDLE_SIZE + 5:
                return name
        return None

    def _poly_contains(self, cx, cy):
        """轴对齐矩形命中检测。"""
        x1, y1, x2, y2 = self.bbox
        cx1, cy1 = self._to_canvas(x1, y1)
        cx2, cy2 = self._to_canvas(x2, y2)
        return cx1 <= cx <= cx2 and cy1 <= cy <= cy2

    def _on_motion(self, event):
        """鼠标移动：检测旋转区域并切换光标。"""
        if self._drag:
            return
        if self._hit_rotation_zone(event.x, event.y):
            self.configure(cursor="exchange")
        else:
            self.configure(cursor="")

    def _clear_hover(self):
        self.configure(cursor="")
        self.winfo_toplevel().configure(cursor="")

    # ── 鼠标事件 ──────────────────────────────────────────────

    def _on_down(self, event):
        # 清理上一次残留的拖拽状态（鼠标可能在 Canvas 外松开导致 _on_up 丢失）
        if self._defer_preview:
            self._defer_preview = False
            if self._preview_deferred and self._on_change:
                self._preview_deferred = False
                self._on_change()
        self._drag = None
        self._drag_box = None
        self._preview_deferred = False

        # 旋转区域
        if self._hit_rotation_zone(event.x, event.y):
            self._drag = "rotate"
            self._drag_sx, self._drag_sy = event.x, event.y
            self._drag_angle = self.angle
            self._defer_preview = True
            self.winfo_toplevel().configure(cursor="exchange")
            return
        h = self._hit_corner(event.x, event.y)
        if h:
            self._drag = self._corner_name(event.x, event.y)
            self._drag_sx, self._drag_sy = event.x, event.y
            self._drag_box = list(self.bbox)
            self._defer_preview = True
            return
        e = self._hit_edge(event.x, event.y)
        if e:
            self._drag = e
            self._drag_sx, self._drag_sy = event.x, event.y
            self._drag_box = list(self.bbox)
            self._defer_preview = True
            return
        if self._poly_contains(event.x, event.y):
            self._drag = "move"
            self._drag_sx, self._drag_sy = event.x, event.y
            self._drag_box = list(self.bbox)
            self._defer_preview = True

    def _on_move(self, event):
        if not self._drag: return

        # 旋转：绕 bbox 中心旋转 → 重绘全画面（图片旋转 + 绿框）
        if self._drag == "rotate":
            cx = (self.bbox[0] + self.bbox[2]) / 2
            cy = (self.bbox[1] + self.bbox[3]) / 2
            ccx, ccy = self._to_canvas(cx, cy)
            angle_start = math.atan2(self._drag_sy - ccy, self._drag_sx - ccx)
            angle_now = math.atan2(event.y - ccy, event.x - ccx)
            delta = math.degrees(angle_now - angle_start)
            if event.state & 0x1:
                delta = round(delta / 15) * 15
            raw = self._drag_angle + delta
            self.angle = max(-45, min(45, round(raw * 2) / 2))
            now = time.perf_counter()
            if now - self._last_redraw >= 0.016:
                self._redraw_rotation_frame()
                self._last_redraw = now
            if self._on_change:
                self._preview_deferred = True
            return

        if self._drag_box is None: return
        dx = (event.x - self._drag_sx) / self.scale
        dy = (event.y - self._drag_sy) / self.scale

        iw, ih = self.pil_img.width, self.pil_img.height
        x1, y1, x2, y2 = self._drag_box
        moves = {
            "move": lambda: (int(max(0, x1+dx)), int(max(0, y1+dy)),
                             int(min(iw, x1+dx+(x2-x1))), int(min(ih, y1+dy+(y2-y1))))
                     if x1+dx+(x2-x1) <= iw and y1+dy+(y2-y1) <= ih else None,
            "nw": lambda: (int(min(max(0, x1+dx), x2-10)), int(min(max(0, y1+dy), y2-10)), x2, y2),
            "ne": lambda: (x1, int(min(max(0, y1+dy), y2-10)), int(max(x1+10, min(iw, x2+dx))), y2),
            "sw": lambda: (int(min(max(0, x1+dx), x2-10)), y1, x2, int(max(y1+10, min(ih, y2+dy)))),
            "se": lambda: (x1, y1, int(max(x1+10, min(iw, x2+dx))), int(max(y1+10, min(ih, y2+dy)))),
            "n":  lambda: (x1, int(min(max(0, y1+dy), y2-10)), x2, y2),
            "s":  lambda: (x1, y1, x2, int(max(y1+10, min(ih, y2+dy)))),
            "w":  lambda: (int(min(max(0, x1+dx), x2-10)), y1, x2, y2),
            "e":  lambda: (x1, y1, int(max(x1+10, min(iw, x2+dx))), y2),
        }
        fn = moves.get(self._drag)
        if fn:
            new = fn()
            if new and len(new) == 4:
                self.bbox = list(new)
                # 拖拽期间只重绘叠加层（绿框 + 控制点），不重建背景图
                now = time.perf_counter()
                if now - self._last_redraw >= 0.016:
                    self._draw_overlay()
                    self._last_redraw = now
                if self._on_change:
                    self._preview_deferred = True

    def _on_up(self, event):
        was_dragging = self._drag is not None
        was_rotating = self._drag == "rotate"
        self._drag = None
        self._drag_box = None
        if was_rotating:
            self.winfo_toplevel().configure(cursor="")
        if self._defer_preview:
            self._defer_preview = False
            if was_dragging:
                # 松手后做一次完整重绘（确保背景图位置正确）
                self._redraw()
            if self._preview_deferred and self._on_change:
                self._preview_deferred = False
                self._on_change()

    def _on_wheel(self, event):
        if not self.pil_img: return
        mx, my = event.x, event.y
        old_ix, old_iy = self._to_image(mx, my)
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        self.scale = max(0.02, min(5.0, self.scale * factor))
        self.ox = mx - old_ix * self.scale
        self.oy = my - old_iy * self.scale
        # 16ms 帧率节流，取消之前的延迟重绘
        if self._wheel_id:
            self.after_cancel(self._wheel_id)
        self._wheel_id = self.after(16, self._redraw)

    def _on_pan_down(self, event):
        self._pan_data = (self.ox, self.oy, event.x, event.y)

    def _on_pan_move(self, event):
        if self._pan_data:
            ox0, oy0, mx0, my0 = self._pan_data
            self.ox = ox0 + (event.x - mx0)
            self.oy = oy0 + (event.y - my0)
            # 16ms 帧率节流
            if self._pan_id:
                self.after_cancel(self._pan_id)
            self._pan_id = self.after(16, self._redraw)

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
        self.minsize(900, 600)  # 防止窗口过小导致布局崩溃
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
        self._mask_a = None
        self._mask_b = None
        self.annotations: dict[str, dict] = {}
        self.input_dir: Optional[Path] = None
        self._preview_zoom = 1.0
        self._liquified: Optional[Image.Image] = None  # 液化修改后的结果
        self._preview_placeholder = None  # 预览水印
        self._load_seq = 0              # 异步加载序列号，防止竞态
        self._spinner_id = None         # 加载旋转动画 canvas id
        self._spinner_after = None      # spinner after() id

        # 流式处理
        self._proc_done = 0
        self._proc_total = 0
        self._results: list = []
        self._first_loaded = False

        self.btn_debug = None

        # Logo（窗口图标 + 标题栏）
        LOGO_DIR = Path(__file__).parent
        ico = LOGO_DIR / "logo.ico"
        if ico.exists():
            try:
                self.iconbitmap(str(ico))
            except Exception:
                pass
        self._logo_img = None
        self._logo_photo = None
        logo_png = LOGO_DIR / "logo_toolbar.png"
        if logo_png.exists():
            self._logo_photo = ImageTk.PhotoImage(Image.open(str(logo_png)))
            self._logo_img = self._logo_photo  # 保持引用防止 GC

        # 编辑器空状态 placeholder：极暗深灰水印（若隐若现高级感）
        logo_placeholder_png = LOGO_DIR / "logo_placeholder.png"
        if logo_placeholder_png.exists():
            ph = Image.open(str(logo_placeholder_png)).convert("RGBA")
            BBoxEditor.set_placeholder(ph)
            self._preview_placeholder = ph
        else:
            self._preview_placeholder = None

        self._build_ui()
        self.update_idletasks()
        if self._preview_placeholder:
            self._draw_preview_placeholder()
        self._t0 = time.time()
        self._status_loader.configure(text="模型加载中...")
        self.processor.prewarm()
        self.after(200, self._check_prewarm)

    def _check_prewarm(self):
        try:
            if getattr(self.processor, '_warmed', False):
                elapsed = time.time() - self._t0
                self._status_loader.configure(text=f"模型就绪 ({elapsed:.1f}s)")
                return
            # 显示加载进度
            elapsed = time.time() - self._t0
            self._status_loader.configure(text=f"模型加载中 ({elapsed:.1f}s)")
            self.after(200, self._check_prewarm)
        except Exception:
            self.after(200, self._check_prewarm)

    # ── UI ────────────────────────────────────────────────────

    def _build_ui(self):
        bar = ctk.CTkFrame(self)
        bar.pack(fill="x", padx=6, pady=(6, 2))

        ctk.CTkLabel(bar, text="文件夹:").pack(side="left", padx=(4, 2))
        self.entry_dir = ctk.CTkEntry(bar, width=200)
        self.entry_dir.pack(side="left", padx=2)
        ctk.CTkButton(bar, text="浏览", width=45, command=self._pick_dir).pack(side="left", padx=1)
        ctk.CTkButton(bar, text="AI 处理", width=60, command=lambda: self._start_process()).pack(side="left", padx=(6, 1))
        self.btn_debug = ctk.CTkButton(bar, text="调试", width=40,
                                        command=self._start_debug, state="disabled")
        self.btn_debug.pack(side="left", padx=1)

        ctk.CTkFrame(bar, width=1, height=20, fg_color="#555").pack(side="left", padx=6)

        self.btn_prev = ctk.CTkButton(bar, text="◀", width=30, command=self._prev_pair, state="disabled")
        self.btn_prev.pack(side="left", padx=1)
        self.lbl_idx = ctk.CTkLabel(bar, text="0 / 0", font=ctk.CTkFont(size=13, weight="bold"), width=55)
        self.lbl_idx.pack(side="left", padx=1)
        self.btn_next = ctk.CTkButton(bar, text="▶", width=30, command=self._next_pair, state="disabled")
        self.btn_next.pack(side="left", padx=1)

        ctk.CTkFrame(bar, width=1, height=20, fg_color="#555").pack(side="left", padx=6)

        self._angle_mode = "theilsen"
        self._btn_angle_mode = ctk.CTkButton(bar, text="TheilSen", width=65,
                                             command=self._toggle_angle_mode)
        self._btn_angle_mode.pack(side="left", padx=1)
        ctk.CTkButton(bar, text="互换", width=45, command=self._swap_fb).pack(side="left", padx=1)
        ctk.CTkButton(bar, text="重置AI", width=55, command=self._reset_ai).pack(side="left", padx=1)
        ctk.CTkButton(bar, text="液化", width=40, fg_color="#6B3FA0", hover_color="#56338A",
                       command=self._liquify).pack(side="left", padx=1)

        ctk.CTkFrame(bar, width=1, height=20, fg_color="#555").pack(side="left", padx=6)

        ctk.CTkButton(bar, text="导出", width=45, fg_color="#2B8C3C", hover_color="#236E30",
                       command=self._export_single).pack(side="left", padx=1)
        ctk.CTkButton(bar, text="批量", width=45, command=self._export_all).pack(side="left", padx=1)

        # Logo（右侧，点击显示关于）
        if self._logo_img:
            self._lbl_logo = tk.Label(bar, image=self._logo_img, bg=ctk.ThemeManager.theme["CTkFrame"]["fg_color"][1],
                                      cursor="hand2")
            self._lbl_logo.pack(side="right", padx=(4, 2))
            self._lbl_logo.bind("<Button-1>", self._show_about)
        else:
            self._lbl_logo = tk.Label(bar, text="关于", fg="#999",
                                      bg=ctk.ThemeManager.theme["CTkFrame"]["fg_color"][1],
                                      font=ctk.CTkFont(size=10), cursor="hand2")
            self._lbl_logo.pack(side="right", padx=(4, 2))
            self._lbl_logo.bind("<Button-1>", self._show_about)

        self.lbl_fname = ctk.CTkLabel(bar, text="", font=ctk.CTkFont(size=9), text_color="#999")
        self.lbl_fname.pack(side="right", padx=8)

        # 主体 — grid 布局比 pack 更可控：左列固定预览，右列弹性编辑器
        main = ctk.CTkFrame(self)
        main.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        main.grid_columnconfigure(0, weight=0)   # 预览列：固定宽度
        main.grid_columnconfigure(1, weight=1)   # 编辑列：弹性扩展
        main.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(main, width=500, height=500)
        left.grid(row=0, column=0, sticky="ns", padx=(4, 2), pady=4)
        left.pack_propagate(False)
        ctk.CTkLabel(left, text="拼接预览", font=ctk.CTkFont(weight="bold")).pack(pady=(6, 2))
        self.preview_canvas = tk.Canvas(left, bg="#1E1E1E", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.preview_canvas.bind("<MouseWheel>", self._on_preview_wheel)
        self.preview_canvas.bind("<Enter>", lambda e: self.preview_canvas.focus_set())
        self.preview_canvas.bind("<Configure>", self._on_preview_configure)
        # 预览 1:1 正方形 = 容器高度（原始设计），窄屏时压缩以保编辑器 ≥420px
        main.bind("<Configure>", lambda e: left.configure(
            width=max(200, min(e.height - 8, e.width - 420))))

        # 右：编辑 — grid 等分正面/反面（窄窗时两侧等比压缩）
        right = ctk.CTkFrame(main)
        right.grid(row=0, column=1, sticky="nsew", padx=(2, 4), pady=4)
        right.grid_columnconfigure(0, weight=1)  # 正面列
        right.grid_columnconfigure(1, weight=1)  # 反面列
        right.grid_rowconfigure(0, weight=1)

        # -- 正面（左） --
        frame_a = ctk.CTkFrame(right)
        frame_a.grid(row=0, column=0, sticky="nsew", padx=(2, 2), pady=4)

        row_a = ctk.CTkFrame(frame_a)
        row_a.pack(fill="x", padx=4, pady=(4, 2))
        ctk.CTkLabel(row_a, text="正面", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(row_a, text="↻顺", width=30, command=lambda: self._adj_angle('a', +0.5)).pack(side="right", padx=1)
        self.lbl_angle_a = ctk.CTkLabel(row_a, text="0.0°", width=40, font=ctk.CTkFont(size=11))
        self.lbl_angle_a.pack(side="right", padx=2)
        ctk.CTkButton(row_a, text="↺逆", width=30, command=lambda: self._adj_angle('a', -0.5)).pack(side="right", padx=1)
        self.btn_reset_angle_a = ctk.CTkButton(row_a, text="⟳", width=25,
                                               command=lambda: self._reset_angle('a'))
        self.btn_reset_angle_a.pack(side="right", padx=1)

        self.editor_a = BBoxEditor(frame_a, on_change=self._on_bbox_changed)
        self.editor_a.pack(fill="both", expand=True, padx=4, pady=(2, 4))

        # -- 反面（右） --
        frame_b = ctk.CTkFrame(right)
        frame_b.grid(row=0, column=1, sticky="nsew", padx=(2, 2), pady=4)

        row_b = ctk.CTkFrame(frame_b)
        row_b.pack(fill="x", padx=4, pady=(4, 2))
        ctk.CTkLabel(row_b, text="反面", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(row_b, text="↻顺", width=30, command=lambda: self._adj_angle('b', +0.5)).pack(side="right", padx=1)
        self.lbl_angle_b = ctk.CTkLabel(row_b, text="0.0°", width=40, font=ctk.CTkFont(size=11))
        self.lbl_angle_b.pack(side="right", padx=2)
        ctk.CTkButton(row_b, text="↺逆", width=30, command=lambda: self._adj_angle('b', -0.5)).pack(side="right", padx=1)
        self.btn_reset_angle_b = ctk.CTkButton(row_b, text="⟳", width=25,
                                               command=lambda: self._reset_angle('b'))
        self.btn_reset_angle_b.pack(side="right", padx=1)

        self.editor_b = BBoxEditor(frame_b, on_change=self._on_bbox_changed)
        self.editor_b.pack(fill="both", expand=True, padx=4, pady=(2, 4))

        # 状态栏（左侧状态文字 + 右侧加载提示）
        status_frame = ctk.CTkFrame(self, fg_color="transparent")
        status_frame.pack(fill="x", padx=12, pady=(0, 6))
        self.status = ctk.CTkLabel(status_frame, text="就绪 — 选择文件夹并点击「AI 处理」", anchor="w")
        self.status.pack(side="left")
        self._status_loader = ctk.CTkLabel(status_frame, text="", text_color="#FFA500",
                                            font=ctk.CTkFont(size=11))
        self._status_loader.pack(side="right", padx=(8, 0))

        # 键盘
        self.bind("<Right>", lambda e: self._next_pair())
        self.bind("<Left>", lambda e: self._prev_pair())
        self.bind("<e>", lambda e: self._export_single()); self.bind("<E>", lambda e: self._export_single())
        self.bind("<s>", lambda e: self._export_single()); self.bind("<S>", lambda e: self._export_single())
        self.bind("<f>", lambda e: self._fit_editors()); self.bind("<F>", lambda e: self._fit_editors())
        self.bind("<x>", lambda e: self._swap_fb()); self.bind("<X>", lambda e: self._swap_fb())
        self.bind("<r>", lambda e: self._reset_rotation()); self.bind("<R>", lambda e: self._reset_rotation())
        self.bind("<F1>", lambda e: self._show_about())
        # 句号逗号微调角度
        self.bind("<comma>", lambda e: self._adj_angle('a', -0.5))
        self.bind("<comma>", lambda e: self._adj_angle('b', -0.5), add=True)
        self.bind("<period>", lambda e: self._adj_angle('a', +0.5))
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
        self._liquified = None  # 角度变了，液化结果作废
        self._update_preview()
        self._auto_save_debounce()

    def _reset_rotation(self):
        self.angle_a = 0.0; self.angle_b = 0.0
        self.editor_a.angle = 0.0; self.editor_b.angle = 0.0
        self.lbl_angle_a.configure(text="0.0°"); self.lbl_angle_b.configure(text="0.0°")
        self.editor_a._redraw(); self.editor_b._redraw()
        self._liquified = None
        self._update_preview()
        self._auto_save_debounce()

    def _toggle_angle_mode(self):
        """切换倾斜修正算法：Theil-Sen ↔ Template。"""
        if self._angle_mode == "theilsen":
            self._angle_mode = "template"
            self._btn_angle_mode.configure(text="Template", fg_color="#6B3FA0")
        else:
            self._angle_mode = "theilsen"
            self._btn_angle_mode.configure(text="TheilSen", fg_color="#555")
        self._recalc_angle()

    def _recalc_angle(self):
        """根据当前模式调用对应角度算法。"""
        if self._mask_a is None or self._mask_b is None:
            return
        if self._angle_mode == "theilsen":
            angle_a, angle_b = self.processor.tilt_theil_sen(
                self._mask_a, self._mask_b, self.bbox_a, self.bbox_b)
        else:
            angle_a, angle_b = self.processor.mask_centerline_angle(
                self._mask_a, self._mask_b, self.bbox_a, self.bbox_b)
        self.angle_a = angle_a
        self.angle_b = angle_b
        self.editor_a.angle = angle_a
        self.lbl_angle_a.configure(text=f"{angle_a:+.1f}°")
        self.editor_b.angle = angle_b
        self.lbl_angle_b.configure(text=f"{angle_b:+.1f}°")

    def _reset_angle(self, which):
        """重置单面角度为零（杆子）并应用于该面。"""
        if which == 'a':
            self.angle_a = 0.0
            self.editor_a.angle = 0.0
            self.lbl_angle_a.configure(text="0.0°")
            self.editor_a._redraw()
        else:
            self.angle_b = 0.0
            self.editor_b.angle = 0.0
            self.lbl_angle_b.configure(text="0.0°")
            self.editor_b._redraw()
        self._liquified = None
        self._update_preview()
        self._auto_save_debounce()

    # ── 流式处理 ──────────────────────────────────────────────

    def _pick_dir(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="选择文件夹中任意一张图片",
            filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")])
        if not path:
            return
        d = Path(path).parent
        self.entry_dir.delete(0, "end")
        self.entry_dir.insert(0, str(d))
        ann = d / "annotations.json"
        self.update_idletasks()
        if ann.exists():
            self.after_idle(lambda: self._start_process(auto_load=True))
        else:
            self.status.configure(text="已选文件夹，点击「AI 处理」开始")

    def _start_process(self, auto_load=False):
        # 模型还在加载中，不允许开始处理
        if not getattr(self.processor, '_warmed', False):
            self.status.configure(text="模型仍在加载中，请稍候...")
            return

        # Prevent re-entry while worker is running
        if getattr(self, '_processing', False):
            self.status.configure(text="AI 正在处理中，请稍候...")
            return

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

        # 清除上次结果
        self.annotations = {}
        self._results = []
        self._liquified = None
        self._first_loaded = False
        self.btn_debug.configure(state="normal")

        # 加载标注文件
        ann_path = self.input_dir / "annotations.json"
        if ann_path.exists():
            data = json.loads(ann_path.read_text("utf-8"))
            for a in data.get("annotations", []):
                self.annotations[a["file"]] = a

        # auto_load 模式：浏览文件夹时静默加载已有标注
        # 标注覆盖全 → 秒开，不覆盖全 → 等待用户点击「AI 处理」
        if auto_load:
            paired_names = {p[0].name for p in self.pairs} | {p[1].name for p in self.pairs}
            covered = paired_names & set(self.annotations.keys())
            if paired_names <= set(self.annotations.keys()):
                self._proc_total = len(self.pairs)
                self._proc_done = self._proc_total
                self._results = [None] * self._proc_total
                self.pair_idx = 0
                self._first_loaded = True
                self._load_current_pair()
                self.lbl_idx.configure(text=f"1 / {self._proc_total}")
                self.btn_prev.configure(state="disabled")
                self.btn_next.configure(state="normal" if self._proc_done > 1 else "disabled")
                self.status.configure(text=f"已加载 {self._proc_total} 对标注")
                return
            else:
                # 标注不完整或为空：只加载已有标注，不跑 AI
                missing = len(paired_names) - len(covered)
                self._proc_total = len(self.pairs)
                self._proc_done = 0
                self._results = [None] * self._proc_total
                self._first_loaded = False
                self.pair_idx = 0
                self.btn_prev.configure(state="disabled")
                self.btn_next.configure(state="disabled")
                self.lbl_idx.configure(text="0 / 0")
                if covered:
                    self.status.configure(
                        text=f"已加载 {len(covered)}/{len(paired_names)} 对标注，"
                             f"剩余 {missing} 对请点击「AI 处理」补全")
                else:
                    self.status.configure(text="已有标注文件但未匹配当前图片，点击「AI 处理」重新检测")
                return
        else:
            # 手动点击：丢弃标注，强制重跑
            self.annotations = {}

        self._proc_total = len(self.pairs)
        self._proc_done = 0
        self._results = [None] * self._proc_total
        self.pair_idx = 0
        self._first_loaded = False

        self.btn_prev.configure(state="disabled")
        self.btn_next.configure(state="disabled")
        self.lbl_idx.configure(text="0 / 0")
        self.status.configure(text=f"AI+CV 处理中... 0/{self._proc_total}")
        self._status_loader.configure(text="AI 处理中...")

        self._processing = True
        n = self._proc_total

        def worker():
            # 在本线程用第一张真实图片跑一次预热推理，触发 ONNX 的所有惰性初始化。
            # 不能用 32×32 小图——ONNX 对小图和大图走不同的内存规划/kernel，
            # 小图预热后的内存缓冲区在真实图片面前要重新分配，等于没预热。
            if not getattr(self.processor, '_warmed_up', False):
                try:
                    pa0, _ = self.pairs[0]
                    img0 = ImageOps.exif_transpose(Image.open(pa0)).convert("RGB")
                    self.processor._single_pipe(img0)
                except Exception:
                    pass
                self.processor._warmed_up = True

            try:
                for i, (pa, pb) in enumerate(self.pairs):
                    try:
                        img_a = ImageOps.exif_transpose(Image.open(pa)).convert("RGB")
                        img_b = ImageOps.exif_transpose(Image.open(pb)).convert("RGB")
                        bb_a, bb_b, mask_a, mask_b = self.processor._joint_detect(
                            img_a, img_b, stem_a=str(pa), stem_b=str(pb))
                        self._results[i] = (bb_a, bb_b, mask_a, mask_b)
                    except Exception:
                        self._results[i] = (None, None)
                    self.after(0, lambda idx=i: self._on_one_done(idx))
                self.after(0, self._on_all_done)
            finally:
                self._processing = False
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
        self._update_nav_buttons()

    def _on_all_done(self):
        self._proc_done = self._proc_total
        self._save_ai_results()  # AI 完成后写入 annotations.json
        self.status.configure(text=f"全部完成 — {self._proc_total} 对已就绪")
        self._status_loader.configure(text="AI 就绪")
        self._update_nav_buttons()

    def _save_ai_results(self):
        """将 AI 检测结果写入 annotations.json。再次运行时覆盖旧结果。"""
        if not self.input_dir or not self._results:
            return
        ann_path = self.input_dir / "annotations.json"
        for i, res in enumerate(self._results):
            if res is None or i >= len(self.pairs):
                continue
            pa, pb = self.pairs[i]
            bb_a = res[0]; bb_b = res[1]
            if bb_a:
                self.annotations[pa.name] = {"file": pa.name, "bbox": list(bb_a), "angle": 0.0}
            if bb_b:
                self.annotations[pb.name] = {"file": pb.name, "bbox": list(bb_b), "angle": 0.0}
        data = {"source_dir": str(self.input_dir), "annotations": list(self.annotations.values())}
        ann_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

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

        self._liquified = None

        pa, pb = self.pairs[self.pair_idx]
        self.lbl_idx.configure(text=f"{self.pair_idx + 1} / {self._proc_total}")
        self.lbl_fname.configure(text=f"{pa.stem}  +  {pb.stem}")

        # 解析 bbox / mask（不依赖图片，从 AI 结果和标注获取）
        res = self._results[self.pair_idx]
        if res and len(res) == 4:
            self.ai_bbox_a, self.ai_bbox_b, self._mask_a, self._mask_b = res
        elif res:
            self.ai_bbox_a, self.ai_bbox_b = res[0], res[1]
            self._mask_a = self._mask_b = None
        else:
            self.ai_bbox_a = self.ai_bbox_b = None
            self._mask_a = self._mask_b = None

        ann_a = self.annotations.get(pa.name, {})
        ann_b = self.annotations.get(pb.name, {})

        # bbox：标注优先 → AI 结果 → 兜底（图片加载后覆盖）
        if "bbox" in ann_a:
            self.bbox_a = list(ann_a["bbox"])
        elif self.ai_bbox_a:
            self.bbox_a = list(self.ai_bbox_a)
        else:
            self.bbox_a = None  # 需要图片才能算

        if "bbox" in ann_b:
            self.bbox_b = list(ann_b["bbox"])
        elif self.ai_bbox_b:
            self.bbox_b = list(self.ai_bbox_b)
        else:
            self.bbox_b = None

        # 角度：已标注用标注值，否则用 AI 默认
        if self.bbox_a is not None and self.bbox_b is not None:
            if "angle" in ann_a:
                self.angle_a = ann_a["angle"]
            else:
                self._recalc_angle()
            if "angle" not in ann_b:
                pass  # _recalc_angle 已同时设置 a 和 b
        else:
            self.angle_a = ann_a.get("angle", 0.0)
            self.angle_b = ann_b.get("angle", 0.0)

        # 显示角度
        self.lbl_angle_a.configure(text=f"{self.angle_a:+.1f}°")
        self.lbl_angle_b.configure(text=f"{self.angle_b:+.1f}°")

        # 后台异步加载图片 + 显示旋转 spinner
        self._load_seq += 1
        seq = self._load_seq
        self._show_loading()
        self.status.configure(text="加载图片...")

        def _load():
            try:
                ia = ImageOps.exif_transpose(Image.open(pa)).convert("RGB")
                ib = ImageOps.exif_transpose(Image.open(pb)).convert("RGB")
            except Exception:
                ia = ib = None

            def _done():
                if seq != self._load_seq:
                    return  # 用户已切到其他对，丢弃
                self._hide_loading()
                if ia is None:
                    self.status.configure(text="图片加载失败")
                    return
                self.img_a = ia; self.img_b = ib
                # 兜底 bbox
                if self.bbox_a is None:
                    self.bbox_a = [ia.width//4, ia.height//4,
                                   ia.width*3//4, ia.height*3//4]
                if self.bbox_b is None:
                    self.bbox_b = [ib.width//4, ib.height//4,
                                   ib.width*3//4, ib.height*3//4]
                self.editor_a.set_image(ia, self.bbox_a, self.ai_bbox_a, self.angle_a)
                self.editor_b.set_image(ib, self.bbox_b, self.ai_bbox_b, self.angle_b)
                self._update_preview()
                self.status.configure(text=f"当前: {pa.stem} + {pb.stem}")
                self.after(100, self._fit_editors)
            self.after(0, _done)

        threading.Thread(target=_load, daemon=True).start()

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
        bcy = (y1 + y2) / 2
        crop_h = crop_w * 2
        # 黄金比例：外侧（近画幅边缘）∶ 中间总间隙（正+反内侧） = φ ∶ 1
        # 设 inner_a ≈ inner_b = inner，则 outer / (2*inner) = φ
        # → inner = extra / (2φ + 1), outer = extra - inner
        bw = x2 - x1
        extra = max(0, crop_w - bw)
        phi = 1.618
        inner = int(extra / (2 * phi + 1))  # 近中线侧 ≈24%
        outer = extra - inner               # 近边缘侧 ≈76%
        if anchor == "right":
            cx = x2 + inner - crop_w
        else:
            cx = x1 - inner
        if cx < 0: cx = 0
        if cx + crop_w > w: cx = w - crop_w
        cy = int(bcy - crop_h / 2)
        if cy < 0: cy = 0
        if cy + crop_h > h: cy = h - crop_h
        return img.crop((cx, cy, cx + crop_w, cy + crop_h))

    def _update_preview(self):
        if not self.img_a or not self.img_b: return
        if self._liquified:
            preview = self._liquified
            th = preview.width
        else:
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
            left = crop_a.resize((hw, th), Image.BILINEAR)
            right = crop_b.resize((hw, th), Image.BILINEAR)
            preview = Image.new("RGB", (th, th), (255, 255, 255))
            preview.paste(left, (0, 0)); preview.paste(right, (hw, 0))

        c = self.preview_canvas
        cw_canvas = max(c.winfo_width(), 100); ch_canvas = max(c.winfo_height(), 100)
        display_size = min(cw_canvas, ch_canvas)
        ds = max(int(display_size * self._preview_zoom), 100)
        if th != ds:
            preview = preview.resize((ds, ds), Image.BILINEAR)

        self._preview_img = ImageTk.PhotoImage(preview)
        c.delete("spinner")  # 清除 spinner（如果正在加载）
        # 清除画布但不删 spinner tag（已在上面删）
        items = c.find_all()
        for item in items:
            if "spinner" not in c.gettags(item):
                c.delete(item)
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

    # ── 加载动画 ────────────────────────────────────────────────

    def _show_loading(self):
        """在预览 canvas 中央显示半透明旋转加载弧。"""
        c = self.preview_canvas
        cw = max(c.winfo_width(), 100)
        ch = max(c.winfo_height(), 100)
        r = min(cw, ch) // 10
        cx, cy = cw // 2, ch // 2
        self._spinner_angle = 0
        self._spinner_id = c.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=0, extent=270, outline="#BBBBBB", width=3,
            style="arc", tags="spinner")
        self._spin()

    def _spin(self):
        if not self._spinner_id:
            return
        self._spinner_angle = (self._spinner_angle + 20) % 360
        c = self.preview_canvas
        try:
            c.itemconfigure(self._spinner_id, start=self._spinner_angle)
        except Exception:
            self._spinner_id = None
            return
        self._spinner_after = self.after(40, self._spin)

    def _hide_loading(self):
        if self._spinner_after:
            self.after_cancel(self._spinner_after)
            self._spinner_after = None
        self._spinner_id = None
        self.preview_canvas.delete("spinner")

    def _on_preview_configure(self, event=None):
        """预览 canvas 尺寸变化时：如果无图片，重绘水印 logo。"""
        if not self.img_a or not self.img_b:
            self._draw_preview_placeholder()

    def _draw_preview_placeholder(self):
        """空状态：预览 canvas 居中显示半透明水印。"""
        c = self.preview_canvas
        if not self._preview_placeholder:
            return
        cw_canvas = max(c.winfo_width(), 100)
        ch_canvas = max(c.winfo_height(), 100)
        ph = self._preview_placeholder
        pw, ph_h = ph.size
        dw = min(BBoxEditor.PLACEHOLDER_W, cw_canvas - 20)
        dh = int(dw * ph_h / pw)
        photo = ImageTk.PhotoImage(ph.resize((dw, dh), Image.LANCZOS))
        c.delete("all")
        c.create_image(cw_canvas // 2, ch_canvas // 2, anchor=tk.CENTER, image=photo)
        self._preview_photo = photo  # keep ref

    def _on_preview_wheel(self, event):
        z = self._preview_zoom
        z *= 1.1 if event.delta > 0 else 1 / 1.1
        self._preview_zoom = max(1.0, min(5.0, z))
        self._update_preview()

    def _on_bbox_changed(self):
        self.bbox_a = list(self.editor_a.bbox); self.bbox_b = list(self.editor_b.bbox)
        self.angle_a = self.editor_a.angle; self.angle_b = self.editor_b.angle
        self.lbl_angle_a.configure(text=f"{self.angle_a:+.1f}°")
        self.lbl_angle_b.configure(text=f"{self.angle_b:+.1f}°")
        self._liquified = None
        self._update_preview()
        self._auto_save_debounce()

    # ── 按钮操作 ──────────────────────────────────────────────

    def _auto_save_debounce(self):
        if hasattr(self, '_auto_save_id'): self.after_cancel(self._auto_save_id)
        self._auto_save_id = self.after(800, self._auto_save)

    def _liquify(self):
        """对当前拼接结果打开液化工具"""
        if not self.img_a or not self.img_b: return
        stitched = self._stitch_current()
        pa = self.pairs[self.pair_idx][0]
        tool = LiquifyTool(stitched, f"液化 — {pa.stem}.png",
                           on_apply=lambda result: self._on_liquify_done(result, pa))
        self.wait_window(tool)

    def _on_liquify_done(self, result, pa):
        if result:
            self._liquified = result
            self._update_preview()
            self.status.configure(text=f"液化已应用，导出时生效")

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
        self._liquified = None
        self._update_preview()
        self.lbl_fname.configure(text=f"{pb.stem}  +  {pa.stem}")
        self.status.configure(text="已互换正反面")
        self._auto_save()

    def _reset_ai(self):
        if self.ai_bbox_a:
            self.bbox_a = list(self.ai_bbox_a)
            self.editor_a.bbox = list(self.ai_bbox_a)
        if self.ai_bbox_b:
            self.bbox_b = list(self.ai_bbox_b)
            self.editor_b.bbox = list(self.ai_bbox_b)
        self._recalc_angle()
        self._update_preview(); self.editor_a._redraw(); self.editor_b._redraw()
        self._auto_save()

    def _auto_save(self):
        if not self.pairs or not self.input_dir: return
        pa, pb = self.pairs[self.pair_idx]
        self.annotations[pa.name] = {"file": pa.name, "bbox": list(self.bbox_a), "angle": self.angle_a}
        self.annotations[pb.name] = {"file": pb.name, "bbox": list(self.bbox_b), "angle": self.angle_b}
        data = {"source_dir": str(self.input_dir), "annotations": list(self.annotations.values())}
        json_text = json.dumps(data, ensure_ascii=False, indent=2)
        ann_path = self.input_dir / "annotations.json"
        # 后台写入，不阻塞 UI
        threading.Thread(target=lambda: ann_path.write_text(json_text, encoding="utf-8"), daemon=True).start()

    # ── 导出 ──────────────────────────────────────────────────

    def _export_single(self):
        if not self.pairs: return
        pa, pb = self.pairs[self.pair_idx]
        out_dir = self.input_dir / "审核输出" if self.input_dir else Path("审核输出")
        out_dir.mkdir(parents=True, exist_ok=True)
        result = self._liquified or self._stitch_current()
        result.save(out_dir / f"{pa.stem}.png", "PNG")
        self.status.configure(text=f"已导出 {pa.stem}.png")

    def _stitch_current(self):
        """生成当前 bbox/角度的拼接结果（不含液化），用于导出和液化入口。"""
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
        return result

    def _export_all(self):
        if not self.input_dir: return
        out_dir = self.input_dir / "审核输出"; out_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for i in range(self._proc_done):
            pa, pb = self.pairs[i]
            if i == self.pair_idx:
                if self._liquified:
                    try:
                        self._liquified.save(out_dir / f"{pa.stem}.png", "PNG")
                        ok += 1
                    except Exception:
                        pass
                    continue
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
        """对当前正在查看的那对运行调试检测并弹出可视化窗口。"""
        if not self.pairs or self._proc_done < 1:
            return
        pa, pb = self.pairs[self.pair_idx]
        try:
            ia = ImageOps.exif_transpose(Image.open(pa)).convert("RGB")
            ib = ImageOps.exif_transpose(Image.open(pb)).convert("RGB")
            _, _, debug_entries = self.processor._joint_detect_debug(
                ia, ib, angle_mode=self._angle_mode,
                stem_a=str(pa), stem_b=str(pb))
            DebugWindow(self, debug_entries, f"调试 — {pa.stem} + {pb.stem}")
        except Exception as e:
            self.status.configure(text=f"调试失败: {e}")

    def _show_about(self, event=None):
        """弹出关于窗口。"""
        logo = None
        logo_about_png = Path(__file__).parent / "logo_about.png"
        if logo_about_png.exists():
            logo = ImageTk.PhotoImage(Image.open(str(logo_about_png)))
        w = AboutWindow(self, logo)
        w.transient(self)
        w.grab_set()
        w.lift()


class AboutWindow(tk.Toplevel):
    """关于对话框。"""

    def __init__(self, parent, logo):
        super().__init__(parent)
        self.title("关于")
        self.geometry("780x400")
        self.configure(bg="#1E1E1E")
        self.resizable(False, False)
        self.transient(parent)

        if logo:
            lbl = tk.Label(self, image=logo, bg="#1E1E1E")
            lbl.image = logo
            lbl.pack(pady=(30, 8))

        info = tk.Label(self,
                        text="Garment Front-Back Stitcher\n"
                             "服装样品正反面 AI+CV 拼接工具\n\n"
                             "技术栈： rembg / onnxruntime / NumPy / SciPy / Pillow\n"
                             "桌面框架： customtkinter + tkinter",
                        bg="#1E1E1E", fg="#CCC",
                        font=("Microsoft YaHei UI", 11),
                        justify="center")
        info.pack(pady=(4, 20))


class DebugWindow(tk.Toplevel):
    """Scrollable debug image viewer - each step shows A/B side by side."""

    def __init__(self, parent, entries, title="debug"):
        super().__init__(parent)
        self.title(title)
        self.geometry("1050x800")
        self.configure(bg="#1E1E1E")

        # Group entries by step name (without trailing " A"/" B" suffix)
        steps = {}   # step_name -> {"img_a": Image, "img_b": Image}
        order = []   # insertion order
        for label, img in entries:
            if label.endswith(" A") or label.endswith(" B"):
                name = label[:-2]
                side = label[-1]  # "A" or "B"
            else:
                name = label
                side = None
            if name not in steps:
                steps[name] = {}
                order.append(name)
            if side == "B":
                steps[name]["img_b"] = img
            else:
                steps[name]["img_a"] = img

        info = tk.Label(self, text=f"共 {len(order)} 步", bg="#1E1E1E", fg="#999",
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

        self._photos = []
        PAIR_W = 460   # 每张 A/B 配对图宽度
        FULL_W = 900   # 单张全宽图（图表等）

        for name in order:
            data = steps[name]
            lbl = tk.Label(scroll_frame, text=name, bg="#1E1E1E", fg="#CCC",
                           font=("Microsoft YaHei UI", 11, "bold"))
            lbl.pack(pady=(12, 2))

            img_a = data.get("img_a")
            img_b = data.get("img_b")
            pair_frame = tk.Frame(scroll_frame, bg="#1E1E1E")
            pair_frame.pack(pady=(0, 6))

            for img in [img_a, img_b]:
                if img is None:
                    continue
                w, h = img.size
                max_w = PAIR_W if img_b is not None else FULL_W
                if w > max_w:
                    img = img.resize((max_w, int(h * max_w / w)), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._photos.append(photo)
                side = "left" if img is img_a else "left"  # pack both side=left for flow
                tk.Label(pair_frame, image=photo, bg="#1E1E1E").pack(side="left", padx=3)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.bind("<MouseWheel>", _on_mousewheel)


def main():
    ReviewerApp().mainloop()

if __name__ == "__main__":
    main()
