"""液化修图工具 — PS 风格操控

鼠标：
  左键拖拽      前推变形
  空格+左键拖   平移视图
  Alt+右键拖    缩放（上下拖动）
  Ctrl+滚轮     调整笔刷大小
  滚轮          缩放视图

键盘：
  [ / ]         笔刷大小 +/-5
  Space (按住)  临时切换到手型平移
  Ctrl+Z        撤销
  Ctrl+Shift+Z  重做
  R             重置全部
  M             网格显示
  Enter         应用并返回
  Escape        取消
"""

import math
from pathlib import Path
from typing import Optional
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk

try:
    from scipy.ndimage import zoom as _sp_zoom, map_coordinates as _sp_map
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _fast_resize(grid: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    gh, gw = grid.shape
    if _HAS_SCIPY:
        return _sp_zoom(grid, (out_h / gh, out_w / gw), order=1, mode='nearest')
    ys = np.linspace(0, gh - 1, out_h, dtype=np.float32)
    xs = np.linspace(0, gw - 1, out_w, dtype=np.float32)
    y0 = np.floor(ys).astype(int); x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, gh - 1); x1 = np.minimum(x0 + 1, gw - 1)
    fy = (ys - y0).reshape(-1, 1).astype(np.float32)
    fx = (xs - x0).reshape(1, -1).astype(np.float32)
    q00 = grid[y0[:, None], x0[None, :]]; q10 = grid[y1[:, None], x0[None, :]]
    q01 = grid[y0[:, None], x1[None, :]]; q11 = grid[y1[:, None], x1[None, :]]
    return ((1 - fy) * ((1 - fx) * q00 + fx * q01) +
            fy * ((1 - fx) * q10 + fx * q11)).astype(np.float32)


class LiquifyEngine:
    def __init__(self, image: Image.Image, grid_spacing: int = 8):
        self.original = image.copy()
        self.w, self.h = image.size
        self.grid_spacing = grid_spacing
        self.gw = self.w // grid_spacing + 1
        self.gh = self.h // grid_spacing + 1
        self.dx = np.zeros((self.gh, self.gw), dtype=np.float32)
        self.dy = np.zeros((self.gh, self.gw), dtype=np.float32)
        self._dirty = True
        self._cache_scale = 0.0
        self._cache_img: Optional[Image.Image] = None
        self._full_arr: Optional[np.ndarray] = None   # 全帧 numpy 缓存 (uint8, H×W×3)
        self._dirty_rect: Optional[list] = None        # 脏矩形 [x1,y1,x2,y2] 图片坐标, None=需全刷
        self._history: list[tuple[np.ndarray, np.ndarray]] = []

    def push_history(self):
        self._history.append((self.dx.copy(), self.dy.copy()))
        if len(self._history) > 50: self._history.pop(0)

    def undo(self) -> bool:
        if not self._history: return False
        self.dx, self.dy = self._history.pop()
        self._dirty = True
        self._dirty_rect = None   # 全量重绘
        return True

    def reset(self):
        if self._history:
            self.dx, self.dy = self._history[0][0].copy(), self._history[0][1].copy()
        else:
            self.dx.fill(0); self.dy.fill(0)
        self._dirty = True
        self._dirty_rect = None   # 全量重绘

    def brush_stroke(self, x, y, prev_x, prev_y, radius, pressure):
        delta_x = x - prev_x; delta_y = y - prev_y
        if max(abs(delta_x), abs(delta_y)) < 0.5: return
        self.push_history()
        gx_min = max(0, int((x - radius) / self.grid_spacing) - 1)
        gx_max = min(self.gw - 1, int((x + radius) / self.grid_spacing) + 1)
        gy_min = max(0, int((y - radius) / self.grid_spacing) - 1)
        gy_max = min(self.gh - 1, int((y + radius) / self.grid_spacing) + 1)
        gy_arr, gx_arr = np.mgrid[gy_min:gy_max + 1, gx_min:gx_max + 1]
        px_arr = gx_arr * self.grid_spacing; py_arr = gy_arr * self.grid_spacing
        dx_arr = (px_arr - x).astype(np.float32); dy_arr = (py_arr - y).astype(np.float32)
        dist_arr = np.sqrt(dx_arr * dx_arr + dy_arr * dy_arr)
        norm = dist_arr / max(radius, 1)
        falloff = np.clip(1.0 - norm, 0.0, 1.0)
        falloff = falloff * falloff
        self.dx[gy_min:gy_max + 1, gx_min:gx_max + 1] += delta_x * falloff * pressure
        self.dy[gy_min:gy_max + 1, gx_min:gx_max + 1] += delta_y * falloff * pressure
        self._dirty = True

        # 追踪脏矩形（图片坐标，含余量覆盖插值边界）
        margin = self.grid_spacing * 2
        dr_x1 = max(0, int(x - radius - margin))
        dr_y1 = max(0, int(y - radius - margin))
        dr_x2 = min(self.w, int(x + radius + margin))
        dr_y2 = min(self.h, int(y + radius + margin))
        if self._dirty_rect is None:
            self._dirty_rect = [dr_x1, dr_y1, dr_x2, dr_y2]
        else:
            self._dirty_rect[0] = min(self._dirty_rect[0], dr_x1)
            self._dirty_rect[1] = min(self._dirty_rect[1], dr_y1)
            self._dirty_rect[2] = max(self._dirty_rect[2], dr_x2)
            self._dirty_rect[3] = max(self._dirty_rect[3], dr_y2)

    def get_warped(self, display_scale: float = 1.0) -> Image.Image:
        """返回变形后的 PIL Image。自动选择全量或增量渲染路径。"""
        if not self._dirty and self._cache_scale == display_scale and self._cache_img:
            return self._cache_img

        dw = int(self.w * display_scale)
        dh = int(self.h * display_scale)

        # 判断是否需要全量渲染
        need_full = (
            self._full_arr is None
            or self._cache_scale != display_scale
            or self._dirty_rect is None
        )
        if not need_full:
            dr = self._dirty_rect
            dirty_area = (dr[2] - dr[0]) * (dr[3] - dr[1])
            if dirty_area > self.w * self.h * 0.5:
                need_full = True

        if need_full:
            return self._full_render(display_scale, dw, dh)
        else:
            return self._partial_render(display_scale, dw, dh)

    def _full_render(self, display_scale: float, dw: int, dh: int) -> Image.Image:
        """全量像素重映射。"""
        dx_full = _fast_resize(self.dx, dw, dh) * display_scale
        dy_full = _fast_resize(self.dy, dw, dh) * display_scale
        src = self.original.resize((dw, dh), Image.LANCZOS) if display_scale < 1.0 else self.original
        arr = np.array(src, dtype=np.float32)
        ys, xs = np.mgrid[0:dh, 0:dw].astype(np.float32)
        src_xs = xs - dx_full
        src_ys = ys - dy_full
        if _HAS_SCIPY:
            result = np.zeros((dh, dw, 3), dtype=np.uint8)
            coords = np.stack([src_ys.ravel(), src_xs.ravel()], axis=0)
            for c in range(3):
                result[:, :, c] = _sp_map(arr[:, :, c], coords.reshape(2, dh, dw),
                                          order=1, mode='constant', cval=255, prefilter=False)
            result = np.clip(result, 0, 255).astype(np.uint8)
        else:
            src_xs = np.clip(src_xs, 0, dw - 1.001)
            src_ys = np.clip(src_ys, 0, dh - 1.001)
            x0 = src_xs.astype(int); y0 = src_ys.astype(int)
            x1 = np.minimum(x0 + 1, dw - 1); y1 = np.minimum(y0 + 1, dh - 1)
            fx = src_xs - x0; fy = src_ys - y0
            fx = fx[:, :, np.newaxis]; fy = fy[:, :, np.newaxis]
            q00 = arr[y0, x0]; q10 = arr[y0, x1]; q01 = arr[y1, x0]; q11 = arr[y1, x1]
            result = np.clip((1 - fy) * ((1 - fx) * q00 + fx * q10) +
                             fy * ((1 - fx) * q01 + fx * q11), 0, 255).astype(np.uint8)
        self._full_arr = result
        self._dirty_rect = None
        self._dirty = False
        self._cache_scale = display_scale
        self._cache_img = Image.fromarray(result)
        return self._cache_img

    def _partial_render(self, display_scale: float, dw: int, dh: int) -> Image.Image:
        """仅重算脏矩形内像素，其余从 _full_arr 缓存拷贝。"""
        dr = self._dirty_rect
        margin = 3
        dx1 = max(0, int(dr[0] * display_scale) - margin)
        dy1 = max(0, int(dr[1] * display_scale) - margin)
        dx2 = min(dw, int(dr[2] * display_scale) + margin)
        dy2 = min(dh, int(dr[3] * display_scale) + margin)
        sub_w = dx2 - dx1
        sub_h = dy2 - dy1

        if sub_w <= 0 or sub_h <= 0:
            self._dirty_rect = None
            self._dirty = False
            return self._cache_img

        # 全量位移场（变形网格小，_fast_resize 开销可忽略）
        dx_full = _fast_resize(self.dx, dw, dh) * display_scale
        dy_full = _fast_resize(self.dy, dw, dh) * display_scale

        sub_dx = dx_full[dy1:dy2, dx1:dx2]
        sub_dy = dy_full[dy1:dy2, dx1:dx2]

        src = self.original.resize((dw, dh), Image.LANCZOS) if display_scale < 1.0 else self.original
        arr = np.array(src, dtype=np.float32)

        ys, xs = np.mgrid[dy1:dy2, dx1:dx2].astype(np.float32)
        src_xs = xs - sub_dx
        src_ys = ys - sub_dy

        if _HAS_SCIPY:
            sub_result = np.zeros((sub_h, sub_w, 3), dtype=np.uint8)
            coords = np.stack([src_ys.ravel(), src_xs.ravel()], axis=0)
            for c in range(3):
                sub_result[:, :, c] = _sp_map(arr[:, :, c], coords.reshape(2, sub_h, sub_w),
                                              order=1, mode='constant', cval=255, prefilter=False)
            sub_result = np.clip(sub_result, 0, 255).astype(np.uint8)
        else:
            src_xs = np.clip(src_xs, 0, dw - 1.001)
            src_ys = np.clip(src_ys, 0, dh - 1.001)
            x0 = src_xs.astype(int); y0 = src_ys.astype(int)
            x1 = np.minimum(x0 + 1, dw - 1); y1 = np.minimum(y0 + 1, dh - 1)
            fx = src_xs - x0; fy = src_ys - y0
            fx = fx[:, :, np.newaxis]; fy = fy[:, :, np.newaxis]
            q00 = arr[y0, x0]; q10 = arr[y0, x1]; q01 = arr[y1, x0]; q11 = arr[y1, x1]
            sub_result = np.clip((1 - fy) * ((1 - fx) * q00 + fx * q10) +
                                 fy * ((1 - fx) * q01 + fx * q11), 0, 255).astype(np.uint8)

        # 将子区域结果贴回全帧缓存
        self._full_arr[dy1:dy2, dx1:dx2] = sub_result

        self._dirty_rect = None
        self._dirty = False
        self._cache_img = Image.fromarray(self._full_arr)
        return self._cache_img


class LiquifyCanvas(tk.Canvas):
    """PS 风格液化画布：左键变形 / 空格平移 / 滚轮缩放 / Ctrl+滚轮调笔刷"""

    def __init__(self, parent, engine: LiquifyEngine, **kw):
        super().__init__(parent, bg="#1E1E1E", highlightthickness=0, **kw)
        self.engine = engine
        self.scale = 0.35
        self.ox = 0; self.oy = 0           # 图像左上角在 canvas 上的偏移
        self.brush_radius = 40
        self.pressure = 0.3
        self.show_mesh = False

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._mode: Optional[str] = None     # 'warp' | 'pan' | 'zoom'
        self._prev_cx = 0; self._prev_cy = 0
        self._prev_ix = 0.0; self._prev_iy = 0.0
        self._space_down = False
        self._alt_down = False
        self._ctrl_down = False
        self._render_pending = False
        self._initialized = False

        # 鼠标事件
        self.bind("<ButtonPress-1>", self._on_down)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_up)
        self.bind("<ButtonPress-3>", self._on_r_down)
        self.bind("<B3-Motion>", self._on_r_drag)
        self.bind("<ButtonRelease-3>", self._on_r_up)
        self.bind("<MouseWheel>", self._on_wheel)
        self.bind("<Configure>", lambda e: self._render())
        self.bind("<Motion>", self._on_motion)   # 笔刷光标跟随
        self.bind("<Leave>", self._on_leave)      # 离开画布隐藏光标

        # 键盘
        self.bind_all("<KeyPress-space>", self._space_press)
        self.bind_all("<KeyRelease-space>", self._space_release)
        self.bind_all("<KeyPress-Alt_L>", self._alt_press); self.bind_all("<KeyPress-Alt_R>", self._alt_press)
        self.bind_all("<KeyRelease-Alt_L>", self._alt_release); self.bind_all("<KeyRelease-Alt_R>", self._alt_release)
        self.bind_all("<KeyPress-Control_L>", self._ctrl_press); self.bind_all("<KeyPress-Control_R>", self._ctrl_press)
        self.bind_all("<KeyRelease-Control_L>", self._ctrl_release); self.bind_all("<KeyRelease-Control_R>", self._ctrl_release)

    def _space_press(self, e): self._space_down = True
    def _space_release(self, e): self._space_down = False
    def _alt_press(self, e): self._alt_down = True
    def _alt_release(self, e): self._alt_down = False
    def _ctrl_press(self, e): self._ctrl_down = True
    def _ctrl_release(self, e): self._ctrl_down = False

    # ── 坐标 ──────────────────────────────────────────────────

    def _canv_to_img(self, cx, cy):
        return ((cx - self.ox) / self.scale, (cy - self.oy) / self.scale)

    # ── 渲染 ──────────────────────────────────────────────────

    def _render(self):
        """重绘全图（包含图像 + 平移/缩放位移）"""
        dw = int(self.engine.w * self.scale); dh = int(self.engine.h * self.scale)
        img = self.engine.get_warped(self.scale)
        self._photo = ImageTk.PhotoImage(img)
        self.delete("img", "brush")
        cw, ch = self.winfo_width(), self.winfo_height()
        # 首次自动居中
        if not self._initialized:
            self._initialized = True
            self.ox = (cw - dw) // 2; self.oy = (ch - dh) // 2
        # 限制不超出画布太远
        self.ox = max(min(self.ox, cw - 100), -dw + 100)
        self.oy = max(min(self.oy, ch - 100), -dh + 100)
        self.create_image(self.ox, self.oy, anchor=tk.NW, image=self._photo, tags="img")
        self._render_pending = False

    def _schedule_render(self):
        if not self._render_pending:
            self._render_pending = True
            self.after(16, self._render)  # ~60fps（脏区域后单帧成本极低）

    # ── 左键: 变形或平移 ─────────────────────────────────────

    def _on_down(self, event):
        if self._space_down:
            self._mode = 'pan'
            self._prev_cx, self._prev_cy = event.x, event.y
        else:
            self._mode = 'warp'
            ix, iy = self._canv_to_img(event.x, event.y)
            self._prev_ix, self._prev_iy = ix, iy

    def _on_drag(self, event):
        if not self._mode: return
        if self._mode == 'pan':
            dx, dy = event.x - self._prev_cx, event.y - self._prev_cy
            self.ox += dx; self.oy += dy
            self._prev_cx, self._prev_cy = event.x, event.y
            self._schedule_render()
        elif self._mode == 'warp':
            ix, iy = self._canv_to_img(event.x, event.y)
            self.engine.brush_stroke(ix, iy, self._prev_ix, self._prev_iy,
                                     self.brush_radius, self.pressure)
            self._prev_ix, self._prev_iy = ix, iy
            self._schedule_render()
            # 笔刷变形预览（canvas 坐标，三层）
            r = int(self.brush_radius * self.scale)
            r1 = max(2, int(r * 0.3))
            self.create_oval(event.x - r1, event.y - r1, event.x + r1, event.y + r1,
                             fill="#FFFFFF", outline="", stipple="gray50", tags="brush")
            r2 = max(r1 + 2, int(r * 0.7))
            self.create_oval(event.x - r2, event.y - r2, event.x + r2, event.y + r2,
                             outline="#00BFFF", width=1, tags="brush")
            self.create_oval(event.x - r, event.y - r, event.x + r, event.y + r,
                             outline="#00BFFF", width=1, dash=(3, 5), tags="brush")

    def _on_up(self, event):
        self._mode = None
        self._render()

    def _on_motion(self, event):
        """鼠标移动时显示三层同心笔刷光标"""
        self.delete("cursor")
        if self._mode in ('pan', 'zoom'):
            return
        r = int(self.brush_radius * self.scale)
        if r < 3:
            return
        cx, cy = event.x, event.y
        # 内圈: 核心压力区，半透明白色填充
        r1 = max(2, int(r * 0.3))
        self.create_oval(cx - r1, cy - r1, cx + r1, cy + r1,
                         fill="#FFFFFF", outline="", stipple="gray50", tags="cursor")
        # 中圈: 有效变形区，实线
        r2 = max(r1 + 2, int(r * 0.7))
        self.create_oval(cx - r2, cy - r2, cx + r2, cy + r2,
                         outline="#00BFFF", width=1, tags="cursor")
        # 外圈: 衰减边界，虚线
        self.create_oval(cx - r, cy - r, cx + r, cy + r,
                         outline="#00BFFF", width=1, dash=(4, 6), tags="cursor")

    def _on_leave(self, event):
        self.delete("cursor")

    # ── 右键: 缩放（Alt+右键拖上下）─────────────────────────

    def _on_r_down(self, event):
        if self._alt_down:
            self._mode = 'zoom'
            self._prev_cy = event.y
            self._prev_scale = self.scale

    def _on_r_drag(self, event):
        if self._mode != 'zoom': return
        dy = self._prev_cy - event.y
        factor = 1.0 + dy * 0.005
        new_scale = max(0.05, min(2.0, self._prev_scale * factor))
        # 以鼠标位置为中心缩放
        mx, my = event.x, event.y
        self.ox = mx - (mx - self.ox) * (new_scale / self.scale)
        self.oy = my - (my - self.oy) * (new_scale / self.scale)
        self.scale = new_scale
        self._schedule_render()

    def _on_r_up(self, event):
        self._mode = None
        self._render()

    # ── 滚轮: 缩放 或 Ctrl+滚轮调笔刷 ────────────────────────

    def _on_wheel(self, event):
        if (event.state & 0x4) or self._ctrl_down:  # Ctrl = 调笔刷大小
            self.set_brush(radius=self.brush_radius + (-10 if event.delta < 0 else 10))
            self.event_generate("<<BrushChanged>>")
        else:  # 普通滚轮 = 缩放
            factor = 1.1 if event.delta > 0 else 1 / 1.1
            new_scale = max(0.05, min(2.0, self.scale * factor))
            mx, my = event.x, event.y
            self.ox = mx - (mx - self.ox) * (new_scale / self.scale)
            self.oy = my - (my - self.oy) * (new_scale / self.scale)
            self.scale = new_scale
            self._schedule_render()
        return "break"

    def set_brush(self, radius=None, pressure=None):
        if radius is not None: self.brush_radius = max(5, min(300, radius))
        if pressure is not None: self.pressure = max(0.05, min(2.0, pressure))

    def toggle_mesh(self): self.show_mesh = not self.show_mesh; self._render()
    def undo(self):
        if self.engine.undo(): self._render()
    def reset(self): self.engine.reset(); self._render()
    def fit(self):
        """适应窗口"""
        self.update_idletasks()
        cw, ch = self.winfo_width(), self.winfo_height()
        s = min(cw / self.engine.w, ch / self.engine.h) * 0.85
        self.scale = max(0.05, s)
        dw = int(self.engine.w * self.scale); dh = int(self.engine.h * self.scale)
        self.ox = (cw - dw) // 2; self.oy = (ch - dh) // 2
        self._render()


class LiquifyTool(tk.Toplevel):
    def __init__(self, image: Image.Image, title: str = "液化修图", on_apply=None):
        super().__init__()
        self.title(title); self.geometry("1200x850"); self.configure(bg="#2B2B2B")
        self.result: Optional[Image.Image] = None
        self.on_apply = on_apply
        self.engine = LiquifyEngine(image)

        # ── 工具栏 ──────────────────────────────────────────────
        bar = tk.Frame(self, bg="#3C3C3C"); bar.pack(fill=tk.X)
        b = {"bg": "#555", "fg": "white", "relief": tk.FLAT,
             "font": ("Microsoft YaHei UI", 9), "padx": 10, "pady": 4}
        l = {"bg": "#3C3C3C", "fg": "#CCC", "font": ("Microsoft YaHei UI", 9)}

        tk.Button(bar, text="应用(Enter)", command=self._apply, **b).pack(side=tk.LEFT, padx=3, pady=4)
        tk.Button(bar, text="取消(Esc)", command=self._cancel, **b).pack(side=tk.LEFT, padx=3, pady=4)
        tk.Frame(bar, width=1, height=28, bg="#666").pack(side=tk.LEFT, padx=6, pady=4)

        tk.Label(bar, text="笔刷:", **l).pack(side=tk.LEFT, padx=(8, 2))
        self._bs = tk.Scale(bar, from_=5, to=300, orient=tk.HORIZONTAL, length=150,
                            bg="#555", fg="white", highlightbackground="#555",
                            command=lambda v: self._on_brush(float(v)))
        self._bs.set(40); self._bs.pack(side=tk.LEFT, padx=2)
        self._bl = tk.Label(bar, text="40px", bg="#3C3C3C", fg="#CCC",
                            font=("Consolas", 9), width=6); self._bl.pack(side=tk.LEFT)
        tk.Frame(bar, width=1, height=28, bg="#666").pack(side=tk.LEFT, padx=6, pady=4)

        tk.Label(bar, text="压力:", **l).pack(side=tk.LEFT, padx=(8, 2))
        self._ps = tk.Scale(bar, from_=5, to=200, orient=tk.HORIZONTAL, length=120,
                            bg="#555", fg="white", highlightbackground="#555",
                            command=lambda v: self._on_press(float(v) / 100))
        self._ps.set(30); self._ps.pack(side=tk.LEFT, padx=2)
        self._pl = tk.Label(bar, text="0.30", bg="#3C3C3C", fg="#CCC",
                            font=("Consolas", 9), width=5); self._pl.pack(side=tk.LEFT)
        tk.Frame(bar, width=1, height=28, bg="#666").pack(side=tk.LEFT, padx=6, pady=4)

        tk.Button(bar, text="适应(F)", command=self._fit, **b).pack(side=tk.LEFT, padx=3, pady=4)
        tk.Button(bar, text="撤销(Ctrl+Z)", command=self._undo, **b).pack(side=tk.LEFT, padx=3, pady=4)
        tk.Button(bar, text="重置(R)", command=self._reset, **b).pack(side=tk.LEFT, padx=3, pady=4)

        self._info = tk.Label(bar, text="", bg="#3C3C3C", fg="#999", font=("Consolas", 9))
        self._info.pack(side=tk.RIGHT, padx=10)

        # ── 画布 ────────────────────────────────────────────────
        self.canvas = LiquifyCanvas(self, self.engine)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 笔刷变化时更新滑块
        self.canvas.bind("<<BrushChanged>>", lambda e: self._sync_ui())

        # ── 键盘 ────────────────────────────────────────────────
        self.bind("<Return>", lambda e: self._apply())
        self.bind("<Escape>", lambda e: self._cancel())
        self.bind("<f>", lambda e: self._fit()); self.bind("<F>", lambda e: self._fit())
        self.bind("<r>", lambda e: self._reset()); self.bind("<R>", lambda e: self._reset())
        self.bind("<Control-z>", lambda e: self._undo()); self.bind("<Control-Z>", lambda e: self._undo())
        self.bind("<bracketleft>", lambda e: self._adj_brush(-5))
        self.bind("<bracketright>", lambda e: self._adj_brush(+5))
        # 防止操作系统截获空格
        self.bind("<Key-space>", lambda e: "break")

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.update()
        self._fit(); self._upd()
        self.after(100, self.grab_set)

    def _sync_ui(self):
        self._bs.set(int(self.canvas.brush_radius))
        self._bl.config(text=f"{int(self.canvas.brush_radius)}px")
        self._upd()

    def _upd(self):
        self._info.config(text=f"缩放:{self.canvas.scale:.0%} 笔刷:{self.canvas.brush_radius:.0f}px 压力:{self.canvas.pressure:.2f}")

    def _on_brush(self, v):
        self.canvas.set_brush(radius=v); self._bl.config(text=f"{int(v)}px"); self._upd()

    def _on_press(self, v):
        self.canvas.set_brush(pressure=v); self._pl.config(text=f"{v:.2f}"); self._upd()

    def _adj_brush(self, d):
        r = self.canvas.brush_radius + d; self._bs.set(int(r)); self._on_brush(r)

    def _fit(self): self.canvas.fit(); self._upd()
    def _undo(self): self.canvas.undo(); self._upd()
    def _reset(self): self.canvas.reset(); self._upd()

    def _apply(self):
        self.result = self.canvas.engine.get_warped(1.0)
        if self.on_apply: self.on_apply(self.result)
        self.grab_release(); self.destroy()

    def _cancel(self):
        self.grab_release(); self.destroy()


def liquify_image(image: Image.Image, title: str = "液化修图") -> Optional[Image.Image]:
    tool = LiquifyTool(image, title); tool.wait_window(); return tool.result


def main():
    from tkinter import filedialog, Tk
    root = Tk(); root.withdraw()
    path = filedialog.askopenfilename(title="选择图片", filetypes=[("图片", "*.png *.jpg *.jpeg *.bmp")])
    root.destroy()
    if not path: return
    img = Image.open(path).convert("RGB")
    r = liquify_image(img, f"液化 — {Path(path).name}")
    if r:
        out = Path(path).parent / f"{Path(path).stem}_liquified.png"
        r.save(out, "PNG"); print(f"已保存: {out}")

if __name__ == "__main__":
    main()
