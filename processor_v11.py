"""v11 处理器：单管道 rembg + 联合轮廓共识匹配（简洁版）

与 v22+ 的区别：
- 单管道（不做多管道预处理/投票）
- 共识区间直接作为 bbox（不做 trim-only 约束）
- 无顶部/底部宽度裁剪
- 无 PAIR_HEIGHT_RATIO_MAX
"""

import threading
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


CATEGORIES = [
    (0.60, 0.07, 0.60),
    (0.80, 0.06, 0.80),
    (0.95, 0.05, 0.90),
    (1.00, 0.04, 0.95),
]

CONSENSUS_RATIO_THRESHOLD = 1.35


class ImageProcessorV11:

    def __init__(self):
        self._session = None
        self._session_lock = threading.Lock()

    def _get_session(self):
        """线程安全地返回 rembg session。使用 double-check 锁确保只创建一次。"""
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    from rembg import new_session
                    self._session = new_session()
        return self._session

    def prewarm(self):
        """后台线程加载 u2net 模型权重到内存。

        _warmed 在 try/finally 中设置，确保即使加载失败也不会导致轮询死循环。
        失败时 _warmed 仍为 True（只是 session 为 None），
        工作线程会在首次推理时重新创建 session。
        """
        self._warmed = False
        def _load():
            try:
                self._get_session()
            except Exception:
                pass  # 加载失败，工作线程稍后重试
            finally:
                self._warmed = True
        threading.Thread(target=_load, daemon=True).start()

    # -- 公共入口 --------------------------------------------------------

    def find_pairs(self, input_dir: Path) -> list[tuple[Path, Path]]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        files = sorted(
            [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in exts],
            key=lambda f: f.name,
        )
        return [(files[i], files[i + 1]) for i in range(0, len(files) - 1, 2)]

    def process_pair(self, img_path_a: Path, img_path_b: Path) -> Image.Image:
        img_a = ImageOps.exif_transpose(Image.open(img_path_a)).convert("RGB")
        img_b = ImageOps.exif_transpose(Image.open(img_path_b)).convert("RGB")

        bbox_a, bbox_b, _mask_a, _mask_b = self._joint_detect(img_a, img_b)

        # 统一crop_w
        unified_cw = self._unified_crop_w(img_a, bbox_a, img_b, bbox_b)

        crop_a = self._crop_1x2(img_a, bbox_a, anchor="right", forced_cw=unified_cw)
        crop_b = self._crop_1x2(img_b, bbox_b, anchor="left", forced_cw=unified_cw)

        if crop_a.width != crop_b.width:
            tw = max(crop_a.width, crop_b.width)
            th = tw * 2
            tmp_a = Image.new("RGB", (tw, th), (255, 255, 255))
            tmp_a.paste(crop_a, ((tw - crop_a.width) // 2, (th - crop_a.height) // 2))
            crop_a = tmp_a
            tmp_b = Image.new("RGB", (tw, th), (255, 255, 255))
            tmp_b.paste(crop_b, ((tw - crop_b.width) // 2, (th - crop_b.height) // 2))
            crop_b = tmp_b

        return self._stitch(crop_a, crop_b)

    def process_all(self, input_dir: Path, output_dir: Path, progress_callback=None) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        pairs = self.find_pairs(input_dir)
        ok, fail = 0, []
        for i, (a, b) in enumerate(pairs):
            if progress_callback:
                progress_callback(i + 1, len(pairs))
            try:
                r = self.process_pair(a, b)
                r.save(output_dir / f"{a.stem}.png", "PNG")
                ok += 1
            except Exception as e:
                fail.append((a.name, b.name, str(e)))
        return {"ok": ok, "skip": 0, "fail": fail}

    # -- rembg 单管道 ----------------------------------------------------

    def _single_pipe(self, img: Image.Image) -> tuple[Optional[tuple[int, int, int, int]], np.ndarray]:
        """单管道：对比度增强 → rembg → 杆子/支架色彩过滤 → (bbox, mask)。

        rembg 对纯白服装易把金属杆误检为服装。rembg 后用 HSV 滤除
        mask 前景中的深灰 / 低饱和像素（金属杆/支架的典型特征）。
        """
        enhanced = ImageEnhance.Contrast(img).enhance(1.4)
        from rembg import remove
        mask = remove(enhanced, session=self._get_session(), only_mask=True)
        w, h = img.size
        if mask.size != (w, h):
            if mask.size == (h, w):
                mask = mask.transpose(Image.Transpose.TRANSPOSE)
            else:
                mask = mask.resize((w, h), Image.LANCZOS)
        mask_arr = np.array(mask)

        # 杆子/支架色彩过滤：深灰+低饱和 → 从 mask 前景中扣除
        hsv = np.array(img.convert("HSV"), dtype=np.float32)
        rod_mask = (hsv[..., 1] < 50) & (hsv[..., 2] < 160)  # 低饱和 + 中低亮度
        mask_arr[(mask_arr > 30) & rod_mask] = 0

        rows, cols = np.where(mask_arr > 30)
        if len(rows) < 100:
            return None, mask_arr
        bbox = (int(cols.min()), int(rows.min()),
                int(cols.max()), int(rows.max()))
        return bbox, mask_arr

    def _single_pipe_bbox(self, img: Image.Image):
        """向后兼容：只返回 bbox。新代码请用 _single_pipe。"""
        bbox, _ = self._single_pipe(img)
        return bbox

    def _get_mask_arr(self, img: Image.Image) -> np.ndarray:
        """向后兼容：只返回 mask。新代码请用 _single_pipe。"""
        _, mask = self._single_pipe(img)
        return mask

    # -- 联合轮廓分析 ----------------------------------------------------

    @staticmethod
    def mask_centerline_angle(mask_a, mask_b, bbox_a=None, bbox_b=None, debug_cb=None):
        """模板对齐法（5步）。如果 debug_cb(label, img) 传入则每步回调可视化。

        1. 双面 ±0.5°×0.1° 网格搜索，像素重叠最大化 → (angle_a, angle_b)
        2. 模板 = 正面(a) ∩ 翻背面(b) → 共识服装轮廓
        3. 模板 Theil-Sen 中轴倾角 → 摆正模板
        4. 正面独立搜索：与正模板重叠最大的角 → angle_a
        5. 翻背面独立搜索：与正模板重叠最大的角 → angle_b

        返回 (angle_a, angle_b)，精度 0.1°。
        """
        PA, PB = ImageProcessorV11._prepare_aligned(mask_a, mask_b, bbox_a, bbox_b)

        def _rot(mask_bool, deg):
            h, w = mask_bool.shape
            if abs(deg) < 0.005:
                return (mask_bool > 0.5).astype(np.float32)
            rad = np.radians(deg); c = np.cos(rad); sx = np.sin(rad)
            cx, cy = w / 2.0, h / 2.0
            ys, xs = np.where(mask_bool > 0.5)
            nx = np.round((xs - cx) * c - (ys - cy) * sx + cx).astype(int)
            ny = np.round((xs - cx) * sx + (ys - cy) * c + cy).astype(int)
            pl, pr = max(0, -nx.min()), max(0, nx.max() - w + 1)
            pt, pb = max(0, -ny.min()), max(0, ny.max() - h + 1)
            out = np.zeros((h + pt + pb, w + pl + pr), dtype=np.float32)
            out[ny + pt, nx + pl] = 1.0
            return out

        def _overlap(m1, m2):
            h = min(m1.shape[0], m2.shape[0])
            w = min(m1.shape[1], m2.shape[1])
            return int((m1[:h, :w] * m2[:h, :w]).sum())

        def _tilt(mask):
            ys, xs = np.where(mask > 0.5)
            if len(ys) < 30: return 0.0
            mid = {}
            for y in np.unique(ys):
                row = xs[ys == y]
                if len(row) >= 5: mid[y] = (row[0] + row[-1]) * 0.5
            if len(mid) < 20: return 0.0
            my = np.array(list(mid.keys()), dtype=float)
            mx = np.array([mid[int(y)] for y in my], dtype=float)
            n = len(my); g = max(1, n // 3); slopes = []
            for i in range(0, n - g, max(1, n // 100)):
                dy = my[i + g] - my[i]
                if dy > 0: slopes.append((mx[i + g] - mx[i]) / dy)
            return np.degrees(np.arctan(float(np.median(slopes)))) if slopes else 0.0

        # === Step 1: 双面网格搜索 ===
        fa = (PA > 0.5).astype(np.float32)
        fb = (PB > 0.5).astype(np.float32)
        baseline = _overlap(fa, fb)
        angles = np.arange(-0.5, 0.51, 0.1)
        best_a = 0.0; best_b = 0.0; best_ov = baseline
        # collect grid for debug heatmap
        score_grid = np.zeros((len(angles), len(angles)), dtype=int)

        for ai, a in enumerate(angles):
            ra = _rot(PA > 0.5, a)
            for bi, b in enumerate(angles):
                rb = _rot(PB > 0.5, b)
                ov = _overlap(ra, rb)
                score_grid[ai, bi] = ov
                if ov > best_ov:
                    best_ov = ov; best_a = float(a); best_b = float(b)

        if best_ov <= baseline:
            return 0.0, 0.0

        # Debug: step 1 heatmap
        if debug_cb:
            debug_cb(f"S1 网格搜索 (best A:{best_a:+.1f} B:{best_b:+.1f} ov:{best_ov}) baseline={baseline}",
                     ImageProcessorV11._debug_score_grid(angles, score_grid, best_a, best_b))

        # === Step 2: 模板 ===
        ra_opt = _rot(PA > 0.5, best_a)
        rb_opt = _rot(PB > 0.5, best_b)
        h_t = min(ra_opt.shape[0], rb_opt.shape[0])
        w_t = min(ra_opt.shape[1], rb_opt.shape[1])
        tpl = ((ra_opt[:h_t, :w_t] > 0.5) & (rb_opt[:h_t, :w_t] > 0.5)).astype(np.float32)
        if int(tpl.sum()) < 50:
            return 0.0, 0.0

        # Debug: step 2 alignment + step 3 template
        if debug_cb:
            debug_cb(f"S2 对齐重叠 (A:{best_a:+.1f} B:{best_b:+.1f}) green=front blue=flipped_back yellow=overlap",
                     ImageProcessorV11._debug_mirror_overlay(ra_opt, rb_opt, best_a, best_b))
            debug_cb(f"S3 模板mask (交集={int(tpl.sum())}px) yellow=template green=front_only blue=back_only",
                     ImageProcessorV11._debug_template_only(ra_opt[:h_t, :w_t], rb_opt[:h_t, :w_t], tpl))

        # === Step 3: 摆正模板 ===
            for b in angles:
                rb = _rot(PB > 0.5, b)
                ov = _overlap(ra, rb)
                if ov > best_ov: best_ov = ov; best_a = float(a); best_b = float(b)

        if best_ov <= baseline:
            return 0.0, 0.0

        # --- Step 2: 模板 = 最优角下的交集 ---
        ra_opt = _rot(PA > 0.5, best_a)
        rb_opt = _rot(PB > 0.5, best_b)
        h_t = min(ra_opt.shape[0], rb_opt.shape[0])
        w_t = min(ra_opt.shape[1], rb_opt.shape[1])
        tpl = ((ra_opt[:h_t, :w_t] > 0.5) & (rb_opt[:h_t, :w_t] > 0.5)).astype(np.float32)
        if int(tpl.sum()) < 50:
            return 0.0, 0.0

        # --- Step 3: 模板中轴倾角 → 摆正 ---
        tpl_tilt = _tilt(tpl)
        tpl_up = _rot(tpl > 0.5, -tpl_tilt) if abs(tpl_tilt) >= 0.05 else tpl

        if debug_cb:
            debug_cb(f"S4 摆正模板 (tilt={tpl_tilt:+.1f}°) 黄色=摆正后的模板mask",
                     ImageProcessorV11._debug_upright_template(tpl_up))

        # --- Step 4: 正面独立搜索（与正模板重叠最大）---
        a_curve = []
        a_best = 0.0; a_max = 0
        for a in angles:
            s = _overlap(_rot(PA > 0.5, a), tpl_up)
            a_curve.append((float(a), s))
            if s > a_max: a_max = s; a_best = float(a)

        # --- Step 5: 翻背面独立搜索（与正模板重叠最大）---
        b_curve = []
        b_best = 0.0; b_max = 0
        for b in angles:
            s = _overlap(_rot(PB > 0.5, b), tpl_up)
            b_curve.append((float(b), s))
            if s > b_max: b_max = s; b_best = float(b)

        if debug_cb:
            debug_cb(f"S5 独立搜索曲线 (best A:{a_best:+.1f}° B:{b_best:+.1f}°)",
                     ImageProcessorV11._debug_search_curves(a_curve, b_curve, a_best, b_best))

        return round(a_best * 10) / 10, round(b_best * 10) / 10

    @staticmethod
    def tilt_theil_sen(mask_a, mask_b, bbox_a=None, bbox_b=None):
        """Theil-Sen + 3σ 剔除 — bbox 内逐行中点拟合，删异常行重算。
        与模板对齐法互补：向量空间法 PK 像素重叠法。
        返回 (angle_a, angle_b)，精度 0.1°。"""
        def _ts_tilt_robust(mask, bb):
            if bb is None: return 0.0
            x1, y1, x2, y2 = [int(v) for v in bb]
            mid_xs, mid_ys = [], []
            for yi in range(y1, min(y2, mask.shape[0])):
                cols = np.where(mask[yi] > 30)[0]
                if len(cols) >= 5:
                    mid_xs.append((cols[0] + cols[-1]) * 0.5)
                    mid_ys.append(yi)
            if len(mid_ys) < 30: return 0.0
            mx = np.array(mid_xs, dtype=float); my = np.array(mid_ys, dtype=float)
            n = len(my); g = max(1, n // 3); slopes = []
            for i in range(0, n - g, max(1, n // 100)):
                dy = my[i + g] - my[i]
                if dy > 0: slopes.append((mx[i + g] - mx[i]) / dy)
            if not slopes: return 0.0
            a = float(np.median(slopes))
            ym = my.mean(); xm = mx.mean()
            res = np.abs(mx - (xm + a * (my - ym)))
            keep = res <= res.std() * 3.0
            if keep.sum() < 20 or keep.all(): return np.degrees(np.arctan(a))
            ix, iy = mx[keep], my[keep]
            ni = len(iy); gi = max(1, ni // 3); slopes2 = []
            for j in range(0, ni - gi, max(1, ni // 100)):
                dy = iy[j + gi] - iy[j]
                if dy > 0: slopes2.append((ix[j + gi] - ix[j]) / dy)
            if not slopes2: return np.degrees(np.arctan(a))
            return np.degrees(np.arctan(float(np.median(slopes2))))
        a = -_ts_tilt_robust(mask_a, bbox_a) if bbox_a else 0.0
        b = -_ts_tilt_robust(mask_b, bbox_b) if bbox_b else 0.0
        return round(a * 10) / 10, round(b * 10) / 10

    @staticmethod
    def _vertical_profile(mask_arr):
        h = mask_arr.shape[0]
        ys, ls, rs = [], [], []
        for y in range(h):
            c = np.where(mask_arr[y] > 30)[0]
            if len(c) > 10:
                ys.append(y); ls.append(c.min()); rs.append(c.max())
        return (np.array(ys), np.array(ls), np.array(rs)) if ys else (np.array([]), np.array([]), np.array([]))

    @staticmethod
    def _largest_consensus_interval(y, mask):
        bs, be, bl = y[0], y[-1], 0
        cs = None
        for i, v in enumerate(mask):
            if v and cs is None:
                cs = y[i]
            elif not v and cs is not None:
                l = y[i - 1] - cs
                if l > bl:
                    bl, bs, be = l, cs, y[i - 1]
                cs = None
        if cs is not None and y[-1] - cs > bl:
            bs, be = cs, y[-1]
        return bs, be

    @staticmethod
    def _bbox_in_range(mask_arr, y_min, y_max):
        h = mask_arr.shape[0]
        yi, ya = max(0, int(y_min)), min(h, int(y_max) + 1)
        if ya <= yi:
            return None
        r, c = np.where(mask_arr[yi:ya, :] > 30)
        if len(r) < 50:
            return None
        return (int(c.min()), yi + int(r.min()),
                min(mask_arr.shape[1], int(c.max())), yi + int(r.max()))

    def _joint_detect(self, img_a, img_b):
        """Returns (bbox_a, bbox_b, mask_a, mask_b) — masks kept for angle detection."""
        bbox_a, mask_a = self._single_pipe(img_a)
        bbox_b, mask_b = self._single_pipe(img_b)

        ys_a, lefts_a, rights_a = self._vertical_profile(mask_a)
        ys_b, lefts_b, rights_b = self._vertical_profile(mask_b)

        if len(ys_a) >= 20 and len(ys_b) >= 20:
            wa = rights_a - lefts_a
            wb = rights_b - lefts_b
            y_min = max(ys_a.min(), ys_b.min())
            y_max = min(ys_a.max(), ys_b.max())

            if y_max > y_min:
                uh = y_max - y_min
                uy = np.linspace(y_min, y_max, uh)
                wi_a = np.interp(uy, ys_a, wa.astype(float))
                wi_b = np.interp(uy, ys_b, wb.astype(float))
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.maximum(wi_a, wi_b) / np.maximum(np.minimum(wi_a, wi_b), 1)
                consensus = ratio < CONSENSUS_RATIO_THRESHOLD
                cy_min, cy_max = self._largest_consensus_interval(uy, consensus)

                if cy_max - cy_min >= 50:
                    j_bbox_a = self._bbox_in_range(mask_a, cy_min, cy_max)
                    j_bbox_b = self._bbox_in_range(mask_b, cy_min, cy_max)
                    if j_bbox_a:
                        bbox_a = j_bbox_a
                    if j_bbox_b:
                        bbox_b = j_bbox_b

        # v11.1: 杆子底部裁剪
        if bbox_a:
            bbox_a = self._trim_rod_bottom(img_a, mask_a, bbox_a)
        if bbox_b:
            bbox_b = self._trim_rod_bottom(img_b, mask_b, bbox_b)

        return bbox_a, bbox_b, mask_a, mask_b

    def _joint_detect_debug(self, img_a, img_b, angle_mode="theilsen"):
        """联合检测 + 收集每一步的调试图像。
        angle_mode: "theilsen" | "template" — 决定角度算法的调试输出。
        """
        debug = []

        bbox_a, mask_a = self._single_pipe(img_a)
        bbox_b, mask_b = self._single_pipe(img_b)

        # 1. AI 分割：rembg (u2net) -> Mask
        debug.append(("① AI分割(rembg) A", self._debug_mask_overlay(img_a, mask_a)))
        debug.append(("① AI分割(rembg) B", self._debug_mask_overlay(img_b, mask_b)))

        # 2. Mask -> BBox
        if bbox_a:
            debug.append(("② 初步BBox A", self._debug_bbox_overlay(img_a, bbox_a, color=(255, 165, 0))))
        if bbox_b:
            debug.append(("② 初步BBox B", self._debug_bbox_overlay(img_b, bbox_b, color=(255, 165, 0))))

        # 3. CV 联合轮廓分析：宽度分布 + 共识区间
        ys_a, lefts_a, rights_a = self._vertical_profile(mask_a)
        ys_b, lefts_b, rights_b = self._vertical_profile(mask_b)

        if len(ys_a) >= 20 and len(ys_b) >= 20:
            wa = rights_a - lefts_a
            wb = rights_b - lefts_b
            y_min = max(ys_a.min(), ys_b.min())
            y_max = min(ys_a.max(), ys_b.max())

            if y_max > y_min:
                uh = y_max - y_min
                uy = np.linspace(y_min, y_max, uh)
                wi_a = np.interp(uy, ys_a, wa.astype(float))
                wi_b = np.interp(uy, ys_b, wb.astype(float))
                with np.errstate(divide='ignore', invalid='ignore'):
                    ratio = np.maximum(wi_a, wi_b) / np.maximum(np.minimum(wi_a, wi_b), 1)
                consensus = ratio < CONSENSUS_RATIO_THRESHOLD
                cy_min, cy_max = self._largest_consensus_interval(uy, consensus)

                debug.append(("③ CV宽度分布&共识区间",
                              self._debug_profile_chart(uy, wi_a, wi_b, ratio, consensus, cy_min, cy_max)))

                if cy_max - cy_min >= 50:
                    j_bbox_a = self._bbox_in_range(mask_a, cy_min, cy_max)
                    j_bbox_b = self._bbox_in_range(mask_b, cy_min, cy_max)
                    if j_bbox_a:
                        debug.append(("④ CV共识提炼(橙=前绿=后) A",
                                      self._debug_rod_compare(img_a, bbox_a, j_bbox_a)))
                        bbox_a = j_bbox_a
                    if j_bbox_b:
                        debug.append(("④ CV共识提炼(橙=前绿=后) B",
                                      self._debug_rod_compare(img_b, bbox_b, j_bbox_b)))
                        bbox_b = j_bbox_b

        # 5. 杆子裁剪前后对比
        if bbox_a:
            rod_bbox_a = self._trim_rod_bottom(img_a, mask_a, bbox_a)
            if rod_bbox_a != bbox_a:
                debug.append(("⑤ CV杆子裁剪(橙=前绿=后) A", self._debug_rod_compare(img_a, bbox_a, rod_bbox_a)))
            bbox_a = rod_bbox_a
        if bbox_b:
            rod_bbox_b = self._trim_rod_bottom(img_b, mask_b, bbox_b)
            if rod_bbox_b != bbox_b:
                debug.append(("⑤ CV杆子裁剪(橙=前绿=后) B", self._debug_rod_compare(img_b, bbox_b, rod_bbox_b)))
            bbox_b = rod_bbox_b

        # 6. 自动角度 (依 angle_mode 切换)
        angle_a = angle_b = 0.0
        if bbox_a and bbox_b:
            if angle_mode == "template":
                self._debug_angle_template(debug, img_a, img_b, mask_a, mask_b, bbox_a, bbox_b)
            else:
                self._debug_angle_theilsen(debug, img_a, img_b, mask_a, mask_b, bbox_a, bbox_b)

        # 7. 最终 bbox
        if bbox_a:
            debug.append(("⑦ 最终结果 A", self._debug_bbox_overlay(img_a, bbox_a, color=(0, 255, 0))))
        if bbox_b:
            debug.append(("⑦ 最终结果 B", self._debug_bbox_overlay(img_b, bbox_b, color=(0, 255, 0))))

        return bbox_a, bbox_b, debug


    # -- 调试可视化辅助方法 ----------------------------------------------

    @staticmethod
    def _debug_mask_overlay(img, mask_arr):
        """原图上叠加半透明绿色 mask。"""
        overlay = img.copy().convert("RGBA")
        green = np.array([0, 180, 0, 90], dtype=np.uint8)
        mask_bool = mask_arr > 30
        arr = np.array(overlay)
        arr[mask_bool] = ((arr[mask_bool].astype(np.uint16) * 0.5 +
                           green.astype(np.uint16) * 0.5).clip(0, 255).astype(np.uint8))
        out = Image.fromarray(arr).convert("RGB")
        w, h = out.size
        out = out.resize((500, int(500 * h / w)), Image.LANCZOS)
        return out

    @staticmethod
    def _debug_bbox_overlay(img, bbox, color=(0, 255, 0)):
        """原图上画 bbox 矩形框，返回 400px 宽缩略图。"""
        from PIL import ImageDraw
        out = img.copy()
        draw = ImageDraw.Draw(out)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, (x2 - x1) // 150))
        w, h = out.size
        out = out.resize((500, int(500 * h / w)), Image.LANCZOS)
        return out

    @staticmethod
    def _debug_rod_compare(img, before_bbox, after_bbox):
        """杆子裁剪前后比较：黄框=前，绿框=后。"""
        from PIL import ImageDraw
        out = img.copy()
        draw = ImageDraw.Draw(out)
        bx1, by1, bx2, by2 = [int(v) for v in before_bbox]
        ax1, ay1, ax2, ay2 = [int(v) for v in after_bbox]
        w = max(2, (bx2 - bx1) // 150)
        draw.rectangle([bx1, by1, bx2, by2], outline=(255, 180, 0), width=w)
        draw.rectangle([ax1, ay1, ax2, ay2], outline=(0, 255, 0), width=w)
        w_i, h_i = out.size
        out = out.resize((500, int(500 * h_i / w_i)), Image.LANCZOS)
        return out

    @staticmethod
    def _debug_profile_chart(uy, wi_a, wi_b, ratio, consensus, cy_min, cy_max):
        """绘制宽度分布 + 共识区间图表（PIL 纯绘图，不依赖 matplotlib）。"""
        from PIL import ImageDraw, ImageFont
        w_img, h_img = 900, 520
        pad_t, pad_b, pad_l, pad_r = 50, 30, 80, 20
        pw = w_img - pad_l - pad_r
        ph = h_img - pad_t - pad_b

        img = Image.new("RGB", (w_img, h_img), (28, 28, 30))
        draw = ImageDraw.Draw(img)

        try:
            font_s = ImageFont.truetype("msyh.ttc", 11)
        except Exception:
            try:
                font_s = ImageFont.truetype("simhei.ttf", 11)
            except Exception:
                font_s = ImageFont.load_default()

        n = len(uy)
        y2px = lambda v: pad_t + int(ph * (v - uy[0]) / (uy[-1] - uy[0] + 1))
        val2px = lambda v, vmin, vmax: pad_l + int(pw * (v - vmin) / max(vmax - vmin, 1))

        w_max = max(wi_a.max(), wi_b.max())
        r_max = max(ratio.max(), CONSENSUS_RATIO_THRESHOLD + 0.5)
        r_min = 1.0

        # 共识区域绿色背景
        for i in range(n - 1):
            if consensus[i]:
                y0, y1 = y2px(uy[i]), y2px(uy[i + 1])
                draw.rectangle([pad_l, y0, pad_l + pw, y1], fill=(35, 65, 35))

        # 共识区间边界绿框
        if cy_max > cy_min:
            yc0, yc1 = y2px(cy_min), y2px(cy_max)
            draw.rectangle([pad_l, yc0, pad_l + pw, yc1], outline=(0, 220, 0), width=3)

        # 宽度曲线：正面（蓝）、反面（红）
        step = max(1, n // 600)
        pts_a = [(val2px(wi_a[i], 0, w_max), y2px(uy[i])) for i in range(0, n, step)]
        pts_b = [(val2px(wi_b[i], 0, w_max), y2px(uy[i])) for i in range(0, n, step)]
        for pts, color, w in [(pts_a, (60, 140, 255), 2), (pts_b, (255, 90, 90), 2)]:
            for j in range(len(pts) - 1):
                draw.line([pts[j][0], pts[j][1], pts[j + 1][0], pts[j + 1][1]],
                          fill=color, width=w)

        # 比率曲线（黄色）
        r_pts = [(val2px(ratio[i], r_min, r_max), y2px(uy[i])) for i in range(0, n, step)]
        for j in range(0, len(r_pts) - 1, 2):
            draw.line([r_pts[j][0], r_pts[j][1], r_pts[j + 1][0], r_pts[j + 1][1]],
                      fill=(255, 230, 80), width=1)

        # 阈值线（黄色虚线）
        thr_x = val2px(CONSENSUS_RATIO_THRESHOLD, r_min, r_max)
        for yy in range(pad_t, pad_t + ph, 12):
            draw.line([thr_x, yy, thr_x, min(yy + 6, pad_t + ph)],
                      fill=(255, 230, 80), width=1)

        # Y 轴刻度标记
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            yy = pad_t + int(ph * frac)
            draw.line([pad_l - 4, yy, pad_l, yy], fill=(150, 150, 150))

        # 图例
        lx, ly = pad_l + 8, 8
        draw.rectangle([lx, ly, lx + 14, ly + 10], fill=(60, 140, 255))
        draw.text((lx + 18, ly - 1), "front width(px)", fill=(180, 180, 180), font=font_s)
        draw.rectangle([lx + 160, ly, lx + 174, ly + 10], fill=(255, 90, 90))
        draw.text((lx + 178, ly - 1), "back width(px)", fill=(180, 180, 180), font=font_s)
        draw.rectangle([lx + 340, ly, lx + 354, ly + 10], fill=(255, 230, 80))
        draw.text((lx + 358, ly - 1), "width ratio max/min", fill=(180, 180, 180), font=font_s)
        draw.text((lx + 490, ly - 1),
                  f"threshold={CONSENSUS_RATIO_THRESHOLD}", fill=(255, 230, 80), font=font_s)
        draw.rectangle([lx, ly + 14, lx + 14, ly + 24], fill=(35, 65, 35))
        draw.text((lx + 18, ly + 13), "consensus (ratio<1.35)", fill=(180, 180, 180), font=font_s)

        return img

    @staticmethod
    def _prepare_aligned(mask_a, mask_b, bbox_a, bbox_b):
        """为 mask_a 和 mask_b 做 crop+flip+pad，返回 (PA, PB)。"""
        def _crop_pad(mask, bbox):
            if bbox:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                bw, bh = x2 - x1, y2 - y1
                px, py = int(bw * 0.15), int(bh * 0.15)
                x1 = max(0, x1 - px); x2 = min(mask.shape[1], x2 + px)
                y1 = max(0, y1 - py); y2 = min(mask.shape[0], y2 + py)
                return mask[y1:y2, x1:x2]
            return mask
        ma = _crop_pad(mask_a, bbox_a)
        mb = _crop_pad(mask_b, bbox_b)
        flipped = mb[:, ::-1]
        target_h = 400
        sa = max(1, ma.shape[0] // target_h)
        sb = max(1, flipped.shape[0] // target_h)
        small_a = (ma[::sa, ::sa] > 30)
        small_f = (flipped[::sb, ::sb] > 30)
        H = max(small_a.shape[0], small_f.shape[0])
        W = max(small_a.shape[1], small_f.shape[1])
        pad_h, pad_w = H // 2, W // 2
        HH, WW = H + pad_h * 2, W + pad_w * 2
        PA = np.zeros((HH, WW), dtype=np.float32)
        PB = np.zeros((HH, WW), dtype=np.float32)
        dya = (HH - small_a.shape[0]) // 2; dxa = (WW - small_a.shape[1]) // 2
        dyb = (HH - small_f.shape[0]) // 2; dxb = (WW - small_f.shape[1]) // 2
        PA[dya:dya + small_a.shape[0], dxa:dxa + small_a.shape[1]] = small_a.astype(np.float32)
        PB[dyb:dyb + small_f.shape[0], dxb:dxb + small_f.shape[1]] = small_f.astype(np.float32)
        return PA, PB

    @staticmethod
    def _render_mask_pair(ra, rb, text):
        """通用：绿=正面，蓝=翻转背面，黄=重叠。返回 350px 高 PIL Image。"""
        from PIL import ImageDraw
        H = max(ra.shape[0], rb.shape[0]); W = max(ra.shape[1], rb.shape[1])
        disp_h = min(H, 350); disp_w = int(disp_h * W / H) if H > 0 else 350
        yi = np.linspace(0, H - 1, disp_h).astype(int)
        xi = np.linspace(0, W - 1, disp_w).astype(int)
        img_out = Image.new("RGB", (disp_w, disp_h + 26), (30, 30, 30))
        draw = ImageDraw.Draw(img_out)
        for yd in range(disp_h):
            for xd in range(0, disp_w, 2):
                a_on = ra[yi[yd], xi[xd]] > 0.5 if yi[yd] < ra.shape[0] and xi[xd] < ra.shape[1] else False
                b_on = rb[yi[yd], xi[xd]] > 0.5 if yi[yd] < rb.shape[0] and xi[xd] < rb.shape[1] else False
                if a_on and b_on: draw.point((xd, yd), fill=(255, 200, 60))
                elif a_on: draw.point((xd, yd), fill=(60, 180, 60))
                elif b_on: draw.point((xd, yd), fill=(60, 120, 255))
        draw.text((4, disp_h + 6), text, fill=(200, 200, 200))
        return img_out

    @staticmethod
    def _debug_score_grid(angles, grid, best_a, best_b):
        """S1: 重叠热力图 — 行=A角度，列=B角度，亮=高重叠，圆=最优。"""
        from PIL import ImageDraw
        n = len(angles); cell = 50; pad = 60
        w = pad + n * cell + 200; h = pad + n * cell + 40
        img = Image.new("RGB", (w, h), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        vmin, vmax = grid.min(), grid.max()
        for i in range(n):
            for j in range(n):
                t = (grid[i, j] - vmin) / max(vmax - vmin, 1)
                c = int(40 + t * 200)
                color = (c, c, min(255, c + 60))
                x0, y0, x1, y1 = pad + j * cell, pad + i * cell, pad + (j + 1) * cell - 4, pad + (i + 1) * cell - 4
                draw.rectangle([x0, y0, x1, y1], fill=color)
                draw.text((x0 + 16, y0 + 14), str(grid[i, j]), fill=(255, 255, 255))
        # axis labels
        for i in range(n):
            draw.text((pad + i * cell + 14, pad + n * cell + 4), f"{angles[i]:+.1f}", fill=(200, 200, 200))
            draw.text((pad - 40, pad + i * cell + 14), f"{angles[i]:+.1f}", fill=(200, 200, 200))
        draw.text((pad + n * cell // 2 - 30, pad + n * cell + 22), "B:", fill=(120, 180, 255))
        draw.text((pad - 58, pad + n * cell // 2 - 10), "A:", fill=(120, 255, 120))
        # mark best position
        ai = int(np.abs(angles - best_a).argmin()); bi = int(np.abs(angles - best_b).argmin())
        cx, cy = pad + bi * cell + cell // 2, pad + ai * cell + cell // 2
        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10], fill=None, outline=(255, 255, 0), width=3)
        draw.text((pad + n * cell + 8, 4), f"best: A={best_a:+.1f} B={best_b:+.1f}", fill=(255, 255, 0))
        return img

    @staticmethod
    def _debug_mirror_overlay(ra, rb, a, b):
        """S2: 对齐重叠图 — 正面(绿) + 翻转背面(蓝) + 重叠(黄)。"""
        return ImageProcessorV11._render_mask_pair(ra, rb,
            f"对齐重叠 (A:{a:+.1f} B:{b:+.1f}) green=front blue=flipped_back yellow=overlap")

    @staticmethod
    def _debug_template_only(ra, rb, tpl):
        """S3: 模板mask = 交集(黄)，仅正面=绿，仅背面=蓝。"""
        H = max(ra.shape[0], rb.shape[0]); W = max(ra.shape[1], rb.shape[1])
        h_t, w_t = tpl.shape
        pa = np.zeros((H, W), dtype=np.float32); pb = np.zeros((H, W), dtype=np.float32)
        pt = np.zeros((H, W), dtype=np.float32)
        pa[:h_t, :w_t] = ra[:h_t, :w_t]; pb[:h_t, :w_t] = rb[:h_t, :w_t]; pt[:h_t, :w_t] = tpl
        from PIL import ImageDraw
        disp_h = min(H, 350); disp_w = int(disp_h * W / H) if H > 0 else 350
        yi = np.linspace(0, H - 1, disp_h).astype(int)
        xi = np.linspace(0, W - 1, disp_w).astype(int)
        img_out = Image.new("RGB", (disp_w, disp_h + 26), (30, 30, 30))
        draw = ImageDraw.Draw(img_out)
        for yd in range(disp_h):
            for xd in range(0, disp_w, 2):
                t = pt[yi[yd], xi[xd]] > 0.5 if yi[yd] < pt.shape[0] and xi[xd] < pt.shape[1] else False
                a = pa[yi[yd], xi[xd]] > 0.5 if yi[yd] < pa.shape[0] and xi[xd] < pa.shape[1] else False
                b = pb[yi[yd], xi[xd]] > 0.5 if yi[yd] < pb.shape[0] and xi[xd] < pb.shape[1] else False
                if t: draw.point((xd, yd), fill=(255, 200, 60))
                elif a: draw.point((xd, yd), fill=(60, 180, 60))
                elif b: draw.point((xd, yd), fill=(60, 120, 255))
        draw.text((4, disp_h + 6), f"模板={int(tpl.sum())}px yellow=template green=front_only blue=back_only", fill=(200, 200, 200))
        return img_out

    @staticmethod
    def _debug_upright_template(tpl_up):
        """S4: 摆正后的模板mask — 白色区域就是共识服装轮廓。"""
        from PIL import ImageDraw
        H, W = tpl_up.shape
        disp_h = min(H, 350); disp_w = int(disp_h * W / H) if H > 0 else 350
        yi = np.linspace(0, H - 1, disp_h).astype(int)
        xi = np.linspace(0, W - 1, disp_w).astype(int)
        img_out = Image.new("RGB", (disp_w, disp_h + 26), (30, 30, 30))
        draw = ImageDraw.Draw(img_out)
        for yd in range(disp_h):
            for xd in range(0, disp_w, 2):
                if yi[yd] < tpl_up.shape[0] and xi[xd] < tpl_up.shape[1] and tpl_up[yi[yd], xi[xd]] > 0.5:
                    draw.point((xd, yd), fill=(255, 200, 60))
        draw.text((4, disp_h + 6), f"摆正模板={int(tpl_up.sum())}px yellow=upright_template", fill=(200, 200, 200))
        return img_out

    @staticmethod
    def _debug_search_curves(a_curve, b_curve, a_best, b_best):
        """S5: 搜索曲线 — 红=A独立搜索 蓝=B独立搜索 圆标记=最优角。"""
        from PIL import ImageDraw
        w, h = 600, 350; pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 40
        img = Image.new("RGB", (w, h), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        pw = w - pad_l - pad_r; ph = h - pad_t - pad_b
        all_v = [v for _, v in a_curve + b_curve]
        vmin, vmax = min(all_v), max(all_v)
        if vmax <= vmin: vmax = vmin + 1
        # A curve (red)
        for curve, color in [(a_curve, (255, 100, 100)), (b_curve, (100, 150, 255))]:
            pts = []
            for ang, val in curve:
                x = pad_l + int(pw * (ang + 0.5) / 1.0)
                y = pad_t + int(ph * (1 - (val - vmin) / (vmax - vmin)))
                pts.append((x, y))
            for j in range(len(pts) - 1):
                draw.line([pts[j][0], pts[j][1], pts[j + 1][0], pts[j + 1][1]], fill=color, width=2)
        # Mark best
        for ang_best, curve, color in [(a_best, a_curve, (255, 100, 100)), (b_best, b_curve, (100, 150, 255))]:
            idx = min(range(len(curve)), key=lambda k: abs(curve[k][0] - ang_best))
            x = pad_l + int(pw * (curve[idx][0] + 0.5) / 1.0)
            y = pad_t + int(ph * (1 - (curve[idx][1] - vmin) / (vmax - vmin)))
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=color)
        draw.text((pad_l, pad_t + ph + 6),
                  f"red=A: {a_best:+.1f}° blue=B: {b_best:+.1f}°", fill=(200, 200, 200))
        return img


    def _debug_angle_template(self, debug, img_a, img_b, mask_a, mask_b, bbox_a, bbox_b):
        """模板对齐法多步调试：S1网格→S2重叠→S3模板→S4摆正→S5曲线。"""
        tpl_steps = []
        aa, ab = ImageProcessorV11.mask_centerline_angle(
            mask_a, mask_b, bbox_a, bbox_b,
            debug_cb=lambda label, img: tpl_steps.append((label, img)))
        debug.extend(tpl_steps)

    def _debug_angle_theilsen(self, debug, img_a, img_b, mask_a, mask_b, bbox_a, bbox_b):
        """Theil-Sen 中轴法调试：逐行中点+拟合线可视化。"""
        aa, ab = ImageProcessorV11.tilt_theil_sen(mask_a, mask_b, bbox_a, bbox_b)
        debug.append((f"Theil-Sen A (corr={aa:+.1f}deg)",
                      ImageProcessorV11._debug_theilsen_chart(img_a, mask_a, bbox_a, -aa)))
        debug.append((f"Theil-Sen B (corr={ab:+.1f}deg)",
                      ImageProcessorV11._debug_theilsen_chart(img_b, mask_b, bbox_b, -ab)))

    @staticmethod
    def _debug_theilsen_chart(img, mask, bbox, tilt_deg):
        """TS辅助：mask(绿) + 逐行中点(黄) + Theil-Sen拟合线(绿) + 垂参(红)。"""
        from PIL import ImageDraw
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1: return img
        w_c, h_c = x2 - x1, y2 - y1
        dw, dh = 500, max(1, int(500 * h_c / w_c))
        sx, sy = dw / max(w_c, 1), dh / max(h_c, 1)
        crop = img.crop((x1, y1, x2, y2)).convert('RGBA')
        arr = np.array(crop)
        for yi in range(h_c):
            cols = np.where(mask[y1 + yi] > 30)[0]
            if cols.size >= 5:
                xl, xr = max(0, cols[0] - x1), min(w_c, cols[-1] - x1 + 1)
                arr[yi, int(xl):int(xr), :] = ((arr[yi, int(xl):int(xr), :].astype(np.uint16) * 0.6 +
                    np.array([0, 160, 0, 255], dtype=np.uint16) * 0.4).clip(0, 255).astype(np.uint8))
        out = Image.fromarray(arr, 'RGBA').convert('RGB').resize((dw, dh), Image.LANCZOS)
        draw = ImageDraw.Draw(out)
        mid_pts = [(float((c2[0] + c2[-1]) * 0.5), float(yi)) for yi in range(y1, y2)
                   if len(c2 := np.where(mask[yi] > 30)[0]) >= 5]
        if len(mid_pts) < 20: return out
        step = max(1, len(mid_pts) // 80)
        for i in range(0, len(mid_pts), step):
            px, py = int((mid_pts[i][0] - x1) * sx), int((mid_pts[i][1] - y1) * sy)
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 255, 100))
        if abs(tilt_deg) > 0.001:
            mx = np.array([p[0] for p in mid_pts]); my = np.array([p[1] for p in mid_pts])
            ym = my.mean(); xm = mx.mean()
            a = np.tan(np.radians(tilt_deg))
            ty, by = my[0], my[-1]
            draw.line([int((xm + a*(ty - ym) - x1)*sx), int((ty - y1)*sy),
                       int((xm + a*(by - ym) - x1)*sx), int((by - y1)*sy)],
                      fill=(0, 255, 200), width=2)
        cx = (x1 + x2) / 2
        draw.line([int((cx - x1)*sx), 0, int((cx - x1)*sx), dh], fill=(255, 80, 80), width=1)
        draw.text((4, 2), f'Theil-Sen: tilt={tilt_deg:+.1f} green=fitted red=vertical yellow=midpoints', fill=(255, 255, 255))
        return out

    @staticmethod
    def _debug_fusion_summary(aa, ab, ta, tb, fa, fb):
        """融合摘要：模板对齐 vs Theil-Sen 对比表。"""
        from PIL import ImageDraw
        img = Image.new("RGB", (600, 200), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        headers = [("方法", (80, 40)), ("A 面角度", (250, 40)), ("B 面角度", (420, 40))]
        for (txt, (x, y)), c in zip(headers, [(255,255,255), (120,255,120), (120,180,255)]):
            draw.text((x, y), txt, fill=c)
        rows = [
            ("模板对齐", aa, ab, (255, 200, 100) if abs(aa - fa) < 0.3 else (255, 100, 100)),
            ("Theil-Sen", ta, tb, (255, 200, 100) if abs(ta - fa) < 0.3 else (255, 100, 100)),
        ]
        for ri, (label, a_v, b_v, color) in enumerate(rows):
            y = 80 + ri * 40
            draw.text((80, y), label, fill=(200, 200, 200))
            draw.text((250, y), f"{a_v:+.1f}", fill=color)
            draw.text((420, y), f"{b_v:+.1f}", fill=color)
        draw.line([80, 120, 520, 120], fill=(100, 100, 100), width=1)
        draw.text((80, 140), f"融合 → A: {fa:+.1f}°  B: {fb:+.1f}° (一致=平均 分歧=取TS)", fill=(255, 255, 0))
        return img



    def _trim_rod_bottom(self, img, mask_arr, bbox):
        """基于 mask 宽度占比裁剪杆子/人台底部。

        杆子区域: mask宽度 < bbox宽度的12% (细杆/人台腿)。
        服装区域: mask宽度通常占bbox宽度的40%+。
        """
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        if bh < 200:
            return bbox

        bw = x2 - x1
        if bw < 30:
            return bbox

        # 逐行计算 mask 宽度占比
        rows_data = []
        for y in range(y1, y2):
            c = np.where(mask_arr[y] > 30)[0]
            if len(c) >= 5:
                rows_data.append((y, (c.max() - c.min()) / bw))
            else:
                rows_data.append((y, 0))

        if len(rows_data) < 40:
            return bbox

        # 平滑（小窗口，保持边界敏感）
        n = len(rows_data)
        win = max(n // 60, 2)
        ys_arr = [rows_data[i][0] for i in range(n)]
        ratios = [rows_data[i][1] for i in range(n)]
        smoothed = np.zeros(n)
        for i in range(n):
            lo, hi = max(0, i - win), min(n, i + win + 1)
            smoothed[i] = np.mean(ratios[lo:hi])

        # 计算身体参考宽度：上半部中位数
        body_ref = float(np.median(smoothed[:n // 2]))
        if body_ref < 0.15:
            return bbox  # 整体都窄，不裁剪

        # 从上半部往下扫描：找宽度骤降点（杆子/人台腿开始处）
        # 方法：取局部 1/8 区域的宽度中位数，和 body_ref 比较
        seg_len = max(n // 20, 5)
        cutoff_y = y2
        for i in range(n // 2, n - seg_len):
            seg_median = float(np.median(ratios[i:i + seg_len]))
            if seg_median < body_ref * 0.20 and seg_median < 0.10:
                # 这个区段显著窄，检查下方是否也持续窄
                below_median = float(np.median(ratios[i:]))
                if below_median < body_ref * 0.25:
                    cutoff_y = ys_arr[i]
                    break

        if cutoff_y < y2 and y2 - cutoff_y > bh * 0.05 and cutoff_y - y1 >= 150:
            return (x1, y1, x2, cutoff_y)
        return bbox

    # -- 裁切 -------------------------------------------------------------

    @staticmethod
    def _classify(ch, img_h):
        ratio = ch / img_h
        for mp, mg, fl in CATEGORIES:
            if ratio <= mp:
                return mg, fl
        return 0.04, 0.95

    def _unified_crop_w(self, img_a, bbox_a, img_b, bbox_b):
        def natural_cw(img, bbox):
            if bbox is None:
                return img.size[1] // 2
            x1, y1, x2, y2 = bbox
            bw = x2 - x1
            bh = y2 - y1
            _, fl = self._classify(bh, img.size[1])
            crop_h = min(int(bh / fl), img.size[1])
            crop_h += crop_h % 2
            cw2 = crop_h // 2
            if cw2 < bw * 1.05:
                cw2 = int(bw / 0.85)
                if cw2 % 2:
                    cw2 += 1
            return cw2
        cw_a = natural_cw(img_a, bbox_a)
        cw_b = natural_cw(img_b, bbox_b)
        return max(cw_a, cw_b)

    def _crop_1x2(self, img, bbox, anchor, forced_cw=None):
        w, h = img.size
        if bbox is None:
            cw = forced_cw if forced_cw else h // 2
            return img.crop((max(0, (w - cw) // 2), 0,
                             max(0, (w - cw) // 2) + cw, h))

        x1, y1, x2, y2 = bbox
        cw, ch = x2 - x1, y2 - y1
        cy = (y1 + y2) // 2
        mr, fl = self._classify(ch, h)

        if forced_cw:
            crop_w = forced_cw
            crop_h = crop_w * 2
            if crop_h > h:
                crop_h = h - (h % 2)
                crop_w = crop_h // 2
        else:
            crop_h = min(int(ch / fl), h)
            crop_h += crop_h % 2
            crop_w = crop_h // 2
            if crop_w < cw * 1.05:
                crop_w = int(cw / 0.85)
                if crop_w % 2:
                    crop_w += 1
                crop_h = crop_w * 2
                if crop_h > h:
                    crop_h = h - (h % 2)
                    crop_w = crop_h // 2

        margin = int(crop_w * mr)
        for m in [margin, max(margin // 2, 1), 0]:
            ok, cx = self._try_anchor(w, crop_w, x1, x2, anchor, m)
            if ok:
                cy1 = self._fit_vertical(h, cy, crop_h)
                return img.crop((cx, cy1, cx + crop_w, cy1 + crop_h))

        if anchor == "right":
            cx = max(0, w - crop_w)
        else:
            cx = 0
        if cx + crop_w > w:
            crop_w = w - cx
            if crop_w % 2:
                crop_w -= 1
            crop_h = crop_w * 2
        cy1 = self._fit_vertical(h, cy, crop_h)
        return img.crop((cx, cy1, cx + crop_w, cy1 + crop_h))

    @staticmethod
    def _try_anchor(iw, cw, cx1, cx2, a, m):
        if a == "right":
            x = min(iw, cx2 + m) - cw
        else:
            x = max(0, cx1 - m)
        if x < 0:
            return False, 0
        if x + cw > iw and a == "left":
            return False, 0
        return True, x

    @staticmethod
    def _fit_vertical(ih, cy, ch):
        ch = min(ch, ih)
        y = cy - ch // 2
        if y < 0:
            y = 0
        if y + ch > ih:
            y = ih - ch
        return y

    @staticmethod
    def _stitch(left, right):
        th = min(left.height, right.height)
        th += th % 2
        hw = th // 2
        left = left.resize((hw, th), Image.LANCZOS)
        right = right.resize((hw, th), Image.LANCZOS)
        c = Image.new("RGB", (th, th), (255, 255, 255))
        c.paste(left, (0, 0))
        c.paste(right, (hw, 0))
        return c
