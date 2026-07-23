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
        """单管道：对比度增强 → rembg → (bbox, mask)。

        v11 用对比度增强做 rembg，对暗色衣物/人台分离效果较好。
        一次 rembg 推理同时产出 bbox 和 mask，避免重复调用。
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
    def mask_centerline_angle(mask_a, mask_b, bbox_a=None, bbox_b=None):
        """Theil-Sen + 3σ 剔除 — bbox 内逐行中点拟合，删异常行重算。"""
        def _ts_tilt_robust(mask, bb):
            if bb is None:
                return 0.0
            x1, y1, x2, y2 = [int(v) for v in bb]
            mid_xs, mid_ys = [], []
            for yi in range(y1, min(y2, mask.shape[0])):
                cols = np.where(mask[yi] > 30)[0]
                if len(cols) >= 5:
                    mid_xs.append((cols[0] + cols[-1]) * 0.5)
                    mid_ys.append(yi)
            if len(mid_ys) < 30:
                return 0.0
            mx = np.array(mid_xs, dtype=float)
            my = np.array(mid_ys, dtype=float)

            # Theil-Sen 中位数斜率
            n = len(my); g = max(1, n // 3)
            slopes = []
            for i in range(0, n - g, max(1, n // 100)):
                dy = my[i + g] - my[i]
                if dy > 0: slopes.append((mx[i + g] - mx[i]) / dy)
            if not slopes:
                return 0.0
            a = float(np.median(slopes))

            # 3σ 剔除异常行
            ym = my.mean(); xm = mx.mean()
            res = np.abs(mx - (xm + a * (my - ym)))
            keep = res <= res.std() * 3.0
            if keep.sum() < 20 or keep.all():
                return np.degrees(np.arctan(a))

            # 仅内点重算
            ix, iy = mx[keep], my[keep]
            ni = len(iy); gi = max(1, ni // 3)
            slopes2 = []
            for j in range(0, ni - gi, max(1, ni // 100)):
                dy = iy[j + gi] - iy[j]
                if dy > 0: slopes2.append((ix[j + gi] - ix[j]) / dy)
            if not slopes2:
                return np.degrees(np.arctan(a))
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

    def _joint_detect_debug(self, img_a, img_b):
        """联合检测 + 收集每一步的调试图像。

        Returns: (bbox_a, bbox_b, debug_entries)
          debug_entries: list of (label: str, image: Image.Image)
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

        # 6. Theil-Sen 中轴线 — 逐行中点 + 3sigma 剔除
        angle_a = angle_b = 0.0
        if bbox_a and bbox_b:
            angle_a, angle_b = ImageProcessorV11.mask_centerline_angle(
                mask_a, mask_b, bbox_a, bbox_b)
        if bbox_a and mask_a is not None:
            # -angle = raw tilt (correction = -raw_tilt)
            debug.append((f"⑥ Theil-Sen A (tilt={-angle_a:+.1f}° corr={angle_a:+.1f}°) green=fitted red=vertical",
                          ImageProcessorV11._debug_ts_chart(img_a, mask_a, bbox_a, -angle_a)))
        if bbox_b and mask_b is not None:
            debug.append((f"⑥ Theil-Sen B (tilt={-angle_b:+.1f}° corr={angle_b:+.1f}°) green=fitted red=vertical",
                          ImageProcessorV11._debug_ts_chart(img_b, mask_b, bbox_b, -angle_b)))

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
    def _debug_ts_chart(img, mask_arr, bbox, angle_deg):
        """Theil-Sen: mask(绿) + 内点(黄) + 剔除点(蓝) + 拟合线(绿) + 垂直(红)。"""
        from PIL import ImageDraw
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1 or angle_deg == 0:
            return ImageProcessorV11._debug_bbox_overlay(img, bbox, color=(0, 255, 0))
        w_c, h_c = x2 - x1, y2 - y1
        dw, dh = 500, max(1, int(500 * h_c / w_c))
        sx, sy = dw / max(w_c, 1), dh / max(h_c, 1)
        crop = img.crop((x1, y1, x2, y2)).convert('RGBA')
        arr = np.array(crop)
        for yi in range(h_c):
            cols = np.where(mask_arr[y1 + yi] > 30)[0]
            if cols.size >= 5:
                xl, xr = max(0, cols[0] - x1), min(w_c, cols[-1] - x1 + 1)
                arr[yi, int(xl):int(xr), :] = ((arr[yi, int(xl):int(xr), :].astype(np.uint16) * 0.6 +
                    np.array([0, 160, 0, 255], dtype=np.uint16) * 0.4).clip(0, 255).astype(np.uint8))
        out = Image.fromarray(arr, 'RGBA').convert('RGB').resize((dw, dh), Image.LANCZOS)
        draw = ImageDraw.Draw(out)
        mid_pts = [(float((c2[0] + c2[-1]) * 0.5), float(yi)) for yi in range(y1, y2)
                   if len(c2 := np.where(mask_arr[yi] > 30)[0]) >= 5]
        if len(mid_pts) < 20:
            return out
        mx = np.array([p[0] for p in mid_pts]); my = np.array([p[1] for p in mid_pts])
        n = len(my); g = max(1, n // 3)
        slopes = [(mx[i+g]-mx[i])/(my[i+g]-my[i]) for i in range(0, n-g, max(1, n//100)) if my[i+g] > my[i]]
        a = float(np.median(slopes)) if slopes else 0.0
        ym, xm = my.mean(), mx.mean()
        res = np.abs(mx - (xm + a * (my - ym)))
        thresh = res.std() * 3.0
        keep = res <= thresh
        step = max(1, n // 80)
        for i in range(0, n, step):
            px, py = int((mid_pts[i][0] - x1) * sx), int((mid_pts[i][1] - y1) * sy)
            c = (255, 255, 100) if keep[i] else (100, 140, 255)
            r = 3 if keep[i] else 2
            draw.ellipse([px - r, py - r, px + r, py + r], fill=c)
        ix, iy = mx[keep], my[keep]
        if len(ix) >= 20:
            ym2, xm2 = iy.mean(), ix.mean()
            a_f = np.tan(np.radians(angle_deg))
            ty, by = iy[0], iy[-1]
            draw.line([int((xm2 + a_f*(ty-ym2) - x1)*sx), int((ty - y1)*sy),
                       int((xm2 + a_f*(by-ym2) - x1)*sx), int((by - y1)*sy)],
                      fill=(0, 255, 200), width=2)
        cx_mid = (x1 + x2) / 2
        draw.line([int((cx_mid - x1)*sx), 0, int((cx_mid - x1)*sx), dh], fill=(255, 80, 80), width=1)
        n_in, n_out = int(keep.sum()), n - int(keep.sum())
        draw.text((4, 2), f'TS: {angle_deg:+.1f} in={n_in}/{n} out={n_out} 3sig={thresh:.0f}px yellow=kept blue=rej green=fitted red=vertical', fill=(255, 255, 255))
        return out



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
