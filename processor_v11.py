"""v11 处理器：单管道 rembg + 联合轮廓共识匹配（简洁版）

与 v22+ 的区别：
- 单管道（不做多管道预处理/投票）
- 共识区间直接作为 bbox（不做 trim-only 约束）
- 无顶部/底部宽度裁剪
- 无 PAIR_HEIGHT_RATIO_MAX
"""

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

    def _get_session(self):
        if self._session is None:
            from rembg import new_session
            self._session = new_session()
        return self._session

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

        bbox_a, bbox_b = self._joint_detect(img_a, img_b)

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
        """v11: 单管道 rembg bbox + 联合轮廓共识匹配 + 杆子底部裁剪。

        每张图只跑一次 rembg（_single_pipe 同时返回 bbox + mask），
        共识区间直接作为最终 bbox，不做 trim-only 约束。
        """
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

        # v11.1: 杆子底部裁剪 — 用原图色彩区分服装 vs 人台/杆子
        if bbox_a:
            bbox_a = self._trim_rod_bottom(img_a, mask_a, bbox_a)
        if bbox_b:
            bbox_b = self._trim_rod_bottom(img_b, mask_b, bbox_b)

        return bbox_a, bbox_b

    def _joint_detect_debug(self, img_a, img_b):
        """联合检测 + 收集每一步的调试图像。

        Returns: (bbox_a, bbox_b, debug_entries)
          debug_entries: list of (label: str, image: Image.Image)
        """
        debug = []

        bbox_a, mask_a = self._single_pipe(img_a)
        bbox_b, mask_b = self._single_pipe(img_b)

        # 1. 初步检测：rembg 分割 → Mask
        debug.append(("① rembg 分割 A", self._debug_mask_overlay(img_a, mask_a)))
        debug.append(("① rembg 分割 B", self._debug_mask_overlay(img_b, mask_b)))

        # 2. 从 Mask 提取 BBox
        if bbox_a:
            debug.append(("② 初步 BBox A", self._debug_bbox_overlay(img_a, bbox_a, color=(255, 165, 0))))
        if bbox_b:
            debug.append(("② 初步 BBox B", self._debug_bbox_overlay(img_b, bbox_b, color=(255, 165, 0))))

        # 3. 联合轮廓匹配：宽度分布 + 共识区间
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

                debug.append(("③ 宽度分布 & 共识区间",
                              self._debug_profile_chart(uy, wi_a, wi_b, ratio, consensus, cy_min, cy_max)))

                if cy_max - cy_min >= 50:
                    j_bbox_a = self._bbox_in_range(mask_a, cy_min, cy_max)
                    j_bbox_b = self._bbox_in_range(mask_b, cy_min, cy_max)
                    # 快照 consensus 提炼前 → 后对比
                    if j_bbox_a:
                        debug.append(("④ 共识提炼 A (橙=前 绿=后)",
                                      self._debug_rod_compare(img_a, bbox_a, j_bbox_a)))
                        bbox_a = j_bbox_a
                    if j_bbox_b:
                        debug.append(("④ 共识提炼 B (橙=前 绿=后)",
                                      self._debug_rod_compare(img_b, bbox_b, j_bbox_b)))
                        bbox_b = j_bbox_b

        # 5. 杆子裁剪前后对比
        if bbox_a:
            rod_bbox_a = self._trim_rod_bottom(img_a, mask_a, bbox_a)
            if rod_bbox_a != bbox_a:
                debug.append(("⑤ 杆子裁剪 A (橙=前 绿=后)", self._debug_rod_compare(img_a, bbox_a, rod_bbox_a)))
            bbox_a = rod_bbox_a
        if bbox_b:
            rod_bbox_b = self._trim_rod_bottom(img_b, mask_b, bbox_b)
            if rod_bbox_b != bbox_b:
                debug.append(("⑤ 杆子裁剪 B (橙=前 绿=后)", self._debug_rod_compare(img_b, bbox_b, rod_bbox_b)))
            bbox_b = rod_bbox_b

        # 6. 最终 bbox
        if bbox_a:
            debug.append(("⑥ 最终结果 A", self._debug_bbox_overlay(img_a, bbox_a, color=(0, 255, 0))))
        if bbox_b:
            debug.append(("⑥ 最终结果 B", self._debug_bbox_overlay(img_b, bbox_b, color=(0, 255, 0))))

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
        from PIL import ImageDraw
        w_img, h_img = 700, 400
        pad_t, pad_b, pad_l, pad_r = 40, 30, 60, 30
        pw = w_img - pad_l - pad_r
        ph = h_img - pad_t - pad_b

        img = Image.new("RGB", (w_img, h_img), (30, 30, 30))
        draw = ImageDraw.Draw(img)

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
                draw.rectangle([pad_l, y0, pad_l + pw, y1], fill=(40, 70, 40))

        # 共识区间绿框
        if cy_max > cy_min:
            yc0, yc1 = y2px(cy_min), y2px(cy_max)
            draw.rectangle([pad_l, yc0, pad_l + pw, yc1], outline=(0, 200, 0), width=3)

        # 宽度曲线 A（蓝）、B（红）
        step = max(1, n // 500)
        pts_a = [(val2px(wi_a[i], 0, w_max), y2px(uy[i])) for i in range(0, n, step)]
        pts_b = [(val2px(wi_b[i], 0, w_max), y2px(uy[i])) for i in range(0, n, step)]
        for pts, color in [(pts_a, (0, 150, 255)), (pts_b, (255, 100, 100))]:
            for j in range(len(pts) - 1):
                draw.line([pts[j][0], pts[j][1], pts[j + 1][0], pts[j + 1][1]], fill=color, width=2)

        # 比率曲线（黄色虚线）
        r_pts = [(val2px(ratio[i], r_min, r_max), y2px(uy[i])) for i in range(0, n, step)]
        for j in range(0, len(r_pts) - 1, 2):
            draw.line([r_pts[j][0], r_pts[j][1], r_pts[j + 1][0], r_pts[j + 1][1]],
                      fill=(255, 255, 100), width=1)

        # 阈值线
        thr_x = val2px(CONSENSUS_RATIO_THRESHOLD, r_min, r_max)
        # 阈值线（手绘虚线：每 6px 画 6px 段，兼容低版本 Pillow）
        for yy in range(pad_t, pad_t + ph, 12):
            draw.line([thr_x, yy, thr_x, min(yy + 6, pad_t + ph)],
                      fill=(255, 255, 100), width=1)

        # 图例
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("consola.ttf", 12)
        except Exception:
            font = ImageFont.load_default()
        draw.text((5, pad_t + 5), "W", fill=(0, 150, 255), font=font)
        draw.text((5, pad_t + 20), "B", fill=(255, 100, 100), font=font)
        draw.text((5, pad_t + 35), "R", fill=(255, 255, 100), font=font)
        draw.text((w_img - 55, pad_t + 5), f"thr={CONSENSUS_RATIO_THRESHOLD}", fill=(255, 255, 100), font=font)

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
