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

    def _single_pipe_bbox(self, img: Image.Image) -> Optional[tuple[int, int, int, int]]:
        """单管道：原始图 → rembg → bbox（v11: 只用对比度增强，效果最好）"""
        # v11 用对比度增强做 rembg，对暗色衣物/人台分离效果较好
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
            return None
        return (int(cols.min()), int(rows.min()),
                int(cols.max()), int(rows.max()))

    def _get_mask_arr(self, img: Image.Image) -> np.ndarray:
        """获取原始图的mask（不增强，用于轮廓分析）"""
        from rembg import remove
        mask = remove(img, session=self._get_session(), only_mask=True)
        w, h = img.size
        if mask.size != (w, h):
            if mask.size == (h, w):
                mask = mask.transpose(Image.Transpose.TRANSPOSE)
            else:
                mask = mask.resize((w, h), Image.LANCZOS)
        return np.array(mask)

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
        """v11: 单管道 rembg bbox + 简单共识匹配。

        共识区间直接作为最终 bbox，不做 trim-only 约束。
        """
        bbox_a = self._single_pipe_bbox(img_a)
        bbox_b = self._single_pipe_bbox(img_b)

        # 获取原始 mask 跑轮廓分析
        mask_a = self._get_mask_arr(img_a)
        mask_b = self._get_mask_arr(img_b)

        ys_a, lefts_a, rights_a = self._vertical_profile(mask_a)
        ys_b, lefts_b, rights_b = self._vertical_profile(mask_b)

        if len(ys_a) < 20 or len(ys_b) < 20:
            return bbox_a, bbox_b

        wa = rights_a - lefts_a
        wb = rights_b - lefts_b
        y_min = max(ys_a.min(), ys_b.min())
        y_max = min(ys_a.max(), ys_b.max())

        if y_max <= y_min:
            return bbox_a, bbox_b

        uh = y_max - y_min
        uy = np.linspace(y_min, y_max, uh)
        wi_a = np.interp(uy, ys_a, wa.astype(float))
        wi_b = np.interp(uy, ys_b, wb.astype(float))
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.maximum(wi_a, wi_b) / np.maximum(np.minimum(wi_a, wi_b), 1)
        consensus = ratio < CONSENSUS_RATIO_THRESHOLD
        cy_min, cy_max = self._largest_consensus_interval(uy, consensus)

        if cy_max - cy_min < 50:
            return bbox_a, bbox_b

        j_bbox_a = self._bbox_in_range(mask_a, cy_min, cy_max)
        j_bbox_b = self._bbox_in_range(mask_b, cy_min, cy_max)

        # v11: 共识结果直接用，不做 trim-only
        if j_bbox_a:
            bbox_a = j_bbox_a
        if j_bbox_b:
            bbox_b = j_bbox_b

        return bbox_a, bbox_b

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
