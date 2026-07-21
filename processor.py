"""图像处理核心：多管道并行 + 置信度投票 + 正反向约束

v23 — v22基础上修复左右放大系数不一致：
正反面使用统一的crop_w，确保stitch时缩放比例完全相同。
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

# 置信度参数
BBOX_AGREE_THRESHOLD = 0.06   # y1/y2 的 std/图像高度 < 此值 → 一致
PAIR_HEIGHT_RATIO_MAX = 1.18  # 正反面bbox高度比不能超过此值


class ImageProcessor:

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

        # 统一crop_w: 正反面用同样的放大系数，取较大值保证两边都能容纳商品
        unified_cw = self._unified_crop_w(img_a, bbox_a, img_b, bbox_b)

        crop_a = self._crop_1x2(img_a, bbox_a, anchor="right", forced_cw=unified_cw)
        crop_b = self._crop_1x2(img_b, bbox_b, anchor="left", forced_cw=unified_cw)

        # 如果回退路径导致宽度不一致, 补齐到相同宽度(保持放大系数一致)
        if crop_a.width != crop_b.width:
            target_w = max(crop_a.width, crop_b.width)
            target_h = target_w * 2
            crop_a = self._pad_to_size(crop_a, target_w, target_h)
            crop_b = self._pad_to_size(crop_b, target_w, target_h)

        return self._stitch(crop_a, crop_b)

    def process_all(self, input_dir: Path, output_dir: Path, progress_callback=None) -> dict:
        output_dir.mkdir(parents=True, exist_ok=True)
        pairs = self.find_pairs(input_dir)
        ok, fail, skip = 0, [], 0
        for i, (a, b) in enumerate(pairs):
            if progress_callback:
                progress_callback(i + 1, len(pairs))
            try:
                r = self.process_pair(a, b)
                r.save(output_dir / f"{a.stem}.png", "PNG")
                ok += 1
            except Exception as e:
                fail.append((a.name, b.name, str(e)))
        return {"ok": ok, "skip": skip, "fail": fail}

    # -- 多管道预处理 ----------------------------------------------------

    @staticmethod
    def _preprocess_pipelines(img: Image.Image) -> list[Image.Image]:
        """生成3个预处理变体。"""
        variants = []

        # 管道0: 原始
        variants.append(img)

        # 管道1: 对比度增强 (对暗色衣物/人台分离效果最好)
        v2 = ImageEnhance.Contrast(img).enhance(1.4)
        variants.append(v2)

        # 管道2: 直方图均衡
        gray = img.convert("L")
        eq_gray = ImageOps.equalize(gray)
        r, g, b = img.split()
        eq_arr = np.array(eq_gray, dtype=float)
        gray_arr = np.array(gray, dtype=float) + 1
        ratio = eq_arr / gray_arr
        ratio = np.clip(ratio, 0.5, 2.0)

        r_arr = np.clip(np.array(r, dtype=float) * ratio, 0, 255).astype(np.uint8)
        g_arr = np.clip(np.array(g, dtype=float) * ratio, 0, 255).astype(np.uint8)
        b_arr = np.clip(np.array(b, dtype=float) * ratio, 0, 255).astype(np.uint8)

        v3 = Image.merge("RGB", (
            Image.fromarray(r_arr),
            Image.fromarray(g_arr),
            Image.fromarray(b_arr),
        ))
        variants.append(v3)

        return variants

    def _multi_pipeline_bbox(self, img: Image.Image) -> Optional[tuple[int, int, int, int]]:
        """多管道并行获取bbox, 置信度投票。

        v21: 极端分歧时信任最紧管道 (对比度增强对裤子人台分离更好)。
        """
        variants = self._preprocess_pipelines(img)

        bboxes = []
        for v in variants:
            bb = self._single_pipe_bbox(v)
            if bb is not None:
                bboxes.append(bb)

        if len(bboxes) < 2:
            return bboxes[0] if bboxes else None

        h = img.size[1]

        if len(bboxes) == 2:
            y1_diff = abs(bboxes[0][1] - bboxes[1][1]) / h
            y2_diff = abs(bboxes[0][3] - bboxes[1][3]) / h
            if y1_diff < BBOX_AGREE_THRESHOLD and y2_diff < BBOX_AGREE_THRESHOLD:
                return (
                    int((bboxes[0][0] + bboxes[1][0]) / 2),
                    int((bboxes[0][1] + bboxes[1][1]) / 2),
                    int((bboxes[0][2] + bboxes[1][2]) / 2),
                    int((bboxes[0][3] + bboxes[1][3]) / 2),
                )
            else:
                h0 = bboxes[0][3] - bboxes[0][1]
                h1 = bboxes[1][3] - bboxes[1][1]
                return bboxes[1] if h1 < h0 else bboxes[0]

        # 3个管道
        y1s = np.array([b[1] for b in bboxes], dtype=float)
        y2s = np.array([b[3] for b in bboxes], dtype=float)
        heights = y2s - y1s

        y1_dev = np.abs(y1s - np.median(y1s)) / h
        y2_dev = np.abs(y2s - np.median(y2s)) / h
        max_dev = (y1_dev + y2_dev).max()

        if max_dev < BBOX_AGREE_THRESHOLD:
            # 全部一致 → 中位数
            return (
                int(np.median([b[0] for b in bboxes])),
                int(np.median(y1s)),
                int(np.median([b[2] for b in bboxes])),
                int(np.median(y2s)),
            )

        # 极端分歧: 最紧高度 < 最大高度*0.8 → 取最紧的
        # (对比度增强管道对暗色衣物/人台分离效果更好)
        if heights.min() < heights.max() * 0.8 and heights.min() > h * 0.35:
            tight_idx = int(np.argmin(heights))
            return bboxes[tight_idx]

        # 一般分歧 → 取中位数
        mid_idx = int(np.argsort(heights)[1])
        return bboxes[mid_idx]

    def _single_pipe_bbox(self, img: Image.Image) -> Optional[tuple[int, int, int, int]]:
        """单管道: 增强图像 → rembg → bbox。"""
        from rembg import remove
        mask = remove(img, session=self._get_session(), only_mask=True)
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
        return (
            int(cols.min()), int(rows.min()),
            int(cols.max()), int(rows.max()),
        )

    # -- 联合轮廓分析 ----------------------------------------------------

    def _get_mask_arr(self, img: Image.Image) -> np.ndarray:
        from rembg import remove
        mask = remove(img, session=self._get_session(), only_mask=True)
        w, h = img.size
        if mask.size != (w, h):
            if mask.size == (h, w):
                mask = mask.transpose(Image.Transpose.TRANSPOSE)
            else:
                mask = mask.resize((w, h), Image.LANCZOS)
        return np.array(mask)

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
    def _mask_width_data(mask_arr, y1, y2):
        """提取mask在[y1,y2]区间的每行宽度."""
        rows_data = []
        for y in range(y1, y2):
            c = np.where(mask_arr[y] > 30)[0]
            if len(c) >= 5:
                rows_data.append((y, len(c)))
        if len(rows_data) < 20:
            return None, None
        ys_arr = np.array([r[0] for r in rows_data])
        ws_arr = np.array([r[1] for r in rows_data], dtype=float)
        return ys_arr, ws_arr

    @staticmethod
    def _smooth(ws_arr):
        n = len(ws_arr)
        win = max(n // 30, 2)
        sm = np.zeros_like(ws_arr)
        for i in range(n):
            lo, hi = max(0, i - win), min(n, i + win + 1)
            sm[i] = ws_arr[lo:hi].mean()
        return sm

    def _joint_detect(self, img_a, img_b):
        """v22: 多管道bbox + 联合轮廓匹配(仅用于缩小,不扩展)。

        核心策略变更: 共识匹配只能TRIM管道bbox, 绝不扩展。
        管道bbox找的是商品+人台, 共识匹配去掉不对称的噪声。
        """
        # 多管道获取每张图的共识bbox
        bbox_a = self._multi_pipeline_bbox(img_a)
        bbox_b = self._multi_pipeline_bbox(img_b)

        if bbox_a is None and bbox_b is None:
            return None, None

        # 用原始mask跑联合轮廓匹配
        mask_a = self._get_mask_arr(img_a)
        mask_b = self._get_mask_arr(img_b)

        # 先用mask宽度裁剪顶部和底部
        if bbox_a is not None:
            bbox_a = self._trim_narrow_top(mask_a, bbox_a)
            bbox_a = self._trim_narrow_bottom(mask_a, bbox_a)
        if bbox_b is not None:
            bbox_b = self._trim_narrow_top(mask_b, bbox_b)
            bbox_b = self._trim_narrow_bottom(mask_b, bbox_b)

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

        # 共识区间内bbox
        j_bbox_a = self._bbox_in_range(mask_a, cy_min, cy_max)
        j_bbox_b = self._bbox_in_range(mask_b, cy_min, cy_max)

        # --- 合并规则 (v22): 共识只能缩小, 不能扩大 ---
        # 正反面高度一致性 → 共识不可靠则完全忽略
        j_reliable = True
        if j_bbox_a and j_bbox_b:
            jh_a = j_bbox_a[3] - j_bbox_a[1]
            jh_b = j_bbox_b[3] - j_bbox_b[1]
            if max(jh_a, jh_b) > min(jh_a, jh_b) * PAIR_HEIGHT_RATIO_MAX:
                j_reliable = False

        def merge(pipe_bb, j_bb, mask_arr):
            """合并管道bbox和共识bbox: 共识只能trim, 不能扩展."""
            if pipe_bb is None:
                return j_bb
            if j_bb is None or not j_reliable:
                return pipe_bb

            px1, py1, px2, py2 = pipe_bb
            jx1, jy1, jx2, jy2 = j_bb
            h = mask_arr.shape[0]

            # y1: 共识上移(去掉顶部) → 取max; 共识下移 → 检查是否合理
            if jy1 > py1:
                # 共识想把顶部切掉 → 检查被切区域是否有大量mask(商品)
                above_rows = np.sum(mask_arr[py1:jy1, :] > 30, axis=1)
                covered = np.sum(above_rows > 5) / max(jy1 - py1, 1)
                if covered < 0.25:
                    # 被切区域稀疏 → 接受共识的trim
                    ny1 = jy1
                else:
                    # 被切区域有商品 → 保留管道
                    ny1 = py1
            else:
                # 共识想扩展顶部(不应该发生, 但保守处理)
                ny1 = py1

            # y2: 共识下移(去掉底部) → 取min; 共识上移 → 检查
            if jy2 < py2:
                # 共识想把底部切掉 → 检查被切区域
                below_rows = np.sum(mask_arr[jy2:py2, :] > 30, axis=1)
                covered = np.sum(below_rows > 5) / max(py2 - jy2, 1)
                if covered < 0.25:
                    ny2 = jy2
                else:
                    ny2 = py2
            else:
                ny2 = py2

            if ny2 - ny1 < 50:
                return pipe_bb
            # x从管道取(管道x范围更可靠)
            return (px1, ny1, px2, ny2)

        merged_a = merge(bbox_a, j_bbox_a, mask_a)
        merged_b = merge(bbox_b, j_bbox_b, mask_b)

        return merged_a, merged_b

    def _trim_narrow_top(self, mask_arr, bbox):
        """顶部窄区裁剪: 顶部宽度 < 主体宽度45% → 裁剪到衣领。

        人台头部/颈部比衣物主体窄, 通过宽度突变检测。
        """
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        if bh < 100:
            return bbox

        result = self._mask_width_data(mask_arr, y1, y2)
        if result[0] is None:
            return bbox
        ys_arr, ws_arr = result
        sm = self._smooth(ws_arr)
        n = len(sm)

        bw = float(np.median(sm[n // 4:3 * n // 4]))
        if bw < 10:
            return bbox

        # 顶部宽度 < 主体45% → 窄区, 下移
        top_avg = float(sm[:max(n // 20, 3)].mean())
        if top_avg < bw * 0.45:
            new_y1 = y1
            for i in range(n):
                if sm[i] >= bw * 0.60:
                    new_y1 = ys_arr[i]
                    break
            if y2 - new_y1 >= 50:
                return (x1, new_y1, x2, y2)
        return bbox

    @staticmethod
    def _trim_narrow_bottom(mask_arr, bbox):
        """底部窄区裁剪: 底部宽度 < 主体宽度15% → 裁剪到衣摆。"""
        x1, y1, x2, y2 = bbox
        bh = y2 - y1
        if bh < 100:
            return bbox

        # 复用 _mask_width_data 的逻辑但这里需要内联
        rows_data = []
        for y in range(y1, y2):
            c = np.where(mask_arr[y] > 30)[0]
            if len(c) >= 5:
                rows_data.append((y, len(c)))
        if len(rows_data) < 20:
            return bbox

        ys_arr = np.array([r[0] for r in rows_data])
        ws_arr = np.array([r[1] for r in rows_data], dtype=float)

        win = max(len(ws_arr) // 30, 2)
        sm = np.zeros_like(ws_arr)
        for i in range(len(ws_arr)):
            lo, hi = max(0, i - win), min(len(ws_arr), i + win + 1)
            sm[i] = ws_arr[lo:hi].mean()

        n = len(sm)
        bw = float(np.median(sm[n // 4:3 * n // 4]))
        if bw < 10:
            return bbox

        if sm[-1] < bw * 0.15:
            new_y2 = y2
            for i in range(n - 1, -1, -1):
                if sm[i] >= bw * 0.40:
                    new_y2 = ys_arr[i]
                    break
            if new_y2 - y1 >= 50:
                return (x1, y1, x2, new_y2)
        return bbox

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

    # -- 裁切拼接 ---------------------------------------------------------

    @staticmethod
    def _classify(ch, img_h):
        ratio = ch / img_h
        for mp, mg, fl in CATEGORIES:
            if ratio <= mp:
                return mg, fl
        return 0.04, 0.95

    def _unified_crop_w(self, img_a, bbox_a, img_b, bbox_b):
        """计算统一的crop_w，确保正反面放大系数一致。

        正反面各自算natural crop_w，取较大值。
        这样stitch时两边resize到相同尺寸，缩放比例完全相同。
        """
        def natural_cw(img, bbox):
            if bbox is None:
                return img.size[1] // 2
            x1, y1, x2, y2 = bbox
            bw = x2 - x1       # bbox宽度
            bh = y2 - y1       # bbox高度
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
            return img.crop((max(0, (w - cw) // 2), 0, max(0, (w - cw) // 2) + cw, h))

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

        # 回退: 裁剪框贴边
        if anchor == "right":
            cx = max(0, w - crop_w)
        else:
            cx = 0
        # 如果仍然超出, 缩小到图像宽度
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
    def _pad_to_size(img, target_w, target_h):
        """白色填充到目标尺寸，内容居中。"""
        if img.width == target_w and img.height == target_h:
            return img
        result = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        ox = (target_w - img.width) // 2
        oy = (target_h - img.height) // 2
        result.paste(img, (ox, oy))
        return result

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
