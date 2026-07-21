# -*- coding: utf-8 -*-
"""为所有配对生成诊断过程图 — 原图+AI框+mask+结果"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from processor import ImageProcessor, CONSENSUS_RATIO_THRESHOLD

SRC = Path("素材/7-21p图")
OUT = Path("素材/诊断过程")
# 先清空再重建
import shutil
if OUT.exists(): shutil.rmtree(OUT)
OUT.mkdir(parents=True, exist_ok=True)

processor = ImageProcessor()
pairs = processor.find_pairs(SRC)

PIPE_COLORS = [(255, 80, 80), (80, 140, 255), (255, 180, 40)]  # R/G/O
LABELS = ["P0:原图", "P1:对比度", "P2:直方图均衡"]
FINAL_COLOR = (0, 255, 0)  # green

for idx, (pa, pb) in enumerate(pairs):
    print(f"[{idx+1}/{len(pairs)}] {pa.stem} + {pb.stem}")

    img_a = ImageOps.exif_transpose(Image.open(pa)).convert("RGB")
    img_b = ImageOps.exif_transpose(Image.open(pb)).convert("RGB")
    wa, ha = img_a.size
    wb, hb = img_b.size

    # --- 管道bboxes ---
    variants_a = processor._preprocess_pipelines(img_a)
    variants_b = processor._preprocess_pipelines(img_b)
    bboxes_a = [(i, processor._single_pipe_bbox(v)) for i, v in enumerate(variants_a)]
    bboxes_a = [(i, b) for i, b in bboxes_a if b is not None]
    bboxes_b = [(i, processor._single_pipe_bbox(v)) for i, v in enumerate(variants_b)]
    bboxes_b = [(i, b) for i, b in bboxes_b if b is not None]

    # --- Mask + 联合分析 ---
    mask_a = processor._get_mask_arr(img_a)
    mask_b = processor._get_mask_arr(img_b)
    ys_a, lefts_a, rights_a = processor._vertical_profile(mask_a)
    ys_b, lefts_b, rights_b = processor._vertical_profile(mask_b)

    consensus_info = None
    if len(ys_a) >= 20 and len(ys_b) >= 20:
        wa_arr = rights_a - lefts_a
        wb_arr = rights_b - lefts_b
        y_min = max(ys_a.min(), ys_b.min())
        y_max = min(ys_a.max(), ys_b.max())
        if y_max > y_min:
            uh = y_max - y_min
            uy = np.linspace(y_min, y_max, uh)
            wi_a = np.interp(uy, ys_a, wa_arr.astype(float))
            wi_b = np.interp(uy, ys_b, wb_arr.astype(float))
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = np.maximum(wi_a, wi_b) / np.maximum(np.minimum(wi_a, wi_b), 1)
            consensus = ratio < CONSENSUS_RATIO_THRESHOLD
            cy_min, cy_max = processor._largest_consensus_interval(uy, consensus)
            consensus_info = (int(cy_min), int(cy_max), int(consensus.mean() * 100))

    # --- 最终结果 ---
    final_a, final_b = processor._joint_detect(img_a, img_b)

    # ====================================================================
    # 绘制函数
    # ====================================================================
    def draw_diag(img, w, h, bboxes, final_bb, mask_arr, cons_info, label):
        """绘制一张诊断图: 原图叠加bbox"""
        scale = 1200 / max(w, h)
        dw, dh = int(w * scale), int(h * scale)
        canvas = img.resize((dw, dh), Image.LANCZOS).copy()
        draw = ImageDraw.Draw(canvas)

        # 管道bbox
        for vi, (x1, y1, x2, y2) in bboxes:
            color = PIPE_COLORS[vi % 3]
            sx1, sy1 = int(x1 * scale), int(y1 * scale)
            sx2, sy2 = int(x2 * scale), int(y2 * scale)
            draw.rectangle([sx1, sy1, sx2, sy2], outline=color, width=2)
            draw.text((sx1 + 2, sy1 + 2), LABELS[vi], fill=color)

        # 最终bbox (绿色实心粗线)
        if final_bb:
            x1, y1, x2, y2 = final_bb
            sx1, sy1 = int(x1 * scale), int(y1 * scale)
            sx2, sy2 = int(x2 * scale), int(y2 * scale)
            draw.rectangle([sx1 - 1, sy1 - 1, sx2 + 1, sy2 + 1], outline=FINAL_COLOR, width=3)
            # 中心十字
            cx, cy = (sx1 + sx2) // 2, (sy1 + sy2) // 2
            draw.line([(cx - 15, cy), (cx + 15, cy)], fill=FINAL_COLOR, width=3)
            draw.line([(cx, cy - 15), (cx, cy + 15)], fill=FINAL_COLOR, width=3)

        # 共识区间线 (橙色虚线)
        if cons_info:
            cy1, cy2, cpct = cons_info
            scy1, scy2 = int(cy1 * scale), int(cy2 * scale)
            for y_pos in range(scy1, scy2, 25):
                draw.line([(0, y_pos), (dw, y_pos)], fill=(255, 160, 30), width=1)
            # 标注共识率
            draw.text((dw - 180, scy1 + 5),
                      f"CV共识:{cpct}% [{cy1},{cy2}]", fill=(255, 160, 30))

        draw.text((5, 5), label, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        return canvas

    def draw_mask(mask_arr, w, h, final_bb, label):
        """绘制mask + 最终中心线"""
        scale = 1200 / max(w, h)
        dw, dh = int(w * scale), int(h * scale)
        m_img = Image.fromarray(mask_arr).resize((dw, dh), Image.LANCZOS)
        m_arr = np.array(m_img, dtype=float)
        # 热力图: 蓝(0) → 绿(128) → 红(255)
        norm = np.clip(m_arr / 255.0, 0, 1)
        r = (norm * 255).astype(np.uint8)
        g = (np.clip(1 - abs(norm - 0.5) * 2, 0, 1) * 255).astype(np.uint8)
        b = (np.clip(1 - norm, 0, 1) * 200).astype(np.uint8)
        heat = np.stack([r, g, b], axis=-1)
        canvas = Image.fromarray(heat)
        draw = ImageDraw.Draw(canvas)

        # 最终bbox叠加
        if final_bb:
            x1, y1, x2, y2 = final_bb
            sx1, sy1 = int(x1 * scale), int(y1 * scale)
            sx2, sy2 = int(x2 * scale), int(y2 * scale)
            draw.rectangle([sx1 - 1, sy1 - 1, sx2 + 1, sy2 + 1], outline=FINAL_COLOR, width=3)
            # 中心线
            cx = (sx1 + sx2) // 2
            draw.line([(cx, sy1), (cx, sy2)], fill=FINAL_COLOR, width=2)

        draw.text((5, 5), label, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
        return canvas

    # 生成
    da = draw_diag(img_a, wa, ha, bboxes_a, final_a, mask_a, consensus_info, f"A:{pa.stem}")
    db = draw_diag(img_b, wb, hb, bboxes_b, final_b, mask_b, consensus_info, f"B:{pb.stem}")
    ma = draw_mask(mask_a, wa, ha, final_a, f"mask A:{pa.stem}")
    mb = draw_mask(mask_b, wb, hb, final_b, f"mask B:{pb.stem}")

    # 最终拼接结果
    result = processor.process_pair(pa, pb)
    res_h = 600
    rw, rh = result.size
    res_rs = result.resize((int(rw * res_h / rh), res_h), Image.LANCZOS)

    # --- 组装大图 (3列 x 2行 + 结果行) ---
    cells = [da, ma, res_rs, db, mb, res_rs]
    # 统一每行高度
    row1_h = max(da.height, ma.height, res_rs.height)
    row2_h = max(db.height, mb.height, res_rs.height)
    pad = 10

    # 标准化到行高
    def fix_h(img_obj, th):
        iw, ih = img_obj.size
        tw = int(iw * th / ih)
        return img_obj.resize((tw, th), Image.LANCZOS)

    da2 = fix_h(da, row1_h)
    ma2 = fix_h(ma, row1_h)
    rr2 = fix_h(res_rs, row1_h)
    db2 = fix_h(db, row2_h)
    mb2 = fix_h(mb, row2_h)
    rr3 = fix_h(res_rs, row2_h)

    cw = max(da2.width, db2.width)
    mw2 = max(ma2.width, mb2.width)
    rw2 = max(rr2.width, rr3.width)
    total_w = cw + mw2 + rw2 + pad * 2
    total_h = row1_h + row2_h + pad + 40

    canvas = Image.new("RGB", (total_w, total_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)

    # 行1
    canvas.paste(da2, (0, 20))
    canvas.paste(ma2, (cw + pad, 20))
    canvas.paste(rr2, (cw + mw2 + pad * 2, 20))
    # 行2
    canvas.paste(db2, (0, row1_h + pad + 20))
    canvas.paste(mb2, (cw + pad, row1_h + pad + 20))
    canvas.paste(rr3, (cw + mw2 + pad * 2, row1_h + pad + 20))

    # 列标题
    draw.text((cw // 2 - 40, 2), "AI BBox 分析", fill=(200, 200, 200))
    draw.text((cw + pad + mw2 // 2 - 40, 2), "AI Mask", fill=(200, 200, 200))
    draw.text((cw + mw2 + pad * 2 + rw2 // 2 - 40, 2), "拼接结果", fill=(200, 200, 200))

    # 保存
    out_path = OUT / f"{idx+1:02d}_{pa.stem}.png"
    canvas.save(out_path, "PNG", optimize=True)
    print(f"  -> {out_path.name}")

print(f"\n完成! 共{len(pairs)}对 → {OUT}/")
