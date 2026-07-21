# -*- coding: utf-8 -*-
"""基于用户标注的量化评估 + 问题分析"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from processor import ImageProcessor

ANNOTATIONS = Path("素材/annotations.json")
SRC = Path("素材/7-21p图")

with open(ANNOTATIONS, encoding='utf-8') as f:
    data = json.load(f)
    anns = {a["file"]: a["bbox"] for a in data["annotations"]}

processor = ImageProcessor()

# 对每个标注图片单独评估管道bbox
errors = []
for fname, gt_bbox in anns.items():
    p = SRC / fname
    if not p.exists(): continue
    img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    w, h = img.size

    # 多管道bbox
    bb = processor._multi_pipeline_bbox(img)
    if bb is None: continue

    gt_x1, gt_y1, gt_x2, gt_y2 = gt_bbox
    dy1 = bb[1] - gt_y1
    dy2 = bb[3] - gt_y2
    dy1_norm = dy1 / h
    dy2_norm = dy2 / h

    # 各管道单独结果
    variants = processor._preprocess_pipelines(img)
    pipe_results = []
    for vi, v in enumerate(variants):
        pb = processor._single_pipe_bbox(v)
        if pb:
            pipe_results.append((vi, pb[1], pb[3]))

    # 找到最接近GT的管道
    best_pipe = min(pipe_results, key=lambda p: abs(p[1]-gt_y1) + abs(p[2]-gt_y2))

    errors.append({
        "file": fname,
        "gt_y1": gt_y1, "gt_y2": gt_y2,
        "pipe_y1": bb[1], "pipe_y2": bb[3],
        "dy1": dy1, "dy2": dy2,
        "dy1_norm": dy1_norm, "dy2_norm": dy2_norm,
        "pipes": pipe_results,
        "best_pipe": best_pipe,
        "h": h,
    })

# ============================================================
print("=" * 70)
print("  管道Bbox vs 用户标注 (误差 = 算法 - 标注)")
print("=" * 70)

dy1s = [e["dy1"] for e in errors]
dy2s = [e["dy2"] for e in errors]
print(f"\n样本数: {len(errors)}")
print(f"\nY1误差: mean={np.mean(dy1s):.0f}px med={np.median(dy1s):.0f}px min={min(dy1s):+d}px max={max(dy1s):+d}px")
print(f"  bbox偏上(含人台): {sum(1 for d in dy1s if d < -200)}/{len(errors)}")
print(f"  bbox准(+-200):    {sum(1 for d in dy1s if -200 <= d <= 200)}/{len(errors)}")
print(f"  bbox偏下:         {sum(1 for d in dy1s if d > 200)}/{len(errors)}")

print(f"\nY2误差: mean={np.mean(dy2s):.0f}px med={np.median(dy2s):.0f}px min={min(dy2s):+d}px max={max(dy2s):+d}px")
print(f"  bbox偏低(含腿): {sum(1 for d in dy2s if d > 200)}/{len(errors)}")
print(f"  bbox准(+-200):  {sum(1 for d in dy2s if -200 <= d <= 200)}/{len(errors)}")
print(f"  bbox偏高(切商品):{sum(1 for d in dy2s if d < -200)}/{len(errors)}")

# 管道分歧分析
print(f"\n管道一致性分析:")
agree = 0
diverge = 0
extreme = 0
for e in errors:
    pipes = e["pipes"]
    if len(pipes) >= 2:
        y1s = [p[1] for p in pipes]
        y2s = [p[2] for p in pipes]
        hs = [p[2]-p[1] for p in pipes]
        dev = (np.std(y1s) + np.std(y2s)) / e["h"]
        if dev < 0.06:
            agree += 1
        else:
            diverge += 1
            if min(hs) < max(hs) * 0.8:
                extreme += 1
print(f"  一致: {agree}, 分歧: {diverge}, 极端分歧: {extreme}")

# 管道1优势分析
print(f"\n管道1(对比度增强)优势分析:")
p1_better = 0
p1_worse = 0
for e in errors:
    pipes = e["pipes"]
    if len(pipes) < 2: continue
    p0_err = abs(pipes[0][1]-e["gt_y1"]) + abs(pipes[0][2]-e["gt_y2"])
    p1_err = abs(pipes[1][1]-e["gt_y1"]) + abs(pipes[1][2]-e["gt_y2"])
    if p1_err < p0_err * 0.8: p1_better += 1
    elif p0_err < p1_err * 0.8: p1_worse += 1

print(f"  管道1明显更好(误差<80%管道0): {p1_better}")
print(f"  管道0明显更好(误差<80%管道1): {p1_worse}")

# 列出问题文件
print(f"\n{'='*70}")
print(f"  Y1误差最大 (bbox太高, 含人台头部) - 需要trimTop")
print(f"{'='*70}")
for e in sorted(errors, key=lambda x: x["dy1"])[:20]:
    best_tag = ""
    if e["best_pipe"][0] != 2 and abs(e["best_pipe"][1]-e["gt_y1"]) < abs(e["pipe_y1"]-e["gt_y1"]):
        best_tag = f" -> 最好管{e['best_pipe'][0]}:y1={e['best_pipe'][1]}"
    print(f"  {e['file']}: dy1={e['dy1']:+d} gt_y1={e['gt_y1']} pipe_y1={e['pipe_y1']}{best_tag}")

print(f"\n{'='*70}")
print(f"  Y2误差最大 (bbox太低, 含人台腿) - 需要trimBottom")
print(f"{'='*70}")
for e in sorted(errors, key=lambda x: -x["dy2"])[:20]:
    best_tag = ""
    if e["best_pipe"][0] != 2 and abs(e["best_pipe"][2]-e["gt_y2"]) < abs(e["pipe_y2"]-e["gt_y2"]):
        best_tag = f" -> 最好管{e['best_pipe'][0]}:y2={e['best_pipe'][2]}"
    print(f"  {e['file']}: dy2={e['dy2']:+d} gt_y2={e['gt_y2']} pipe_y2={e['pipe_y2']}{best_tag}")
