"""验证 v15 人台形状匹配 vs 用户标注"""
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from processor import ImageProcessor

data = json.load(open("素材/annotations.json", encoding="utf-8"))
ground_truth = {a["file"]: a for a in data["annotations"]}

SRC = Path("素材/7-21p图")
REF = Path("素材/人台")
processor = ImageProcessor(mannequin_ref_dir=REF)

annotated_files = sorted(ground_truth.keys())

print(f"{'图片':<16} {'标注y1':>6} {'v15 y1':>6} {'差':>6}  "
      f"{'标注y2':>6} {'v15 y2':>6} {'差':>6}  {'判定':>8}")
print("-" * 80)

ok_top = 0
ok_bot = 0
total = 0
top_errs = []
bot_errs = []

for fname in annotated_files:
    path = SRC / fname
    if not path.exists():
        continue

    ann = ground_truth[fname]
    gt_y1, gt_y2 = ann["bbox"][1], ann["bbox"][3]

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    mask = processor._get_mask_arr(img)
    simple = processor._detect_bbox_simple(mask)
    if simple is None:
        continue
    v15 = processor._trim_by_mannequin(mask, simple)

    v15_y1, v15_y2 = v15[1], v15[3]
    img_h = img.size[1]
    total += 1

    d_top = v15_y1 - gt_y1
    d_bot = v15_y2 - gt_y2

    tol = max(60, img_h * 0.03)
    top_ok = abs(d_top) < tol
    bot_ok = abs(d_bot) < tol
    if top_ok:
        ok_top += 1
    else:
        top_errs.append((fname, d_top))
    if bot_ok:
        ok_bot += 1
    else:
        bot_errs.append((fname, d_bot))

    v = "✓" if (top_ok and bot_ok) else "△" if (top_ok or bot_ok) else "✗"
    print(f"{fname:<16} {gt_y1:>6} {v15_y1:>6} {d_top:+6d}  "
          f"{gt_y2:>6} {v15_y2:>6} {d_bot:+6d}  {v:>8}")

print(f"\n--- 准确率 ---")
print(f"顶部: {ok_top}/{total} ({ok_top/total*100:.0f}%)")
print(f"底部: {ok_bot}/{total} ({ok_bot/total*100:.0f}%)")
print(f"综合: {ok_top+ok_bot}/{total*2} ({(ok_top+ok_bot)/(total*2)*100:.0f}%)")

if top_errs:
    print(f"\n顶部问题 ({len(top_errs)}):")
    for f, d in sorted(top_errs, key=lambda x: -abs(x[1]))[:15]:
        print(f"  {f}: {d:+d}px")
if bot_errs:
    print(f"\n底部问题 ({len(bot_errs)}):")
    for f, d in sorted(bot_errs, key=lambda x: -abs(x[1]))[:15]:
        print(f"  {f}: {d:+d}px")
