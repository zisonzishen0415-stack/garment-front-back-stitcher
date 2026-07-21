"""验证 v14 裁剪规则 vs 用户标注"""
import json, sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from processor import ImageProcessor

# 加载标注
data = json.load(open("素材/annotations.json", encoding="utf-8"))
ground_truth = {a["file"]: a for a in data["annotations"]}

SRC = Path("素材/7-21p图")
processor = ImageProcessor()

# 只验证有标注的图片对
pairs = processor.find_pairs(SRC)
# 过滤到有标注的
annotated_files = set(ground_truth.keys())

print(f"{'图片':<16} {'标注y1':>6} {'v14 y1':>6} {'差':>5}  "
      f"{'标注y2':>6} {'v14 y2':>6} {'差':>5}  {'判定':>10}")
print("-" * 80)

ok_top = 0
ok_bot = 0
total = 0
top_errors = []
bot_errors = []

for fname in sorted(annotated_files):
    path = SRC / fname
    if not path.exists():
        continue

    ann = ground_truth[fname]
    gt_bbox = ann["bbox"]  # [x1, y1, x2, y2]

    # 运行 v14 的 bbox 检测
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    mask_arr = processor._get_mask_arr(img)

    # 模拟 _joint_detect 后的裁剪（跳过联合匹配，直接测 _trim_narrow_ends）
    simple_bbox = processor._detect_bbox_simple(mask_arr)
    if simple_bbox is None:
        continue
    v14_bbox = processor._trim_narrow_ends(mask_arr, simple_bbox)

    gt_y1, gt_y2 = gt_bbox[1], gt_bbox[3]
    v14_y1, v14_y2 = v14_bbox[1], v14_bbox[3]
    img_h = img.size[1]

    d_top = v14_y1 - gt_y1
    d_bot = v14_y2 - gt_y2

    total += 1

    # 判定：误差 < 50px 或 < 5% 图像高度 算正确
    top_ok = abs(d_top) < max(50, img_h * 0.03)
    bot_ok = abs(d_bot) < max(50, img_h * 0.03)
    if top_ok: ok_top += 1
    if bot_ok: ok_bot += 1
    if not top_ok: top_errors.append((fname, d_top))
    if not bot_ok: bot_errors.append((fname, d_bot))

    verdict = "✓" if (top_ok and bot_ok) else "△" if (top_ok or bot_ok) else "✗"

    print(f"{fname:<16} {gt_y1:>6} {v14_y1:>6} {d_top:+5d}  "
          f"{gt_y2:>6} {v14_y2:>6} {d_bot:+5d}  {verdict:>10}")

print(f"\n顶部准确率: {ok_top}/{total} ({ok_top/total*100:.0f}%)")
print(f"底部准确率: {ok_bot}/{total} ({ok_bot/total*100:.0f}%)")
print(f"综合准确率: {ok_top+ok_bot}/{total*2} ({(ok_top+ok_bot)/(total*2)*100:.0f}%)")

if top_errors:
    print(f"\n顶部误差大 ({len(top_errors)}):")
    for f, d in sorted(top_errors, key=lambda x: -abs(x[1]))[:10]:
        print(f"  {f}: {d:+d}px")
if bot_errors:
    print(f"\n底部误差大 ({len(bot_errors)}):")
    for f, d in sorted(bot_errors, key=lambda x: -abs(x[1]))[:10]:
        print(f"  {f}: {d:+d}px")
