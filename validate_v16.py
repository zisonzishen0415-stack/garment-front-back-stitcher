"""验证 v16 mask清洗 vs 用户标注"""
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from processor import ImageProcessor

data = json.load(open("素材/annotations.json", encoding="utf-8"))
gt = {a["file"]: a for a in data["annotations"]}

SRC = Path("素材/7-21p图")
REF = Path("素材/人台")
processor = ImageProcessor(mannequin_ref_dir=REF)

print(f"{'图片':<16} {'标注y1':>6} {'v16 y1':>6} {'差':>6}  "
      f"{'标注y2':>6} {'v16 y2':>6} {'差':>6}  {'判定':>6}")
print("-" * 78)

ok_t = ok_b = total = 0
te, be = [], []

for fn in sorted(gt):
    p = SRC / fn
    if not p.exists(): continue
    ann = gt[fn]
    g1, g2 = ann["bbox"][1], ann["bbox"][3]

    img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
    ma = processor._get_mask_arr(img)
    mc = processor._clean_mask(ma)
    # 在清洗后的 mask 上检测 bbox
    bb = processor._detect_bbox_simple(mc)
    if bb is None:
        bb = processor._detect_bbox_simple(ma)
        if bb is None: continue

    v1, v2 = bb[1], bb[3]
    ih = img.size[1]
    total += 1
    dt, db = v1 - g1, v2 - g2
    tol = max(60, ih * 0.03)
    tok = abs(dt) < tol
    bok = abs(db) < tol
    if tok: ok_t += 1
    else: te.append((fn, dt))
    if bok: ok_b += 1
    else: be.append((fn, db))
    v = "✓" if (tok and bok) else "△" if (tok or bok) else "✗"
    print(f"{fn:<16} {g1:>6} {v1:>6} {dt:+6d}  {g2:>6} {v2:>6} {db:+6d}  {v:>6}")

print(f"\n顶部: {ok_t}/{total} ({ok_t/total*100:.0f}%)")
print(f"底部: {ok_b}/{total} ({ok_b/total*100:.0f}%)")
print(f"综合: {ok_t+ok_b}/{total*2} ({(ok_t+ok_b)/(total*2)*100:.0f}%)")

if te:
    print(f"\n顶部异常 ({len(te)}):")
    for f, d in sorted(te, key=lambda x: -abs(x[1]))[:12]:
        print(f"  {f}: {d:+d}px")
if be:
    print(f"\n底部异常 ({len(be)}):")
    for f, d in sorted(be, key=lambda x: -abs(x[1]))[:12]:
        print(f"  {f}: {d:+d}px")
