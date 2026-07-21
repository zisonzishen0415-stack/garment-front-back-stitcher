"""直接用用户手动标注的bbox裁剪拼接 —— 简单直接，不做多余计算。
缺失标注的用AI补。
输出到 素材/手动选择参考/
"""
import json
from pathlib import Path
from PIL import Image, ImageOps
from processor import ImageProcessor

MARGIN = 0.12  # bbox每边加12%边距

with open("素材/annotations.json", encoding='utf-8') as f:
    ann_map = {a["file"]: a["bbox"] for a in json.load(f)["annotations"]}

# AI补漏
processor = ImageProcessor()

# 找配对
input_dir = Path("素材/7-21p图")
exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in exts], key=lambda f: f.name)
pairs = [(files[i], files[i + 1]) for i in range(0, len(files) - 1, 2)]

output_dir = Path("素材/手动选择参考")
output_dir.mkdir(parents=True, exist_ok=True)

def simple_crop(img, bbox, anchor, crop_w):
    """简单裁剪: 以bbox为中心, 1:2比例, anchor控制左右贴边方向."""
    w, h = img.size
    x1, y1, x2, y2 = bbox
    bcx = (x1 + x2) / 2
    bcy = (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1

    # crop_h = crop_w * 2 (1:2比例)
    # 确保crop_h能覆盖bbox高度
    crop_h = crop_w * 2
    if crop_h < bh * (1 + MARGIN):
        crop_h = int(bh * (1 + MARGIN))
        crop_h += crop_h % 2
        crop_w = crop_h // 2

    # 水平位置: 按anchor方向
    if anchor == "right":
        # bbox右边留margin, 从右边取crop_w
        right = min(w, int(x2 + bw * MARGIN))
        cx = right - crop_w
    else:
        # bbox左边留margin, 从左边取crop_w
        left = max(0, int(x1 - bw * MARGIN))
        cx = left

    # 贴边约束
    if cx < 0:
        cx = 0
    if cx + crop_w > w:
        cx = w - crop_w

    # 垂直居中于bbox中心
    cy = int(bcy - crop_h / 2)
    if cy < 0:
        cy = 0
    if cy + crop_h > h:
        cy = h - crop_h

    return img.crop((cx, cy, cx + crop_w, cy + crop_h))


def stitch(left, right):
    """拼接: 统一缩放到相同尺寸, 拼成1:1正方形."""
    th = min(left.height, right.height)
    th += th % 2
    hw = th // 2
    left = left.resize((hw, th), Image.LANCZOS)
    right = right.resize((hw, th), Image.LANCZOS)
    c = Image.new("RGB", (th, th), (255, 255, 255))
    c.paste(left, (0, 0))
    c.paste(right, (hw, 0))
    return c


ok, fail, ai_fallback = 0, [], 0
for pa, pb in pairs:
    try:
        img_a = ImageOps.exif_transpose(Image.open(pa)).convert("RGB")
        img_b = ImageOps.exif_transpose(Image.open(pb)).convert("RGB")

        bbox_a = ann_map.get(pa.name)
        bbox_b = ann_map.get(pb.name)

        # 缺失标注的用AI补充
        if bbox_a is None or bbox_b is None:
            bb_ai_a, bb_ai_b = processor._joint_detect(img_a, img_b)
            if bbox_a is None:
                bbox_a = bb_ai_a; ai_fallback += 1
            if bbox_b is None:
                bbox_b = bb_ai_b; ai_fallback += 1

        if bbox_a is None or bbox_b is None:
            fail.append((pa.name, pb.name, "no bbox"))
            continue

        # 各算natural crop_w
        def natural_w(bbox):
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            cw = int(bw * (1 + MARGIN))
            ch = cw * 2
            if ch < bh * (1 + MARGIN):
                ch = int(bh * (1 + MARGIN))
                ch += ch % 2
                cw = ch // 2
            return cw

        # 统一crop_w = 取较大值
        unified_cw = max(natural_w(bbox_a), natural_w(bbox_b))

        crop_a = simple_crop(img_a, bbox_a, anchor="right", crop_w=unified_cw)
        crop_b = simple_crop(img_b, bbox_b, anchor="left", crop_w=unified_cw)

        # 补到相同宽度
        target_w = max(crop_a.width, crop_b.width)
        target_h = target_w * 2
        if crop_a.size != (target_w, target_h):
            tmp = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            ox, oy = (target_w - crop_a.width) // 2, (target_h - crop_a.height) // 2
            tmp.paste(crop_a, (ox, oy))
            crop_a = tmp
        if crop_b.size != (target_w, target_h):
            tmp = Image.new("RGB", (target_w, target_h), (255, 255, 255))
            ox, oy = (target_w - crop_b.width) // 2, (target_h - crop_b.height) // 2
            tmp.paste(crop_b, (ox, oy))
            crop_b = tmp

        result = stitch(crop_a, crop_b)
        result.save(output_dir / f"{pa.stem}.png", "PNG")
        ok += 1
    except Exception as e:
        fail.append((pa.name, pb.name, str(e)))

print(f"成功: {ok}, 失败: {len(fail)}, AI补漏: {ai_fallback}")
for a, b, e in fail:
    print(f"  FAIL {a} + {b}: {e}")
