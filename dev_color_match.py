"""用参考人台颜色匹配, 在量产图中检测裸露人台行"""
from pathlib import Path
import colorsys
import numpy as np
from PIL import Image, ImageOps
from rembg import remove, new_session

REF_DIR = Path("素材/人台")
SRC_DIR = Path("素材/7-21p图")
session = new_session()

# ========== 1. 提取人台颜色签名 ==========
print("=" * 60)
print("提取人台颜色签名")
print("=" * 60)

all_mannequin_colors = []  # 所有参考人台的前景像素颜色

for f in sorted(REF_DIR.iterdir()):
    if not f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        continue
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    mask = remove(img, session=session, only_mask=True)
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.LANCZOS)
    mask_arr = np.array(mask)
    img_arr = np.array(img)

    fg = mask_arr > 30
    colors = img_arr[fg]  # (N, 3)
    all_mannequin_colors.extend(colors)

    r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]
    print(f"  {f.name}: {len(colors)}px  "
          f"R({r.min()}-{r.max()}) G({g.min()}-{g.max()}) B({b.min()}-{b.max()})")

all_mannequin_colors = np.array(all_mannequin_colors)
ref_mean = all_mannequin_colors.mean(axis=0)
ref_std = all_mannequin_colors.std(axis=0)
print(f"\n人台颜色: mean=({ref_mean[0]:.0f},{ref_mean[1]:.0f},{ref_mean[2]:.0f}) "
      f"std=({ref_std[0]:.0f},{ref_std[1]:.0f},{ref_std[2]:.0f})")

# ========== 2. 在问题图上测试 ==========
print("\n" + "=" * 60)
print("测试: 逐行判定是否为裸露人台")
print("=" * 60)

test_files = ["IMG_6026.JPG", "IMG_6027.JPG",  # 裤子
              "IMG_6035.JPG", "IMG_6036.JPG",  # 裤子
              "IMG_5971.JPG", "IMG_5973.JPG",  # T恤(对照, 无人台露出)
              "IMG_6048.JPG", "IMG_6050.JPG",  # 短裤
              "IMG_5962.JPG", "IMG_5964.JPG"]  # 裤子

COLOR_THRESHOLD = 2.5  # 标准化欧氏距离阈值

for fname in test_files:
    path = SRC_DIR / fname
    if not path.exists():
        continue

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    mask = remove(img, session=session, only_mask=True)
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.LANCZOS)
    mask_arr = np.array(mask)
    img_arr = np.array(img)

    h = mask_arr.shape[0]
    mannequin_rows = 0
    clothing_rows = 0

    # 每10行采样打印
    samples = []

    for y in range(h):
        fg_cols = np.where(mask_arr[y] > 30)[0]
        if len(fg_cols) < 5:
            continue

        # 该行前景像素的颜色
        row_colors = img_arr[y, fg_cols]  # (n, 3)

        # 与参考人台颜色的标准化距离
        diff = (row_colors.astype(float) - ref_mean.reshape(1, 3)) / ref_std.reshape(1, 3)
        dist = np.sqrt((diff ** 2).sum(axis=1)) / np.sqrt(3)

        # 该行中"像人台"的像素占比
        mannequin_pct = (dist < COLOR_THRESHOLD).mean()

        # 该行中"不像人台"的像素占比 (衣服)
        clothing_pct = (dist >= COLOR_THRESHOLD).mean()

        if mannequin_pct > 0.7:
            mannequin_rows += 1
        elif clothing_pct > 0.5:
            clothing_rows += 1

        if y % 408 == 0:  # 约每10%
            row_median = np.median(row_colors, axis=0)
            samples.append((y / h, row_median, mannequin_pct, clothing_pct))

    # 找 顶部裸露人台 → 衣服 的过渡点
    # 从顶部向下: 找第一行 clothing_pct > 0.5
    top_transition = None
    for y in range(h):
        fg_cols = np.where(mask_arr[y] > 30)[0]
        if len(fg_cols) < 5:
            continue
        row_colors = img_arr[y, fg_cols]
        diff = (row_colors.astype(float) - ref_mean.reshape(1, 3)) / ref_std.reshape(1, 3)
        dist = np.sqrt((diff ** 2).sum(axis=1)) / np.sqrt(3)
        cp = (dist >= COLOR_THRESHOLD).mean()
        if cp > 0.5:
            top_transition = y
            break

    # 从底部向上: 找第一行 clothing_pct > 0.5
    bot_transition = None
    for y in range(h - 1, -1, -1):
        fg_cols = np.where(mask_arr[y] > 30)[0]
        if len(fg_cols) < 5:
            continue
        row_colors = img_arr[y, fg_cols]
        diff = (row_colors.astype(float) - ref_mean.reshape(1, 3)) / ref_std.reshape(1, 3)
        dist = np.sqrt((diff ** 2).sum(axis=1)) / np.sqrt(3)
        cp = (dist >= COLOR_THRESHOLD).mean()
        if cp > 0.5:
            bot_transition = y
            break

    print(f"\n{fname}:")
    print(f"  人台颜色行: {mannequin_rows}  衣服颜色行: {clothing_rows}")
    print(f"  顶部颜色过渡(人台→衣服): y={top_transition} (h={top_transition/h:.2f})" if top_transition else "  顶部: 未检测到过渡")
    print(f"  底部颜色过渡(衣服→人台): y={bot_transition} (h={bot_transition/h:.2f})" if bot_transition else "  底部: 未检测到过渡")
    for h_norm, med, mp, cp in samples:
        flag = "←人台" if mp > 0.7 else "←衣服" if cp > 0.5 else " 混合"
        print(f"    h={h_norm:.1f}: median=({med[0]:.0f},{med[1]:.0f},{med[2]:.0f}) "
              f"人台%={mp:.0%} 衣服%={cp:.0%} {flag}")
