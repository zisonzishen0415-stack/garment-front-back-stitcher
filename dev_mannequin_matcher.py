"""开发: 参考人台宽度轮廓匹配

思路:
1. 从参考人台图提取归一化宽度分布 → 人台形状模板
2. 在量产图中, 逐行比对前景宽度 vs 参考模板
3. 宽度接近参考 (=裸露人台) → 裁剪
4. 宽度远大于参考 (=衣服覆盖人台) → 保留
"""
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from rembg import remove, new_session

REF_DIR = Path("素材/人台")
SRC_DIR = Path("素材/7-21p图")
OUT_DIR = Path("素材/诊断/人台匹配")
OUT_DIR.mkdir(parents=True, exist_ok=True)
session = new_session()


# ============================================================
# 1. 构建参考人台宽度模板
# ============================================================

def get_width_profile(img, mask_arr):
    """返回 (y_vals, width_vals, left_vals, right_vals)"""
    h = mask_arr.shape[0]
    ys, ws, ls, rs = [], [], [], []
    for y in range(h):
        fg = np.where(mask_arr[y] > 30)[0]
        if len(fg) >= 5:
            ys.append(y)
            ws.append(len(fg))
            ls.append(fg.min())
            rs.append(fg.max())
    return (np.array(ys), np.array(ws, dtype=float),
            np.array(ls), np.array(rs))


print("=" * 70)
print("构建参考人台宽度模板")
print("=" * 70)

ref_profiles = []
for f in sorted(REF_DIR.iterdir()):
    if not f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        continue
    img = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
    w, h = img.size
    mask = remove(img, session=session, only_mask=True)
    if mask.size != (w, h):
        mask = mask.resize((w, h), Image.LANCZOS)
    ys, ws, ls, rs = get_width_profile(img, np.array(mask))
    if len(ys) < 100:
        continue

    # 归一化高度: 0=头顶, 1=脚底
    y_min, y_max = ys.min(), ys.max()
    norm_h = (ys - y_min) / max(y_max - y_min, 1)

    ref_profiles.append({
        'name': f.stem,
        'norm_h': norm_h,
        'width': ws,
        'img_w': w,
        'img_h': h,
        'y_range': (y_min, y_max),
    })
    print(f"  {f.name}: 高度{y_min}-{y_max} 宽度{ws.min():.0f}-{ws.max():.0f}")

# 合并所有参考: 对每个归一化高度, 取平均宽度
n_bins = 100
ref_width = np.zeros(n_bins)
ref_count = np.zeros(n_bins)
for rp in ref_profiles:
    for i in range(len(rp['norm_h'])):
        bin_idx = min(int(rp['norm_h'][i] * n_bins), n_bins - 1)
        ref_width[bin_idx] += rp['width'][i]
        ref_count[bin_idx] += 1

# 平滑
mask_valid = ref_count >= 2
if mask_valid.any():
    bin_valid = np.where(mask_valid)[0]
    for i in range(n_bins):
        if not mask_valid[i]:
            # 找最近的有效 bin
            nearest = bin_valid[np.argmin(np.abs(bin_valid - i))]
            ref_width[i] = ref_width[nearest]
            ref_count[i] = 1

ref_width /= np.maximum(ref_count, 1)

# 平滑
window = 5
ref_width_smooth = np.zeros_like(ref_width)
for i in range(n_bins):
    lo = max(0, i - window)
    hi = min(n_bins, i + window + 1)
    ref_width_smooth[i] = ref_width[lo:hi].mean()

ref_width = ref_width_smooth

print(f"\n参考人台宽度模板 (归一化):")
for pct in range(0, 101, 10):
    idx = min(pct, 99)
    print(f"  h={pct/100:.1f}: 宽度={ref_width[idx]:.0f}px")

# 人台关键特征:
# h=0.3-0.45: 肩膀最宽处 (~550-580px)
# h=0.5-0.7:  躯干 (~50px, 是杆!)
# h=0.7-1.0:  底部杆 (~50-70px)
shoulder_peak = ref_width[20:50].max()
shoulder_h = 0.20 + np.argmax(ref_width[20:50]) / n_bins
print(f"\n肩膀峰值: {shoulder_peak:.0f}px @ h≈{shoulder_h:.2f}")


# ============================================================
# 2. 在问题图片上匹配
# ============================================================

print("\n" + "=" * 70)
print("在量产图上匹配人台模板")
print("=" * 70)

# 所有需要底部裁剪的 + 顶部有问题的裤子
test_files = [
    "IMG_6026.JPG", "IMG_6027.JPG",  # 裤子
    "IMG_6035.JPG", "IMG_6036.JPG",  # 裤子
    "IMG_6048.JPG", "IMG_6050.JPG",  # 短裤
    "IMG_6060.JPG", "IMG_6062.JPG",  # 长款(大量底部裁剪)
    "IMG_5952.JPG", "IMG_5953.JPG",  # 有底部裁剪
    "IMG_5962.JPG", "IMG_5964.JPG",  # 有顶部+底部
    "IMG_5971.JPG", "IMG_5973.JPG",  # 无底部裁剪(对照)
]

for fname in test_files:
    path = SRC_DIR / fname
    if not path.exists():
        continue

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w_img, h_img = img.size

    mask = remove(img, session=session, only_mask=True)
    if mask.size != (w_img, h_img):
        mask = mask.resize((w_img, h_img), Image.LANCZOS)
    mask_arr = np.array(mask)
    ys, ws, ls, rs = get_width_profile(img, mask_arr)

    if len(ys) < 20:
        print(f"\n{fname}: 检测失败")
        continue

    y_min, y_max = ys.min(), ys.max()
    y_range = max(y_max - y_min, 1)

    # --- 尝试用肩膀特征对齐 ---
    # 在量产图的 mask 中找最大宽度区域 (= 肩膀或衣服主体)
    # 把人台模板缩放到与量产图匹配

    # 方法: 把人台模板的宽度归一化到量产图的尺度
    # 量产图的主体宽度 = 衣服区域的宽度
    n_prod = len(ys)
    mid_lo, mid_hi = n_prod // 4, 3 * n_prod // 4
    prod_body_w = float(np.median(ws[mid_lo:mid_hi]))

    # 人台模板的肩膀宽度 (参考尺度)
    ref_shoulder_w = shoulder_peak

    # 缩放因子: 人台参考图 vs 量产图的尺度比
    # 人台肩膀宽 ≈ 570px, 量产图中躯干/衣服宽 ≈ 600-1000px
    # 但人台和衣服宽度不同, 不能用衣服宽度做缩放

    # 更好的方法: 假设人台在每张图中的实际像素尺寸是固定的
    # 因为拍摄距离/相机应该一致
    # 直接用人台参考的像素宽度做匹配

    scale = prod_body_w / max(ref_shoulder_w, 1) if ref_shoulder_w > 0 else 1.0
    scaled_ref = ref_width * scale

    # 对量产图的每一行, 计算其归一化高度, 然后与参考比较
    prod_norm_h = (ys - y_min) / y_range

    # 对每个量产行, 找最近的参考 bin
    match_scores = []
    for i in range(len(prod_norm_h)):
        h = prod_norm_h[i]
        bin_idx = min(int(h * n_bins), n_bins - 1)
        bin_idx = max(0, min(n_bins - 1, bin_idx))

        ref_w = scaled_ref[bin_idx]
        prod_w = ws[i]

        if ref_w > 5:
            ratio = prod_w / ref_w
        else:
            ratio = 99.0

        match_scores.append((h, ys[i], prod_w, ref_w, ratio))

    # --- 判断: 哪些区域是裸露人台 ---
    # ratio ≈ 1.0: 宽度接近参考人台 → 裸露人台
    # ratio >> 1.0: 宽度远大于参考 → 衣服覆盖

    # 从顶部往下找衣服开始的位置
    top_mannequin_end = None
    for h, y, pw, rw, ratio in match_scores:
        if ratio > 2.0 and pw > 100:  # 宽度至少是参考的2倍 → 衣服开始了
            top_mannequin_end = y
            break

    # 从底部往上找衣服结束的位置
    bot_mannequin_start = None
    for h, y, pw, rw, ratio in reversed(match_scores):
        if ratio > 2.0 and pw > 100:
            bot_mannequin_start = y
            break

    # --- 输出 ---
    # 顶部20行的匹配
    top_ratios = [s[4] for s in match_scores[:20]]
    bot_ratios = [s[4] for s in match_scores[-20:]]
    mid_ratios = [s[4] for s in match_scores[len(match_scores)//3:2*len(match_scores)//3]]

    print(f"\n{fname}:")
    print(f"  主体宽度={prod_body_w:.0f}px  缩放={scale:.2f}x")
    print(f"  顶部ratio: {np.mean(top_ratios):.1f}  中部ratio: {np.mean(mid_ratios):.1f}  底部ratio: {np.mean(bot_ratios):.1f}")
    print(f"  顶部裸露人台到: y={top_mannequin_end} (h={top_mannequin_end/h_img:.2f})" if top_mannequin_end else "  顶部: 未检测到裸露人台")
    print(f"  底部衣服结束于: y={bot_mannequin_start} (h={bot_mannequin_start/h_img:.2f})" if bot_mannequin_start else "  底部: 未检测到衣服终点")

    # 详细: 每隔10%高度打印ratio
    print(f"  高度分段ratio:")
    for pct in range(10, 101, 10):
        h = pct / 100
        near = [s for s in match_scores if abs(s[0] - h) < 0.05]
        if near:
            avg_r = np.mean([s[4] for s in near])
            avg_pw = np.mean([s[2] for s in near])
            avg_rw = np.mean([s[3] for s in near])
            flag = " ←人台" if avg_r < 2.0 else ""
            print(f"    h={h:.1f}: ratio={avg_r:.1f}x prod_w={avg_pw:.0f} ref_w={avg_rw:.0f}{flag}")
