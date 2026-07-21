"""从标注数据学习裁剪规则

分析：在用户标注的边界处，rembg mask 有什么共同特征？
- 顶部：用户下移 y1 → 上面是什么？（衣架？人台肩膀？）
- 底部：用户上移 y2 → 下面是什么？（人台腿？金属杆？）
"""
import json
import colorsys
import numpy as np
from pathlib import Path
from PIL import Image, ImageOps
from rembg import remove, new_session

SRC = Path("素材/7-21p图")
data = json.load(open("素材/annotations.json", encoding="utf-8"))
anns = data["annotations"]
session = new_session()

print("=" * 80)
print("特征分析：用户边界 vs rembg边界")
print("=" * 80)

features = []

for a in anns:
    fname = a["file"]
    b = a["bbox"]        # 用户标注: [x1, y1, x2, y2]
    r = a.get("rembg_bbox")
    if r is None:
        continue
    path = SRC / fname
    if not path.exists():
        continue

    ux1, uy1, ux2, uy2 = b
    rx1, ry1, rx2, ry2 = r

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w_img, h_img = img.size
    mask = remove(img, session=session, only_mask=True)
    if mask.size != (w_img, h_img):
        mask = mask.resize((w_img, h_img), Image.LANCZOS)
    mask_arr = np.array(mask)
    img_arr = np.array(img)

    # ---- 宽度分布 ----
    rows_data = []
    for y in range(h_img):
        fg = np.where(mask_arr[y] > 30)[0]
        if len(fg) >= 5:
            rows_data.append((y, fg.min(), fg.max(), len(fg)))

    ys = np.array([r[0] for r in rows_data])
    ls = np.array([r[1] for r in rows_data])
    rs = np.array([r[2] for r in rows_data])
    ws = np.array([r[3] for r in rows_data], dtype=float)

    if len(ys) < 20:
        continue

    # 中间区域宽度（用户认为的衣服主体）
    user_mid_lo = uy1 + (uy2 - uy1) * 0.25
    user_mid_hi = uy1 + (uy2 - uy1) * 0.75
    user_mid_mask = (ys >= user_mid_lo) & (ys <= user_mid_hi)
    if user_mid_mask.sum() < 5:
        body_width = np.median(ws)
    else:
        body_width = np.median(ws[user_mid_mask])

    # ---- 顶部特征 ----
    top_trim = uy1 - ry1  # 正值=用户下移了顶部

    # 在用户顶部边界处的宽度
    near_user_top = np.where((ys >= uy1 - 10) & (ys <= uy1 + 10))[0]
    width_at_user_top = ws[near_user_top].mean() if len(near_user_top) > 0 else 0

    # rembg顶部到用户顶部之间的宽度趋势
    between_top = (ys >= ry1) & (ys <= uy1)
    if between_top.sum() > 3:
        top_ws = ws[between_top]
        top_max_w = top_ws.max()
        top_min_w = top_ws.min()
        top_mean_w = top_ws.mean()
    else:
        top_max_w = top_min_w = top_mean_w = 0

    # rembg到用户顶部区间内的宽度增长（从用户顶部向下到rembg顶部）
    if between_top.sum() > 3:
        # 从用户顶部向上（到rembg顶部）宽度变化
        # 正的 w_growth 表示往上变窄（可能是衣架，好信号）
        w_at_rembg_top = ws[ys >= ry1].min() if (ys >= ry1).sum() > 0 else 0
        if w_at_rembg_top > 0 and body_width > 0:
            top_narrow_ratio = w_at_rembg_top / body_width
        else:
            top_narrow_ratio = 1.0
    else:
        top_narrow_ratio = 1.0

    # ---- 底部特征 ----
    bot_trim = ry2 - uy2  # 正值=用户上移了底部（裁剪）

    # 在用户底部边界处的宽度
    near_user_bot = np.where((ys >= uy2 - 10) & (ys <= uy2 + 10))[0]
    width_at_user_bot = ws[near_user_bot].mean() if len(near_user_bot) > 0 else 0

    # 用户底部到rembg底部之间的宽度
    between_bot = (ys >= uy2) & (ys <= ry2)
    if between_bot.sum() > 3:
        bot_ws = ws[between_bot]
        bot_mean_w = bot_ws.mean()
        bot_min_w = bot_ws.min()
        bot_max_w = bot_ws.max()
    else:
        bot_mean_w = bot_min_w = bot_max_w = 0

    # 底部宽度 vs 主体宽度
    if body_width > 0:
        bot_narrow_ratio = bot_mean_w / body_width if bot_mean_w > 0 else 0
    else:
        bot_narrow_ratio = 0

    # 用户底部边界处宽度 vs 主体宽度
    if body_width > 0:
        user_bot_ratio = width_at_user_bot / body_width if width_at_user_bot > 0 else 1.0
    else:
        user_bot_ratio = 1.0

    features.append({
        "file": fname,
        "top_trim": top_trim,
        "bot_trim": bot_trim,
        "body_width": body_width,
        "width_at_user_top": width_at_user_top,
        "width_at_user_bot": width_at_user_bot,
        "top_narrow_ratio": top_narrow_ratio,
        "bot_narrow_ratio": bot_narrow_ratio,
        "user_bot_ratio": user_bot_ratio,
        "top_max_w": top_max_w,
        "top_min_w": top_min_w,
        "between_top_sum": between_top.sum() if 'between_top' in dir() else 0,
        "bottom_ws_mean": bot_mean_w,
    })

# ============================================================
# 分组分析
# ============================================================

no_bot = [f for f in features if f["bot_trim"] < 20]
has_bot = [f for f in features if f["bot_trim"] >= 100]
big_bot = [f for f in features if f["bot_trim"] >= 500]
med_bot = [f for f in features if 100 <= f["bot_trim"] < 500]

no_top = [f for f in features if f["top_trim"] < 20]
has_top = [f for f in features if f["top_trim"] >= 100]

print(f"\n--- 底部特征 (bot_trim >= 0) ---")
for label, group in [("无需裁剪(<20px)", no_bot), ("中幅裁剪(100-500)", med_bot), ("大幅裁剪(≥500)", big_bot)]:
    if group:
        bnr = np.mean([f["bot_narrow_ratio"] for f in group])
        ubr = np.mean([f["user_bot_ratio"] for f in group])
        bw = np.mean([f["body_width"] for f in group])
        print(f"  {label}: {len(group)}张  "
              f"底部/主体宽度比={bnr:.2f}  "
              f"用户边界/主体比={ubr:.2f}  "
              f"主体宽={bw:.0f}px")

print(f"\n--- 顶部特征 (top_trim >= 0) ---")
for label, group in [("未改(<20px)", no_top), ("有下移(≥100px)", has_top)]:
    if group:
        tnr = np.mean([f["top_narrow_ratio"] for f in group])
        print(f"  {label}: {len(group)}张  顶部窄度比={tnr:.2f}  "
              f"(1.0=顶部和主体一样宽, <0.5=顶部明显窄)")

# ---- 关键特征分布 ----
print(f"\n--- 底部宽度比的决策边界 ---")
for threshold in [0.3, 0.4, 0.5, 0.6, 0.7]:
    correct = 0
    for f in features:
        predicted_bot = f["body_width"] * threshold
        # 如果预测到的边界在用户边界的 ±20% 范围内就算正确
        actual_bot_ratio = f["user_bot_ratio"]
        correct += 1
    # 检查: 对于不需要裁剪的图，预测也不该裁剪
    pos_true = 0
    neg_true = 0
    for f in features:
        should_trim = f["bot_trim"] >= 50
        pred_trim = f["bot_narrow_ratio"] < threshold
        if should_trim and pred_trim:
            pos_true += 1
        if not should_trim and not pred_trim:
            neg_true += 1
    acc = (pos_true + neg_true) / len(features) * 100
    prec = pos_true / max((sum(1 for f in features if f["bot_narrow_ratio"] < threshold)), 1) * 100
    rec = pos_true / max(sum(1 for f in features if f["bot_trim"] >= 50), 1) * 100
    print(f"  ratio<{threshold}: 准确率={acc:.0f}% 精确率={prec:.0f}% 召回率={rec:.0f}%  "
          f"TP={pos_true} FN={sum(1 for f in features if f['bot_trim'] >= 50) - pos_true}")

# 打印底部边界处的具体数值
print(f"\n--- 底部详情 (所有需要裁剪的图片) ---")
for f in sorted(features, key=lambda x: -x["bot_trim"]):
    if f["bot_trim"] >= 50:
        print(f"  {f['file']}: 裁剪{f['bot_trim']:.0f}px "
              f"下部/主体={f['bot_narrow_ratio']:.2f} "
              f"用户边界/主体={f['user_bot_ratio']:.2f} "
              f"主体宽={f['body_width']:.0f}px 下部平均宽={f['bottom_ws_mean']:.0f}px")

print(f"\n--- 底部无需裁剪的图片 ---")
for f in features:
    if f["bot_trim"] < 20:
        print(f"  {f['file']}: 下部/主体={f['bot_narrow_ratio']:.2f} "
              f"主体宽={f['body_width']:.0f}px")
