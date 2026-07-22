# 液化工具体验提升 + 编辑器布局优化 — 设计文档

## 概述

三项用户体验改进：

1. **液化脏区域渲染** — 解决拖拽变形时的卡顿问题
2. **压力可视化笔刷光标** — 多层同心圆展示笔刷影响范围
3. **编辑器左右排列** — 匹配 9:16 竖向素材图的布局优化

---

## 1. 液化脏区域渲染

### 问题

当前每次鼠标移动都触发 `get_warped()` 对全图逐像素做 scipy `map_coordinates` 重映射。0.35x 缩放下约 150 万像素/帧，单帧耗时 >30ms，~30fps 限速也爆。实际上笔刷 40px 半径只影响约 200×200 px 区域。

### 方案

**`LiquifyEngine` 内部改动：**

- 新增 `_full_arr` — 上一帧完整渲染结果（numpy uint8 数组）
- 新增 `_dirty_rect` — 脏矩形 `[x1, y1, x2, y2]`，图片坐标系
- `brush_stroke()` — 更新 `_dirty_rect`，膨胀范围为笔刷半径 + 1 个 grid_spacing 的像素区域
- `get_warped(display_scale)`：
  - scale 变化或首次调用 → 全刷，`_dirty_rect` 重置
  - scale 不变 → 只对 `_dirty_rect` 内像素做 remap，其余从 `_full_arr` 拷贝
  - 返回前更新 `_full_arr` 缓存
- `undo()` / `reset()` → 标记全刷（设 `_dirty_rect` 为整图范围）
- 缩放时网格变形字段 `dx`/`dy` 不需要额外处理——像素级重映射基于完整网格，脏矩形只影响计算范围

**`LiquifyCanvas` 改动：**

- `_schedule_render()` 限速从 33ms 降到 16ms（~60fps），因为单帧渲染量从 ~150 万降到 ~4 万像素

### 影响范围

仅 `liquify.py` 的 `LiquifyEngine` 和 `LiquifyCanvas` 类，不影响外部接口。

---

## 2. 压力可视化笔刷光标

### 方案

`LiquifyCanvas` 中绘制三层同心圆替代当前单一虚线圆：

| 层 | 半径 | 样式 | 含义 |
|---|---|---|---|
| 内圈 | `r × 0.3` | 实心填充，半透明白色 `#FFFFFF50` | 核心压力区 |
| 中圈 | `r × 0.7` | 实线 1px，`#00BFFF` alpha 0.8 | 有效变形区 |
| 外圈 | `r × 1.0` | 虚线 1px，`#00BFFF` alpha 0.5 | 衰减边界 |

**实现细节：**

- 三个圆使用同一 `tags="cursor"`，每次 `_on_motion` 先 `delete("cursor")` 再一起创建
- `_on_drag()` 中依然绘制变形描边圆圈（`tags="brush"`），保持即时反馈
- 笔刷半径 < 3px 时不绘制光标

### 影响范围

仅 `liquify.py` 的 `LiquifyCanvas._on_motion()` 和 `_on_leave()`。

---

## 3. 编辑器左右排列

### 问题

当前右侧面板中正面/反面编辑器上下堆叠。素材为佳能竖拍 9:16 图片，上下对分后每个编辑器高约 350px，宽度 650px，竖向图在狭高空间内被缩小，水平空间大量留白。

### 方案

**布局改为水平并排：**

```
right（横向面板）
├── editor_frame_a（pack side=left, fill=both, expand=True）
│   ├── 标签行: "正面" + 角度旋钮（↺逆 / ↻顺 / 数值）
│   └── editor_a（fill=both, expand=True）
└── editor_frame_b（pack side=left, fill=both, expand=True）
    ├── 标签行: "反面" + 角度旋钮（↺逆 / ↻顺 / 数值）
    └── editor_b（fill=both, expand=True）
```

**具体改动：**

- 右侧面板宽度从 700px 放宽到 780px（两个编辑器各约 380px 宽、~780px 高）
- 移除 `editor_a`/`editor_b` 的 `height=320` 参数，改为 `fill="both", expand=True`
- 移除 `right.pack_propagate(False)` 或调整宽度
- `BBoxEditor._fit()` 中缩放上限 `min(cw/iw, ch/ih, 0.5)` 改为 `min(cw/iw, ch/ih)`（不再封顶 0.5），让竖向大图自然撑满可用空间

### 影响范围

仅 `reviewer.py` 的 `ReviewerApp._build_ui()` 方法（约 30 行改动）和 `BBoxEditor._fit()`（1 行改动）。

---

## 验证方式

- 液化：打开一张拼接结果图，进入液化工具，快速拖拽笔刷，确认无可见卡顿
- 压力可视化：移动鼠标到画布上，确认看到三层同心圆光标
- 编辑器：启动 reviewer，加载素材文件夹，确认右侧编辑器左右并排，9:16 图片不需要缩放即可完整显示
