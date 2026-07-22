"""将 logo.svg 渲染为应用所需的各种尺寸 PNG + ICO。
需要: pip install svgpathtools pillow numpy
"""
from svgpathtools import svg2paths
from PIL import Image, ImageDraw
import numpy as np

VB_W, VB_H = 10937.61, 3074.01  # SVG viewBox
FILL = "#332C2B"

paths, _ = svg2paths("logo.svg")

def render_logo(target_w, fill_color, path_data):
    scale = target_w / VB_W
    target_h = int(VB_H * scale)
    img = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    for path in path_data:
        pts = []
        for segment in path:
            for t in np.linspace(0, 1, 200):
                pt = segment.point(t)
                pts.append((int(pt.real * scale), int(pt.imag * scale)))
        if len(pts) >= 3:
            draw.polygon(pts, fill=fill_color)
    return img

# 暗色版（工具栏、关于、图标）
for name, target_w, color in [
    ("logo_ico", 170, "#332C2B"),
    ("logo_toolbar", 120, "#332C2B"),
    ("logo_about", 710, "#332C2B"),
]:
    img = render_logo(target_w, color, paths)
    img.save(f"{name}.png")
    print(f"{name}.png  {target_w}x{int(VB_H*target_w/VB_W)}  {color}")

# 水印版（白色半透明 logo，暗底上若隐若现）
img = render_logo(560, "#FFFFFF", paths)
r, g, b, a = img.split()
a = a.point(lambda x: int(x * 0.25))  # 只保留 25% 不透明度
img = Image.merge("RGBA", (r, g, b, a))
img.save("logo_placeholder.png", "PNG")
print(f"logo_placeholder.png  560x{int(VB_H*560/VB_W)}  #FFFFFF alpha=25%")

# Windows 图标
Image.open("logo_ico.png").save("logo.ico", format="ICO")
print("logo.ico  done")
