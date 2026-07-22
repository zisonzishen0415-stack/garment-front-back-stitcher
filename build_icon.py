"""将 logo.svg 渲染为应用所需的各种尺寸 PNG + ICO。
需要: pip install svgpathtools pillow numpy
"""
from svgpathtools import svg2paths
from PIL import Image, ImageDraw
import numpy as np

VB_W, VB_H = 10937.61, 3074.01  # SVG viewBox
FILL = "#332C2B"

paths, _ = svg2paths("logo.svg")

for name, target_w in [("logo_ico", 170), ("logo_toolbar", 120), ("logo_about", 710)]:
    scale = target_w / VB_W
    target_h = int(VB_H * scale)
    img = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    for path in paths:
        pts = []
        for segment in path:
            for t in np.linspace(0, 1, 200):
                pt = segment.point(t)
                pts.append((int(pt.real * scale), int(pt.imag * scale)))
        if len(pts) >= 3:
            draw.polygon(pts, fill=FILL)

    img.save(f"{name}.png")
    print(f"{name}.png  {target_w}x{target_h}")

# Windows 图标
Image.open("logo_ico.png").save("logo.ico", format="ICO")
print("logo.ico  done")
