"""v18 批测: v11 + rembg前置对比度增强"""
from pathlib import Path
from processor import ImageProcessor
r = ImageProcessor().process_all(Path("素材/7-21p图"), Path("素材/批量测试/v18"))
print(f"成功: {r['ok']}, 失败: {len(r['fail'])}")
for a, b, e in r["fail"]: print(f"  FAIL {a} + {b}: {e}")
