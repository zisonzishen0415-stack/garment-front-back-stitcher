"""v23 批测: 统一crop_w 保证左右放大系数一致"""
from pathlib import Path
from processor import ImageProcessor
r = ImageProcessor().process_all(Path("素材/7-21p图"), Path("素材/批量测试/v23"))
print(f"成功: {r['ok']}, 失败: {len(r['fail'])}")
for a, b, e in r["fail"]: print(f"  FAIL {a} + {b}: {e}")
