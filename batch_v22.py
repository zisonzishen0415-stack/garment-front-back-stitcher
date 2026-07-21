"""v22 批测: 共识只trim不扩展 + 顶部裁剪"""
from pathlib import Path
from processor import ImageProcessor
r = ImageProcessor().process_all(Path("素材/7-21p图"), Path("素材/批量测试/v22"))
print(f"成功: {r['ok']}, 失败: {len(r['fail'])}")
for a, b, e in r["fail"]: print(f"  FAIL {a} + {b}: {e}")
