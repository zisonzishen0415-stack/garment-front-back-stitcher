"""v17 批测: 多通道(颜色+联合匹配+底部窄区)"""
from pathlib import Path
from processor import ImageProcessor
processor = ImageProcessor(mannequin_ref_dir=Path("素材/人台"))
r = processor.process_all(Path("素材/7-21p图"), Path("素材/批量测试/v17"))
print(f"成功: {r['ok']}, 失败: {len(r['fail'])}")
for a, b, e in r["fail"]: print(f"  FAIL {a} + {b}: {e}")
