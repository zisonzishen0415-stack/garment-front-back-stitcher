"""v16 批测: mask清洗 + 联合轮廓匹配"""
from pathlib import Path
from processor import ImageProcessor

processor = ImageProcessor(mannequin_ref_dir=Path("素材/人台"))
result = processor.process_all(Path("素材/7-21p图"), Path("素材/批量测试/v16"))
print(f"成功: {result['ok']}, 失败: {len(result['fail'])}")
for a, b, r in result["fail"]:
    print(f"  FAIL {a} + {b}: {r}")
