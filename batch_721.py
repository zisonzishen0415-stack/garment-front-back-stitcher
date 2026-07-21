"""对 7-21p图 全量批测 (v13: 联合轮廓 + 底部窄区排除)"""
from pathlib import Path
from processor import ImageProcessor

SRC = Path("素材/7-21p图")
OUT = Path("素材/批量测试/7-21p图结果_v13")
OUT.mkdir(parents=True, exist_ok=True)

processor = ImageProcessor()
pairs = processor.find_pairs(SRC)
print(f"发现 {len(pairs)} 对")

result = processor.process_all(SRC, OUT)
print(f"\n成功: {result['ok']}, 失败: {len(result['fail'])}")
for a, b, reason in result["fail"]:
    print(f"  FAIL {a} + {b}: {reason}")
