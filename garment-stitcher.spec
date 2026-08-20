# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build config (for NSIS installer)

Prerequisites:
  pip install pyinstaller
  python build_icon.py
  mkdir models && cp ~/.u2net/u2net.onnx models/

Build:
  python -m PyInstaller garment-stitcher.spec
  → dist/GarmentStitcher/  (onedir, model bundled in _internal/models/)

Pack installer (requires NSIS: winget install NSIS.NSIS):
  makensis installer.nsi
  → dist/GarmentStitcher_Setup.exe
"""
from pathlib import Path

PROJECT = Path(SPECPATH)

_datas = [
    ('logo.ico', '.'),
    ('logo_toolbar.png', '.'),
    ('logo_about.png', '.'),
    ('logo_placeholder.png', '.'),
]

# rembg u2net 模型（168MB）—— 构建前确保 models/u2net.onnx 存在
_u2net = PROJECT / 'models' / 'u2net.onnx'
if not _u2net.exists():
    raise SystemExit(f"模型文件不存在: {_u2net}\n请先运行: mkdir models && cp ~/.u2net/u2net.onnx models/")
_datas.append((str(_u2net), 'models'))

# 微调模型（如果存在则打包）
# 注意：该模型为 external-data 格式，权重存在同名 .data 文件里，必须一并打包
_ft_model = PROJECT / 'models' / 'u2net_finetuned.onnx'
if _ft_model.exists():
    _datas.append((str(_ft_model), 'models'))
    _ft_data = PROJECT / 'models' / 'u2net_finetuned.onnx.data'
    if _ft_data.exists():
        _datas.append((str(_ft_data), 'models'))
    else:
        raise SystemExit(f"微调模型缺少外部权重文件: {_ft_data}\n请确保 u2net_finetuned.onnx.data 与 .onnx 在同一目录")

a = Analysis(
    ['reviewer.py'],
    # _shims 优先于 site-packages，用占位 pymatting 顶掉真实 pymatting（避免引入 numba/llvmlite）
    pathex=[str(PROJECT / '_shims')],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        'rembg', 'onnxruntime', 'onnxruntime.capi',
        'skimage', 'pooch',
        'customtkinter', 'PIL', 'PIL.ImageTk', 'PIL.ImageDraw',
        'numpy', 'scipy', 'scipy.ndimage',
        'tkinter.ttk',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'torch', 'tensorflow', 'keras',
        'sympy', 'pandas', 'pytest',
        # 运行时未使用、被 PyInstaller 过度收集的重包
        'numba', 'llvmlite', 'pyarrow', 'sklearn', 'pydantic', 'fsspec', 'git',
    ],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GarmentStitcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon='logo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='GarmentStitcher',
)
