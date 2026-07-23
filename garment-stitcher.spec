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

a = Analysis(
    ['reviewer.py'],
    pathex=[],
    binaries=[],
    datas=_datas,
    hiddenimports=[
        'rembg', 'onnxruntime', 'onnxruntime.capi',
        'skimage', 'pymatting', 'pooch',
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
    upx=True,
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
    upx=True,
    upx_exclude=[],
    name='GarmentStitcher',
)
