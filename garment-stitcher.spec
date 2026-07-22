# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir build config (for NSIS installer)
Usage:
  python build_icon.py
  mkdir models && cp ~/.u2net/u2net.onnx models/
  python -m PyInstaller garment-stitcher.spec
  makensis installer.nsi
Output: dist/GarmentStitcher_Setup.exe
"""
from pathlib import Path

PROJECT = Path(SPECPATH).parent

_datas = [
    ('logo.ico', '.'),
    ('logo_toolbar.png', '.'),
    ('logo_about.png', '.'),
    ('logo_placeholder.png', '.'),
]
_u2net = PROJECT / 'models' / 'u2net.onnx'

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
    a.binaries,
    a.zipfiles,
    [],
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
