# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置
用法:
  python build_icon.py            # 生成 logo PNG/ICO
  mkdir models                     # 准备模型目录
  cp ~/.u2net/u2net.onnx models/  # 复制 rembg 模型（168MB）
  pyinstaller garment-stitcher.spec
"""
import sys
from pathlib import Path

PROJECT = Path(SPECPATH).parent

_datas = [
    ('logo.ico', '.'),
    ('logo_toolbar.png', '.'),
    ('logo_about.png', '.'),
]
# rembg u2net 模型（如存在则捆绑）
_u2net = PROJECT / 'models' / 'u2net.onnx'
if _u2net.exists():
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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'torch', 'tensorflow', 'keras',
        'sympy', 'pandas', 'pytest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GarmentStitcher',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='logo.ico',
)
