"""pymatting 占位包。

rembg 的 bg.py 顶层导入了 pymatting 的三个函数，但它们只在 alpha_matting=True
时才被调用。本应用始终使用 only_mask=True（默认 alpha_matting=False），
因此这里用占位实现替代，避免引入 numba / llvmlite 等重依赖。
"""
