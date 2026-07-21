"""customtkinter GUI：文件夹选择、配对预览、进度反馈"""

import threading
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox

from processor import ImageProcessor


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("PS 样品拼接工具")
        self.geometry("720x560")
        self.resizable(True, True)

        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.processor = ImageProcessor()
        self._pairs: list[tuple[Path, Path]] = []

        self._build_ui()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self):
        # 顶部：文件夹选择
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=16, pady=(16, 8))

        ctk.CTkLabel(top, text="输入文件夹:").grid(row=0, column=0, sticky="w", padx=(8, 4))
        self.entry_input = ctk.CTkEntry(top, width=360)
        self.entry_input.grid(row=0, column=1, padx=4)
        ctk.CTkButton(top, text="浏览...", width=80, command=self._pick_input).grid(
            row=0, column=2, padx=4
        )

        ctk.CTkLabel(top, text="输出文件夹:").grid(row=1, column=0, sticky="w", padx=(8, 4), pady=(8, 0))
        self.entry_output = ctk.CTkEntry(top, width=360)
        self.entry_output.grid(row=1, column=1, padx=4, pady=(8, 0))
        ctk.CTkButton(top, text="浏览...", width=80, command=self._pick_output).grid(
            row=1, column=2, padx=4, pady=(8, 0)
        )

        # 扫描按钮
        ctk.CTkButton(top, text="扫描配对", width=100, command=self._scan).grid(
            row=2, column=1, pady=(12, 4)
        )

        # 中部：配对列表
        mid = ctk.CTkFrame(self)
        mid.pack(fill="both", expand=True, padx=16, pady=(4, 8))

        ctk.CTkLabel(mid, text="配 对 列 表", font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=8, pady=(8, 4)
        )

        self.listbox = ctk.CTkTextbox(mid, font=ctk.CTkFont(size=13))
        self.listbox.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # 底部：进度 + 按钮
        bottom = ctk.CTkFrame(self)
        bottom.pack(fill="x", padx=16, pady=(0, 16))

        self.progress = ctk.CTkProgressBar(bottom, width=360)
        self.progress.pack(side="left", padx=(8, 16), pady=12)
        self.progress.set(0)

        self.lbl_status = ctk.CTkLabel(bottom, text="就绪")
        self.lbl_status.pack(side="left", padx=8)

        self.btn_process = ctk.CTkButton(
            bottom, text="开始处理", width=120, command=self._start_process
        )
        self.btn_process.pack(side="right", padx=8, pady=8)

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------

    def _pick_input(self):
        path = filedialog.askdirectory(title="选择原图文件夹")
        if path:
            self.entry_input.delete(0, "end")
            self.entry_input.insert(0, path)

    def _pick_output(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.entry_output.delete(0, "end")
            self.entry_output.insert(0, path)

    def _scan(self):
        input_dir = self.entry_input.get().strip()
        if not input_dir or not Path(input_dir).is_dir():
            messagebox.showwarning("提示", "请先选择有效的输入文件夹。")
            return

        self._pairs = self.processor.find_pairs(Path(input_dir))
        self.listbox.delete("1.0", "end")

        if not self._pairs:
            self.listbox.insert("end", "未找到配对的图片文件。\n")
            self.lbl_status.configure(text=f"发现 0 对")
            return

        lines = []
        for i, (a, b) in enumerate(self._pairs, 1):
            lines.append(f"  {i:02d}.  {a.name}  +  {b.name}")
        self.listbox.insert("end", "\n".join(lines))
        self.lbl_status.configure(text=f"发现 {len(self._pairs)} 对，就绪")

    def _start_process(self):
        if not self._pairs:
            self._scan()
        if not self._pairs:
            messagebox.showwarning("提示", "没有可处理的配对，请先扫描。")
            return

        output_dir = self.entry_output.get().strip()
        if not output_dir:
            messagebox.showwarning("提示", "请先选择输出文件夹。")
            return

        self.btn_process.configure(state="disabled", text="处理中...")
        self.progress.set(0)

        def _on_progress(current, total):
            self.after(0, self._update_progress, current, total)

        def _worker():
            result = self.processor.process_all(
                Path(self.entry_input.get().strip()),
                Path(output_dir),
                progress_callback=_on_progress,
            )
            self.after(0, self._on_done, result)

        threading.Thread(target=_worker, daemon=True).start()

    def _update_progress(self, current, total):
        self.progress.set(current / total)
        self.lbl_status.configure(text=f"处理中 {current}/{total} ...")

    def _on_done(self, result: dict):
        self.btn_process.configure(state="normal", text="开始处理")
        self.progress.set(1)
        ok, fails = result["ok"], result["fail"]
        msg = f"完成：成功 {ok} 张"
        if fails:
            msg += f"，失败 {len(fails)} 对:\n"
            msg += "\n".join(
                f"  • {a} + {b}  ({reason})" for a, b, reason in fails
            )
        else:
            msg += "，全部处理成功！"
        self.lbl_status.configure(text=f"完成，成功 {ok}，失败 {len(fails)}")
        messagebox.showinfo("处理结果", msg)
        self.progress.set(0)
