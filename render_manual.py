# -*- coding: utf-8 -*-
"""生成正式软件说明书 PDF：封皮 + 目录（含页码）+ 正文（页眉页脚）。

依赖：reportlab + Pillow。中文字体：微软雅黑（msyh.ttc / msyhbd.ttc）。
运行：python render_manual.py  →  dist/使用说明书.pdf
"""
import os
from pathlib import Path

from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, white, Color
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFontFamily
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
    NextPageTemplate, Table, TableStyle, KeepTogether, HRFlowable,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).parent
OUT_PDF = Path(os.environ.get("MANUAL_OUT", ROOT / "dist" / "使用说明书.pdf"))
LOGO = ROOT / "logo_about.png"
LOGO_TINT = ROOT / "build" / "logo_cover.png"

FONT_DIR = Path("C:/Windows/Fonts")
PAGE_W, PAGE_H = A4  # 595.27 x 841.89 pt

# ── 统一配色（绿色主色 + 藏青深色，避免灰白/彩色混搭）──────────
DARK = HexColor("#1f2430")        # 藏青：标题、表头
DARK_RGB = (0x1f, 0x24, 0x30)
ACCENT = HexColor("#2B8C3C")      # 绿：主强调色
ACCENT_DARK = HexColor("#236E30")
GRAY = HexColor("#5b6472")        # 正文次级文字（加深，避免发灰发白）
BODY = HexColor("#2b2f36")        # 正文
BORDER = HexColor("#d5dfd7")      # 边框（带一点绿调）
ROW_ALT = HexColor("#f0f7f2")     # 表格隔行：浅绿，替换浅灰
WARN_BG = HexColor("#fff8e1")
WARN_BAR = HexColor("#f0b429")

# ── 字体 ──────────────────────────────────────────────────────
pdfmetrics.registerFont(TTFont("YaHei", FONT_DIR / "msyh.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("YaHeiBd", FONT_DIR / "msyhbd.ttc", subfontIndex=0))
pdfmetrics.registerFont(TTFont("Consolas", FONT_DIR / "consola.ttf"))
registerFontFamily("YaHei", normal="YaHei", bold="YaHeiBd",
                   italic="YaHei", boldItalic="YaHeiBd")


# ── 样式（CJK 换行 + 左对齐，避免中文字符被拉散）────────────────
def _s(name, **kw):
    base = dict(fontName="YaHei", fontSize=10.5, leading=17, textColor=BODY,
                wordWrap="CJK", alignment=TA_LEFT)
    base.update(kw)
    return ParagraphStyle(name, **base)


ST = {
    "H1": _s("H1", fontName="YaHeiBd", fontSize=16, leading=21,
             textColor=ACCENT_DARK, spaceBefore=16, spaceAfter=2, keepWithNext=1),
    "H2": _s("H2", fontName="YaHeiBd", fontSize=12.5, leading=18,
             textColor=DARK, spaceBefore=11, spaceAfter=4, keepWithNext=1),
    "Body": _s("Body", spaceAfter=6),
    "BodyC": _s("BodyC", alignment=TA_CENTER),
    "Li": _s("Li", leftIndent=16, bulletIndent=4, spaceAfter=3),
    "Cell": _s("Cell", fontSize=10, leading=15),
    "CellHead": _s("CellHead", fontName="YaHeiBd", fontSize=10, leading=15,
                   textColor=white),
    "TocTitle": _s("TocTitle", fontName="YaHeiBd", fontSize=20, leading=26,
                   textColor=DARK, spaceAfter=10),
    "Note": _s("Note", fontSize=10, leading=16, textColor=HexColor("#6b5b12")),
}

# 目录条目样式（level 0 = 章，level 1 = 节）
TOC_LEVEL_STYLES = [
    ParagraphStyle("TOC0", fontName="YaHeiBd", fontSize=11.5, leading=20,
                   leftIndent=0, firstLineIndent=0, textColor=DARK, spaceBefore=5,
                   wordWrap="CJK"),
    ParagraphStyle("TOC1", fontName="YaHei", fontSize=10, leading=17,
                   leftIndent=20, firstLineIndent=0, textColor=GRAY, spaceBefore=2,
                   wordWrap="CJK"),
]


class ManualDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            name = flowable.style.name
            if name == "H1":
                self.notify("TOCEntry", (0, flowable.getPlainText(), self.page))
            elif name == "H2":
                self.notify("TOCEntry", (1, flowable.getPlainText(), self.page))


def _prepare_logo():
    """把 logo 统一染成主题藏青色，避免封皮颜色不协调。"""
    LOGO_TINT.parent.mkdir(parents=True, exist_ok=True)
    im = PILImage.open(LOGO).convert("RGBA")
    px = im.load()
    for y in range(im.size[1]):
        for x in range(im.size[0]):
            r, g, b, a = px[x, y]
            if a > 0:
                px[x, y] = (*DARK_RGB, a)
    im.save(LOGO_TINT)
    return LOGO_TINT


def cover_page(canvas, doc):
    """封皮（第 1 页）。"""
    canvas.saveState()
    canvas.setFillColor(white)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    logo = _prepare_logo()
    iw, ih = 250, 250 * 199 / 710  # ≈70pt
    logo_top = PAGE_H - 80
    canvas.drawImage(str(logo), (PAGE_W - iw) / 2, logo_top - ih,
                     width=iw, height=ih, mask="auto")

    # 主标题（与 logo 之间留足间距，避免重叠）
    y = logo_top - ih - 46
    canvas.setFillColor(DARK)
    canvas.setFont("YaHeiBd", 30)
    canvas.drawCentredString(PAGE_W / 2, y, "GarmentStitcher")

    # 副标题
    y -= 34
    canvas.setFont("YaHei", 15)
    canvas.setFillColor(DARK)
    canvas.drawCentredString(PAGE_W / 2, y, "服装正反面拼接工具")

    # 文档类型
    y -= 26
    canvas.setFont("YaHei", 12.5)
    canvas.setFillColor(GRAY)
    canvas.drawCentredString(PAGE_W / 2, y, "使用说明书 · User Manual")

    # 强调色线
    y -= 26
    canvas.setStrokeColor(ACCENT)
    canvas.setLineWidth(2.5)
    canvas.line(PAGE_W / 2 - 60, y, PAGE_W / 2 + 60, y)

    # 版本徽标
    y -= 28
    canvas.setFillColor(ACCENT)
    canvas.roundRect(PAGE_W / 2 - 44, y, 88, 24, 12, fill=1, stroke=0)
    canvas.setFillColor(white)
    canvas.setFont("YaHeiBd", 12)
    canvas.drawCentredString(PAGE_W / 2, y + 8, "版本 v2.0")

    # 底部信息
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.7)
    canvas.line(60, 112, PAGE_W - 60, 112)
    canvas.setFillColor(GRAY)
    canvas.setFont("YaHei", 10.5)
    canvas.drawCentredString(PAGE_W / 2, 90, "2025 年 8 月")
    canvas.drawCentredString(PAGE_W / 2, 68, "Powered by zisonzishen")
    canvas.restoreState()


def body_page(canvas, doc):
    """正文页眉页脚。"""
    canvas.saveState()
    canvas.setStrokeColor(BORDER)
    canvas.setLineWidth(0.6)
    canvas.line(48, PAGE_H - 44, PAGE_W - 48, PAGE_H - 44)
    canvas.setFillColor(GRAY)
    canvas.setFont("YaHei", 8.5)
    canvas.drawString(48, PAGE_H - 36, "GarmentStitcher 使用说明书")
    canvas.drawRightString(PAGE_W - 48, PAGE_H - 36, "v2.0")

    canvas.setStrokeColor(BORDER)
    canvas.line(48, 44, PAGE_W - 48, 44)
    canvas.setFillColor(GRAY)
    canvas.setFont("YaHei", 9)
    canvas.drawCentredString(PAGE_W / 2, 26, f"第 {doc.page} 页")
    canvas.restoreState()


# ── 内容构造辅助 ──────────────────────────────────────────────
def chapter(text):
    """章标题 + 绿色分隔线。"""
    return [
        Paragraph(text, ST["H1"]),
        HRFlowable(width="100%", thickness=1.2, color=ACCENT,
                   spaceBefore=2, spaceAfter=8),
    ]


def H2(text):
    return Paragraph(text, ST["H2"])


def P(text):
    return Paragraph(text, ST["Body"])


def bullets(items):
    return [Paragraph(it, ST["Li"], bulletText="•") for it in items]


def steps(items):
    return [Paragraph(it, ST["Li"], bulletText=f"{i}.") for i, it in enumerate(items, 1)]


def table(header, rows, widths=None):
    data = [[Paragraph(h, ST["CellHead"]) for h in header]]
    for r in rows:
        data.append([Paragraph(c, ST["Cell"]) for c in r])
    n = len(header)
    if widths is None:
        widths = [(PAGE_W - 96) / n] * n
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), DARK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, ROW_ALT]),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def note(text):
    p = Paragraph(text, ST["Note"])
    t = Table([[p]], colWidths=[PAGE_W - 96])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), WARN_BG),
        ("LINEBEFORE", (0, 0), (0, -1), 3, WARN_BAR),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    return t


def qa(q, a):
    return KeepTogether([
        Paragraph(q, _s("Q", fontName="YaHeiBd", spaceBefore=8, spaceAfter=2)),
        Paragraph(a, ST["Body"]),
    ])


# ── 正文内容 ──────────────────────────────────────────────────
def build_story(toc):
    s = [NextPageTemplate("Body"), PageBreak()]
    s.append(Paragraph("目　录", ST["TocTitle"]))
    s.append(Spacer(1, 4))
    s.append(toc)
    s.append(PageBreak())

    s += chapter("1. 软件简介")
    s.append(P("GarmentStitcher 是一款<b>本地离线</b>的服装样品处理工具。"
               "它利用 AI（rembg 图像分割）结合轮廓匹配算法，自动识别<b>正面、反面</b>两张照片中的服装区域，"
               "经裁剪后拼接为 <b>1:1 正方形</b>输出图，并支持人工微调。"))
    s.append(P("主要用途：电商商品上架、服装打样归档等需要「正反面合并为一张方图」的场景。"
               "全程离线运行，<b>无需联网、无需安装 Python</b>。"))

    s += chapter("2. 系统要求")
    s.append(table(
        ["项目", "要求"],
        [["操作系统", "Windows 10 / 11（64 位）"],
         ["内存", "建议 8 GB 及以上"],
         ["磁盘空间", "安装后约 530 MB"],
         ["网络", "<b>不需要</b>（模型已内置，全程离线）"],
         ["Python", "<b>不需要</b>（已打包为独立程序）"]],
        widths=[120, PAGE_W - 96 - 120],
    ))

    s += chapter("3. 安装与启动")
    s.append(H2("3.1 方式一：安装版（推荐）"))
    s.extend(steps([
        "双击 <font name='Consolas'>GarmentStitcher_Setup.exe</font>；",
        "按提示完成安装（默认安装到 <font name='Consolas'>C:\\Program Files\\GarmentStitcher</font>）；",
        "安装程序会自动创建<b>桌面</b>与<b>开始菜单</b>快捷方式；",
        "若系统缺少 VC++ 运行库，安装程序会自动补装。",
    ]))
    s.append(H2("3.2 方式二：免安装便携版"))
    s.extend(steps([
        "解压 <font name='Consolas'>GarmentStitcher_2.0.zip</font>；",
        "进入解压出的 <font name='Consolas'>GarmentStitcher</font> 文件夹；",
        "双击 <font name='Consolas'>GarmentStitcher.exe</font> 即可运行。",
    ]))
    s.append(note("<b>注意：</b>便携版必须整体解压（GarmentStitcher.exe 与 _internal 文件夹必须在同一目录），"
                  "不能只拷贝 exe 文件。"))

    s += chapter("4. 快速上手")
    s.extend(steps([
        "<b>启动程序</b>：首次启动自动加载 AI 模型，状态栏显示「模型加载中…」，数秒后变为「模型就绪」；",
        "<b>准备图片</b>：将同一件衣服的正面、反面照片放入同一文件夹，文件名按顺序命名"
        "（如 <font name='Consolas'>01_正面.jpg、01_反面.jpg、02_正面.jpg、02_反面.jpg</font>）；",
        "<b>选择文件夹</b>：点击工具栏文件夹图标（或输入路径后回车）；",
        "点击<b>「AI 处理」</b>：程序按顺序逐个处理，完成一组立即可审，无需等待全部完成；",
        "<b>审核与调整</b>：对识别不准的选框进行拖动、旋转微调；",
        "<b>导出</b>：点「导出」输出当前一组，或点「批量」一次性输出全部。",
    ]))

    s += chapter("5. 文件准备与配对规则")
    s.append(P("程序将文件夹内图片<b>按文件名排序</b>、<b>两两配对</b>：第 1、2 张为第 1 组，第 3、4 张为第 2 组，依此类推。"))
    s.append(P("每组中：<b>前一张 = 正面</b>，<b>后一张 = 反面</b>。若识别反了，可点「交换」按钮或按 <b>X</b> 键互换正反面。"))
    s.append(table(
        ["项目", "说明"],
        [["支持格式", "JPG / JPEG / PNG / BMP / TIFF"],
         ["建议分辨率", "单张 ≤ 4080px（过大会拖慢界面）"],
         ["命名建议", "01_正面.jpg / 01_反面.jpg（按名称排序配对）"]],
        widths=[110, PAGE_W - 96 - 110],
    ))

    s += chapter("6. 界面说明")
    s.append(P("窗口左半区为<b>预览</b>，右半区为<b>编辑</b>（可拖动绿色选框）。顶部工具栏按钮如下："))
    s.append(table(
        ["按钮", "功能"],
        [["AI 处理", "开始对全部图片进行 AI 识别（流式，完成一组即可审）"],
         ["◀ / ▶", "上一组 / 下一组"],
         ["重置AI", "放弃手动调整，恢复为 AI 自动识别的框"],
         ["液化", "打开液化工具，微调拼接图细节"],
         ["导出", "导出当前这一组"],
         ["批量", "一次性导出全部已处理图片"],
         ["关于", "（右上角）显示软件信息"]],
        widths=[110, PAGE_W - 96 - 110],
    ))

    s += chapter("7. 鼠标操作（编辑区）")
    s.append(table(
        ["操作", "效果"],
        [["拖动选框四角", "调整选框大小"],
         ["拖动选框边中点", "单边缩放"],
         ["拖动选框内部", "移动选框"],
         ["在选框外、靠近角点拖动", "旋转选框（绕中心）"],
         ["按住 Shift 再旋转", "以 15° 为步进吸附"],
         ["滚动滚轮", "缩放视图"],
         ["中键 / 右键拖动", "平移视图"]],
        widths=[170, PAGE_W - 96 - 170],
    ))

    s += chapter("8. 键盘快捷键")
    s.append(table(
        ["按键", "功能"],
        [["← / →", "上一组 / 下一组"],
         ["E 或 S", "导出当前组"],
         ["X", "交换正反面"],
         ["R", "旋转角度归零"],
         ["F", "编辑区适配窗口"],
         [",", "逆时针旋转 0.5°"],
         [".", "顺时针旋转 0.5°"],
         ["F1", "关于对话框"]],
        widths=[110, PAGE_W - 96 - 110],
    ))

    s += chapter("9. 液化工具（可选）")
    s.append(P("点击「液化」会生成当前组的全分辨率拼接预览，并打开液化编辑窗口："))
    s.extend(bullets([
        "用画笔在图上拖动，进行 PS 风格的局部变形微调；",
        "支持撤销（最多 50 步）；",
        "点「应用」后，效果保存到输出目录，并替换当前预览，直到切换到其他组。",
    ]))

    s += chapter("10. 输出说明")
    s.extend(bullets([
        "导出结果统一保存到输入文件夹下的 <font name='Consolas'>审核输出</font> 子目录；",
        "输出为 <b>PNG</b> 格式，文件名与每组的<b>正面图片</b>同名；",
        "输出为 <b>1:1 正方形</b>（正面靠右、反面靠左）；",
        "每次手工调整自动保存到 <font name='Consolas'>annotations.json</font>（约 0.8 秒防抖），"
        "下次打开该文件夹时优先读取手工标注。",
    ]))

    s += chapter("11. 常见问题（FAQ）")
    s.append(qa("<b>Q1：</b>提示「模型加载中」很久没反应？",
                "首次启动加载模型需要几秒到十几秒，属正常现象。若超过 1 分钟，请检查安装目录 "
                "<font name='Consolas'>_internal/models/</font> 下是否存在模型文件。"))
    s.append(qa("<b>Q2：</b>AI 识别的框不准怎么办？",
                "直接用鼠标拖动 / 旋转选框修正即可，修正后会自动保存。"))
    s.append(qa("<b>Q3：</b>正反面识别反了？",
                "按 <b>X</b> 键或点「交换」即可。"))
    s.append(qa("<b>Q4：</b>输出的图是方图吗？",
                "是的，最终输出统一为 1:1 正方形。"))
    s.append(qa("<b>Q5：</b>新电脑上能直接运行吗？",
                "可以。程序完全自包含（含模型与运行库），安装版会自动处理 VC++ 运行库，便携版解压即用，均无需联网。"))
    s.append(qa("<b>Q6：</b>可以处理多少张图？",
                "按文件名排序两两成对即可，没有数量限制，批量导出会按顺序全部处理。"))

    s += chapter("12. 技术支持与版权")
    s.append(P("本软件由 zisonzishen 开发，仅供服装样品正反面拼接处理使用。"))
    s.append(Paragraph("Powered by zisonzishen", ST["BodyC"]))
    return s


def main():
    toc = TableOfContents()
    toc.levelStyles = TOC_LEVEL_STYLES
    toc.dotsMinLevel = 0

    story = build_story(toc)

    frame_cover = Frame(0, 0, PAGE_W, PAGE_H, id="cover")
    frame_body = Frame(48, 48, PAGE_W - 96, PAGE_H - 96, id="body")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    doc = ManualDoc(
        str(OUT_PDF),
        pagesize=A4,
        pageTemplates=[
            PageTemplate(id="Cover", frames=[frame_cover], onPage=cover_page),
            PageTemplate(id="Body", frames=[frame_body], onPage=body_page),
        ],
        title="GarmentStitcher 使用说明书",
        author="zisonzishen",
        subject="服装正反面拼接工具使用说明书",
    )
    doc.multiBuild(story)
    print("PDF 已生成:", OUT_PDF)
    print("大小:", OUT_PDF.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
