"""PPT Agent: creates and edits designed PowerPoint (.pptx) presentations in workspace/.

    task -> tool-calling loop (read / create / add / update / delete slides) -> "Created X" -> result

KEY IDEA: the LLM writes CONTENT as structured data (a Slide: layout + title + bullets / stats / chart ...);
Python renders it into a DESIGNED slide (16:9, colour theme, accent shapes, native charts, slide numbers)
and ENFORCES limits (e.g. max bullets) so slides stay readable. The LLM never places shapes itself.

Layouts: bullets · two_column · stats (big numbers) · chart (native, editable) · quote · section.
Themes:  ocean · forest · sunset · slate (stored in the file, so added slides match the deck).

Run the standalone test from the project root:  python -m agents.ppt
"""
import re
from typing import Literal

from langchain_core.tools import tool
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt
from pydantic import BaseModel, Field

from agents.document import _safe_path, list_files  # same workspace + same security check
from agents.tool_agent import run_tool_agent

TITLE_ONLY_LAYOUT = 5  # default template's "Title Only": keeps a real title placeholder (read/rename/delete use it)
MAX_BULLETS, MAX_BULLET_CHARS = 6, 120  # more than this overflows a slide
MAX_STATS, MAX_CATEGORIES = 4, 12
HEAD_FONT, BODY_FONT = "Segoe UI Semibold", "Segoe UI"

# name -> colours. dark: title/section backgrounds · primary/accent: shapes · text/muted: body · soft: cards
THEMES = {
    "ocean": dict(dark="0F2A44", primary="1F6FEB", accent="22B8CF", text="1E293B", muted="64748B", soft="EEF4FB",
                  series=["1F6FEB", "22B8CF", "0F2A44", "7C9CBF", "F59E0B", "94A3B8"]),
    "forest": dict(dark="12372A", primary="2F855A", accent="F6AD55", text="1C2B24", muted="5F7268", soft="EEF6F0",
                   series=["2F855A", "F6AD55", "12372A", "68D391", "C05621", "A0AEC0"]),
    "sunset": dict(dark="2D1B2E", primary="E4572E", accent="F3A712", text="2B2024", muted="7A6A70", soft="FBF1EC",
                   series=["E4572E", "F3A712", "2D1B2E", "A8C686", "669BBC", "B8A9AE"]),
    "slate": dict(dark="111827", primary="4F46E5", accent="10B981", text="1F2937", muted="6B7280", soft="F3F4F6",
                  series=["4F46E5", "10B981", "111827", "F59E0B", "EC4899", "9CA3AF"]),
}
Theme = Literal["ocean", "forest", "sunset", "slate"]


class Stat(BaseModel):
    value: str = Field(description="The big number, short, e.g. '82%', '1,200', '$4.1M'.")
    label: str = Field(description="What it means, a few words.")


class Chart(BaseModel):
    type: Literal["bar", "line", "pie"] = "bar"
    categories: list[str] = Field(description=f"X-axis labels / pie slices (max {MAX_CATEGORIES}).")
    values: list[float] = Field(description="One number per category, plain numbers (no units, no commas).")
    series_name: str = Field(default="", description="What the numbers are, e.g. 'Revenue'.")


class Slide(BaseModel):
    title: str
    layout: Literal["bullets", "two_column", "stats", "chart", "quote", "section"] = Field(
        default="bullets", description="bullets = normal list · two_column = compare two lists · stats = 1-4 big "
        "numbers · chart = a chart (+ optional bullets beside it) · quote = one key sentence · section = divider.")
    bullets: list[str] = Field(default_factory=list, description=f"At most {MAX_BULLETS} short bullet points "
                               "(left column in two_column). 'Label: detail' shows the label in bold.")
    right: list[str] = Field(default_factory=list, description="two_column only: the right column's bullets.")
    left_heading: str = Field(default="", description="two_column only: heading above the left column.")
    right_heading: str = Field(default="", description="two_column only: heading above the right column.")
    stats: list[Stat] = Field(default_factory=list, description=f"stats only: 1-{MAX_STATS} big numbers.")
    chart: Chart | None = Field(default=None, description="chart only: the data to plot.")
    text: str = Field(default="", description="quote: the sentence to highlight · section: optional subtitle.")
    notes: str = Field(default="", description="Optional speaker notes.")


# --- checks: the error goes back to the LLM, which then fixes its call (e.g. splits the content) ---
def _check(slide: Slide) -> None:
    for name, items in (("bullets", slide.bullets), ("right", slide.right)):
        if len(items) > MAX_BULLETS:
            raise ValueError(f"Slide '{slide.title}' has {len(items)} {name}; max is {MAX_BULLETS}. "
                             "Split it into more slides.")
        too_long = [b for b in items if len(b) > MAX_BULLET_CHARS]
        if too_long:
            raise ValueError(f"Bullets must be under {MAX_BULLET_CHARS} characters. Shorten: {too_long[0][:60]}...")
    if slide.layout == "stats" and not 1 <= len(slide.stats) <= MAX_STATS:
        raise ValueError(f"A stats slide needs 1-{MAX_STATS} stats; got {len(slide.stats)}.")
    if any(len(s.value) > 14 for s in slide.stats):
        raise ValueError("Stat values must be short (max 14 characters), e.g. '82%' or '$4.1M'.")
    if slide.layout == "chart":
        c = slide.chart
        if c is None or not c.categories:
            raise ValueError("A chart slide needs `chart` with categories and values.")
        if len(c.categories) != len(c.values):
            raise ValueError(f"Chart has {len(c.categories)} categories but {len(c.values)} values; they must match.")
        if len(c.categories) > MAX_CATEGORIES:
            raise ValueError(f"Charts take at most {MAX_CATEGORIES} categories.")
    if slide.layout == "two_column" and not (slide.bullets or slide.right):
        raise ValueError("A two_column slide needs `bullets` (left) and/or `right`.")
    if slide.layout == "quote" and not 0 < len(slide.text) <= 300:
        raise ValueError("A quote slide needs `text` (max 300 characters).")


# --- drawing helpers ---
def _rgb(hex_: str) -> RGBColor:
    return RGBColor.from_string(hex_)


def _theme(prs) -> dict:
    """The deck's theme is stored in its properties, so slides added later match (old decks -> ocean)."""
    return THEMES.get(prs.core_properties.category, THEMES["ocean"])


def _shape(slide, kind, x, y, w, h, fill: str, name: str = "deco"):
    s = slide.shapes.add_shape(kind, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = _rgb(fill)
    s.line.fill.background()  # no outline
    s.shadow.inherit = False
    s.name = name  # "deco" shapes are decoration: skipped when reading the slide's text
    return s


def _box(slide, x, y, w, h, name: str):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tb.text_frame.word_wrap = True
    return tb.text_frame


def _font(run, size, color, *, bold=None, italic=False, face=BODY_FONT):
    run.font.size, run.font.name, run.font.italic = Pt(size), face, italic
    run.font.color.rgb = _rgb(color)
    if bold is not None:
        run.font.bold = bold


def _bullet(p, color: str) -> None:
    """A real PowerPoint bullet (coloured square, hanging indent) via the paragraph's XML properties."""
    pPr = p._p.get_or_add_pPr()
    pPr.set("marL", str(Emu(Pt(22))))
    pPr.set("indent", str(-Emu(Pt(22))))
    etree.SubElement(etree.SubElement(pPr, qn("a:buClr")), qn("a:srgbClr"), val=color)
    etree.SubElement(pPr, qn("a:buFont"), typeface="Arial")
    etree.SubElement(pPr, qn("a:buChar"), char="■")


LABEL = re.compile(r"^([^:\d*]{2,40}):\s")  # "Vacation: 24 days" -> bold label (not "3:00 PM")


def _write(tf, lines, size, color, t, *, bullets=False, bold=None, italic=False, face=BODY_FONT, align=None,
           space=10, first_heading: str = ""):
    """Replace a text frame's content with styled paragraphs. **bold** works; bullets get a themed marker."""
    tf.clear()
    paras = ([("h", first_heading)] if first_heading else []) + [("b", x) for x in lines]
    for i, (kind, text) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if kind == "h":
            r = p.add_run()
            r.text = text
            _font(r, size + 4, t["primary"], bold=True, face=HEAD_FONT)
            p.space_after = Pt(space + 4)
            continue
        for j, piece in enumerate((LABEL.sub(r"**\1:** ", text) if bullets else text).split("**")):
            if piece:  # odd pieces sat between ** markers -> bold
                r = p.add_run()
                r.text, r.font.bold = piece, j % 2 == 1
        for r in p.runs:
            _font(r, size, color, bold=r.font.bold if bold is None else bold, italic=italic, face=face)
        p.space_after = Pt(space)
        if align is not None:
            p.alignment = align
        if bullets:
            _bullet(p, t["accent"])


def _title(slide, text, t, *, x=0.8, y=0.45, w=11.7, h=0.95, size=None, color=None, anchor=MSO_ANCHOR.BOTTOM):
    ph = slide.shapes.title
    ph.left, ph.top, ph.width, ph.height = Inches(x), Inches(y), Inches(w), Inches(h)
    tf = ph.text_frame
    tf.word_wrap, tf.auto_size, tf.vertical_anchor = True, MSO_AUTO_SIZE.NONE, anchor
    tf.text = text
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    for r in p.runs:
        _font(r, size or (30 if len(text) <= 48 else 25), color or t["dark"], bold=False, face=HEAD_FONT)


def _footer(slide, prs, t) -> None:
    """Deck title bottom-left, a LIVE slide-number field bottom-right (stays right after moves/deletes)."""
    left = _box(slide, 0.8, 6.95, 8, 0.35, "deco")
    left.text = prs.core_properties.title or ""
    for r in left.paragraphs[0].runs:
        _font(r, 10, t["muted"])
    num = _box(slide, 11.5, 6.95, 1.0, 0.35, "deco").paragraphs[0]
    num.alignment = PP_ALIGN.RIGHT
    fld = etree.SubElement(num._p, qn("a:fld"), id="{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}", type="slidenum")
    rpr = etree.SubElement(fld, qn("a:rPr"), lang="en-US", sz="1000")
    etree.SubElement(etree.SubElement(rpr, qn("a:solidFill")), qn("a:srgbClr"), val=t["muted"])
    etree.SubElement(fld, qn("a:t")).text = "‹#›"


def _content_frame(slide, prs, spec: Slide, t) -> None:
    """White slide + left accent bar + title with underline + footer: shared by every content layout."""
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = _rgb("FFFFFF")
    _shape(slide, MSO_SHAPE.RECTANGLE, 0, 0, 0.18, 7.5, t["primary"])
    _title(slide, spec.title, t)
    _shape(slide, MSO_SHAPE.RECTANGLE, 0.8, 1.48, 1.1, 0.07, t["accent"])
    _footer(slide, prs, t)


def _bullets_size(n: int) -> int:
    return 28 if n <= 3 else 24 if n <= 5 else 22


def _chart(slide, c: Chart, t, x, y, w, h) -> None:
    data = CategoryChartData()
    data.categories = c.categories
    data.add_series(c.series_name or "Value", c.values)
    kind = {"bar": XL_CHART_TYPE.COLUMN_CLUSTERED, "line": XL_CHART_TYPE.LINE_MARKERS, "pie": XL_CHART_TYPE.PIE}
    frame = slide.shapes.add_chart(kind[c.type], Inches(x), Inches(y), Inches(w), Inches(h), data)
    frame.name = "chart"
    chart = frame.chart
    chart.has_title = False  # the slide title already says what it is
    chart.font.size, chart.font.name = Pt(12), BODY_FONT
    chart.font.color.rgb = _rgb(t["muted"])
    plot, series = chart.plots[0], chart.plots[0].series[0]
    plot.has_data_labels = True
    labels = plot.data_labels
    labels.font.size, labels.font.bold = Pt(12), True
    labels.font.color.rgb = _rgb(t["text"])
    if c.type == "pie":
        chart.has_legend, chart.legend.position, chart.legend.include_in_layout = True, XL_LEGEND_POSITION.RIGHT, False
        chart.legend.font.size = Pt(14)
        labels.show_percentage, labels.show_value, labels.number_format_is_linked = True, False, False
        labels.number_format = "0%"
        labels.font.color.rgb = _rgb("FFFFFF")
        for i, point in enumerate(series.points):
            point.format.fill.solid()
            point.format.fill.fore_color.rgb = _rgb(t["series"][i % len(t["series"])])
        return
    chart.has_legend = False
    labels.number_format, labels.number_format_is_linked = "General", False
    if c.type == "bar":
        plot.gap_width = 70
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = _rgb(t["primary"])
    else:
        series.smooth = False
        series.format.line.color.rgb = _rgb(t["primary"])
        series.format.line.width = Pt(3)
        series.marker.style, series.marker.size = XL_MARKER_STYLE.CIRCLE, 9
        series.marker.format.fill.solid()
        series.marker.format.fill.fore_color.rgb = _rgb(t["accent"])
        labels.position = XL_LABEL_POSITION.ABOVE
    axis = chart.value_axis
    axis.has_major_gridlines = True
    if c.type == "bar" and min(c.values) >= 0:
        axis.minimum_scale = 0  # bars must start at zero, or small differences look huge
    axis.major_gridlines.format.line.color.rgb = _rgb("E2E8F0")
    axis.format.line.fill.background()
    chart.category_axis.format.line.color.rgb = _rgb("CBD5E1")
    chart.category_axis.tick_labels.font.size = Pt(12)


def _render(prs, spec: Slide):
    """Build one designed slide from its spec."""
    t = _theme(prs)
    slide = prs.slides.add_slide(prs.slide_layouts[TITLE_ONLY_LAYOUT])

    if spec.layout == "section":
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = _rgb(t["primary"])
        _shape(slide, MSO_SHAPE.OVAL, 9.6, 3.2, 5.2, 5.2, t["dark"])
        _title(slide, spec.title, t, x=0.9, y=2.2, w=9, h=1.6, size=40, color="FFFFFF")
        _shape(slide, MSO_SHAPE.RECTANGLE, 0.9, 3.95, 1.3, 0.08, t["accent"])
        if spec.text:
            _write(_box(slide, 0.9, 4.2, 8.5, 1.2, "subtitle"), [spec.text], 20, "FFFFFF", t)
    else:
        _content_frame(slide, prs, spec, t)

    if spec.layout == "bullets":
        tf = _box(slide, 0.8, 1.85, 11.7, 4.9, "body")
        _write(tf, spec.bullets, _bullets_size(len(spec.bullets)), t["text"], t, bullets=True, space=18)
    elif spec.layout == "two_column":
        size = _bullets_size(max(len(spec.bullets), len(spec.right)))
        for x, items, head, name in ((0.8, spec.bullets, spec.left_heading, "left"),
                                     (6.85, spec.right, spec.right_heading, "right")):
            _shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.9, 5.65, 4.75, t["soft"]).adjustments[0] = 0.05
            _write(_box(slide, x + 0.35, 2.1, 5.0, 4.4, name), items, size - 2, t["text"], t, bullets=True,
                   first_heading=head)
    elif spec.layout == "stats":
        n, gap = len(spec.stats), 0.35
        w = (11.7 - gap * (n - 1)) / n
        for i, s in enumerate(spec.stats):
            x = 0.8 + i * (w + gap)
            card = _shape(slide, MSO_SHAPE.ROUNDED_RECTANGLE, x, 2.3, w, 3.1, t["soft"], name="stat")
            card.adjustments[0] = 0.06
            _shape(slide, MSO_SHAPE.RECTANGLE, x + 0.4, 2.3, w - 0.8, 0.09, t["primary"])
            tf = card.text_frame
            tf.word_wrap, tf.vertical_anchor = True, MSO_ANCHOR.MIDDLE
            tf.margin_left = tf.margin_right = Inches(0.25)
            _write(tf, [s.value, s.label], 16, t["muted"], t, align=PP_ALIGN.CENTER, space=4)
            _font(tf.paragraphs[0].runs[0], 54 if n <= 3 else 44, t["primary"], bold=True, face=HEAD_FONT)
    elif spec.layout == "chart":
        if spec.bullets:  # chart left, takeaways right
            _chart(slide, spec.chart, t, 0.7, 1.8, 7.4, 4.95)
            _write(_box(slide, 8.5, 2.0, 4.0, 4.7, "body"), spec.bullets, 17, t["text"], t, bullets=True, space=12)
        else:
            _chart(slide, spec.chart, t, 0.8, 1.8, 11.7, 4.95)
    elif spec.layout == "quote":
        mark = _box(slide, 0.75, 1.6, 1.5, 1.6, "deco")
        mark.text = "“"
        _font(mark.paragraphs[0].runs[0], 120, t["accent"], face="Georgia")
        tf = _box(slide, 1.9, 2.3, 10.2, 3.8, "quote")
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        _write(tf, [spec.text], 30 if len(spec.text) <= 140 else 24, t["dark"], t, italic=True, face="Georgia")

    if spec.notes:
        slide.notes_slide.notes_text_frame.text = spec.notes
    return slide


def _render_title(prs, title: str, subtitle: str) -> None:
    t = _theme(prs)
    slide = prs.slides.add_slide(prs.slide_layouts[TITLE_ONLY_LAYOUT])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = _rgb(t["dark"])
    _shape(slide, MSO_SHAPE.OVAL, 8.6, -1.6, 6.4, 6.4, t["primary"])
    _shape(slide, MSO_SHAPE.OVAL, 11.0, 4.6, 3.2, 3.2, t["accent"])
    _title(slide, title, t, x=0.9, y=1.6, w=8.2, h=2.5, size=44 if len(title) <= 40 else 36, color="FFFFFF")
    _shape(slide, MSO_SHAPE.RECTANGLE, 0.9, 4.3, 1.4, 0.09, t["accent"])
    if subtitle:
        _write(_box(slide, 0.9, 4.55, 8.0, 1.3, "subtitle"), [subtitle], 20, "CBD5E1", t)


# --- reading (also used by the UI preview) ---
def slide_text(slide) -> tuple[str, list[str]]:
    """(title, text lines) for any slide: designed ones (named shapes) and older placeholder-based ones.
    Charts are described as text, so agents and the preview can "see" them."""
    title = slide.shapes.title.text if slide.shapes.title is not None else ""
    lines = []
    for s in slide.shapes:
        if s == slide.shapes.title or s.name == "deco":
            continue
        if s.has_text_frame:
            lines += [p.text for p in s.text_frame.paragraphs if p.text]
        elif getattr(s, "has_chart", False) and s.has_chart:
            plot = s.chart.plots[0]
            pairs = ", ".join(f"{c} {v:g}" for c, v in zip(plot.categories, plot.series[0].values))
            lines.append(f"[chart: {plot.series[0].name}: {pairs}]")
    return title, lines


def _body(slide):
    """The shape `bullets` updates go into: designed 'body' / 'subtitle', or an old deck's placeholder 1."""
    for s in slide.shapes:
        if s.name in ("body", "subtitle"):
            return s
    return slide.placeholders[1] if len(slide.placeholders) > 1 else None


# --- helpers ---
def _open(filename: str):
    path = _safe_path(filename)
    if path.suffix.lower() != ".pptx":
        raise ValueError("Only .pptx files are supported.")
    if not path.exists():
        raise FileNotFoundError(f"'{filename}' not found. Use list_files to see what exists.")
    return path, Presentation(path)


def _slide_index(prs, slide_number: int) -> int:
    """1-based slide number (what humans and the LLM use) -> 0-based index, with a helpful error."""
    if not 1 <= slide_number <= len(prs.slides):
        raise ValueError(f"slide_number must be 1..{len(prs.slides)}")
    return slide_number - 1


def _move_last_to(prs, position: int) -> None:
    """python-pptx has no 'move slide' API; slide order is the order of <p:sldId> elements in the XML."""
    order = prs.slides._sldIdLst
    last = order[-1]
    order.remove(last)
    order.insert(position, last)


# --- TOOLS ---
@tool
def read_presentation(filename: str) -> str:
    """Show every slide: its number, title, text (charts as data) and speaker notes."""
    _, prs = _open(filename)
    lines = [f"{filename}: {len(prs.slides)} slides"]
    for n, slide in enumerate(prs.slides, start=1):
        title, text = slide_text(slide)
        lines.append(f"Slide {n}: {title or '(no title)'}" + "".join(f"\n  - {x}" for x in text))
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame.text:
            lines.append(f"  notes: {slide.notes_slide.notes_text_frame.text}")
    return "\n".join(lines)


@tool
def create_presentation(filename: str, title: str, slides: list[Slide], subtitle: str = "",
                        theme: Theme = "ocean") -> str:
    """Create a NEW designed 16:9 .pptx: a title slide (title + subtitle) followed by one slide per item in `slides`.
    Pick each slide's `layout` to fit its content, and a colour `theme`. Refuses to overwrite an existing file."""
    path = _safe_path(filename)
    if path.suffix.lower() != ".pptx":
        raise ValueError("filename must end with .pptx")
    if path.exists():  # overwriting needs human approval -> Phase 9
        raise FileExistsError(f"'{filename}' already exists. Pick a new name or edit it with the other tools.")
    for spec in slides:
        _check(spec)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)  # 16:9 widescreen
    prs.core_properties.title, prs.core_properties.category = title, theme
    _render_title(prs, title, subtitle)
    for spec in slides:
        _render(prs, spec)
    prs.save(path)
    return f"Created {filename} with {len(prs.slides)} slides (theme {theme})"


@tool
def add_slide(filename: str, slide: Slide, position: int = 0) -> str:
    """Add a slide (any layout) in the deck's theme. position = slide number it should become (1 = first);
    0 = at the end."""
    path, prs = _open(filename)
    _check(slide)
    _render(prs, slide)
    if position:
        _move_last_to(prs, _slide_index(prs, position))
    prs.save(path)
    return f"Added slide '{slide.title}' to {filename} ({len(prs.slides)} slides now)"


@tool
def update_slide(filename: str, slide_number: int, title: str = "", bullets: list[str] | None = None,
                 notes: str | None = None) -> str:
    """Change an existing slide (slide_number starts at 1). Only the fields you pass change; the rest keep
    their current content (e.g. pass only `title` to rename a slide). `bullets` works on bullet slides, chart
    slides with bullets and the title slide's subtitle; to change a chart, stats or columns, delete the slide and
    add a new one."""
    path, prs = _open(filename)
    target = prs.slides[_slide_index(prs, slide_number)]
    t = _theme(prs)
    if title:
        ph = target.shapes.title
        old = ph.text_frame.paragraphs[0].runs  # keep the designed font, size and colour
        style = (old[0].font.size, old[0].font.name, old[0].font.color.rgb) if old and old[0].font.size else None
        ph.text_frame.text = title
        if style:
            for r in ph.text_frame.paragraphs[0].runs:
                r.font.size, r.font.name, r.font.color.rgb = style
    if bullets is not None:
        body = _body(target)
        if body is None:
            raise ValueError(f"Slide {slide_number} has no bullet area. Delete it and add a new slide instead.")
        _check(Slide(title=title or "x", bullets=bullets))
        if body.name == "subtitle":
            _write(body.text_frame, bullets[:1], 20, "CBD5E1" if slide_number == 1 else "FFFFFF", t)
        elif body.name == "body":
            size = 17 if any(s.name == "chart" for s in target.shapes) else _bullets_size(len(bullets))
            _write(body.text_frame, bullets, size, t["text"], t, bullets=True, space=12)
        else:  # older placeholder-based deck
            body.text_frame.clear()
            body.text_frame.text = bullets[0] if bullets else ""
            for b in bullets[1:]:
                body.text_frame.add_paragraph().text = b
    if notes is not None:  # None = keep existing notes; "" = clear them
        target.notes_slide.notes_text_frame.text = notes
    prs.save(path)
    return f"Updated slide {slide_number} of {filename}"


@tool
def delete_slide(filename: str, slide_number: int) -> str:
    """Request deleting one slide (slide_number starts at 1). Deleting can't be undone, so the slide is only
    removed after the user approves."""
    from approvals import queue_file_action  # imported here to avoid a circular import

    _, prs = _open(filename)
    title = prs.slides[_slide_index(prs, slide_number)].shapes.title
    title = title.text if title is not None else ""
    action_id = queue_file_action({"filename": filename, "slide_number": slide_number, "title": title})
    return f"Deletion of slide {slide_number} ('{title}') requested as file:{action_id}. NOT deleted until the user approves."


def delete_slide_now(filename: str, slide_number: int, title: str | None = None) -> str:
    """HUMAN-APPROVED deletion (called by approvals.execute, never by the LLM).
    Checks the title too: if an earlier deletion shifted the numbers, find the slide by title instead."""
    path, prs = _open(filename)
    titles = [s.shapes.title.text if s.shapes.title is not None else "" for s in prs.slides]
    index = _slide_index(prs, slide_number)
    if title is not None and titles[index] != title:
        if title not in titles:
            raise ValueError(f"Slide '{title}' is no longer in {filename}; nothing deleted.")
        index = titles.index(title)
    order = prs.slides._sldIdLst
    entry = order[index]
    prs.part.drop_rel(entry.rId)  # unlink the slide part...
    order.remove(entry)           # ...and remove it from the slide order
    prs.save(path)
    return f"Deleted slide '{titles[index]}' from {filename} ({len(prs.slides)} slides left)"


TOOLS = [list_files, read_presentation, create_presentation, add_slide, update_slide, delete_slide]

SYSTEM = f"""You are the PPT Agent. You create and edit designed PowerPoint (.pptx) presentations in the user's workspace.
Design (the tools draw everything; you choose the layout per slide and a theme per deck):
- Vary layouts to fit the content, don't make every slide bullets:
  numbers/KPIs -> "stats" (1-{MAX_STATS} big numbers) · data over categories or time -> "chart" (bar / line / pie,
  with up to 3 takeaway bullets beside it) · comparisons (before/after, pros/cons) -> "two_column" ·
  one key message -> "quote" · decks of 7+ slides -> "section" dividers · everything else -> "bullets".
- Themes: ocean (default, corporate), forest (calm, sustainability, HR), sunset (bold, marketing), slate (minimal, tech).
- Good slides: a short title (under ~50 characters), at most {MAX_BULLETS} concise bullets, one idea per slide.
  "Label: detail" bullets show the label in bold. Put extra detail in speaker notes.
Rules:
- Use only facts given in the task; don't invent numbers, names or dates. Chart values must come from the given data.
  Don't add units or currency the data doesn't have (no "M", "$" or "%" unless given).
- Before editing an existing file, call read_presentation to see its slides and their numbers.
- Slide numbers change after adding/deleting slides: re-read the presentation if you need them again.
- delete_slide only REQUESTS a deletion; the user approves it afterwards. Report it as requested, not as done.
- End with a short message: what you did and which file you created or changed."""


def ppt_agent(state: dict) -> dict:
    answer = run_tool_agent("ppt_agent", SYSTEM, state["task"], TOOLS)
    print(f"[ppt_agent] {answer[:80]}...")
    return {"agent_results": {"ppt_agent": answer}}


def _titles(filename: str) -> list[str]:
    _, prs = _open(filename)
    return [s.shapes.title.text for s in prs.slides]


if __name__ == "__main__":
    import sys

    from userdata import use_temp_data, workspace_dir

    use_temp_data()
    WORKSPACE = workspace_dir()

    sys.stdout.reconfigure(encoding="utf-8")

    # 1. Tools alone (no LLM)
    (WORKSPACE / "t.pptx").unlink(missing_ok=True)
    create_presentation.invoke({"filename": "t.pptx", "title": "T", "subtitle": "s", "theme": "forest", "slides": [
        {"title": "A", "bullets": ["a1"]}, {"title": "C"},
        {"title": "Numbers", "layout": "stats", "stats": [{"value": "82%", "label": "satisfied"}]},
        {"title": "Sales", "layout": "chart", "chart": {"categories": ["N", "S"], "values": [120, 90.5],
                                                         "series_name": "Revenue"}, "bullets": ["North leads"]},
        {"title": "Compare", "layout": "two_column", "bullets": ["old"], "right": ["new"], "left_heading": "Before"},
        {"title": "Big idea", "layout": "quote", "text": "Less is more."},
        {"title": "Part 2", "layout": "section", "text": "Details"}]})
    add_slide.invoke({"filename": "t.pptx", "slide": {"title": "B", "notes": "note"}, "position": 3})
    assert _titles("t.pptx") == ["T", "A", "B", "C", "Numbers", "Sales", "Compare", "Big idea", "Part 2"]
    update_slide.invoke({"filename": "t.pptx", "slide_number": 2, "title": "A2"})  # rename only
    deck = read_presentation.invoke({"filename": "t.pptx"})
    assert "- a1" in deck, "renaming must keep the bullets"
    assert "[chart: Revenue: N 120, S 90.5]" in deck and "- 82%" in deck and "- Before" in deck, deck
    assert "Less is more." in deck and "- North leads" in deck, deck
    update_slide.invoke({"filename": "t.pptx", "slide_number": 2, "bullets": ["x"]})  # bullets only
    update_slide.invoke({"filename": "t.pptx", "slide_number": 6, "bullets": ["South grows"]})  # chart takeaways
    try:
        update_slide.invoke({"filename": "t.pptx", "slide_number": 5, "bullets": ["y"]})  # stats: no bullet area
        raise AssertionError("stats slide has no bullets to update")
    except ValueError:
        pass
    _, prs = _open("t.pptx")
    assert _theme(prs) is THEMES["forest"] and prs.slide_width == Inches(13.333)
    msg = delete_slide.invoke({"filename": "t.pptx", "slide_number": 4})  # only a REQUEST now
    assert _titles("t.pptx")[3] == "C" and "NOT deleted" in msg, msg
    import approvals
    approvals.execute(f"file:{msg.split('file:')[1].split('.')[0]}")  # what the approval node does on "approve"
    assert "C" not in _titles("t.pptx"), _titles("t.pptx")
    deck = read_presentation.invoke({"filename": "t.pptx"})
    assert "notes: note" in deck and "- x" in deck and "- a1" not in deck and "South grows" in deck, deck
    for bad in (Slide(title="x", bullets=["b"] * 7), Slide(title="x", layout="stats"),
                Slide(title="x", layout="chart", chart=Chart(categories=["a", "b"], values=[1]))):
        try:
            _check(bad)
            raise AssertionError(f"should be rejected: {bad}")
        except ValueError:
            pass
    (WORKSPACE / "t.pptx").unlink()
    print("tool checks OK\n")
    if "--offline" in sys.argv:
        sys.exit()

    # 2. Create a deck from facts given in the task (the agent should pick fitting layouts)
    deck = WORKSPACE / "onboarding.pptx"
    deck.unlink(missing_ok=True)
    ppt_agent({"task": "Create onboarding.pptx: title slide 'Welcome to Nimbus Labs', then one slide each on "
                       "Vacation (24 paid days a year, 5 carry over), Remote work (up to 3 days a week, "
                       "Tue/Thu in office) and Expenses (submit receipts within 30 days)."})
    titles = _titles("onboarding.pptx")
    print(titles)
    assert len(titles) == 4 and any("vacation" in t.lower() for t in titles), titles

    # 3. Edit: add a slide at the end + rename slide 2
    ppt_agent({"task": "In onboarding.pptx, add a final slide titled 'Questions?' and rename slide 2 to 'Time off'."})
    titles = _titles("onboarding.pptx")
    print(titles)
    assert titles[1] == "Time off" and titles[-1] == "Questions?" and len(titles) == 5, titles

    # 4. Delete by meaning (agent must read first to find the right slide) -> only QUEUED until approved
    ppt_agent({"task": "Remove the slide about expenses from onboarding.pptx."})
    queued = [a for a in approvals.list_pending() if a["kind"] == "file"]
    print(queued)
    assert len(_titles("onboarding.pptx")) == 5 and len(queued) == 1 and "expense" in queued[0]["summary"].lower()
    approvals.execute(queued[0]["id"])
    titles = _titles("onboarding.pptx")
    assert len(titles) == 4 and not any("expense" in t.lower() for t in titles), titles

    # 5. Data -> the agent should choose a chart slide with the given numbers
    ppt_agent({"task": "Create sales.pptx titled 'Q3 Sales' with a chart of revenue by region: "
                       "North 120, South 90, East 75, West 60."})
    deck = read_presentation.invoke({"filename": "sales.pptx"})
    print(deck)
    assert "[chart:" in deck and "North 120" in deck, deck
    (WORKSPACE / "sales.pptx").unlink()

    print("\nPPT agent OK")
