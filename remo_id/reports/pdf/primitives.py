"""PDF generation utilities for remo-id reports."""

import io
import matplotlib.pyplot as plt
import pandas as pd
from typing import List

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib import colors


def fig_to_png_bytes(fig) -> bytes:
    """Convert a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def wrap_lines(c: canvas.Canvas, text: str, font: str, size: int, max_width: float) -> List[str]:
    """
    Wrap `text` into lines that fit `max_width` using the canvas' stringWidth.
    Also respects explicit newlines in `text`.
    """
    lines = []
    for para in str(text).splitlines():
        words = para.split()
        if not words:
            lines.append("")  # blank line
            continue

        cur = words[0]
        for w in words[1:]:
            test = f"{cur} {w}"
            if c.stringWidth(test, font, size) <= max_width:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def pdf_title(c: canvas.Canvas, title: str, subtitle: str = ""):
    """Draw a title and optional subtitle at the top of a PDF page."""
    w, h = letter
    left = 0.75 * inch
    right = w - 0.75 * inch
    max_width = right - left

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.black)
    c.drawString(left, h - 0.9 * inch, title)

    # Subtitle (wrapped)
    y = h - 1.15 * inch
    if subtitle:
        font = "Helvetica"
        size = 10
        leading = 12

        c.setFont(font, size)
        c.setFillColor(colors.grey)

        for line in wrap_lines(c, subtitle, font, size, max_width):
            c.drawString(left, y, line)
            y -= leading

        c.setFillColor(colors.black)

    # Horizontal rule just below subtitle (or at default position if no subtitle)
    rule_y = min(h - 1.25 * inch, y - 6) if subtitle else (h - 1.25 * inch)
    c.line(left, rule_y, right, rule_y)

    return y


def pdf_paragraph(
    c: canvas.Canvas,
    x: float,
    y: float,
    text: str,
    width: float = 7.0*inch,
    leading: float = 12,
    font: str = "Helvetica",
    size: int = 10,
) -> float:
    """
    Draw wrapped text that also respects explicit newlines in `text`.
    Returns the new y after drawing.
    """
    c.setFont(font, size)

    def wrap_one_paragraph(para: str) -> list[str]:
        words = para.split()
        if not words:
            return [""]  # blank line

        lines = []
        cur = words[0]
        for w in words[1:]:
            test = f"{cur} {w}"
            if c.stringWidth(test, font, size) <= width:
                cur = test
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
        return lines

    lines_out = []
    for para in str(text).split("\n"):
        lines_out.extend(wrap_one_paragraph(para))

    for ln in lines_out:
        c.drawString(x, y, ln)
        y -= leading

    return y


def pdf_table(c: canvas.Canvas, x: float, y: float, df: pd.DataFrame,
              col_widths: List[float], row_h: int = 14) -> float:
    """
    Simple table renderer. Expects col_widths in points.
    Returns the new y after drawing.
    """
    c.setFont("Helvetica-Bold", 9)
    # header
    cx = x
    for j, col in enumerate(df.columns):
        c.drawString(cx + 2, y, str(col)[:40])
        cx += col_widths[j]
    y -= row_h
    c.setFont("Helvetica", 9)
    # rows
    for i in range(len(df)):
        cx = x
        for j, col in enumerate(df.columns):
            val = df.iloc[i, j]
            s = f"{val}"
            c.drawString(cx + 2, y, s[:40])
            cx += col_widths[j]
        y -= row_h
    return y


def add_png_page(c: canvas.Canvas, page_title: str, png_bytes: bytes):
    """Add a page with a title and a PNG image."""
    w, h = letter
    pdf_title(c, page_title, "")
    img = ImageReader(io.BytesIO(png_bytes))
    c.drawImage(
        img,
        0.75 * inch,
        1.1 * inch,
        width=7.0 * inch,
        height=9.2 * inch,
        preserveAspectRatio=True,
        mask="auto",
    )
    c.showPage()
