"""PDF report generation utilities."""

from remo_id.reports.pdf.primitives import (
    fig_to_png_bytes,
    wrap_lines,
    pdf_title,
    pdf_paragraph,
    pdf_table,
    add_png_page,
)

__all__ = [
    "fig_to_png_bytes",
    "wrap_lines",
    "pdf_title",
    "pdf_paragraph",
    "pdf_table",
    "add_png_page",
]
