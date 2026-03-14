from pathlib import Path
import textwrap

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def write_line(pdf: canvas.Canvas, text: str, x: float, y: float, font_name: str, font_size: int, leading: int):
    pdf.setFont(font_name, font_size)
    pdf.drawString(x, y, text)
    return y - leading


def export_markdown_to_pdf(md_path: Path, pdf_path: Path) -> None:
    width, height = A4
    left_margin = 50
    right_margin = 50
    top_margin = 50
    bottom_margin = 50
    usable_width = width - left_margin - right_margin

    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    y = height - top_margin
    in_code_block = False

    content = md_path.read_text(encoding="utf-8").splitlines()

    for raw_line in content:
        line = raw_line.rstrip("\n")

        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            y -= 6
            continue

        font_name = "Helvetica"
        font_size = 10
        leading = 14

        if in_code_block:
            font_name = "Courier"
            font_size = 9
            leading = 12
        elif line.startswith("# "):
            font_name = "Helvetica-Bold"
            font_size = 18
            leading = 24
            line = line[2:].strip()
        elif line.startswith("## "):
            font_name = "Helvetica-Bold"
            font_size = 14
            leading = 20
            line = line[3:].strip()
        elif line.startswith("### "):
            font_name = "Helvetica-Bold"
            font_size = 12
            leading = 16
            line = line[4:].strip()
        elif line.startswith("- "):
            line = "- " + line[2:].strip()

        if not line.strip():
            y -= 8
        else:
            # Estimate chars per line for wrapping based on font size.
            approx_chars = max(40, int(usable_width / (font_size * 0.52)))
            wrapped = textwrap.wrap(line, width=approx_chars) or [""]

            for part in wrapped:
                if y <= bottom_margin:
                    pdf.showPage()
                    y = height - top_margin

                y = write_line(pdf, part, left_margin, y, font_name, font_size, leading)

        if y <= bottom_margin:
            pdf.showPage()
            y = height - top_margin

    pdf.save()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a markdown file to PDF")
    parser.add_argument("--input", help="Input markdown file path")
    parser.add_argument("--output", help="Output PDF file path")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if args.input:
        md_file = Path(args.input)
    else:
        md_file = project_root / "docs" / "project-architecture-overview.md"

    if args.output:
        pdf_file = Path(args.output)
    else:
        pdf_file = md_file.with_suffix(".pdf")

    export_markdown_to_pdf(md_file, pdf_file)
    print(f"PDF exported: {pdf_file}")
