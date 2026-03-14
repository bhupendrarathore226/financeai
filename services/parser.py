"""
PDF parsing service for FinanceAI.

Responsibility
--------------
This module extracts raw transaction text from uploaded PDF bank statements.
It forms the very first step of the ingestion pipeline:

    PDF file on disk
        └── parse_pdf()           (this module)
                └── list of text strings  →  services/ingest.py

Approach
--------
Banks typically export statements as PDFs that contain structured tables
(date, description, amount columns).  This module uses `pdfplumber` to:
  1. Open the PDF.
  2. Iterate over every page.
  3. Extract all tables from each page.
  4. Convert each row into a single pipe-delimited string.

The result is a flat list of strings like:
    ["2026-01-05 | Netflix | -12.99", "2026-01-06 | Salary | +3000.00", ...]

These strings are then embedded and stored as individual vector documents in
ChromaDB, one document per transaction row.

Why pipe-delimited text?
------------------------
When the LLM is given multiple transaction rows as context, a consistent
delimiter makes it easier to read column boundaries.  Pipes are visually
clear and rarely appear in financial descriptions.
"""

import pdfplumber  # Third-party library for extracting text and tables from PDFs
import logging     # Standard library: structured log messages


# Module-level logger.  Using __name__ means the log records will show
# "services.parser" as the source, making it easy to filter logs.
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _clean_row(row) -> str:
    """
    Convert a single table row (list of cell values) into a clean text string.

    pdfplumber returns table rows as Python lists where each element is either
    a string or None.  This helper:
      1. Converts every cell to a string and strips leading/trailing whitespace.
      2. Discards empty or whitespace-only cells (None values, merged cells, etc.).
      3. Joins the remaining non-empty cells with " | " as a delimiter.

    Example
    -------
    Input row from pdfplumber:
        ["2026-01-05", "Netflix subscription", None, "-12.99"]
    Output string:
        "2026-01-05 | Netflix subscription | -12.99"

    Parameters
    ----------
    row : list
        A single table row as returned by pdfplumber (list of cell values,
        which may be strings, numbers, or None).

    Returns
    -------
    str
        A pipe-delimited string of non-empty cell values, or an empty string
        if every cell in the row was empty.
    """
    # Build a list that includes only cells that have actual content.
    # `str(cell).strip()` handles None → "None" which is then falsy after checking
    # `if cell` first, ignoring genuinely None or empty cells.
    cleaned_cells = [str(cell).strip() for cell in row if cell and str(cell).strip()]

    # Join surviving cells with a visual separator.  If cleaned_cells is empty
    # (e.g. a header divider row), this returns "" which the caller will discard.
    return " | ".join(cleaned_cells)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def parse_pdf(filepath):
    """
    Extract and return all transaction rows from a PDF bank statement.

    Opens the file at `filepath`, iterates over every page, extracts all
    tables from each page, and converts each non-empty row into a single
    pipe-delimited string via `_clean_row()`.

    Parameters
    ----------
    filepath : str or Path
        Absolute or relative path to the PDF file on disk.

    Returns
    -------
    list[str]
        A flat list of transaction strings, one per non-empty table row
        across all pages of the document.  Returns an empty list if the PDF
        contains no extractable tables (e.g. a scanned image-only PDF).

    Notes
    -----
    - Image-only (scanned) PDFs will return an empty list because pdfplumber
      cannot extract text from rasterised images without OCR.
    - The caller (ingest_file) raises IngestionError if the returned list is
      empty, so this function does not need to handle that case itself.
    """
    extracted_rows = []  # Accumulates all cleaned transaction strings

    # pdfplumber.open() is a context manager: it opens the file and guarantees
    # the file handle is closed when the `with` block exits, even on error.
    with pdfplumber.open(filepath) as pdf:

        # Iterate over every page in the document (0-indexed internally but
        # pdfplumber exposes them as a list so we just iterate).
        for page in pdf.pages:

            # extract_tables() finds all table structures on the page and
            # returns them as a list of tables.  Each table is a list of rows.
            # Each row is a list of cell values (strings or None).
            tables = page.extract_tables()

            for table in tables:

                for row in table:

                    # Skip completely empty rows (None or []) that pdfplumber
                    # sometimes returns for header/footer separators.
                    if row:
                        clean_row = _clean_row(row)

                        # Only keep rows that have at least one non-empty cell
                        # after cleaning.  Blank divider rows are discarded here.
                        if clean_row:
                            extracted_rows.append(clean_row)

    logger.info("Parsed %s rows from %s", len(extracted_rows), filepath)
    return extracted_rows