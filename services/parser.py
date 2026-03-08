import pdfplumber
import logging


logger = logging.getLogger(__name__)


def _clean_row(row) -> str:
    # Normalize cells and remove empty values so each transaction string is stable.
    cleaned_cells = [str(cell).strip() for cell in row if cell and str(cell).strip()]
    return " | ".join(cleaned_cells)

def parse_pdf(filepath):
    extracted_rows = []

    with pdfplumber.open(filepath) as pdf:

        for page in pdf.pages:

            tables = page.extract_tables()

            for table in tables:

                for row in table:

                    if row:
                        clean_row = _clean_row(row)
                        if clean_row:
                            extracted_rows.append(clean_row)

    logger.info("Parsed %s rows from %s", len(extracted_rows), filepath)
    return extracted_rows