import pdfplumber

def parse_pdf(filepath):
    rows = []

    with pdfplumber.open(filepath) as pdf:

        for page in pdf.pages:

            tables = page.extract_tables()

            for table in tables:

                for row in table:

                    if row:

                        clean_row = " | ".join(
                            [str(cell) for cell in row if cell]
                        )

                        rows.append(clean_row)

    return rows