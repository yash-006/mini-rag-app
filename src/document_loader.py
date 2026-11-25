
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def extract_text(uploaded_file):
    """
    Extract text from an uploaded PDF or TXT file.

    Returns:
        str: Extracted document text.
    """

    filename = uploaded_file.name
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        pdf_file = BytesIO(uploaded_file.getvalue())
        reader = PdfReader(pdf_file)

        pages_text = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages_text.append(
                    f"\n--- Page {page_number} ---\n{text}"
                )

        extracted_text = "\n".join(pages_text)

    elif extension == ".txt":
        extracted_text = uploaded_file.getvalue().decode(
            "utf-8-sig",
            errors="replace"
        )

    else:
        raise ValueError(
            "Unsupported file format. Please upload a PDF or TXT file."
        )

    if not extracted_text.strip():
        raise ValueError(
            "No readable text found. The PDF may contain scanned images."
        )

    return extracted_text.strip()
