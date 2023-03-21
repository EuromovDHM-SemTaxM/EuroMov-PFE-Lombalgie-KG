from pathlib import Path

import camelot
import fitz
import pandas
import pdfplumber
import pytesseract as pytesseract
from PIL import Image
from tqdm import trange
import io


def _extract_and_ocr_images(fitz_document, page_index):
    image_list = fitz_document.get_page_images(page_index)
    final_images = []
    if image_list:
        # print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        for image_index, img in enumerate(image_list, start=1):
            # get the XREF of the image
            xref = img[0]

            pix = fitz.Pixmap(fitz_document, xref)
            if pix.n >= 5:  # CMYK: convert to RGB first
                pix = fitz.Pixmap(fitz.csRGB, pix)

            image_text = pytesseract.image_to_string(Image.open(io.BytesIO(pix.pil_tobytes(format="png"))))
            final_images.append((pix, image_text, f"image_{page_index}_{image_index}"))

    return final_images


def _extract_tables(pdf_file):
    extracted_tables = camelot.read_pdf(str(pdf_file), pages='all')
    return [
        extracted_tables[i].df
        for i in trange(extracted_tables.n, desc="Extracting tables")
    ]


def extract_pdf(pdf_file: Path) -> tuple[str, list[fitz.Pixmap], list[pandas.DataFrame]]:
    text_pages = []

    pdf_file_fitz = fitz.open(pdf_file)
    global_images = []
    for page_index in trange(len(pdf_file_fitz), desc="Extracting pages"):
        page = pdf_file_fitz[page_index]
        page_text = page.get_text()
        images = _extract_and_ocr_images(pdf_file_fitz, page_index)
        global_images.extend(images)
        text_pages.append(page_text)
    full_text = ''.join(text_pages)

    tables = _extract_tables(pdf_file)

    return full_text, global_images, tables
