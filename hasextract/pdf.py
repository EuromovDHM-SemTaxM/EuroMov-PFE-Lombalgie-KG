import camelot
import fitz
import pdfplumber
import pandas
import pytesseract as pytesseract
from PIL import Image
from pathlib import Path
from tqdm import trange


def _extract_and_ocr_images(pdf_file_fitz, page_index):
    fitz_page = pdf_file_fitz[page_index]
    image_list = fitz_page.getImageList()
    final_images = []
    if image_list:
        print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        for image_index, img in enumerate(image_list, start=1):
            # get the XREF of the image
            xref = img[0]

            pix = fitz.Pixmap(pdf_file_fitz, xref)
            if pix.n >= 5:  # CMYK: convert to RGB first
                pix = fitz.Pixmap(fitz.csRGB, pix)

            image_text = pytesseract.image_to_string(Image.open(pix.to_bytes(format="raw")))
            final_images.append((pix, image_text, f"image_{page_index}_{image_index}"))

    return final_images


def _extract_tables(pdf_file):
    extracted_tables = camelot.read_pdf(pdf_file, pages='all')
    dfs = []
    for i in trange(extracted_tables.n):
        dfs.append(extracted_tables[i].df)
    return dfs


def extract_pdf(pdf_file: Path) -> tuple[str, list[fitz.Pixmap], list[pandas.Dataframe]]:
    text_pages = []
    with pdfplumber.open(pdf_file) as pdf:
        pdf_file_fitz = fitz.open(pdf_file)
        global_images = []
        for page_index in range(pdf.pages):
            page_text = pdf.pages[page_index].extract_text()

            images = _extract_and_ocr_images(pdf_file_fitz, page_index)
            global_images.extend(images)
            text_pages.append(page_text)
        full_text = ''.join(text_pages)

        tables = _extract_tables(pdf_file)

        return full_text, global_images, tables
