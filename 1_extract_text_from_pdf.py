import argparse
from pathlib import Path

from tqdm import tqdm

from hasextract.util.pdf import extract_pdf

parser = argparse.ArgumentParser(description="PDF Extraction Tool")

parser.add_argument(
    "input_path",
    nargs=1,
    help="Specify input pdf file or input folder containing pdf files to extract",
)

parser.add_argument(
    "--target",
    nargs=1,
    default=["extracted_data"],
    dest="target_directory",
    help="Target directory where the outputs of the extraction will be stored. Default: ./extracted_data",
)


def process_extraction(pdf_file: Path, target_directory: Path):
    pdf_path = pdf_file
    basename = pdf_path.stem

    final_target_directory = Path(target_directory, f"{basename}_extraction")
    final_target_directory.mkdir(exist_ok=True)

    full_text, images, tables = extract_pdf(pdf_file)

    images_path = Path(final_target_directory, "images")
    images_path.mkdir(exist_ok=True)
    for image, image_text, id in images:
        image_save_path = Path(images_path, f"{id}.png")
        image.save(image_save_path)

        image_text_path = Path(images_path, f"{id}.txt")
        with open(image_text_path, "w") as f:
            f.write(image_text)

    with open(
        Path(final_target_directory, "full_text.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(full_text)

    tables_path = Path(final_target_directory, "tables")
    tables_path.mkdir(exist_ok=True)

    for df_index in range(len(tables)):
        tables[df_index].to_csv(Path(tables_path, f"table_{df_index}.csv"), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    target_directory = Path(args.target_directory[0])
    target_directory.mkdir(exist_ok=True)

    input_path = Path(args.input_path[0])

    pdf_files = []
    if input_path.is_file():
        pdf_files.append(input_path)
    else:
        pdf_files.extend(list(input_path.glob("*.pdf")))

    for pdf_file in tqdm(pdf_files, desc="Processing pdf files"):
        process_extraction(pdf_file, target_directory)
