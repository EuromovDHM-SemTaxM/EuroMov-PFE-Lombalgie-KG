import argparse
from pathlib import Path

from tqdm import tqdm

from hasextract.util.sparql import sparql_query


parser = argparse.ArgumentParser(description="PDF Extraction Tool")

parser.add_argument(
    "input_path",
    nargs=1,
    help="Specify input text file containing the sparql query to execute to retrieve the documents. Please make sure to have exactly two variables in the query: ?id and ?document_text.",
)

parser.add_argument(
    "--endpoint",
    nargs=1,
    default=[],
    dest="endpoint",
    help="Endpoint to use to execute the sparql query.",
    required=True,
)

parser.add_argument(
    "--target",
    nargs=1,
    default=["extracted_data"],
    dest="target_directory",
    help="Target directory where the outputs of the extraction will be stored. Default: ./extracted_data",
)


def process_extraction(query_file: Path, target_directory: Path, endpoint: str):
    basename = query_file.stem

    with open(query_file, "r") as f:
        query = f.read()

    if "?id" not in query or "?document_text" not in query:
        raise ValueError(
            "The query must contain exactly two variables: ?id and ?document_text."
        )

    final_target_directory = Path(target_directory, f"{basename}_extraction")
    final_target_directory.mkdir(exist_ok=True)

    df = sparql_query(query, endpoint)

    for _, row in tqdm(df.iterrows(), desc="Processing queried documents"):
        doc_id = row["id"].strip().replace(" ", "_")
        full_text = row["document_text"]

    with open(
        Path(final_target_directory, f"{doc_id}_full_text.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(full_text)


if __name__ == "__main__":
    args = parser.parse_args()
    target_directory = Path(args.target_directory[0])
    target_directory.mkdir(exist_ok=True)

    input_path = Path(args.input_path[0])

    if input_path.is_file():
        raise ValueError(
            "The input_path must be a single file containing the sparql query."
        )
    process_extraction(input_path, target_directory)
