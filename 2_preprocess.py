import argparse
import re
from pathlib import Path

from tqdm import tqdm


def pre_process(text: str):
    text = text.replace("-\n", "")
    text = re.sub("\\d+\\s?\n", "\n", text)
    text = re.sub(r"HAS [•\-/].*\n", "\n", text)
    text = re.sub(r"\d+\.\d+\.\n", "", text)
    return text


parser = argparse.ArgumentParser(
    description="Pre-processing step"
)

parser.add_argument(
    "input_path", nargs=1,
    help="Specify input extraction or input folder containing extractions")

parser.add_argument("--corpus_file", nargs=1, dest="corpus_file", default=['full_text.txt'],
                    help="if the input is a directory name, this parameter gives the name "
                         "of the text file containing the extracted corpus. Default: full_text.txt")

args = parser.parse_args()

if __name__ == "__main__":

    input_path = Path(args.input_path[0])

    input_files = []
    if input_path.is_file():
        input_files.append(input_path)
    else:
        files = list(input_path.glob("*.txt"))
        if len(files) > 0:
            input_files.extend(files)
        else:
            for file in input_path.glob("*"):
                input_files.extend(list(file.glob(f"{args.corpus_file[0]}")))

    for file in tqdm(input_files):
        with open(file, "r") as f:
            input_text = f.read()
        output_text = pre_process(input_text)
        with open(file, "w") as f:
            f.write(output_text)
