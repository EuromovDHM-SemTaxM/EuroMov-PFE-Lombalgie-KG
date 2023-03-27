import argparse
import json
import logging
from pathlib import Path

from tqdm import tqdm
from hasextract.evaluation.evaluator import (
    CompositeKnowledgeEvaluator,
    DescriptiveStatisticsEvaluator,
    OverlapEvaluator,
)

from hasextract.kext.knowledgeextractor import CompositeKnowledgeExtractor
from hasextract.rdf_mapper.mapper import LLODKGMapper
from hasextract.rdf_mapper.merger import FlatMerger
from hasextract.kext.modules.entityfishing import EntityFishingKnowledgeExtractor
from hasextract.kext.modules.ncboannotator import NCBOAnnotatorKnowledgeExtractor
from hasextract.kext.modules.spotlight import SpotlightKnowledgeExtractor
from hasextract.kext.modules.tbx import TBXExtractor
from hasextract.kext.modules.termsuite import TermsuiteKnowledgeExtractor
from hasextract.kext.modules.text2tcs import Text2TCSExtractor
from hasextract.kext.modules.usea import USEAKnowledgeExtractor
from hasextract.util.logging import setup_logging

setup_logging(
    console_log_output="stdout",
    console_log_level="debug",
    console_log_color=True,
    logfile_file="kext.log",
    logfile_log_level="debug",
    logfile_log_color=False,
    log_line_template="%(color_on)s[%(created)d] [%(threadName)s] [%(levelname)-8s] %(message)s%(color_off)s",
)
logger = logging.getLogger()

parser = argparse.ArgumentParser(description="Knowledge Extractor")

parser.add_argument(
    "path",
    nargs=1,
    help="Specify input extraction or input folder containing extractions",
)

parser.add_argument(
    "--extractors",
    "-x",
    nargs="+",
    default=["termsuite", "text2tcs"],
    help="List of knowledge extractors to include. Possible: tbx, text2tcs, termsuite. Default: text2tcs, termsuite.",
)

parser.add_argument(
    "--endpoint",
    "-e",
    nargs=1,
    type=str,
    help="SPARQL Endpoint where to materialize the model",
)

parser.add_argument(
    "--corpus_file",
    nargs=1,
    dest="corpus_file",
    default=["deduplicated.txt"],
    help="if the input is a directory name, this parameter gives the name "
    "of the text file containing the extracted corpus. Default: full_text_deduplicated.txt",
)

parser.add_argument(
    "--prefix-name",
    nargs=1,
    dest="prefix_name",
    default=["kext"],
    help="Prefix name for the model. Default: kext",
)

parser.add_argument(
    "--prefix-uri",
    nargs=1,
    dest="prefix_uri",
    default=["http://w3id.org/kext/"],
    help="Prefix URI for the model. Default: http://w3id.org/kext/",
)


if __name__ == "__main__":
    args = parser.parse_args()

    composite_term_extractor = CompositeKnowledgeExtractor()
    composite_term_extractor.add_extractor(
        [
            TermsuiteKnowledgeExtractor('method contains "termsuite"'),
            TBXExtractor('method contains "tbxtools"'),
            Text2TCSExtractor("method contains text2tcs"),
            EntityFishingKnowledgeExtractor('method contains "entityfishing"'),
            SpotlightKnowledgeExtractor('method contains "spotlight"'),
            NCBOAnnotatorKnowledgeExtractor('method contains "ncboannotator"'),
            USEAKnowledgeExtractor('method contains "usea"'),
        ]
    )

    composite_evaluator = CompositeKnowledgeEvaluator()
    composite_evaluator.add_evaluator(
        [DescriptiveStatisticsEvaluator(), OverlapEvaluator()]
    )

    kext_params = {"method": args.extractors, "source_language": "fr"}

    input_path = Path(args.path[0])

    input_files = []
    if input_path.is_file():
        input_files.append(input_path)
    elif files := list(input_path.glob("*.txt")):
        input_files.append(files)
    else:
        input_files.extend(
            list(file.glob(f"*{args.corpus_file[0]}")) for file in input_path.glob("*")
        )
    result_name = "kext_result"
    eval_name = "eval.json"
    for input_file in tqdm(input_files, desc="Processing files"):

        input_text = ""
        for file in input_file:
            with open(file) as f:
                input_text += f" {f.read()}"
        logger.info(f"Running knowledge extraction pipeline on {file.parent.name}...")
        extraction = composite_term_extractor(input_text, kext_params)

        logger.info("Running evaluation pipeline...")
        evaluation = composite_evaluator(extraction)
        logger.info(json.dumps(evaluation, indent=2))

        with open(Path(file.parent, eval_name), "w") as f:
            json.dump(evaluation, f, indent=2)

        for extractor in extraction:
            with open(Path(file.parent, f"{result_name}_{extractor}.json"), "w") as f:
                f.write(
                    extraction[extractor]
                    .json(exclude_none=True)
                    .encode()
                    .decode("unicode-escape")
                )

        merger = FlatMerger()
        merged_extracted = merger(extraction)

        rdf_mapper = LLODKGMapper(
            base_uri=args.prefix_uri[0],
            prefix_name=args.prefix_name[0],
            kg_description=f"Knowledge extracted from {file.parent.name}",
        )
        rdf_mapper(
            document_name=file.parent.name,
            extracted_knowledge=merged_extracted,
            target_file=f'{str(Path(file.parent, f"{file.parent.name}.ttl"))}',
        )
