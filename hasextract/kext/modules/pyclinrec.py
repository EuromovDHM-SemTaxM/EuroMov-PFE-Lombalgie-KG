import hashlib
import json
import logging
from pathlib import Path
from typing import Dict
from urllib.parse import urlencode
import SPARQLWrapper

import requests
from confz import ConfZ, ConfZFileSource
from pydantic import AnyUrl
from tqdm import tqdm

from pyclinrec.dictionary import MgrepDictionaryLoader
from pyclinrec.recognizer import IntersStemConceptRecognizer

from hasextract.kext.knowledgeextractor import (
    ExtractedKnowledge,
    KGConcept,
    KnowledgeExtractor,
    Mention,
    OntologyRelation,
)
from hasextract.util.cached_requests import get
from hasextract.util.segmentation import break_up_sentences


logger = logging.getLogger()


class PyClinRecConfig(ConfZ):
    dictionary: Path
    resource_path: Path
    CONFIG_SOURCES = ConfZFileSource(file="config/pyclinrec.json")


class SpotlightKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self, trigger_condition: str):
        super().__init__(trigger_condition)
        self.dictionary_loader = MgrepDictionaryLoader(PyClinRecConfig().dictionary)
        self.resource_path = PyClinRecConfig().resource_path

    def __call__(self, corpus: str, parameters: Dict[str, str] = None):
        import spacy

        lang = parameters["source_language"]
        self.concept_recognizer = IntersStemConceptRecognizer(
            self.dictionary_loader,
            str(self.resource_path / f"stopwords{lang}.txt"),
            str(self.resource_path / f"data/termination_terms{lang}.txt"),
        )
        self.concept_recognizer.initialize()

        concept_index = {}

        annotations = self.concept_recognizer.annotate(corpus)

        for annotation in annotations:
            idx = annotation.concept_id
            if idx not in concept_index:
                concept = KGConcept(
                    idx=idx,
                    label=idx.split("/")[-1],
                )
                concept_index[idx] = concept

            if not concept_index[idx].instances:
                concept_index[idx].instances = []

            concept_index[idx].instances.append(
                (
                    Mention(
                        start=annotation.start,
                        end=annotation.end,
                        text=annotation.text,
                    )
                )
            )

        concepts = list(concept_index.values())

        relations = []

        return ExtractedKnowledge(
            name="Pyclinrec Entities",
            agent="Pyclinrec",
            language=parameters["source_language"],
            source_text=corpus,
            concepts=concepts,
            relations=relations,
        )
