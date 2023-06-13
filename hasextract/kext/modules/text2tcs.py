import hashlib
from logging import getLogger
from pathlib import Path
import pickle
from typing import Dict

from confz import ConfZ, ConfZFileSource
from elg import Service
from tqdm import tqdm

from hasextract.util.segmentation import chunk_sentences


logger = getLogger()

from hasextract.kext.knowledgeextractor import (
    Concept,
    ConceptRelation,
    ExtractedKnowledge,
    KnowledgeExtractor,
    Relation,
    TermConcept,
)


class Text2TCSConfig(ConfZ):
    elg_app_id: int = 8122

    CONFIG_SOURCES = ConfZFileSource(file="config/text2tcs.json")


class Text2TCSExtractor(KnowledgeExtractor):
    def __init__(self, trigger_condition: str):
        super().__init__(trigger_condition)
        self.service = None

    def __call__(
        self, corpus: str, parameters: Dict[str, str] = None
    ) -> ExtractedKnowledge:
        cache_dir = Path("./.cache")
        cache_dir.mkdir(exist_ok=True)

        if not self.service:
            logger.debug("Creating ELG Service...")
            self.service = Service.from_id(Text2TCSConfig().elg_app_id)

        from nltk.tokenize import sent_tokenize

        max_chars = 7500

        logger.debug("Chunking sentences to fit API limits... ")
        sentences = sent_tokenize(corpus)
        if len(corpus) > max_chars:
            chunks = chunk_sentences(sentences, max_chars)
        else:
            chunks = [corpus]

        concepts = []
        concept_index = {}
        relations = []
        for sen in tqdm(chunks, desc="Extracting knowledge with Text2TCS"):
            try:
                m = hashlib.sha256()
                m.update(sen.encode("utf-8"))
                # Creating a unique key for the cache.
                key = f"text2tcs_{m.hexdigest()}"
                pickle_path = Path(cache_dir, f"{key}.pkl")
                if pickle_path.exists():
                    with open(pickle_path, "rb") as f:
                        response = pickle.load(f)
                else:
                    response = self.service(request_input=sen, request_type="text")
                    with open(pickle_path, "wb") as f:
                        pickle.dump(response, f)
                for concept in response.annotations:
                    for annotation in response.annotations[concept]:
                        idx = "text2tcs_" + annotation.features["id"]
                        mention = TermConcept(
                            idx=idx, label=annotation.features["term"]
                        )
                        concepts.append(mention)
                        concept_index[idx] = mention
                        if len(annotation.features["relations"]) > 0:
                            for relation in annotation.features["relations"]:
                                relation["source"] = idx
                                relation["related concept"] = (
                                    "text2tcs_" + relation["related concept"]
                                )
                            relations.extend(annotation.features["relations"])
            except Exception as e:
                logger.debug(e)

        relations = [
            ConceptRelation(
                source=concept_index[relation["source"]],
                target=concept_index[relation["related concept"]],
                name=relation["type"],
            )
            for relation in relations
        ]

        return ExtractedKnowledge(
            name="Text2TCS Terminology extraction result",
            agent="Text2TCS ELG",
            language=parameters["source_language"],
            source_text=corpus,
            concepts=concepts,
            relations=relations,
        )
