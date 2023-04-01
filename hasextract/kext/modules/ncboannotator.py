import hashlib
import json
import logging
from typing import Dict
from urllib.parse import urlencode


from confz import ConfZ, ConfZFileSource
from pydantic import AnyUrl

from hasextract.kext.knowledgeextractor import (
    Concept,
    ExtractedKnowledge,
    KGConcept,
    KnowledgeExtractor,
    ConceptType,
    Mention,
)
from hasextract.util.cached_requests import get
from hasextract.util.segmentation import chunk_sentences_by_span

logger = logging.getLogger()


class NCBOAnnotatorConfig(ConfZ):
    endpoint: AnyUrl
    apikey: str
    CONFIG_SOURCES = ConfZFileSource(file="config/ncboannotator.json")


class NCBOAnnotatorKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self, trigger_condition: str):
        super().__init__(trigger_condition)

    def __call__(self, corpus: str, parameters: Dict[str, str] = None):
        from nltk.tokenize import TreebankWordTokenizer as twt

        max_chars = 2000

        logger.debug("Chunking sentences to fit API limits... ")
        sentence_spans = twt().span_tokenize(corpus)
        if len(corpus) > max_chars:
            chunks_spans = chunk_sentences_by_span(corpus, sentence_spans, max_chars)
        else:
            chunks_spans = [(0, len(corpus))]

        concepts = []
        for chunk_span in chunks_spans:
            chunk = corpus[chunk_span[0] : chunk_span[1]]
            m = hashlib.sha256()
            m.update(chunk.encode("utf-8"))
            # Creating a unique key for the cache.
            key = f"5ncboannotator_{m.hexdigest()}"
            request_params = {
                "text": chunk.encode("utf-8"),
                "longest_only": True,
                "expand_mappings": True,
                "apikey": NCBOAnnotatorConfig().apikey,
            }
            if response := get(
                f"{NCBOAnnotatorConfig().endpoint}?{urlencode(request_params)}",
                data=request_params,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"apikey token={NCBOAnnotatorConfig().apikey}",
                },
                key=key,
            ):
                response = json.loads(response)
                for entity in response:
                    annotated_class = entity["annotatedClass"]
                    idx = f"{annotated_class['@id']}"
                    concept = KGConcept(
                        idx=idx,
                        label="",
                        concept_type=ConceptType.LINKED_ENTITY,
                    )

                    concept.instances = [
                        Mention(
                            start=chunk_span[0] + annotation["from"],
                            end=chunk_span[0] + annotation["to"],
                            text=corpus[chunk_span[0] + annotation["from"] : chunk_span[0] + annotation["to"]],
                        )
                        for annotation in entity["annotations"]
                    ]
                    concept.label = concept.instances[0].text
                    concept.mappings = []
                    if "cui" in annotated_class:
                        concept.mappings = [
                            (f"UMLS_{cui}", cui) for cui in annotated_class["cui"]
                        ]

                    concepts.append(concept)

        relations = []
        return ExtractedKnowledge(
            name="NCBO Annotator Entities",
            language=parameters["source_language"],
            agent="NCBO Annotator",
            source_text=corpus,
            concepts=concepts,
            relations=relations,
        )
