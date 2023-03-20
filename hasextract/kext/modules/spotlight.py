import hashlib
import json
import logging
from typing import Dict
from urllib.parse import urlencode
import SPARQLWrapper

import requests
from confz import ConfZ, ConfZFileSource
from pydantic import AnyUrl
from tqdm import tqdm 

from hasextract.kext.knowledgeextractor import (
    Concept,
    ExtractedKnowledge,
    KnowledgeExtractor,
    ConceptType,
    Mention,
    RelationInstance,
)
from hasextract.util import _break_up_sentences, get, post


logger = logging.getLogger()


class SpotlightConfig(ConfZ):
    endpoint: AnyUrl
    dbpedia_prefix_uri: AnyUrl
    dbpedia_sparql_endpoint: AnyUrl
    CONFIG_SOURCES = ConfZFileSource(file="config/spotlight.json")


def query_relations(uri):
    try:

        relations = []
        # wikidata_id = wikidata_id[1:]
        endpoint = SpotlightConfig().dbpedia_sparql_endpoint
        params = {
            "query": f"select distinct ?rel ?target where {{<{uri}> ?rel ?target.}}",
            "format": "application/sparql-results+json",
            "timeout":0,
            "signal_void":"on"
        }
        if result := get(f"{endpoint}?{urlencode(params)}",headers= {}):
            ret = json.loads(result)
            relations.extend(
                (r["rel"]["value"], r["target"]["value"])
                for r in ret["results"]["bindings"]
            )
            
    except json.decoder.JSONDecodeError:
        return None

    return relations


class SpotlightKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self, trigger_condition: str):
        super().__init__(trigger_condition)

    def __call__(self, corpus: str, parameters: Dict[str, str] = None):

        import spacy

        lang = parameters["source_language"]
        if lang == "en":
            nlp = spacy.load("en_core_web_sm")
        else:
            nlp = spacy.load(f"{lang}_core_news_sm")

        concept_index = {}
        relation_index = {}

        max_chars = 500

        logger.debug("Chunking sentences to fit API limits... ")

        doc = nlp(corpus)
        sentence_spans = [(sent.start_char, sent.end_char) for sent in doc.sents]
        chunks_spans = _break_up_sentences(corpus, sentence_spans, max_chars)

        for chunk_span in tqdm(chunks_spans, "Processing sentences with Spotlight"):
            chunk = (
                corpus[chunk_span[0] : chunk_span[1]]
                .replace("\\n", " ")
                .replace("’", "'")
            )
            if len(chunk.strip()) > 0:
                chunk_doc = nlp(chunk)
                m = hashlib.sha256()
                m.update(chunk.encode("utf-8"))
                # Creating a unique key for the cache.
                key = f"ef_{m.hexdigest()}"
                request_params = {
                    "text": chunk,
                }
                if not (
                    response := get(
                        f"{SpotlightConfig().endpoint}?{urlencode(request_params)}",
                        headers={
                            "Accept": "application/json"
                        },
                        key=key,
                    )
                ):
                    return []

                response = json.loads(response)
                if 'Resources' in response:
                    for entity in response["Resources"]:
                        idx = f"{entity['@URI']}"
                        if idx not in concept_index:
                            concept = Concept(
                                idx=idx,
                                label=idx.split("/")[-1],
                                concept_type=ConceptType.LINKED_ENTITY,
                            )
                            concept_index[idx] = concept

                        if not concept_index[idx].instances:
                            concept_index[idx].instances = []
                        offset = int(entity['@offset'])
                        concept_index[idx].instances.append(
                            (
                                Mention(
                                    start=chunk_span[0] + offset,
                                    end=chunk_span[0] + offset + len(entity['@surfaceForm']),
                                    text=entity['@surfaceForm'],
                                )
                            )
                        )

                        if idx not in relation_index:
                            relation_index[idx] = query_relations(idx)

        concepts = list(concept_index.values())

        relations = []
        for concept in concepts:
            concept_relations = relation_index[concept.idx]
            relations.extend(
                RelationInstance(
                    source=concept,
                    target=concept_index[relation[1]],
                    name=relation[0],
                )
                for relation in concept_relations
                if relation[1] in concept_index
            )
        return ExtractedKnowledge(
            name="Spotlight Identified DBPedia Entities",
            agent="Spotlight",
            language=parameters["source_language"],
            source_text=corpus,
            concepts=concepts,
            relations=relations,
        )
