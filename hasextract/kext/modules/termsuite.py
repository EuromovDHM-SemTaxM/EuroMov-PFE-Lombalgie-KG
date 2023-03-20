import hashlib
import json
from typing import Dict

import requests
from confz import ConfZ, ConfZFileSource
from pydantic import AnyUrl

from hasextract.kext.knowledgeextractor import (
    Concept,
    ExtractedKnowledge,
    KnowledgeExtractor,
    ConceptType,
    RelationInstance,
)
from hasextract.util import post


class TermSuiteConfig(ConfZ):
    endpoint: AnyUrl
    CONFIG_SOURCES = ConfZFileSource(file="config/termsuite.json")


class TermsuiteKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self, trigger_condition: str):
        super().__init__(trigger_condition)

    def __call__(self, corpus: str, parameters: Dict[str, str] = None):
        m = hashlib.sha256()
        m.update(corpus.encode("utf-8"))
        # Creating a unique key for the cache.
        key = f"termsuite_{m.hexdigest()}"
        
        if not (
            response := post(
                f"{TermSuiteConfig().endpoint}?language={parameters['source_language']}",
                data=corpus.encode("utf-8"),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "plain/text",
                },
                key=key
            )
        ):
            return []
        concepts = []
        concept_index = {}
        response = json.loads(response)
        for term_id, term in enumerate(response["terms"]):
            idx = f"termsuite_{str(term_id)}"
            props = term["props"]
            term = props["pilot"]
            concept = Concept(
                idx=idx,
                label=term,
                concept_type=ConceptType.EXTRACTED_TERM,
                rank=props["rank"],
                rule=props["rule"],
            )
            concept_index[props["key"]] = concept
            concepts.append(concept)
        relations = [
            RelationInstance(
                source=concept_index[relation["from"]],
                target=concept_index[relation["to"]],
                name=relation["type"],
            )
            for relation in response["relations"]
        ]
        return ExtractedKnowledge(
            name="Termsuite Terminology extraction result",
            agent="Termsuite REST",
            language=parameters["source_language"],
            source_text=corpus,
            concepts=concepts,
            relations=relations
        )
