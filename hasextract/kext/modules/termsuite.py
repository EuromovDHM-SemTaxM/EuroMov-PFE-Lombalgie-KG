import json
from typing import Dict

import requests
from confz import ConfZ, ConfZFileSource
from pydantic import AnyUrl

from hasextract.kext.knowledgeextractor import (
    ConceptMention,
    ExtractedKnowledge,
    KnowledgeExtractor,
    MentionType,
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
        if not (
            response := post(
                f"{TermSuiteConfig().endpoint}?language={parameters['source_language']}",
                data=corpus.encode("utf-8"),
                headers={
                    "Accept": "application/json",
                    "Content-Type": "plain/text",
                },
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
            concept = ConceptMention(
                id=idx,
                matched_text=term,
                mention_type=MentionType.EXTRACTED_TERM,
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
            relations=relations,
        )
