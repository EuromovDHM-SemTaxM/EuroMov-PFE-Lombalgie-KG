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
    Mention,
    RelationInstance,
)
from hasextract.util import get, post


class EntityFishingConfig(ConfZ):
    endpoint: AnyUrl
    wikidata_prefix_uri: AnyUrl
    CONFIG_SOURCES = ConfZFileSource(file="config/entityfishing.json")


# _mapping_types = {
#     "P8814": "Wordnet Synset Id",
#     "P2581": "BabelNet Synset Id",
#     "P11143": "WikiProjectMedID",
#     "P2892": "UMLS CUI",
#     "P494": "ICD-10",
#     "P486": "MeSH Descriptor ID",
#     "P3417": "Quora Topic ID",
#     "P10283": "OpenAlexID",
# }


def request_entity_fishing_concept_lookup(wikidata_id):
    """
    Wrapper around Entity-fishing (language set in English)
    :param text: string, text to be annotated
    :param lang: string, language model to use
    :return: annotations in JSON
    """
    try:
        # wikidata_id = wikidata_id[1:]
        endpoint = EntityFishingConfig().endpoint
        get_url = f"{endpoint}/kb/concept/{wikidata_id}"

        response = get(get_url, headers={})

        return json.loads(response)

    except json.decoder.JSONDecodeError:
        return None


class EntityFishingKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self, trigger_condition: str):
        super().__init__(trigger_condition)

    def __call__(self, corpus: str, parameters: Dict[str, str] = None):
        m = hashlib.sha256()
        m.update(corpus.encode("utf-8"))
        # Creating a unique key for the cache.
        key = f"ef_{m.hexdigest()}"
        request_params = {
            "text": corpus,
            "language": {"lang": parameters["source_language"]},
            "mentions": ["wikipedia"],
            "nbest": False,
            "sentence": False,
            "customisation": "generic",
        }
        if not (
            response := post(
                f"{EntityFishingConfig().endpoint}/disambiguate/",
                json=request_params,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                key=key,
            )
        ):
            return []
        concept_index = {}
        relation_index = {}
        response = json.loads(response)
        for entity in response["entities"]:
            idx = f"{EntityFishingConfig().wikidata_prefix_uri}{entity['wikidataId']}"
            if idx not in concept_index:
                concept = Concept(
                    idx=idx,
                    label=entity["rawName"],
                    concept_type=ConceptType.LINKED_ENTITY,
                    confidence_score=entity["nerd_selection_score"],
                )
                concept_index[idx] = concept

            if not concept_index[idx].instances:
                concept_index[idx].instances = []
            concept_index[idx].instances.append(
                (Mention(start=entity["offsetStart"], end=entity["offsetEnd"], text = corpus[entity["offsetStart"]:entity["offsetEnd"]]))
            )

            concept_relations = request_entity_fishing_concept_lookup(
                entity["wikidataId"]
            )
            relation_index[idx] = []
            concept.mappings = {}
            if "statements" in concept_relations:
                relation_index[idx].extend(
                    [
                        (
                            f"{EntityFishingConfig().wikidata_prefix_uri}{stmt['propertyId']}",
                            f"{EntityFishingConfig().wikidata_prefix_uri}{stmt['value']}",
                        )
                        for stmt in concept_relations["statements"]
                        if not isinstance(stmt["value"], dict)
                        and stmt["value"].startswith("Q")
                    ]
                )

                concept.mappings = [(stmt['propertyId'],stmt['value']) for stmt in concept_relations["statements"] if "valueType" in stmt and stmt['valueType'] == 'external-id']

        concepts = list(concept_index.values())

        relations = []
        for concept in concepts:
            concept_relations = relation_index[concept.idx]
            relations.extend(
                RelationInstance(
                    source=concept, target=concept_index[relation[1]], name=relation[0]
                )
                for relation in concept_relations
                if relation[1] in concept_index
            )
        return ExtractedKnowledge(
            name="Entity Fishing Identified Wikidata Entities",
            agent="Entity Fishing",
            language=parameters["source_language"],
            source_text=corpus,
            concepts=concepts,
            relations=relations,
        )
