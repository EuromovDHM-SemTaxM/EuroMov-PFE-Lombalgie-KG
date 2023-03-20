from abc import ABC, abstractmethod
from itertools import combinations, product
import logging

from jellyfish import levenshtein_distance
from tqdm import tqdm

from hasextract.kext.knowledgeextractor import ExtractedKnowledge

logger = logging.getLogger()


class ExtractedKnowledgeMerger(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, extracted: list[ExtractedKnowledge]):
        pass


class FlatMerger(ExtractedKnowledgeMerger):
    def __init__(self):
        super().__init__()

    def __call__(self, extracted: dict[str, ExtractedKnowledge]) -> ExtractedKnowledge:
        logger.debug("Invoking Flat Merger...")
        global_concepts = []
        global_relations = []
        global_semantic_roles = []
        global_amrs = []
        source_text = None
        lang = None
        for current, value in extracted.items():
            source_text = source_text or extracted[current].source_text
            lang = lang or extracted[current].language

            for concept in value.concepts:
                concept.provenance = {'agent': extracted[current].agent}
                global_concepts.append(concept)

            for relation in extracted[current].relations:
                relation.provenance = {'agent': extracted[current].agent}
                global_relations.append(relation)

            if extracted[current].semantic_roles:
                for sematic_role in extracted[current].semantic_roles:
                    sematic_role.provenance = {'agent': extracted[current].agent}
                    global_semantic_roles.append(sematic_role)

            if extracted[current].amr_parses:
                for amr in  extracted[current].amr_parses:
                    amr.provenance = {'agent':  extracted[current].agent}
                    global_amrs.append(amr)

        return ExtractedKnowledge(name="Flat merged extracted knowledge", agent="FlatMerger", source_text=source_text,language=lang, concepts=global_concepts, relations=global_relations, semantic_roles=global_semantic_roles, amr_parses=global_amrs)



