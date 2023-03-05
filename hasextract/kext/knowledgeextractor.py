from enum import Enum
import logging
from abc import ABC, abstractmethod
import os
from typing import Dict, List, Optional

from pycond import parse_cond
from pydantic import BaseModel, root_validator



logger = logging.getLogger()


class MentionType(Enum):
    LINKED_ENTITY = 0
    EXTRACTED_TERM = 0

class ConceptMention(BaseModel):
    id: str
    matched_text: str
    mention_type: MentionType
    instances: Optional[list[tuple[int, int]]]
    rank: Optional[int]
    rule: Optional[str]
    confidence_score: Optional[float]

    def __hash__(self):
         return hash((type(self),) + tuple(self.id))


class RelationInstance(BaseModel):
    source: ConceptMention
    target: ConceptMention
    name: str


class ExtractedKnowledge(BaseModel):
    name: str
    agent: str
    source_text: str
    language: str
    concepts: List[ConceptMention]
    relations: List[RelationInstance]

    @root_validator()
    @classmethod
    def validate_concept_offsets(cls, values):
        concepts = values.get("concepts")
        language = values.get("language")
        source_text = values.get("source_text")
        if not concepts[0].instances:
            from pyclinrec import __path__ as pyclinrec_path
            from pyclinrec.dictionary import StringDictionaryLoader
            from pyclinrec.recognizer import IntersStemConceptRecognizer
            dictionary = [(concept.id, concept.matched_text)
                          for concept in concepts]
            loader = StringDictionaryLoader(dictionary)

            concept_recognizer = IntersStemConceptRecognizer(
                loader,
                os.path.join(pyclinrec_path[0], f"stopwords{language}.txt"),
                os.path.join(pyclinrec_path[0],
                             f"termination_terms{language}.txt"))
            
            concept_recognizer.initialize()
            logger.info("Reidentifying extracted term spans...")
            _, _, annotations = concept_recognizer.annotate(source_text)

            mention_index = dict()
            for concept in concepts:
                mention_index[concept.id] = set()


            for a in annotations:
                mention_index[a.concept_id].add((a.start, a.end))
            
            for concept in concepts:
                concept.instances = mention_index[concept.id]

        return values


class KnowledgeExtractor(ABC):
    def __init__(self, trigger_condition: str):
        self._condition_string = trigger_condition
        self._condition = parse_cond(trigger_condition)[0]

    @abstractmethod
    def __call__(self, corpus: str, parameters: Dict[str, str] = None):
        pass


class CompositeKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self):
        super().__init__("")
        self.registry = []  # type: List[KnowledgeExtractor]

    def add_extractor(self, extractors=List[KnowledgeExtractor]):
        self.registry.extend(extractors)

    def __call__(self, corpus, parameters: Dict[str, str] = None) -> ExtractedKnowledge:
        result = {}
        for extractor in self.registry:
            logger.debug(
                f"Checking {extractor._condition_string} against {parameters}")
            if extractor._condition(state=parameters):
                logger.debug("Triggering processor" + str(extractor))
                result[extractor.__class__.__name__] = extractor(corpus, parameters)
        return result
