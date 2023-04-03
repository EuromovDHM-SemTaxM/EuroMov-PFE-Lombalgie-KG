from enum import Enum
import logging
from abc import ABC, abstractmethod
import os
from typing import Dict, List, Optional

from pycond import parse_cond
from pydantic import AnyUrl, BaseModel, root_validator
from tqdm import tqdm

from hasextract.util.segmentation import get_spacy_pipeline


logger = logging.getLogger()


class Mention(BaseModel):
    start: int
    end: int
    text: str
    score: Optional[float]
    modifiers: Optional[list[str]]

    def __hash__(self):
        return hash((type(self),) + (self.start, self.end, self.text))


class Concept(BaseModel, ABC):
    idx: str
    label: str
    altLabels: Optional[list[str]]
    instances: Optional[list[Mention]]
    mappings: Optional[list[tuple[str, str]]]
    provenance: Optional[dict[str, str]]
    lemma: Optional[str]
    language: Optional[str]

    def __hash__(self):
        return hash((type(self),) + tuple(self.idx))

    # class Config:
    #     arbitrary_types_allowed=True


class TermConcept(Concept):
    rank: Optional[int]
    rule: Optional[str]
    confidence_score: Optional[float]

    def __init__(self, *args, **kwargs):
        super(TermConcept, self).__init__(*args, **kwargs)


class LexicalSense(Concept):
    
    def __init__(self, *args, **kwargs):
        super(LexicalSense, self).__init__(*args, **kwargs)


class KGConcept(Concept):
    def __init__(self, *args, **kwargs):
        super(KGConcept, self).__init__(*args, **kwargs)


class Relation(BaseModel):
    source: Concept
    target: Concept
    name: str
    provenance: Optional[dict[str, str]]

class LexicalRelation(Relation):
    def __init__(self, *args, **kwargs):
        super(LexicalRelation, self).__init__(*args, **kwargs)

class SemanticRelation(Relation):
    def __init__(self, *args, **kwargs):
        super(SemanticRelation, self).__init__(*args, **kwargs)

class ConceptRelation(Relation):
    def __init__(self, *args, **kwargs):
        super(ConceptRelation, self).__init__(*args, **kwargs)

class OntologyRelation(Relation):
    def __init__(self, *args, **kwargs):
        super(OntologyRelation, self).__init__(*args, **kwargs)

class FrameArgument(BaseModel):
    start: int
    end: int
    role: str


class Frame(BaseModel):
    frame_name: str
    start: int
    end: int
    arguments: list[FrameArgument]
    provenance: Optional[dict[str, str]]
    language: Optional[str]


class AMRGraph(BaseModel):
    start: int
    end: int
    graph: str
    provenance: Optional[dict[str, str]]
    language: Optional[str]


class ExtractedKnowledge(BaseModel):
    name: str
    agent: str
    source_text: str
    language: str
    concepts: list[Concept]
    relations: list[Relation]
    semantic_roles: Optional[list[Frame]]
    amr_parses: Optional[list[AMRGraph]]

    @root_validator()
    @classmethod
    def validate(cls, values):
        nlp = get_spacy_pipeline(values.get("language"))
        concepts = values.get("concepts")
        source_text = values.get("source_text")
        logger.info("Validating extracted knowledge...")
        
        import re

        for concept in tqdm(concepts, desc="Validating concepts"):
            if not concept.instances:
                concept.instances = set()
                for match in re.finditer(re.escape(concept.label), source_text):
                    concept.instances.add(
                        Mention(
                            start=match.start(),
                            end=match.end(),
                            text=source_text[match.start() : match.end()],
                        )
                    )
                concept.instances = list(concept.instances)
            
            concept.language = values.get("language")
            
            label = concept.label
            lemmas = [token.lemma_ for token in nlp(label)]
            concept.lemma = " ".join(lemmas)


        
        logger.info("Validation complete.")
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
            logger.debug(f"Checking {extractor._condition_string} against {parameters}")
            if extractor._condition(state=parameters):
                logger.debug(f"Triggering processor{str(extractor)}")
                result[extractor.__class__.__name__] = extractor(corpus, parameters)
        return result
