import hashlib
import json
import logging
from typing import Dict
from urllib.parse import urlencode


from confz import ConfZ, ConfZFileSource
from pydantic import AnyUrl
from tqdm import tqdm

from hasextract.kext.knowledgeextractor import (
    AMRGraph,
    Concept,
    ExtractedKnowledge,
    Frame,
    FrameArgument,
    KnowledgeExtractor,
    ConceptType,
    LexicalConcept,
    Mention,
)
from hasextract.util.cached_requests import (
    post,
)
from hasextract.util.segmentation import _break_up_sentences

logger = logging.getLogger()


class USEAConfig(ConfZ):
    endpoint: AnyUrl
    CONFIG_SOURCES = ConfZFileSource(file="config/usea.json")


def _extract_lexical_concepts(response, token_spans, chunk_span, concept_index):
    concepts = []

    index = 0
    for sense in response["annotations"]["wsd"]:
        start_offset = token_spans[sense["start"]][0]
        end_offset = token_spans[sense["end"] - 1][1]
        synset = sense["features"]["synset"]
        idx = f"http://babelnet.org/rdf/page/{synset.replace('bn:', 's')}"
        if idx not in concept_index:
            concept = LexicalConcept(
                idx=idx,
                label=synset,
                concept_type=ConceptType.LEXICAL_SENSE,
                instances=[],
            )
            concept_index[idx] = concept
            concepts.append(concept)
        else:
            concept = concept_index[idx]

        concept.instances.append(
            Mention(
                start=chunk_span[0] + start_offset,
                end=chunk_span[0] + end_offset,
                text=response["texts"][index]["features"]["lemma"],
            )
        )

    return concepts, concept_index


def _extract_semantic_roles(response, token_spans, chunk_span):
    initial_roles = response["annotations"]["srl"]
    semantic_roles = []
    for role in initial_roles:
        start_offset = chunk_span[0] + token_spans[role["start"]][0]
        end_offset = chunk_span[0] + token_spans[role["end"] - 1][1]
        frame = role["features"]["frame"]
        arguments = role["features"]["arguments"]
        frame_arguments = []
        for argument in arguments:
            arg_start = chunk_span[0] + token_spans[argument["start"]][0]
            arg_end = chunk_span[0] + token_spans[argument["end"] - 1][1]
            arg_role = argument["role"]
            frame_arguments.append(
                FrameArgument(start=arg_start, end=arg_end, role=arg_role)
            )
        semantic_roles.append(
            Frame(
                frame_name=frame,
                start=start_offset,
                end=end_offset,
                arguments=frame_arguments,
            )
        )

    return semantic_roles


class USEAKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self, trigger_condition: str):
        super().__init__(trigger_condition)

    def __call__(self, corpus: str, parameters: Dict[str, str] = None):
        import spacy

        lang = parameters["source_language"]
        if lang == "en":
            nlp = spacy.load("en_core_web_sm")
        else:
            nlp = spacy.load(f"{lang}_core_news_sm")

        max_chars = 500

        logger.debug("Chunking sentences to fit API limits... ")
    
        doc = nlp(corpus)
        sentence_spans = [(sent.start_char, sent.end_char) for sent in doc.sents]
        chunks_spans = _break_up_sentences(corpus, sentence_spans, max_chars)

        concepts = []
        semantic_roles = []
        amrs = []
        for chunk_span in tqdm(chunks_spans, "Processing sentences with USEA"):
            chunk = (
                corpus[chunk_span[0] : chunk_span[1]]
                .replace("\\n", " ")
                .replace("’", "'")
            )
            m = hashlib.sha256()
            m.update(chunk.encode("utf-8"))
            # Creating a unique key for the cache.
            key = f"usea_{m.hexdigest()}"
            request_params = {"content": chunk, "type": "text"}
            if len(chunk.strip()) > 0:
                if response := post(
                    f"{USEAConfig().endpoint}",
                    json=request_params,
                    headers={},
                    key=key,
                ):
                    response = json.loads(response)
                    if "failure" not in response:
                        response = response["response"]["texts"][0]
                        response_tokens = response["texts"]
                        token_spans = []
                        current_offset = 0
                        for token in response_tokens:
                            text = token["content"]
                            offset = chunk.find(text, current_offset)
                            token_spans.append((offset, offset + len(text)))
                            current_offset = offset + len(text)

                        concept_index = {}
                        # Extracting lexical concepts and lexical senses
                        extracted_concepts, concept_index = _extract_lexical_concepts(
                            response, token_spans, chunk_span, concept_index
                        )
                        concepts.extend(extracted_concepts)
                        semantic_roles.extend(
                            _extract_semantic_roles(response, token_spans, chunk_span)
                        )
                        amr_response = response["annotations"]["amr"]
                        amrs.extend(
                            AMRGraph(
                                start=amr["start"],
                                end=amr["end"],
                                graph=amr["features"]["amr_graph"],
                            )
                            for amr in amr_response
                        )
        relations = []
        return ExtractedKnowledge(
            name="USEA Disambiguation",
            language=parameters["source_language"],
            agent="USEA Annotator",
            source_text=corpus,
            concepts=concepts,
            relations=relations,
            semantic_roles=semantic_roles,
            amr_parses=amrs,
        )
