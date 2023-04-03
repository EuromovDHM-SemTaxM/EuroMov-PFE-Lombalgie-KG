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
    LexicalSense,
    Mention,
)
from hasextract.util.cached_requests import (
    get,
    post,
)
from hasextract.util.segmentation import break_up_sentences, get_spacy_pipeline

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
        lemmas = [entry['features']['lemma'] for entry in response['texts'][sense["start"]:sense["end"]]]
        idx = f"http://babelnet.org/rdf/page/{synset.replace('bn:', 's')}"
        if idx not in concept_index:
            concept = LexicalSense(
                idx=idx,
                label=synset,
                instances=[],
                lemma=" ".join(lemmas)
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

# def query_relations(uri):
#     try:

#         relations = []
#         # wikidata_id = wikidata_id[1:]
#         endpoint = USEAConfig().babelnet_sparql_endpoint
#         params = {
#             "query": f"select distinct ?rel ?target where {{<{uri}> ?rel ?target.}}",
#             "format": "application/sparql-results+json",
#             "timeout": 0,
#             "signal_void": "on",
#         }
#         if result := get(f"{endpoint}?{urlencode(params)}", headers={}):
#             ret = json.loads(result)
#             relations.extend(
#                 (r["rel"]["value"], r["target"]["value"])
#                 for r in ret["results"]["bindings"]
#             )

#     except json.decoder.JSONDecodeError:
#         return None

#     return relations

frame_id_cache = {}
def get_frame_ids(name, lemma):
    if name in frame_id_cache:
        return frame_id_cache[name]
    params = {
        "lemma": lemma,
    }
    if not (
        result := get(
            f"{USEAConfig().verbatlas_endpoint}?{urlencode(params)}",
            headers={},
        )
    ):
        return None
    ret = json.loads(result)
    for synset in ret:
        fid = synset["va_frame_id"]
        frame_id_cache[synset["va_frame_name"]] = fid
    return frame_id_cache[name]
    

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
        nlp = get_spacy_pipeline(lang)

        max_chars = 500

        logger.debug("Chunking sentences to fit API limits... ")
    
        doc = nlp(corpus)
        sentence_spans = [(sent.start_char, sent.end_char) for sent in doc.sents]
        chunks_spans = break_up_sentences(corpus, sentence_spans, max_chars)

        concepts = []
        semantic_roles = []
        amrs = []
        for chunk_span in tqdm(chunks_spans, "Processing sentences with USEA"):
            chunk = (
                corpus[chunk_span[0] : chunk_span[1]]
                .replace("\\n", " ")
                .replace("\n", " ")
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
                    #invalidate_callback=lambda x: "failure" in json.loads(x)
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
