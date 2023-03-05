import hashlib
from pathlib import Path
import pickle
from typing import Dict

from confz import ConfZ, ConfZFileSource
from elg import Service
from tqdm import tqdm

from hasextract.kext.knowledgeextractor import ConceptMention, ExtractedKnowledge, KnowledgeExtractor, MentionType, RelationInstance


class Text2TCSConfig(ConfZ):
    elg_app_id: int = 8122

    CONFIG_SOURCES = ConfZFileSource(file='config/text2tcs.json')


class Text2TCSExtractor(KnowledgeExtractor):
    def __init__(self, trigger_condition: str):
        super().__init__(trigger_condition)
        self.service = None

    def __call__(self, corpus: str, parameters: Dict[str, str] = None) -> ExtractedKnowledge:

        cache_dir = Path("./.cache")
        cache_dir.mkdir(exist_ok=True)
            

        if not self.service:
            self.service = Service.from_id(Text2TCSConfig().elg_app_id)

        from nltk.tokenize import sent_tokenize

        max_chars = 7500

        sentences = sent_tokenize(corpus)
        sentences_with_fractional_splits = []
        for sent in sentences:
            sent_len = len(sent)
            if sent_len > max_chars:
                prev_i = 0
                for i in range(0, sent_len, max_chars):
                    sentences_with_fractional_splits.append(sent[prev_i:i])
            else:
                sentences_with_fractional_splits.append(sent)
        chunks = []
        prev_cutoff = 0
        i = 0
        while i < len(sentences):
            cumulative_string = " ".join(sentences[prev_cutoff:i])
            if len(cumulative_string) > max_chars:
                while len(cumulative_string) > max_chars:
                    i-=1
                    cumulative_string = " ".join(sentences[prev_cutoff:i])
                chunks.append(" ".join(sentences[prev_cutoff:i - 1]))
                prev_cutoff = i - 1
            i += 1

        concepts = []
        concept_index = {}
        relations = []
        for sen in tqdm(chunks):
            try:
                m = hashlib.sha256()
                m.update(sen.encode("utf-8"))
                key = "text2tcs_"+m.hexdigest()
                pickle_path = Path(cache_dir, key+".pkl")
                if pickle_path.exists():
                    with open(pickle_path, "rb") as f:
                        response = pickle.load(f)
                else:
                    response = self.service(request_input=sen, request_type="text")
                    with open(pickle_path, "wb") as f:
                        pickle.dump(response, f)
                for concept in response.annotations:
                    for annotation in response.annotations[concept]:
                        id = "text2tcs_" + annotation.features['id']
                        mention = ConceptMention(id=id, matched_text=annotation.features['term'], mention_type=MentionType.EXTRACTED_TERM)
                        concepts.append(mention)
                        concept_index[id] = mention
                        if len(annotation.features['relations']) > 0:
                            for relation in annotation.features['relations']:
                                relation['source'] = id
                                relation['related concept'] = "text2tcs_" + relation['related concept']
                            relations.extend(annotation.features['relations'])
            except Exception as e:
                print(e)
        
        relations = [RelationInstance(source=concept_index[relation['source']],target=concept_index[relation['related concept']], name=relation['type']) for relation in relations]


        return ExtractedKnowledge(name="Text2TCS Terminology extraction result", agent="Text2TCS ELG", language=parameters['source_language'], source_text=corpus, concepts=concepts, relations=relations)
