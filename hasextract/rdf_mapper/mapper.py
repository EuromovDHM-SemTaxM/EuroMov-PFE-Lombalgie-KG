from abc import ABC, abstractmethod
from typing import Any, Type, TypeVar
import uuid

from pydantic import AnyUrl, BaseModel, validator
from rdflib import OWL, RDF, RDFS, XSD, BNode, Graph, Literal, URIRef
import rdflib
from tqdm import tqdm
from hasextract.kext.knowledgeextractor import (
    AMRGraph,
    Concept,
    ConceptRelation,
    ExtractedKnowledge,
    Frame,
    KGConcept,
    LexicalSense,
    LexicalRelation,
    Mention,
    OntologyRelation,
    Relation,
    SemanticRelation,
    TermConcept,
)

from hasextract.util.visitor import visitor

import kglab

from hasextract.verbatlas.verbatlas import VerbAtlas


class KnowledgeGraphMapper(ABC):
    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
        kg=None,
        uri_index=None,
    ) -> None:
        super().__init__()
        self.base_uri = base_uri
        self.description = kg_description or ""
        self.prefix_name = prefix_name

        namespaces = namespaces or {}
        namespaces.update({prefix_name: base_uri})
        if kg:
            self.kg = kg
            for ns in namespaces.items():
                self.kg.add_ns(*ns)
        else:
            self.kg = kglab.KnowledgeGraph(
                name=kg_description, base_uri=base_uri, namespaces=namespaces
            )

        self.uri_index = uri_index or {'item': 'value'}

    @staticmethod
    def init_from(cls, kg_mapper: "KnowledgeGraphMapper") -> "KnowledgeGraphMapper":

        return cls(
            base_uri=kg_mapper.base_uri,
            prefix_name=kg_mapper.prefix_name,
            kg_description=kg_mapper.description,
            kg=kg_mapper.kg,
            uri_index=kg_mapper.uri_index,
        )

    def _ns(self, name):
        return self.kg.get_ns(name)

    def _bns(self):
        return self.kg.get_ns(self.prefix_name)

    def _add(self, subject, predicate, object):
        self.kg.add(subject, predicate, object)

    def _uuid(self, key: str):
        return uuid.uuid5(uuid.UUID("00000000-0000-0000-0000-000000000000"), key)

    @abstractmethod
    def __call__(self, document_uri, document_text, element: Any):
        pass


class CompositeMapper(KnowledgeGraphMapper):
    M = TypeVar("M", bound=BaseModel)

    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
    ) -> None:
        namespaces = namespaces or {}
        namespaces.update(
            {
                "framenet": "https://w3id.org/framester/framenet/tbox/",
                "schema": "https://schema.org/",
            }
        )
        super().__init__(base_uri, prefix_name, kg_description, namespaces)
        self.bindings = (
            {}
        )  # type: dict[Type[CompositeMapper.M], Type[KnowledgeGraphMapper]]

    def bind_mapper(self, mapper: Type[KnowledgeGraphMapper], model_class: Type[M]):
        self.bindings[model_class] = mapper

    def bind_mappers(self, bindings: dict[Type[M], Type[KnowledgeGraphMapper]]):
        self.bindings.update(bindings)

    def __call__(
        self, document_name, extracted_knowledge: ExtractedKnowledge
    ) -> kglab.KnowledgeGraph:

        doc_uri = self._bns()[f"document_{self._uuid(document_name)}"]

        self._add(doc_uri, self._ns("rdf").type, self._ns("schema").Document)
        self._add(doc_uri, self._ns("rdf").type, self._ns("framenet").Document)
        self._add(
            doc_uri,
            self._ns("schema").description,
            rdflib.Literal(extracted_knowledge.name),
        )
        self._add(
            doc_uri,
            self._ns("schema").text,
            rdflib.Literal(extracted_knowledge.source_text),
        )

        coalesced_knowledge = []
        coalesced_knowledge.extend(extracted_knowledge.concepts)
        coalesced_knowledge.extend(extracted_knowledge.relations)
        coalesced_knowledge.extend(extracted_knowledge.semantic_roles)
        coalesced_knowledge.extend(extracted_knowledge.amr_parses)
        
        instance_cache = {}
        
        for item in tqdm(coalesced_knowledge, desc="Mapping extracted items to RDF"):
            if item.__class__ in self.bindings:
                if item.__class__ in instance_cache:
                    mapper = instance_cache[item.__class__]
                else:
                    mapper_class = self.bindings[item.__class__]
                    mapper = mapper_class.init_from(mapper_class, self)
                    instance_cache[item.__class__] = mapper
                mapper(doc_uri, extracted_knowledge.source_text, item)

        return self.kg


class OntolexMapper(KnowledgeGraphMapper):
    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
        kg=None,
        uri_index=None,
    ) -> None:
        namespaces = namespaces or {}
        namespaces.update(
            {
                "ontolex": "http://www.w3.org/ns/lemon/ontolex#",
                "skos": "http://www.w3.org/2004/02/skos/core#",
                "lexinfo": "http://www.lexinfo.net/ontology/2.0/lexinfo#",
                "oa": "http://www.w3.org/ns/oa#",  # The Web Annotation Data Model
                "prov": "http://www.w3.org/ns/prov#",
                "vartrans": "http://www.w3.org/ns/lemon/vartrans#",
                "framester": "https://w3id.org/framester/schema/",
                "framenet": "https://w3id.org/framester/framenet/tbox/",
                "fr": "https://w3id.org/framester/data/framesterrole/",
                "gfe": "https://w3id.org/framester/framenet/abox/gfe/",
            }
        )
        super().__init__(
            base_uri=base_uri,
            prefix_name=prefix_name,
            kg_description=kg_description,
            namespaces=namespaces,
            kg=kg,
            uri_index=uri_index,
        )

    def _create_offset_selector(
        self, start_offset, end_offset, document_uri, document_text
    ) -> URIRef:
        uuid = self._uuid(f"{str(document_uri)}{start_offset}_{end_offset}")
        if uuid not in self.uri_index:
            uri = URIRef(f"{self.base_uri}selector_{str(uuid)}")
            self.uri_index[uuid] = uri
            self._add(uri, RDF.type, self._ns("oa").TextPositionSelector)
            self._add(uri, RDF.type, self._ns("oa").TextQuoteSelector)
            self._add(
                uri,
                self._ns("oa").exact,
                Literal(document_text[start_offset:end_offset]),
            )
            self._add(
                uri, self._ns("oa").start, Literal(start_offset, datatype=XSD.integer)
            )
            self._add(
                uri, self._ns("oa").end, Literal(end_offset, datatype=XSD.integer)
            )
            self._add(
                uri, self._ns("oa").end, Literal(end_offset, datatype=XSD.integer)
            )

        return self.uri_index[uuid]

    def _create_web_annotation(
        self,
        concept_uri: URIRef,
        mentions: list[Mention],
        document_uri: URIRef,
        document_text: str,
    ) -> URIRef:
        uuid = self._uuid(f"annotation_{concept_uri}")
        if uuid not in self.uri_index:
            annotation = URIRef(f"{self.base_uri}annotation_{uuid}")
            self.uri_index[uuid] = annotation

            self._add(annotation, RDF.type, self._ns("oa").Annotation)
            self._add(annotation, self._ns("schema").about, document_uri)
            self._add(annotation, self._ns("oa").hasBody, concept_uri)

            target_bnode = BNode()
            self._add(annotation, self._ns("oa").hasTarget, target_bnode)

            for mention in mentions:
                selector = self._create_offset_selector(
                    mention.start, mention.end, document_uri, document_text
                )
                self._add(target_bnode, self._ns("oa").hasSelector, selector)

        return self.uri_index[uuid]

    def _create_prov_agent(self, provenance) -> URIRef:
        uuid = self._uuid(provenance)
        if uuid in self.uri_index:
            return self.uri_index[uuid]
        source = URIRef(f"{self.base_uri}agent_{uuid}")

        self._add(
            source,
            RDF.type,
            self._ns("prov").Agent,
        )
        self.uri_index[uuid] = source
        return source

    def _create_lexical_entry(
        self,
        canonical_form=None,
        lexical_forms: list[str] = None,
        other_forms=list[str],
    ):
        canonical_form = canonical_form or ""
        if not lexical_forms:
            lexical_forms = []
        if not other_forms:
            other_forms = []

        representation = canonical_form

        le_uuid = self._uuid(f"le_{representation}")
        if le_uuid not in self.uri_index:
            le_uri = URIRef(f"{self.base_uri}le_{le_uuid}")
            self._add(le_uri, RDF.type, self._ns("ontolex").LexicalEntry)
            self.uri_index[le_uuid] = le_uri
        else:
            le_uri = self.uri_index[le_uuid]
        if canonical_form and f"cf_{le_uuid}" not in self.uri_index:
            cf_uri = URIRef(f"{self.base_uri}cf_{le_uuid}")
            self._add(cf_uri, RDF.type, self._ns("ontolex").Form)
            self._add(le_uri, self._ns("ontolex").canonicalForm, cf_uri)
            self.uri_index[f"cf_{le_uuid}"] = cf_uri

        for f_counter, lf in enumerate(lexical_forms, start=1):
            uuid = f"lf_{le_uuid}_{f_counter}"
            if uuid not in self.uri_index:
                lf_uri = URIRef(f"{self.base_uri}{uuid}")
                self._add(lf_uri, RDF.type, self._ns("ontolex").Form)
                self._add(le_uri, self._ns("ontolex").lexicalForm, lf_uri)
                self._add(lf_uri, self._ns("ontolex").writtenRep, Literal(lf))
                self.uri_index[uuid] = lf_uri
        for f_counter, of in enumerate(other_forms, start=1):
            uuid = f"of_{le_uuid}_{f_counter}"
            of_uri = URIRef(f"{self.base_uri}{uuid}")
            self._add(of_uri, RDF.type, self._ns("ontolex").Form)
            self._add(le_uri, self._ns("ontolex").otherForm, of_uri)
            self._add(of_uri, self._ns("ontolex").writtenRep, Literal(of))
            self.uri_index[uuid] = of_uri
        return self.uri_index[le_uuid]

    def _create_lexical_concept(self, label):
        lc_uuid = self._uuid(f"fc_{label}")
        if lc_uuid not in self.uri_index:
            lc_uri = URIRef(f"{self.base_uri}lc_{lc_uuid}")
            self._add(lc_uri, RDF.type, self._ns("ontolex").LexicalConcept)
            self.uri_index[lc_uuid] = lc_uri
        return self.uri_index[lc_uuid]

    def _create_lexical_sense(self, idx):
        ls_uuid = self._uuid(f"ls_{idx}")
        if ls_uuid not in self.uri_index:
            ls_uri = URIRef(f"{self.base_uri}ls_{ls_uuid}")
            self._add(ls_uri, RDF.type, self._ns("ontolex").LexicalSense)
            if "http" in idx and "babelnet" in idx:
                self._add(ls_uri, self._ns("skos").exactMatch, URIRef(idx))
            self.uri_index[ls_uuid] = ls_uri
        return self.uri_index[ls_uuid]

    def _create_ontolex_common(
        self, document_uri: URIRef, element: Concept, document_text
    ) -> tuple[URIRef, URIRef]:
        idx = element.idx
        label = element.label
        prov = self._create_prov_agent(element.provenance["agent"])
        concept_uri = URIRef(idx)
        annotation = self._create_web_annotation(
            concept_uri, element.instances, document_uri, document_text
        )
        self._add(annotation, RDF.type, self._ns("prov").Entity)
        self._add(annotation, self._ns("prov").wasAttributedTo, prov)

        lc_uri = self._create_lexical_concept(label)

        forms = [label]
        forms.extend(element.altLabels or [])
        le_uri = self._create_lexical_entry(
            canonical_form=element.lemma, other_forms=forms
        )
        self._add(le_uri, self._ns("ontolex").denotes, concept_uri)

        return le_uri, lc_uri


class TermConceptMapper(OntolexMapper):
    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
        kg=None,
        uri_index=None,
    ):
        super().__init__(
            base_uri=base_uri,
            prefix_name=prefix_name,
            kg_description=kg_description,
            namespaces=namespaces,
            kg=kg,
            uri_index=uri_index,
        )

    def __call__(self, document_uri, document_text, element: TermConcept):
        idx = element.idx
        label = element.label
        uuid = self._uuid(f"{idx}_{label}")
        if uuid not in self.uri_index:
            element.idx = idx if "http" in idx else str(self._bns()[f"tc_{uuid}"])
            _, _ = self._create_ontolex_common(document_uri, element, document_text)
            self.uri_index[uuid] = URIRef(element.idx)


class LexicalConceptMapper(OntolexMapper):
    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
        kg=None,
        uri_index=None,
    ):
        super().__init__(
            base_uri=base_uri,
            prefix_name=prefix_name,
            kg_description=kg_description,
            namespaces=namespaces,
            kg=kg,
            uri_index=uri_index,
        )

    def __call__(self, document_uri, document_text, element: LexicalSense):
        idx = element.idx
        label = element.label
        uuid = self._uuid(f"{idx}_{label}")
        if uuid not in self.uri_index:
            concept_uri = URIRef(idx)

            le_url, lc_uri = self._create_ontolex_common(
                document_uri, element, document_text
            )
            ls_uri = self._create_lexical_sense(idx)
            self._add(le_url, self._ns("ontolex").sense, ls_uri)
            self._add(lc_uri, self._ns("ontolex").lexicalizedSense, ls_uri)

            self.uri_index[uuid] = concept_uri


class KGConceptMapper(OntolexMapper):
    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
        kg=None,
        uri_index=None,
    ):
        super().__init__(
            base_uri=base_uri,
            prefix_name=prefix_name,
            kg_description=kg_description,
            namespaces=namespaces,
            kg=kg,
            uri_index=uri_index,
        )

    def __call__(self, document_uri, document_text, element: LexicalSense):
        idx = element.idx
        label = element.label
        uuid = self._uuid(f"{idx}_{label}")
        if uuid not in self.uri_index:
            concept_uri = URIRef(idx)
            le_url, lc_uri = self._create_ontolex_common(
                document_uri, element, document_text
            )
            self._add(lc_uri, self._ns("ontolex").isConceptOf, concept_uri)
            self._add(le_url, self._ns("ontolex").denotes, concept_uri)

            self.uri_index[uuid] = concept_uri


class LexicalRelationMapper(OntolexMapper):
    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
        kg=None,
        uri_index=None,
    ):
        super().__init__(
            base_uri=base_uri,
            prefix_name=prefix_name,
            kg_description=kg_description,
            namespaces=namespaces,
            kg=kg,
            uri_index=uri_index,
        )

    def __call__(self, document_uri, document_text, element: LexicalRelation):
        source_le, _ = self._create_ontolex_common(
            document_uri, element.source, document_text
        )
        target_le, _ = self._create_ontolex_common(
            document_uri, element.target, document_text
        )
        rel_uuid = self._uuid(f"lr_deriv_{element.source.idx}_{element.target.idx}")
        if rel_uuid not in self.uri_index:
            rel_uri = URIRef(f"{self.base_uri}lr_deriv_{rel_uuid}")
            self._add(rel_uri, RDF.type, self._ns("vartrans").LexicalRelation)
            self.uri_index[rel_uuid] = rel_uri
        else:
            rel_uri = self.uri_index[rel_uuid]
        self._add(rel_uri, self._ns("vartrans").source, source_le)
        self._add(rel_uri, self._ns("vartrans").target, target_le)


class SemanticRelationMapper(OntolexMapper):
    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
        kg=None,
        uri_index=None,
    ):
        super().__init__(
            base_uri=base_uri,
            prefix_name=prefix_name,
            kg_description=kg_description,
            namespaces=namespaces,
            kg=kg,
            uri_index=uri_index,
        )

    def __call__(self, document_uri, document_text, element: LexicalRelation):
        source_ls = self._create_lexical_sense(element.source.idx)
        target_ls = self._create_lexical_sense(element.source.idx)

        rel_uuid = self._uuid(f"sr_{element.source.idx}_{element.target.idx}")
        if rel_uuid not in self.uri_index:
            rel_uri = URIRef(f"{self.base_uri}sr_{rel_uuid}")
            rel_class_uri = self._bns()[element.name]
            if rel_class_uri not in self.uri_index:
                rel_class = URIRef(rel_class_uri)
                self._add(
                    rel_class, RDFS.subClassOf, self._ns("vartrans").SemanticRelation
                )
                self.uri_index[rel_class_uri] = rel_class
            else:
                rel_class = self.uri_index[rel_class_uri]
            self._add(rel_uri, RDF.type, self._ns("vartrans").SemanticRelation)
            self.uri_index[rel_uuid] = rel_uri
        else:
            rel_uri = self.uri_index[rel_uuid]
        self._add(rel_uri, self._ns("vartrans").source, source_ls)
        self._add(rel_uri, self._ns("vartrans").target, target_ls)


class ConceptRelationMapper(OntolexMapper):
    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
        kg=None,
        uri_index=None,
    ):
        super().__init__(
            base_uri=base_uri,
            prefix_name=prefix_name,
            kg_description=kg_description,
            namespaces=namespaces,
            kg=kg,
            uri_index=uri_index,
        )

    def __call__(self, document_uri, document_text, element: ConceptRelation):
        _, source_lc = self._create_ontolex_common(
            document_uri, element.source, document_text
        )
        _, target_lc = self._create_ontolex_common(
            document_uri, element.target, document_text
        )
        self._add(source_lc, self._ns("skos").narrower, target_lc)
        if "http" in element.name:
            self._add(source_lc, URIRef(element.name), target_lc)
        else:
            self._add(source_lc, self._bns()[element.name], target_lc)


class OntologyRelationMapper(KnowledgeGraphMapper):
    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
        kg=None,
        uri_index=None,
    ):
        super().__init__(
            base_uri=base_uri,
            prefix_name=prefix_name,
            kg_description=kg_description,
            namespaces=namespaces,
            kg=kg,
            uri_index=uri_index,
        )

    def __call__(self, document_uri, document_text, element: ConceptRelation):
        if "http://" in element.name:
            relation_uri = URIRef(element.name)
        else:
            relation_uri = self._bns()[element.name]
        source_uuid = self._uuid(f"{element.source.idx}_{element.source.label}")
        target_uuid = self._uuid(f"{element.target.idx}_{element.target.label}")
        self._add(
            self.uri_index[source_uuid], relation_uri, self.uri_index[target_uuid]
        )
