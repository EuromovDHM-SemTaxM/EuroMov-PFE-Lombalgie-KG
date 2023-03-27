from abc import ABC, abstractmethod
from typing import Any
import uuid

from pydantic import AnyUrl
from rdflib import RDF, RDFS, XSD, BNode, Graph, Literal, URIRef
import rdflib
from hasextract.kext.knowledgeextractor import (
    AMRGraph,
    Concept,
    ConceptRelation,
    ExtractedKnowledge,
    Frame,
    KBConcept,
    LexicalConcept,
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
    ) -> None:
        super().__init__()
        self.base_uri = base_uri
        self.description = kg_description or ""
        self.prefix_name = prefix_name

        namespaces = namespaces or {}

        self.kg = kglab.KnowledgeGraph(
            name=kg_description, base_uri=base_uri, namespaces=namespaces
        )

    def _ns(self, name):
        return self.kg.get_ns(name)

    def _bns(self):
        return self.kg.get_ns(self.prefix_name)

    def _add(self, subject, predicate, object):
        self.kg.add(subject, predicate, object)

    @abstractmethod
    def __call__(
        self, document_name, extracted_knowledge: ExtractedKnowledge, target_file
    ) -> Any:
        pass


class LLODKGMapper(KnowledgeGraphMapper):
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
                "ontolex": "http://www.w3.org/ns/lemon/ontolex#",
                "skos": "http://www.w3.org/2004/02/skos/core#",
                "lexinfo": "http://www.lexinfo.net/ontology/2.0/lexinfo#",
                "oa": "http://www.w3.org/ns/oa#",  # The Web Annotation Data Model
                "framester": "https://w3id.org/framester/schema/",
                "prov": "http://www.w3.org/ns/prov#",
                prefix_name: base_uri,
                "vartrans": "http://www.w3.org/ns/lemon/vartrans#",
                "framester": "https://w3id.org/framester/schema/",
                "framenet": "https://w3id.org/framester/framenet/tbox/"
            }
        )
        super().__init__(
            base_uri=base_uri,
            prefix_name=prefix_name,
            kg_description=kg_description,
            namespaces=namespaces,
        )
        self.uri_index = {}
        
        self.verb_atlas = VerbAtlas()

    def _uuid(self, key: str):
        return uuid.uuid5(uuid.UUID("00000000-0000-0000-0000-000000000000"), key)

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

    def _map_frame(self, document_uri: URIRef, element: Frame, document_text):
        uuid = self._uuid(element.frame_name)
        frame_info = self.verb_atlas.get_frame_by_name(element.frame_name)
        if uuid not in self.uri_index:
            frame_uri = self._bns()[f"frame_{uuid}"]
            self._add(frame_uri, RDF.type, self._ns("framester").ConceptualFrame)
            self.uri_index[uuid] = frame_uri
        pass

    def _map_amr(self, document_uri: URIRef, element: AMRGraph, document_text: str):
        pass

    def _map_relation(self, document_uri: URIRef, document_text, element: Relation):
        if isinstance(element, ConceptRelation):
            self._map_concept_relation(element, document_uri, document_text)
        elif isinstance(element, LexicalRelation):
            self._map_lexical_relation(element, document_uri, document_text)
        elif isinstance(element, SemanticRelation):
            self._map_semantic_relation(element, document_uri, document_text)
        elif isinstance(element, OntologyRelation):
            self._map_ontology_relation(element, document_uri, document_text)

    def _map_concept_relation(
        self, relation: ConceptRelation, document_uri: URIRef, document_text: str
    ):

        _, source_lc = self._create_ontolex_common(
            document_uri, relation.source, document_text
        )
        _, target_lc = self._create_ontolex_common(
            document_uri, relation.target, document_text
        )
        self._add(source_lc, self._ns("skos").narrower, target_lc)
        if "http" in relation.name:
            self._add(source_lc, URIRef(relation.name), target_lc)
        else:
            self._add(source_lc, self._bns()[relation.name], target_lc)

    def _map_lexical_relation(
        self, relation: LexicalRelation, document_uri: URIRef, document_text: str
    ):
        source_le, _ = self._create_ontolex_common(
            document_uri, relation.source, document_text
        )
        target_le, _ = self._create_ontolex_common(
            document_uri, relation.target, document_text
        )
        rel_uuid = self._uuid(f"lr_deriv_{relation.source.idx}_{relation.target.idx}")
        if rel_uuid not in self.uri_index:
            rel_uri = URIRef(f"{self.base_uri}lr_deriv_{rel_uuid}")
            self._add(rel_uri, RDF.type, self._ns("vartrans").LexicalRelation)
            self.uri_index[rel_uuid] = rel_uri
        else:
            rel_uri = self.uri_index[rel_uuid]
        self._add(rel_uri, self._ns("vartrans").source, source_le)
        self._add(rel_uri, self._ns("vartrans").target, target_le)

    def _map_semantic_relation(
        self, relation: SemanticRelation, document_uri: URIRef, document_text: str
    ):
        source_ls = self._create_lexical_sense(relation.source.idx)
        target_ls = self._create_lexical_sense(relation.source.idx)

        rel_uuid = self._uuid(f"sr_{relation.source.idx}_{relation.target.idx}")
        if rel_uuid not in self.uri_index:
            rel_uri = URIRef(f"{self.base_uri}sr_{rel_uuid}")
            rel_class_uri = self._bns()[relation.name]
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

    def _map_ontology_relation(self, relation: OntologyRelation, document_uri: URIRef, document_text: str):
        if "http://" in relation.name:
            relation_uri = URIRef(relation.name)
        else:
            relation_uri = self._bns()[relation.name]
        source_uuid = self._uuid(f"{relation.source.idx}_{relation.source.label}")
        target_uuid = self._uuid(f"{relation.target.idx}_{relation.target.label}")
        self._add(
            self.uri_index[source_uuid], relation_uri, self.uri_index[target_uuid]
        )

    def _map_term_concept(
        self, document_uri: URIRef, element: TermConcept, document_text
    ):
        idx = element.idx
        label = element.label
        uuid = self._uuid(f"{idx}_{label}")
        if uuid not in self.uri_index:
            concept_uri = URIRef(idx)
            le_url, lc_uri = self._create_ontolex_common(
                document_uri, element, document_text
            )
            # ls_uri = self._create_lexical_sense(idx)
            # self._add(le_url,self._ns("ontolex").sense, ls_uri)
            # self._add(lc_uri, self._ns("ontolex").lexicalizedSense, ls_uri)
            self.uri_index[uuid] = concept_uri
        return self.uri_index[uuid]

    def _map_lexical_semantic_concept(
        self, document_uri: URIRef, element: LexicalConcept, document_text
    ):
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
        return self.uri_index[uuid]

    def _map_kg_concept(
        self, document_uri: URIRef, element: KBConcept, document_text
    ) -> tuple[URIRef, URIRef]:
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
        return self.uri_index[uuid]

    def _map_extracted_knowledge(
        self, document_uri: URIRef, element: ExtractedKnowledge, document_text
    ):
        for concept in element.concepts:
            if isinstance(concept, TermConcept):
                self._map_term_concept(document_uri, concept, document_text)
            elif isinstance(concept, KBConcept):
                self._map_kg_concept(document_uri, concept, document_text)
            elif isinstance(concept, LexicalConcept):
                self._map_lexical_semantic_concept(document_uri, concept, document_text)

        for relation in element.relations:
            self._map_relation(document_uri, document_text, relation)

        for frame in element.semantic_roles:
            self._map_frame(document_uri, frame, document_text)

        for amr_graph in element.amr_parses:
            self._map_amr(document_uri, amr_graph, document_text)

    def __call__(
        self, document_name, extracted_knowledge: ExtractedKnowledge, target_file
    ) -> Any:
        doc_uri = URIRef(f"{self.base_uri}/document_{self._uuid(document_name)}")

        self._add(doc_uri, self._ns("rdf").type, self._ns("schema").Document)
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

        self._map_extracted_knowledge(
            doc_uri, extracted_knowledge, extracted_knowledge.source_text
        )
        self.kg.save_rdf(target_file, format="ttl")
