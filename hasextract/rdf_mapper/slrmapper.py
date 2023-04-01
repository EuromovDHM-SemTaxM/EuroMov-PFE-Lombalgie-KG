from pydantic import AnyUrl
from rdflib import OWL, RDF, RDFS, Literal
from hasextract.kext.knowledgeextractor import Frame
from hasextract.rdf_mapper.mapper import KnowledgeGraphMapper, OntolexMapper
from hasextract.verbatlas.verbatlas import VerbAtlas


class SRLMapper(OntolexMapper):
    def __init__(
        self,
        base_uri: AnyUrl,
        prefix_name: str = "base",
        kg_description: str = None,
        namespaces=None,
        kg=None,
        uri_index=None,
    ):
        namespaces = namespaces or {}
        namespaces.update(
            {
                "framester": "https://w3id.org/framester/schema/",
                "framenet": "https://w3id.org/framester/framenet/tbox/",
                "fr": "https://w3id.org/framester/data/framesterrole/",
                "gfe": "https://w3id.org/framester/framenet/abox/gfe/",
                'bn': 'http://babelnet.org/rdf/', 
                "vn": "https://w3id.org/framester/vn/schema/"
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
        self.verb_atlas = VerbAtlas()
        self.frame_role_mapping = {
          "Agent" : ["fr", "agent"],
          "Theme" : ["fr", "theme"],
          "Attribute" : ["gfe", "Attribute"],
          "Source" : ["fr", "source"],
          "Destination" : ["fr", "destination"],
          "Instrument" : ["fr", "instrument"],
          "Purpose" : ["fr", "purpose"],
          "Cause" : ["fr", "cause"],
          "Result" : ["fr", "result"],
          "Value" : ["fr", "value"],
          "Extent" : ["fr", "extent"],
          "Location" : ["fr", "location"],
          "Experiencer" : ["fr", "experiencer"],
          "Stimulus" : ["fr", "stimulus"],
          "Time" : ["fr", "time"],
          "Topic" : ["fr", "topic"],
          "Recipient" : ["fr", "recipient"],
          "Co-Theme" : ["gfe", "Co_theme"],
          "Material" : ["fr", "material"],
          "Co-Agent" : ["gfe", "Co_participant"],
          "Idiom" : ["gfe", "Idiom"],
          "Goal" : ["gfe", "Goal"],
          "Beneficiary" : ["gfe", "Beneficiary"],
          "Patient" : ["gfe", "Patient"],
          "Co-Patient" : ["gfe", "Co_Patient"],
          "Product" : ["gfe", "Product"],
          "Asset" : ["gfe", "Asset"]
        }

    def __call__(self, document_uri, document_text, element: Frame):
        frame_info = self.verb_atlas.get_frame_by_name(element.frame_name)
        uuid = self._uuid(frame_info.idx)
        if uuid not in self.uri_index:
            frame_uri = self._bns()[f"frame_{element.frame_name.lower()}_{uuid}"]
            self._add(frame_uri, RDF.type, self._ns("framester").ConceptualFrame)
            self._add(frame_uri, RDF.type, self._ns("framester").Frame)
            self._add(frame_uri, RDF.type, OWL.Class)
            self.uri_index[uuid] = frame_uri
        else:
            frame_uri = self.uri_index[uuid]

        self._add(
            frame_uri, self._ns("framenet").frame_name, Literal(element.frame_name)
        )
        self._add(frame_uri, RDFS.label, Literal(element.frame_name))
        self._add(frame_uri, RDFS.comment, Literal(frame_info.description))
        self._add(
            frame_uri, self._ns("framenet").definition, Literal(frame_info.description)
        )
        self._add(frame_uri, self._ns("framenet").frame_ID, Literal(frame_info.idx))
        for frame_element in frame_info.frame_elements:
            role_uuid = self._uuid(frame_element)
            if role_uuid not in self.uri_index:
                mapping = self.frame_role_mapping[frame_element]
                role_uri = self._ns(mapping[0])[mapping[1]]
                self._add(frame_uri, self._ns("framenet").vnRole, role_uri)
                self.uri_index[role_uuid] = role_uri
            else:
              role_uri = self.uri_index[role_uuid]
            
            synset = frame_info.synset.split(":")[1]
            ls_uri = self._create_lexical_sense(str(self._ns("bn")[f"s{synset}"]))
            
            frame_element_uri = (
                f"frameelement_{element.frame_name.lower()}.{frame_element}_{uuid}"
            )
            self._add(self._bns()[frame_element_uri], RDF.type, self._ns("framenet").FrameElement)
            self._add(
                frame_uri, self._ns("framenet").hasFrameElement, self._bns()[frame_element_uri]
            )