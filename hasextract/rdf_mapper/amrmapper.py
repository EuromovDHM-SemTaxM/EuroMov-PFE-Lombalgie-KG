from pydantic import AnyUrl
from hasextract.kext.knowledgeextractor import AMRGraph
from hasextract.rdf_mapper.mapper import KnowledgeGraphMapper
from hasextract.verbatlas.verbatlas import VerbAtlas


class AMRMapper(KnowledgeGraphMapper):
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

    def __call__(self, document_uri, document_text, element: AMRGraph):
        pass
