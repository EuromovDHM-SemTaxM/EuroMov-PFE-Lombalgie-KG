from abc import ABC, abstractmethod

from pydantic import AnyUrl
from rdflib import Graph
from hasextract.kext.knowledgeextractor import AMRGraph, Frame, RelationInstance

from hasextract.util.visitor import visitor


class RDFMapper(ABC):
  def __init__(self, base_uri: AnyUrl) -> None:
    super().__init__()
    self.base_uri = base_uri
    
  @abstractmethod
  @visitor(Frame)
  def generate(self, element: Frame):
    pass
  
  @abstractmethod
  @visitor(AMRGraph)
  def generate(self, element: AMRGraph):
    pass
  
  @abstractmethod
  @visitor(RelationInstance)
  def generate(self, element: RelationInstance):
    pass
  
  @abstractmethod
  def __call__(self, *args: Any, **kwds: Any) -> Any:
    pass

  
