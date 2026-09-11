from r4agent import *
from r4agent.utils import setup_logging
from r4agent.rag import DbElement, RAGClient, QDrantDatabase, Chunker, ChunkerConfig, VectorDatabaseConfig
from r4agent.struct.models import CosmosEmbedding, CosmosConfig


setup_logging(force=True)

db_client = RAGClient(
  QDrantDatabase(VectorDatabaseConfig()),
  Chunker(ChunkerConfig),
  CosmosEmbedding(CosmosConfig())
)

# path = Path("/home/enes/Desktop/ollama2.md")
path = "https://tarhannes.com.tr/posts/rag-mimarisi/"

# == Operations ==========================
def add_doc():
  db_client.add_document(path)

def query():
  elements = db_client.query("gömme modelleri")
  print("\n QUERY_ELEMENTS")
  for elem in elements:
      print(elem)

def list_elems():
  elements = db_client.list_elements()
  print("\n LIST_ELEMENTS")
  for elem in elements:
      print(" - ",elem)

def remove_elems():
  title = "tarhannes-posts-gecmisten-bugune-evrisimli-sinir-aglarinin-mimari-gelismeleri"
  db_client.remove_elements(document=title)
  elements = db_client.list_elements()
  db_client.remove_elements(element_ids=[elem.id for elem in elements])

try:
  # add_doc()
  # list_elems()
  query()
  # remove_elems()
finally:
  db_client.close()