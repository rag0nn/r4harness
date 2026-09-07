from pathlib import Path
import os
from pydantic import BaseModel

class ChunkerConfig(BaseModel):
    tokenizer:str = "word"
    context_size:int = 20
    chunk_size:int = 100
    
class VectorDatabaseConfig(BaseModel):
    path: Path | str = Path(__file__).parent / "db" 
    url: str | None = os.getenv("QDRANT_URL") or None
    api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    collection_name:str = "rag_documents"
    embed_vector_size: int = 768