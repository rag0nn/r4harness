from chonkie import RecursiveChunker, OverlapRefinery
import logging
from datetime import datetime
import uuid
from typing import overload
from pathlib import Path
from urllib.parse import unquote, urlparse
from docling.document_converter import DocumentConverter

from .config import ChunkerConfig
from .database import QDrantDatabase, DbElement
from ..struct.models import BaseEmbeddingGenerationModel

# == Chunkers ==========================
class Chunker:
    """
    A simple chunking strategy that splits text into chunks of a specified size.
    """
    
    def __init__(self, config: ChunkerConfig):
        self.config = config
        self.chunker = RecursiveChunker(
                tokenizer=config.tokenizer,
                chunk_size=self.config.chunk_size,
            )
        self.overlap = OverlapRefinery(
            context_size=config.context_size,
            merge = True
        )
    
    def chunk(self, text: str) -> list[str]:
        """
        Splits the input text into chunks of the specified size.

        Args:
            text (str): The input text to be chunked.

        Returns:
            list[str]: A list of text chunks.
        """
        chunkes = self.chunker.chunk(text)
        overlapped_chunks = self.overlap.refine(chunkes)
        return [chunk.text for chunk in overlapped_chunks]
    
# == Parse & Writing ==========================
class DocumentHandler:

    @staticmethod
    def _url_title(source: str) -> str:
        parsed = urlparse(source)
        host = parsed.hostname or "document"
        domain = host.removeprefix("www.").split(".", 1)[0]
        path_parts = [unquote(part) for part in parsed.path.split("/") if part]
        return "-".join([domain, *path_parts])
    
    @staticmethod
    def parse_document(source: Path | str) -> tuple[str, str]:
        """Belgeyi Markdown'a dönüştürür ve kaynak için kararlı bir başlık üretir."""
        parsed_source = urlparse(source) if isinstance(source, str) else None
        is_url = parsed_source is not None and parsed_source.scheme in {"http", "https"} and bool(parsed_source.netloc)
        document_source = source if is_url else Path(source)
        converter = DocumentConverter()
        result = converter.convert(str(document_source))
        content = result.document.export_to_markdown()
        title = DocumentHandler._url_title(source) if is_url else document_source.stem
        logging.info(f"Title: {title} Döküman okundu: {str(document_source)[:15]}...")
        return content, title
    
    @staticmethod
    def write_document(content:str, path:Path)->None:
        """Farklı uzantılarda kaydetmeye yarar"""
        text_extensions = {".txt", ".md", ".markdown", ".html", ".htm"}
        suffix = path.suffix.lower()

        if suffix not in text_extensions:
            raise ValueError(f"Unsupported document extension: {path.suffix}")

        path.write_text(content, encoding="utf-8")
        logging.info(f"Dosya yazıldı: {path}")

# == Client ==========================
class DbClient:
    
    def __init__(self, database: QDrantDatabase, chunker: Chunker, embed_model: BaseEmbeddingGenerationModel):
        self.database = database
        self.chunker = chunker
        self.embed_model = embed_model
        
    def _generate_embedding(self, text:str)->list[float]:
        """Seçili embedding provider üzerinden tek metin vektörü üretir."""
        return self.embed_model.embed(text)
        
    def query(self, content: str)->list[DbElement]:
        """Sorguyu embed eder ve en yakın RAG kayıtlarını veritabanından getirir."""
        embeddings: list[float] = self._generate_embedding(content)
        elements = self.database.query(embeddings)
        info = "\n".join([f"{elem.document} {elem.content[:25]}..."for elem in elements])
        logging.info(f"\n\n RETRIEVED ELEMENTS\n{info}")
        return elements
    
    def add_document(self, path:Path):
        """Belgeyi parse eder, chunk'lar ve embedding'leri tek upsert ile kaydeder."""
        content, document_title = DocumentHandler.parse_document(path)
        chunks = self.chunker.chunk(content)
        elements: list[DbElement] = []
        for chunk in chunks:
            _id = uuid.uuid4()
            embedding = self._generate_embedding(chunk)
            elements.append(
                DbElement(
                    id=_id,
                    content=chunk,
                    document=document_title,
                    metadata={"date" : datetime.now().strftime("%d-%m-%Y_%H-%M-%S")},
                    embedding=embedding,
                )
            )
        
        self.database.add_elements(elements)
        
    def add_content(self, content:str, title:str):
        """Metni chunk'layıp embedding'lerini üreterek başlık altında kaydeder."""
        chunks = self.chunker.chunk(content)
        elements: list[DbElement] = []
        for chunk in chunks:
            _id = uuid.uuid4()
            embedding = self._generate_embedding(chunk)
            elements.append(
                DbElement(
                    id=_id,
                    content=chunk,
                    document=title,
                    metadata={"date" : datetime.now().strftime("%d-%m-%Y_%H-%M-%S")},
                    embedding=embedding,
                )
            )
        
        self.database.add_elements(elements)
        
    @overload
    def remove_elements(self, element_ids:list[int]):...
    
    @overload
    def remove_elements(self, document:str):...

    def remove_elements(self, element_ids:list[int] = None, document:str = None):
        return self.database.remove_elements(
            element_ids=element_ids,
            document=document,
        )
        
    def list_elements(self)->list[DbElement]:
        return self.database.list_elements()

    def close(self) -> None:
        self.database.close()