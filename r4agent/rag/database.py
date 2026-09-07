from dataclasses import dataclass, field
from uuid import UUID
from typing import Any
from qdrant_client import QdrantClient, models
from numbers import Real
import logging

from .config import VectorDatabaseConfig

@dataclass(slots=True)
class DbElement:
    id: UUID
    content: str
    document: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    score: float | None = None
        
    def __repr__(self):
        scr = f"{self.score:.2f}" if self.score else "None"
        doc = self.document.replace("\n", "").replace("\t", "")
        cnt = self.content[:30].replace("\n", "").replace("\t", "")
        return f"score: {scr} document: {doc} content: {cnt}... metadata: {self.metadata}"

class QDrantDatabase:
    
    def __init__(self, config:VectorDatabaseConfig):
        self.config = config
        if config.url:
            self.database_client = QdrantClient(
                url=config.url,
                api_key=config.api_key,
            )
        else:
            database_path = self.config.path
            database_path.mkdir(parents=True, exist_ok=True)
            self.database_client = QdrantClient(path=str(database_path))
        self._ensure_collection()
        
    def _ensure_collection(self):
        """Koleksiyonu oluşturur ve embedding boyutunun config ile uyumunu doğrular."""
        # Eğer koleksiyon henüz eklenmemişse oluştur
        if not self.database_client.collection_exists(self.config.collection_name):
            self.database_client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config=models.VectorParams(
                    size=self.config.embed_vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            return

        collection = self.database_client.get_collection(self.config.collection_name)
        vectors_config = collection.config.params.vectors
        if not isinstance(vectors_config, models.VectorParams):
            raise ValueError(
                f"Collection '{self.config.collection_name}' uses named vectors; "
                "R4Rag expects a single unnamed vector."
            )
        if vectors_config.size != self.config.embed_vector_size:
            raise ValueError(
                f"Collection '{self.config.collection_name}' has vector size "
                f"{vectors_config.size}, but config expects {self.config.embed_vector_size}."
            )    
            
    def _validate_vector(self, vector: list[float] | None) -> list[float]:
        """Vektörü Qdrant'a göndermeden önce tipini, boyutunu ve değerlerini kontrol eder."""
        if vector is None:
            raise ValueError("Each DbElement must have an embedding before it is added.")
        if not isinstance(vector, list):
            raise TypeError(f"Embedding must be a list of numbers. Current type: {type(vector)}")
        if len(vector) != self.config.embed_vector_size:
            raise ValueError(
                f"Embedding size is {len(vector)}; expected {self.config.embed_vector_size}."
            )
        if any(not isinstance(value, Real) or isinstance(value, bool) for value in vector):
            raise TypeError("Embedding values must be numbers.")
        return vector
    
    def close(self) -> None:
        self.database_client.close()
        logging.info("Veri Tabanı Kapatıldı")
        
    def add_elements(self, elements: list[DbElement] | DbElement):
        """Vektör veri tabanına eleman ekler"""
        if not elements:
            return
        if isinstance(elements, DbElement):
            elements = [elements]
           
        points: list[models.PointStruct] = [] 
        for element in elements:
            points.append(
                models.PointStruct(
                    id=element.id,
                    vector=self._validate_vector(element.embedding),
                    payload={
                        "content": element.content,
                        "metadata": dict(element.metadata),
                        "document": element.document,
                    }
                )
            )
        
        self.database_client.upsert(
            collection_name=self.config.collection_name,
            points=points,
            wait=True,
        )
            
        logging.info(
            "%s dökümanlar qdrant koleksiyonlarına eklendi '%s'.",
            len(points),
            self.config.collection_name,
        )
        
    def list_elements(self) -> list[DbElement]:
        """Koleksiyon kayıtlarını vektör payload'ını taşımadan listeler."""
        elements: list[DbElement] = []
        offset: UUID | None = None
        
        while True:
            records, next_offset = self.database_client.scroll(
                collection_name=self.config.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            
            for record in records:
                payload = record.payload or {}
                metadata = payload.get("metadata") or {}
                elements.append(
                    DbElement(
                        id= record.id,
                        content=payload.get("content", ""),
                        document=payload.get("document",""),
                        metadata=dict(metadata) if isinstance(metadata, dict) else {},
                        )
                )
            
            if next_offset is None:
                break
            offset = next_offset
            
        return elements
        
    def query(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float | None = None,
        query_filter: models.Filter | None = None,
    ) -> list[DbElement]:
        """Benzer kayıtları payload ile getirir; embedding vektörlerini response'a taşımaz."""
        if not isinstance(limit, int) or isinstance(limit, bool):
            raise TypeError("limit must be an integer.")
        if limit < 1:
            raise ValueError("limit must be at least 1.")
        if score_threshold is not None and (
            not isinstance(score_threshold, Real) or isinstance(score_threshold, bool)
        ):
            raise TypeError("score_threshold must be a number or None.")

        validated_vector = self._validate_vector(query_vector)
        points = self.database_client.query_points(
            collection_name=self.config.collection_name,
            query=validated_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=(
                float(score_threshold) if score_threshold is not None else None
            ),
            with_payload=True,
            with_vectors=False,
        ).points

        elements: list[DbElement] = []
        for point in points:
            payload = point.payload or {}
            metadata = payload.get("metadata") or {}
            elements.append(
                DbElement(
                    id=point.id,
                    content=payload.get("content", ""),
                    document=payload.get("document",""),
                    metadata=dict(metadata) if isinstance(metadata, dict) else {},
                    score=point.score,
                )
            )
            
        logging.info(f"Sorgu {len(elements)} adet element döndürdü.")
        return elements
    
    def remove_elements(
        self,
        element_ids: list[UUID] | UUID | None = None,
        document: str | None = None,
    ) -> None:
        if element_ids is not None and document is not None:
            raise ValueError("Provide either element_ids or document, not both.")
        if element_ids is None and document is None:
            raise ValueError("Provide element_ids or document.")

        if element_ids is not None:
            ids = element_ids if isinstance(element_ids, list) else [element_ids]
            if not ids:
                return
            selector = models.PointIdsList(points=ids)
        else:
            selector = models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document",
                            match=models.MatchValue(value=document),
                        )
                    ]
                )
            )

        self.database_client.delete(
            collection_name=self.config.collection_name,
            points_selector=selector,
            wait=True,
        )
        logging.info("Veriler başarıyla kaldırıldı.")
