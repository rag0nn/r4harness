import threading
import logging
import json

from .struct.models import (
    BaseContextGenerationModel,
    BaseToolGenerationModel,
    BaseEmbeddingGenerationModel,
    GeminiGenModel,
    GeminiGenConfig,
    GeminiEmbedding,
    GeminiEmbedConfig,
    OllamaGenModel,
    OllamaToolGenModel,
    OllamaEmbeddingGenModel,
    OllamaConfig,
    CosmosEmbedding,
    CosmosConfig,
    FasterWhisper,
    FasterWishperConfig,
)
from .rag import RAGClient, QDrantDatabase, VectorDatabaseConfig, Chunker, ChunkerConfig
from .utils import read_env
from pathlib import Path
from pydantic import BaseModel


read_env(Path(__file__).parent / ".env")

class ModelRegistery:
    context_models: dict[str, tuple[type[BaseContextGenerationModel], type[BaseModel]]] = {
        "ollama": (OllamaGenModel, OllamaConfig),
        "gemini": (GeminiGenModel, GeminiGenConfig),
    }
    toolgen_models: dict[str, tuple[type[BaseToolGenerationModel], type[BaseModel]]] = {
        "ollama": (OllamaToolGenModel, OllamaConfig),
    }
    embed_models: dict[str, tuple[type[BaseEmbeddingGenerationModel], type[BaseModel]]] = {
        "ollama": (OllamaEmbeddingGenModel, OllamaConfig),
        "gemini": (GeminiEmbedding, GeminiEmbedConfig),
        "cosmos": (CosmosEmbedding, CosmosConfig),
    }
    whisper_models: dict[str, tuple[type[FasterWhisper], type[FasterWishperConfig]]] = {
        "faster-whisper": (FasterWhisper, FasterWishperConfig),
    }

    @classmethod
    def _build(cls: "ModelRegistery", key: str, cat:str):
        models: dict = getattr(cls, cat)
        try:
            mdl, cfg = models[key]
            return mdl(cfg())
        except KeyError:
            raise KeyError(
                f"Yanlış model key: {key}. {cat} için bunlardan biri olmalı: {tuple(models.keys())}"
            ) from None

    @staticmethod
    def build_context_model(key: str) -> BaseContextGenerationModel:
        model = ModelRegistery._build(key, "context_models")
        logging.info(f"Context Model: {model.__class__.__name__}")
        return model

    @staticmethod
    def build_toolgen_model(key: str) -> BaseToolGenerationModel:
        model = ModelRegistery._build(key, "toolgen_models")
        logging.info(f"Tool Selection Model: {model.__class__.__name__}")
        return model 

    @staticmethod
    def build_embed_model(key: str) -> BaseEmbeddingGenerationModel:
        model = ModelRegistery._build(key, "embed_models")
        logging.info(f"Embedding Model: {model.__class__.__name__}")
        return model
    
    @staticmethod
    def build_whisper_model( key: str) -> FasterWhisper:
        model = ModelRegistery._build(key, "whisper_models")
        logging.info(f"Whisper Model: {model.__class__.__name__}")
        return model
    
    @staticmethod
    def build_rag_client(embedding_model: BaseEmbeddingGenerationModel) -> RAGClient:
        database = QDrantDatabase(VectorDatabaseConfig())
        logging.info(f"Database: {database.__class__.__name__}")
        chunker = Chunker(ChunkerConfig())
        logging.info(f"Chunker: {chunker.__class__.__name__}")
        client = RAGClient(
            database=database,
            chunker=chunker,
            embed_model=embedding_model
        )
        return client

class RegisterySet(BaseModel):
    context_model : str = "ollama"
    toolgen_model : str = "ollama"
    embed_model : str =  "cosmos"
    whisper_model : str = "faster-whisper"
    system_prompt : str = "Sen bir yapay zeka asistanısın. İstenen çıktıları en net ve gerekli şekliyle ver, yorum katma."

class UsageRegisteryLoader:
        
    USAGE_PATH = Path(__file__).parent / "usage.json"
    
    @classmethod
    def load(cls):
        if not cls.USAGE_PATH.exists():
            raise FileNotFoundError(f"File is missing: {cls.USAGE_PATH}")
        with open(cls.USAGE_PATH, "r", encoding="utf-8") as f:
            content = json.load(f)
        
        prompts_dict:dict = content["prompts"]
        registery_sets: dict[str, RegisterySet] = {}
        for k, v in  content["registery-sets"].items():
            registery_sets.update({k :  RegisterySet.model_validate(v)})
            
        return registery_sets, prompts_dict

class ProviderManager:
    
    def __init__(self, registery_set: RegisterySet | None):
        self._lock = threading.RLock()
        self.registery_set = registery_set or RegisterySet()
        
        self._system_prompt: str = ""
        self._context_model: BaseContextGenerationModel = None
        self._tool_model: BaseToolGenerationModel = None
        self._embed_model: BaseEmbeddingGenerationModel = None
        self._whisper_model: FasterWhisper = None
        
        self._dbclient: RAGClient = None

    @property
    def system_prompt(self):
        self._system_prompt = self.registery_set.system_prompt
        return self._system_prompt
        
    @property
    def context_model(self):
        if self._context_model is None:
            with self._lock:
                if self._context_model is None:
                    self._context_model = ModelRegistery.build_context_model(self.registery_set.context_model)
        return self._context_model

    @property
    def tool_model(self):
        if self._tool_model is None:
            with self._lock:
                if self._tool_model is None:
                    self._tool_model = ModelRegistery.build_toolgen_model(self.registery_set.toolgen_model)
        return self._tool_model
    
    @property
    def embed_model(self):
        if self._embed_model is None:
            with self._lock:
                if self._embed_model is None:
                    self._embed_model = ModelRegistery.build_embed_model(self.registery_set.embed_model)
        return self._embed_model
    
    @property
    def whisper(self):
        if self._whisper_model is None:
            with self._lock:
                if self._whisper_model is None:
                    self._whisper_model = ModelRegistery.build_whisper_model(self.registery_set.whisper_model)
        return self._whisper_model
    
    @property
    def dbclient(self):
        if self._dbclient is None:
            with self._lock:
                if self._dbclient is None:
                    self._dbclient = ModelRegistery.build_rag_client(self.embed_model)
        return self._dbclient
    
    def close(self):
        if self._dbclient is not None:
            with self._lock:   
                if self._dbclient is not None:
                    self._dbclient.close()
        
    def reset(self):
        with self._lock:
            self.close()
            self._system_prompt = ""
            self._context_model = None
            self._tool_model = None
            self._embed_model = None
            self._dbclient = None
            self._whisper_model = None
        
    def change_system_prompt(self, text:str):
        with self._lock:
            self.registery_set.system_prompt = text
            logging.info(f"System prompt değiştirildi: {text}")
    
    def change_context_model(self, model_code:str):
        with self._lock:
            self.registery_set.context_model = model_code
            self._context_model = ModelRegistery.build_context_model(model_code)
            logging.info(f"Context model değiştirildi: {model_code}")
            
    
    def change_tool_model(self, model_code:str):
        with self._lock:
            self.registery_set.toolgen_model = model_code
            self._tool_model = ModelRegistery.build_toolgen_model(model_code)
            logging.info(f"Tool model değiştirildi: {model_code}")
            
        
    def change_embed_model(self, model_code:str):
        with self._lock:
            self.registery_set.embed_model = model_code
            self._embed_model = ModelRegistery.build_embed_model(model_code)
            logging.info(f"Embed model değiştirildi: {model_code}")
            if self._dbclient is not None:
                self._dbclient.close()
                self._dbclient = None
            
    def change_whisper_model(self, model_code:str):
        with self._lock:
            self.registery_set.whisper_model = model_code
            self._whisper_model = ModelRegistery.build_whisper_model(model_code)
            logging.info(f"Whisper model değiştirildi: {model_code}")
    
