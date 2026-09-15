from typing import Literal
from pydantic import BaseModel, Field
from pathlib import Path

# == Abstraction ==========================
class BaseConfig(BaseModel):
    model : str

# == Types ==========================
type gemini_think_level = Literal["minimal", "low", "medium", "high"]

# == Childs ==========================

class OllamaConfig(BaseConfig):
    model : str = "qwen3.5:4b"
    think : bool = False
    context_window : int = 8192

class GeminiGenConfig(BaseConfig):
    model : str = "gemini-3.6-flash"
    stream : bool = True
    stop_sequences : list = Field(default_factory=list)
    seed : int = 42
    thinking_level : gemini_think_level = "minimal"
    max_output_tokens : int = 1024
    context_window : int = 1048576
    
class GeminiEmbedConfig(BaseConfig):
    model : str = "gemini-embedding-2"
    vector_size :int = 768
    
class CosmosConfig(BaseConfig):
    model : str = "cosmos"
    path:Path = Path(__file__).parent.parent / "weights/cosmos-modernbert-embed"
    vector_size: int = 768
    max_seq_length: int = 8192 # ModernBERT max_position_embeddings (token)
    
class FasterWishperConfig(BaseConfig):
    model : str = "faster-whisper"
    model_size : str = "small" # Options: "tiny", "base", "small", "medium", "large"
    compute_type : str = "int8_float16"# 4 GB VRAM için optimum kuantizasyon
    