from typing import Literal
from pydantic import BaseModel, Field
from pathlib import Path

class OllamaConfig(BaseModel):
    model : str = "qwen3.5:4b"
    think : bool = False
    
type gemini_think_level = Literal["minimal", "low", "medium", "high"]
class GeminiGenConfig(BaseModel):
    model : str = "gemini-3.6-flash"
    stream : bool = True
    stop_sequences : list = Field(default_factory=list)
    seed : int = 42
    thinking_level : gemini_think_level = "minimal"
    max_output_tokens : int = 1024
    
class GeminiEmbedConfig(BaseModel):
    model : str = "gemini-embedding-2"
    vector_size :int = 768
    
class CosmosConfig(BaseModel):
    onnx_path:Path = Path(__file__).parent.parent / "weights/cosmos/model.onnx"
    tokenizer_path: Path = Path(__file__).parent.parent / "weights/cosmos/tokenizer"
    vector_size: int = 768
    
class FasterWishperConfig(BaseModel):
    model_size : str = "small" # Options: "tiny", "base", "small", "medium", "large"
    compute_type : str = "int8_float16"# 4 GB VRAM için optimum kuantizasyon
    