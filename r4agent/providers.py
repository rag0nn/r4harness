from typing import TYPE_CHECKING
import atexit
import sys
import threading


from .struct.models import (
    BaseContextGenerationModel,
    BaseToolGenerationModel,
    BaseEmbeddingGenerationModel,
    GeminiGenModel,
    GeminiGenConfig,
    OllamaGenModel,
    OllamaToolGenModel,
    OllamaConfig,
    CosmosEmbedding,
    CosmosConfig,
    FasterWhisper,
    FasterWishperConfig,
    Registery
)
from .rag import DbClient, QDrantDatabase, VectorDatabaseConfig, Chunker, ChunkerConfig
from .utils import read_env
from pathlib import Path

read_env(Path(__file__).parent / ".env")

registery = Registery()

# ---------------------------------------------------------------------------
# lazy-singleton cache
# ---------------------------------------------------------------------------
_lock = threading.Lock()

SYSTEM_PROMPT: str = "Sen bir yapay zeka asistanısın. İstenen çıktıları en net ve gerekli şekliyle ver, yorum katma."
_context_model: BaseContextGenerationModel | None = None
_tool_model: BaseToolGenerationModel | None = None
_embed_model: BaseEmbeddingGenerationModel | None = None
_dbclient:  DbClient | None = None
_whisper: FasterWhisper | None = None


def configure(
    context_model: str | None = None,
    embed_model: str | None = None,
    toolgen_model: str | None = None,
) -> None:
    """Provider seçimlerini günceller ve lazy instance'ları yeniden kurulum için temizler."""
    if context_model is not None:
        registery.set_context_model(context_model)
    if embed_model is not None:
        registery.set_embed_model(embed_model)
    if toolgen_model is not None:
        registery.set_toolgen_model(toolgen_model)
    rebuild()


def rebuild() -> None:
    """Aktif provider kaynaklarını kapatıp sonraki erişimi yeni seçime yönlendirir."""
    global _context_model, _tool_model, _embed_model, _dbclient, _whisper

    with _lock:
        if _dbclient is not None:
            _dbclient.close()
        _context_model = None
        _tool_model = None
        _embed_model = None
        _dbclient = None
        _whisper = None

if TYPE_CHECKING:
    # Bu blok sadece type checker ve IDE'ler tarafından okunur.
    # Çalışma zamanında (runtime) import edilmezler!
    CONTEXT_GEN_MODEL: BaseContextGenerationModel
    TOOL_GEN_MODEL: BaseToolGenerationModel
    EMBED_GEN_MODEL: BaseEmbeddingGenerationModel
    DBCLIENT: DbClient
    WHISPER: FasterWhisper
    
def __getattr__(name: str):
    """İstenen provider'ı kilit altında lazy olarak oluşturup cache'ler."""
    global _context_model, _tool_model, _embed_model, _dbclient, _whisper

    if name == "CONTEXT_GEN_MODEL":
        if _context_model is None:
            with _lock:
                if _context_model is None:
                    _context_model = registery.get_context_model(registery.context_key)
        return _context_model

    if name == "TOOL_GEN_MODEL":
        if _tool_model is None:
            with _lock:
                if _tool_model is None:
                    _tool_model = registery.get_toolgen_model(registery.toolgen_key)
        return _tool_model

    if name == "EMBED_GEN_MODEL":
        if _embed_model is None:
            with _lock:
                if _embed_model is None:
                    _embed_model = registery.get_embed_model(registery.embed_key)
        return _embed_model

    if name == "DBCLIENT":
        if _dbclient is None:
            embed_model = sys.modules[__name__].EMBED_GEN_MODEL
            with _lock:
                if _dbclient is None:
                    database = QDrantDatabase(VectorDatabaseConfig())
                    chunker = Chunker(ChunkerConfig())
                    _dbclient = DbClient(database, chunker, embed_model)
        return _dbclient
    
    if name == "WHISPER":
        if _whisper is None:
            with _lock:
                if _whisper is None:
                    _whisper = FasterWhisper(FasterWishperConfig())
        return _whisper

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
def _close_providers() -> None:
    """Process kapanırken açık veritabanı kaynağını kapatır."""
    if _dbclient is not None:
        _dbclient.close()


atexit.register(_close_providers)


