from typing import TYPE_CHECKING
from .rag.client import DbClient, DocumentHandler
from .rag.config import *
from .struct.model_configs import *
from .struct import MessageSequence, Message

if TYPE_CHECKING:
    from .master import R4Agent

def __getattr__(name):
	if name == "R4Agent":
		from .master import R4Agent
		return R4Agent
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
	"DbClient",
	"DocumentHandler",
	"R4Agent",
]