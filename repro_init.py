"""R4Agent init akışını TUI'siz tekrarlar — başlatma hatasını yakalamak için."""
import logging
import traceback
from time import perf_counter

logging.basicConfig(level=logging.DEBUG)

from r4agent import R4Agent
from r4agent.providers import ProviderManager, UsageRegisteryLoader

print("usage yükleniyor...")
registery_sets, prompts = UsageRegisteryLoader.load()
print("registery set:", registery_sets["assistant-semi-local"].model_dump())

start = perf_counter()
try:
    agent = R4Agent(
        ProviderManager(registery_sets["assistant-semi-local"]),
        True,
    )
    print(f"R4Agent OK ({perf_counter()-start:.1f}s)")
except Exception:
    print(f"HATA ({perf_counter()-start:.1f}s sonra):")
    traceback.print_exc()