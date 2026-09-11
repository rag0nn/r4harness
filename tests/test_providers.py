from r4agent.providers import (
    UsageRegisteryLoader, 
    ModelRegistery, 
    RegisterySet,
    ProviderManager,
    )

import pytest

# == UsageRegisteryLoader ==========================
class TestUsageRegisteryLoader:
    
    def test_load(self):
        registery_sets, prompts = UsageRegisteryLoader.load()
        print(type(prompts))
        assert isinstance(registery_sets, dict)
        assert all(isinstance(k, str) and isinstance(v, RegisterySet) for k,v in registery_sets.items())
        assert isinstance(prompts, dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k,v in prompts.items())
    
# == ProviderManager ==========================
@pytest.fixture
def manager():
    rset = RegisterySet(
        context_model="ollama",
        embed_model="cosmos",
        system_prompt="İlk test promptu",
        whisper_model="faster-whisper",
        toolgen_model="ollama"
    )
    pm = ProviderManager(rset)
    yield pm  # Test metodu çalıştığı an bu nesne verilir
    pm.close() # Test bittiğinde cleanup/close otomatik çalışır


class TestProviderManager:

    def test_init_models(self, manager: ProviderManager):
        manager.context_model
        manager.embed_model
        manager.system_prompt
        manager.whisper
        manager.dbclient      
          
    def test_change_models(self, manager: ProviderManager):
        manager.change_context_model("gemini")
        assert manager.registery_set.context_model == "gemini"
        manager.change_embed_model("gemini")
        assert manager.registery_set.embed_model == "gemini"
        manager.change_system_prompt("Yeni test sistem promptu")
        assert manager.registery_set.system_prompt == "Yeni test sistem promptu"
        manager.change_tool_model("ollama")
        assert manager.registery_set.toolgen_model == "ollama"
        manager.change_whisper_model("faster-whisper")
        assert manager.registery_set.whisper_model == "faster-whisper"
    
    def test_close(self, manager: ProviderManager):
        manager.close()
        
    def test_reset(self, manager: ProviderManager):        
        manager.reset()