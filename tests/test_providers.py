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
        context_model="ollama-qwen3.5:4b",
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

# == Model Kayıt Defteri (Registry) ==========================
class TestModelRegistryWiring:

    def test_toolgen_models_are_tool_generation_classes(self):
        """Tool-seçim kayıtları tool üretim sınıfı olmalı.

        'phi4-mini:latest' önceden context sınıfına (OllamaGenModel) bağlıydı;
        master tool döngüsü tool listesini ikinci konumsal argüman olarak
        geçtiğinden bu değer stream parametresi sayılıp Ollama ChatRequest
        doğrulama hatasına (stream << ListToolsResult) yol açıyordu.
        """
        from r4agent.struct.models import BaseToolGenerationModel, OllamaToolGenModel

        assert ModelRegistery.toolgen_models["phi4-mini:latest"][0] is OllamaToolGenModel
        for key, (model_cls, _) in ModelRegistery.toolgen_models.items():
            assert issubclass(model_cls, BaseToolGenerationModel), (
                f"toolgen_model '{key}' tool üretim sınıfı değil: {model_cls.__name__}"
            )

    def test_context_models_are_context_generation_classes(self):
        """Context kayıtları bağlam üretim sınıfı olmalı.

        'gemma3:1b' önceden tool sınıfına (OllamaToolGenModel) bağlıydı;
        master send döngüsü context_model'e stream parametresi geçtiğinden
        TypeError: ... got an unexpected keyword argument 'stream' oluşuyordu.
        """
        from r4agent.struct.models import BaseContextGenerationModel, OllamaGenModel

        assert ModelRegistery.context_models["gemma3:1b"][0] is OllamaGenModel
        for key, (model_cls, _) in ModelRegistery.context_models.items():
            assert issubclass(model_cls, BaseContextGenerationModel), (
                f"context_model '{key}' bağlam üretim sınıfı değil: {model_cls.__name__}"
            )