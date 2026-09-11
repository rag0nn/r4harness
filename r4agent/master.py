from .struct import *
from .tools.client import MCPClient
from .providers import ProviderManager, RegisterySet
from .utils import log_execution_time

from typing import Generator
from pathlib import Path
import logging

class R4Agent:
    MAX_TOOL_ROUNDS = 5

    def __init__(self,
            provider: ProviderManager | None = None,
            stream : bool = False, ):
        self.stream = stream
        self.provider = provider or ProviderManager(RegisterySet())
        self._build()
        
    def change_provider_manager(self, provider: ProviderManager):
        self.provider = provider
        logging.info("R4Agent providerları yenileniyor...")
        self._build()
        
    @log_execution_time
    def _build(self):
        self.mcp_client = MCPClient(embed_model=self.provider.registery_set.embed_model)
        self.message_sequnce = MessageSequence(
            initial_system_prompt=self.provider.system_prompt+ f" MCP Server Instructions: {self.mcp_client.get_insturactions()}")
        self.provider.context_model
        self.provider.tool_model
        logging.info(f"System prompt: {self.provider.system_prompt}")
        logging.info("R4Agent başlatıldı.")

    def _call_tools_mcp(self, tool_calls: list) -> None:
        """Modelin seçtiği MCP araçlarını sırayla çalıştırıp sonuçlarını geçmişe ekler."""
        for item in tool_calls or []:
            if isinstance(item, tuple) and len(item) == 2:
                name, args = item
            elif isinstance(item, dict):
                fn = item.get("function") if isinstance(item.get("function"), dict) else item
                name = (fn or {}).get("name") if isinstance(fn, dict) else item.get("name")
                args = (fn or {}).get("arguments") if isinstance(fn, dict) else item.get("arguments", {})
            else:
                continue

            if not name:
                continue
            logging.info(f"Tool Çağrılıyor: {name}  args: {args}")
            result_text = self.mcp_client.call_tool(name, args or {})
            self.message_sequnce.add(Message(role=Roles.tool, content=result_text))

    def _run_tool_loop(self) -> None:
        """Araç seçimini sınırlı turda tekrarlar ve sonsuz tool döngüsünü engeller."""
        for round_number in range(self.MAX_TOOL_ROUNDS):
            tool_calls_model, tool_calls_mcp = self.provider.tool_model.send(
                self.message_sequnce,
                self.mcp_client.get_tools(),
            )

            if not tool_calls_mcp:
                return

            self.message_sequnce.add(
                Message(
                    role=Roles.assistant,
                    content="",
                    tool_calls=tool_calls_model,
                )
            )
            self._call_tools_mcp(tool_calls_mcp)

        logging.warning(
            "Tool çağrısı azami tur sayısına ulaştı: %s",
            self.MAX_TOOL_ROUNDS,
        )
            
    def send(self, query:str) ->Generator[tuple[str, str], None,None]:
        """Sorguyu geçmişe ekler, tool döngüsünü çalıştırır ve model çıktısını stream eder."""
        self.message_sequnce.add(Message(role=Roles.user, content=query))
        self._run_tool_loop()
        
        full_content, full_thinking = "", ""
        for cnt, tnk in self.provider.context_model.send(
                self.message_sequnce,
                stream=self.stream):
            full_content += cnt
            full_thinking += tnk
            yield cnt, tnk

        self.message_sequnce.add(Message(role=Roles.assistant, content=full_content))

    def load_messages(self, path: str | Path):
        """
        Mesaj geçmişi yükler. (json)
        """
        self.message_sequnce.load(path=path)

    def save_messages(self, path: str | Path)->Path:
        """Mesaj geçmişini kaydeder. (json), kaydedilmiş yolu döndürür."""
        return self.message_sequnce.save(path=path)

    def reset_messages(self):
        """Mesaj geçmişini resetler. """
        self.message_sequnce.reset(self.provider.system_prompt)
