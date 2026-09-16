from .struct import *
from .tools.client import MCPClient, ToolApprovalManager
from .providers import ProviderManager, RegisterySet
from .utils import log_execution_time

from typing import Generator, Optional
from pathlib import Path
from time import perf_counter
import logging

class R4Agent:
    MAX_TOOL_ROUNDS = 5

    def __init__(self,
            approval: ToolApprovalManager,
            provider: ProviderManager | None = None,
            stream : bool = False, ):
        self.stream = stream
        self.provider = provider or ProviderManager(RegisterySet())
        self.approval = approval
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

            if not self.approval.authorize_execution(name, args or {}):
                logging.info(f"Kullanıcı '{name}' aracını reddetti.")
                self.message_sequnce.add(Message(
                    role=Roles.tool,
                    content=f"Kullanıcı '{name}' aracını çalıştırmayı reddetti; çağrı yapılmadı.",
                    metrics=PerformanceMetrics(tool_duration_ms=0.0),
                ))
                continue

            logging.info(f"Tool çağrılıyor => {name}({args})")
            tool_start = perf_counter()
            success, result_text = self.mcp_client.call_tool(name, args or {})
            call_duration_ms = (perf_counter() - tool_start) * 1000
            self.message_sequnce.add(Message(
                role=Roles.tool,
                content=result_text,
                metrics=PerformanceMetrics(tool_duration_ms=round(call_duration_ms, 2)),
            ))

    def _run_tool_loop(self) -> float:
        """Araç seçimini sınırlı turda tekrarlar; tüm döngünün süresini ms olarak döndürür."""
        tool_start = perf_counter()
        for round_number in range(self.MAX_TOOL_ROUNDS):
            tool_calls_model, tool_calls_mcp = self.provider.tool_model.send(
                self.message_sequnce,
                self.mcp_client.get_tools(),
            )

            if not tool_calls_mcp:
                return (perf_counter() - tool_start) * 1000

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

        return (perf_counter() - tool_start) * 1000

    def send(self, query:str) ->Generator[tuple[str, str, Optional[PerformanceMetrics]], None, None]:
        """Sorguyu geçmişe ekler, tool döngüsünü çalıştırır ve model çıktısını metriklerle stream eder.

        Her parça (content, thinking, metrics) üçlüsü olarak yield edilir; metrics
        duvar saati TTFT'sini ve canlı output_tps/token sayılarını taşır.
        """
        user_message = Message(role=Roles.user, content=query)
        self.message_sequnce.add(user_message)
        send_start = perf_counter()
        tool_duration_ms = self._run_tool_loop()

        full_content, full_thinking = "", ""
        first_chunk_at: float | None = None
        first_chunk_ttft_ms = 0.0
        final_metrics: PerformanceMetrics | None = None

        for cnt, tnk, chunk_metrics in self.provider.context_model.send(
                self.message_sequnce,
                stream=self.stream):
            now = perf_counter()
            if first_chunk_at is None:
                first_chunk_at = now
                first_chunk_ttft_ms = (now - send_start) * 1000
            elif chunk_metrics is not None and chunk_metrics.completion_tokens:
                chunk_metrics.output_tps = round(
                    chunk_metrics.completion_tokens / (now - first_chunk_at), 2)

            full_content += cnt
            full_thinking += tnk

            if chunk_metrics is not None:
                chunk_metrics.ttft_ms = first_chunk_ttft_ms
            final_metrics = chunk_metrics
            yield cnt, tnk, chunk_metrics

        if final_metrics is None:
            final_metrics = PerformanceMetrics()
        final_metrics.ttft_ms = first_chunk_ttft_ms
        final_metrics.tool_duration_ms = tool_duration_ms
        if self.stream and first_chunk_at is not None and final_metrics.completion_tokens:
            elapsed = perf_counter() - first_chunk_at
            if elapsed > 0:
                final_metrics.output_tps = round(final_metrics.completion_tokens / elapsed, 2)

        total_duration_ms = round((perf_counter() - send_start) * 1000, 2)
        final_metrics.total_duration_ms = total_duration_ms
        user_message.metrics = PerformanceMetrics(total_duration_ms=total_duration_ms)

        self.message_sequnce.add(Message(
            role=Roles.assistant,
            content=full_content,
            metrics=final_metrics,
        ))
        self.provider.record_telemetry(final_metrics)

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
