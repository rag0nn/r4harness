import types

import pytest

from r4agent import Message, R4Agent
from r4agent.struct import Roles, MessageSequence
from r4agent.struct.metrics import PerformanceMetrics
from r4agent.backend.structs import QueryStreamChunk


# == PerformanceMetrics ==========================
class TestPerformanceMetrics:

    def test_total_tokens(self):
        m = PerformanceMetrics(prompt_tokens=100, completion_tokens=25)
        assert m.total_tokens == 125

    def test_context_usage_pct(self):
        m = PerformanceMetrics(total_context_tokens=2048, max_context_window=8192)
        assert m.context_usage_pct == 25.0

    def test_context_usage_pct_zero_window(self):
        m = PerformanceMetrics(total_context_tokens=10, max_context_window=0)
        assert m.context_usage_pct == 0.0

    def test_add_cumulative(self):
        a = PerformanceMetrics(
            prompt_tokens=10, completion_tokens=5, ttft_ms=100.0,
            output_tps=10.0, max_context_window=4096,
        )
        b = PerformanceMetrics(
            prompt_tokens=10, completion_tokens=15, ttft_ms=50.0,
            output_tps=30.0, max_context_window=8192,
        )
        c = a + b
        assert c.prompt_tokens == 20
        assert c.completion_tokens == 20
        assert c.total_tokens == 40
        assert c.ttft_ms == 150.0
        assert c.total_context_tokens == 0
        assert c.output_tps == 20.0
        assert c.max_context_window == 8192

    def test_add_invalid_type(self):
        m = PerformanceMetrics()
        with pytest.raises(TypeError):
            m + 5

    def test_to_dict(self):
        d = PerformanceMetrics(prompt_tokens=3, completion_tokens=4).to_dict()
        assert d["prompt_tokens"] == 3
        assert d["completion_tokens"] == 4


# == Message ==========================
class TestMessageMetrics:

    def test_to_dict_excludes_metrics(self):
        msg = Message(
            role=Roles.assistant,
            content="x",
            metrics=PerformanceMetrics(completion_tokens=7),
        )
        assert msg.to_dict() == {"role": "assistant", "content": "x"}

    def test_default_none(self):
        msg = Message(role=Roles.user, content="q")
        assert msg.metrics is None


# == MessageSequence ==========================
class TestMessageSequenceContext:

    def test_live_context_tracking(self):
        seq = MessageSequence("sys")
        assert seq.total_context_tokens == 0
        seq.add(
            Message(
                role=Roles.assistant,
                content="a",
                metrics=PerformanceMetrics(total_context_tokens=1500, max_context_window=8192),
            )
        )
        assert seq.total_context_tokens == 1500
        assert seq.context_usage_pct == round((1500 / 8192) * 100, 2)

    def test_reset_clears_metrics(self):
        seq = MessageSequence("sys")
        seq.add(
            Message(
                role=Roles.assistant,
                content="a",
                metrics=PerformanceMetrics(total_context_tokens=500),
            )
        )
        seq.reset()
        assert seq.total_context_tokens == 0


# == Pydantic / SSE ==========================
class TestQueryStreamChunkMetrics:

    def test_round_trip(self):
        chunk = QueryStreamChunk(
            content="selam",
            thinking="",
            metrics=PerformanceMetrics(
                ttft_ms=120.5,
                completion_tokens=42,
                total_context_tokens=1500,
                max_context_window=8192,
            ),
        )
        payload = chunk.model_dump_json()
        parsed = QueryStreamChunk.model_validate_json(payload)
        assert parsed.metrics is not None
        assert parsed.metrics.ttft_ms == 120.5
        assert parsed.metrics.completion_tokens == 42
        assert parsed.metrics.total_context_tokens == 1500

    def test_none_metrics(self):
        chunk = QueryStreamChunk(content="selam", thinking="")
        assert chunk.metrics is None


# == R4Agent stream zamanlama ==========================
class FakeGenModel:
    """Sıralı chunk ve metrik üreten sahte context modeli."""

    def __init__(self, chunks, metrics_list):
        self.chunks = list(zip(chunks, metrics_list))

    def send(self, message_sequence, stream=False):
        for content, metrics in self.chunks:
            yield content, "", metrics


class FakeToolModel:

    def send(self, message_sequence, tools=None):
        return [], []


class FakeProvider:

    def __init__(self, gen_model):
        self.registery_set = types.SimpleNamespace(
            embed_model="fake-embed",
            system_prompt="Sen bir test asistanısın.",
        )
        self.system_prompt = "Sen bir test asistanısın."
        self.context_model = gen_model
        self.tool_model = FakeToolModel()
        self.recorded: list[PerformanceMetrics] = []

    def record_telemetry(self, metrics: PerformanceMetrics) -> None:
        self.recorded.append(metrics)


class FakeMCPClient:

    def __init__(self, embed_model):
        pass

    def get_insturactions(self):
        return "test talimatları"

    def get_tools(self):
        return []

    def call_tool(self, name, params=None):
        return ""


@pytest.fixture
def stream_agent(monkeypatch):
    monkeypatch.setattr("r4agent.master.MCPClient", FakeMCPClient)
    chunks = ["Merh", "aba", "!"]
    metrics_list = [
        PerformanceMetrics(prompt_tokens=10, completion_tokens=2, total_context_tokens=10),
        PerformanceMetrics(prompt_tokens=10, completion_tokens=5, total_context_tokens=10),
        PerformanceMetrics(prompt_tokens=10, completion_tokens=8, total_context_tokens=10),
    ]
    provider = FakeProvider(FakeGenModel(chunks, metrics_list))
    r4 = R4Agent(provider=provider, stream=True)
    return r4, provider


class TestR4AgentStreamMetrics:

    def test_yields_metrics_with_every_chunk(self, stream_agent):
        r4, _provider = stream_agent
        collected = list(r4.send("test"))
        assert len(collected) == 3
        assert all(len(item) == 3 for item in collected)
        assert "".join(item[0] for item in collected) == "Merhaba!"

    def test_ttft_and_live_tps(self, stream_agent):
        r4, _provider = stream_agent
        first, second, third = (item[2] for item in r4.send("test"))
        assert first.ttft_ms > 0
        assert first.ttft_ms == third.ttft_ms
        assert second.output_tps > 0
        assert third.output_tps > 0

    def test_final_metrics_attached_to_message(self, stream_agent):
        r4, _provider = stream_agent
        list(r4.send("test"))
        last = r4.message_sequnce.sequence[-1]
        assert last.role == Roles.assistant
        assert last.metrics is not None
        assert last.metrics.completion_tokens == 8
        assert last.metrics.ttft_ms > 0
        assert last.metrics.tool_duration_ms >= 0

    def test_telemetry_recorded(self, stream_agent):
        r4, provider = stream_agent
        list(r4.send("test"))
        assert len(provider.recorded) == 1
        assert provider.recorded[0].completion_tokens == 8


class TestR4AgentNonStreamMetrics:

    def test_keeps_adapter_tps(self, monkeypatch):
        monkeypatch.setattr("r4agent.master.MCPClient", FakeMCPClient)
        provider = FakeProvider(
            FakeGenModel(
                ["tam yanıt"],
                [PerformanceMetrics(
                    prompt_tokens=10,
                    completion_tokens=4,
                    output_tps=12.5,
                    total_context_tokens=10,
                )],
            )
        )
        r4 = R4Agent(provider=provider, stream=False)
        collected = list(r4.send("test"))
        assert len(collected) == 1
        assert collected[0][2].output_tps == 12.5
        assert collected[0][2].ttft_ms > 0