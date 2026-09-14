from threading import Event

from r4agent.application.cli.query_controller import QueryController
from r4agent.struct.metrics import PerformanceMetrics


class _FakeR4:
    """R4Agent.send' in stream imzasını (content, thinking, metrics) taklit eder."""

    def __init__(self, chunks):
        self._chunks = chunks

    def send(self, prompt):
        yield from self._chunks


class _FakeHandler:
    def __init__(self, chunks):
        self.r4 = _FakeR4(chunks)


class TestQueryController:

    def test_run_forwards_stream_content_chunks(self):
        chunks = [
            ("Mer", "", None),
            ("haba", "", None),
            (" dünya", "", None),
        ]
        controller = QueryController()
        generation, cancel_event = controller.begin()

        received: list[str] = []
        controller.run(
            _FakeHandler(chunks),
            "selam",
            generation,
            cancel_event,
            on_content=received.append,
        )

        assert "".join(received) == "Merhaba dünya"

    def test_run_forwards_metrics(self):
        chunks = [("cevap", "", PerformanceMetrics(completion_tokens=7))]
        controller = QueryController()
        generation, cancel_event = controller.begin()

        received: list[PerformanceMetrics] = []
        controller.run(
            _FakeHandler(chunks),
            "selam",
            generation,
            cancel_event,
            on_metrics=received.append,
        )

        assert len(received) == 1
        assert received[0].completion_tokens == 7

    def test_run_returns_none_when_cancelled(self):
        controller = QueryController()
        generation, cancel_event = controller.begin()
        cancel_event.set()

        result = controller.run(
            _FakeHandler([("Mer", "", None)]),
            "selam",
            generation,
            cancel_event,
        )

        assert result is None