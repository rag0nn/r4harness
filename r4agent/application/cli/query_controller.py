from __future__ import annotations

from threading import Event
from typing import Callable

from textual.worker import Worker

from r4agent.struct.metrics import PerformanceMetrics


class QueryController:
    """Own query generations and cooperative cancellation outside the view."""

    def __init__(self) -> None:
        self.generation = 0
        self.active_event: Event | None = None
        self.cancelled_generations: set[int] = set()

    def begin(self) -> tuple[int, Event]:
        """Create a generation token and cancellation event for a new query."""
        self.generation += 1
        event = Event()
        self.active_event = event
        return self.generation, event

    def run(
        self,
        handler,
        prompt: str,
        generation: int,
        cancel_event: Event,
        on_metrics: Callable[[PerformanceMetrics], None] | None = None,
    ):
        """Consume an agent stream until it completes or cancellation is requested."""
        stream = handler.r4.send(prompt)
        try:
            result = []
            for item in stream:
                if cancel_event.is_set() or generation in self.cancelled_generations:
                    return None
                if on_metrics is not None and len(item) > 2 and item[2] is not None:
                    on_metrics(item[2])
                result.append(item)
            return result
        finally:
            stream.close()

    def cancel(self, worker: Worker | None) -> bool:
        """Cancel the active worker and discard stale generation markers."""
        if self.active_event is None:
            return False
        if worker is not None:
            worker.cancel()
        self.active_event.set()
        self.cancelled_generations.add(self.generation)
        self.active_event = None
        self._discard_old_generations()
        return True

    def finish(self) -> None:
        """Release the active event after a terminal worker state."""
        self.active_event = None
        self._discard_old_generations()

    def _discard_old_generations(self) -> None:
        """Keep cancellation bookkeeping bounded to the latest generation."""
        cutoff = self.generation - 32
        self.cancelled_generations.intersection_update(
            generation for generation in self.cancelled_generations if generation > cutoff
        )
