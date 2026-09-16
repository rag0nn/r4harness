"""TUI komut ağacı regresyon testleri.

`/change system_prompt` komutu daha önce `args[0]`'a argümansız eriştiği için
IndexError fırlatıyordu; bu testler o durumu ve kayıtlı set adı önerisini kapsar.
"""

import pytest

from r4agent.application.cli.tui import R4TUI


@pytest.fixture(scope="module")
def app() -> R4TUI:
    return R4TUI()


class TestChangeSystemPromptCommand:

    def test_change_system_prompt_without_args_does_not_raise(self, app: R4TUI) -> None:
        assert app.command_runner.execute("/change system_prompt") is True

    def test_change_system_prompt_resolves_to_handler(self, app: R4TUI) -> None:
        node, remaining = app.command_runner.resolve("/change system_prompt")
        assert node is not None
        assert remaining == []
        assert "handler" in node

    def test_change_system_prompt_suggests_registery_sets(self, app: R4TUI) -> None:
        node, remaining = app.command_runner.resolve("/change system_prompt a")
        assert remaining == ["a"]
        suggestions = node["arg_provider"](app, remaining)
        assert suggestions
        assert all(
            suggestion.startswith("/change system_prompt ") for suggestion in suggestions
        )
        # her öneri kayıtlı bir set adına karşılık gelir
        assert all(
            suggestion.removeprefix("/change system_prompt ")
            in app.usage_registery_sets
            for suggestion in suggestions
        )

    def test_change_provider_rejects_single_part(self, app: R4TUI) -> None:
        # Tek parça (ör. sadece "/change system_prompt" icra edilir) sessizce
        # hiçbir iş yapmadan döner; handler kurulumundan önce güvenlidir.
        assert app.change_provider("system_prompt") is None