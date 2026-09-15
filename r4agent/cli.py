from .application.cli.tui import R4TUI

if __name__ == "__main__":
    r4tui = R4TUI(
        chosen_registery_set="assistant-semi-local",
        )
    r4tui.run()