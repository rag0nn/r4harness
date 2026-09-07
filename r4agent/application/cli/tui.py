from textual.containers import Container, VerticalScroll
from textual import on
from textual.widgets import Button, Header, Label, LoadingIndicator, Static, TextArea
from textual.widgets import OptionList
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState
from textual.app import App,ComposeResult
from textual.binding import Binding
from pathlib import Path
from typing import Callable

from .handler import Handler
from .command_parser import CommandParser
from .log_controller import LogController
from .message_renderer import render_messages
from .query_controller import QueryController
from .session_controller import SessionController
from .widgets import ModelBanner, PromptTextArea
from r4agent import R4Agent, Message
from r4agent.struct.base import Roles
import r4agent.providers as provs

class R4TUI(App):
    CSS_PATH = "styles.tcss"
    TITLE= "R4Agent"
    BINDINGS = [
        Binding("ctrl+b", "cancel_query", show=True, priority=True),
    ]
    
    def __init__(self):
        self.handler: Handler | None = None
        self._banner_offset = 0
        self._query_waiting = False
        self._query_status_offset = 0
        self._query_worker: Worker | None = None
        self.query_controller = QueryController()
        self._selected_files: list[Path] = []
        self._voice_recording = False
        self._voice_worker: Worker | None = None
        self.log_controller = LogController(self)
        self.session_controller: SessionController | None = None
        self._active_view = "chat"
        super().__init__()

    def _build_model_banner_text(self) -> str:
        context = provs.registery.context_key
        tool = provs.registery.toolgen_key
        embed = provs.registery.embed_key
        base = f"CG: {context}   TG: {tool}   EG: {embed}"
        repeat = base + "   " + base + "   " + base
        window = 60
        start = self._banner_offset % len(base)
        return repeat[start:start + window]

    def _update_model_banner(self) -> None:
        try:
            banner = self.query_one("#model-banner", ModelBanner)
        except Exception:
            return

        banner.set_text(self._build_model_banner_text())
        self._banner_offset = (self._banner_offset + 1) % 200

    def _update_query_status(self) -> None:
        if not self._query_waiting:
            return

        frames = ("  ...", " ... ", "...  ", " ... ")
        status = self.query_one("#query-status", Label)
        status.update(frames[self._query_status_offset])
        self._query_status_offset = (self._query_status_offset + 1) % len(frames)

    def compose(self)->ComposeResult:
        """
        Ekranı renderlar
        """
        yield Header()
        yield ModelBanner("CG: loading... TG: loading... EG: loading...", id="model-banner")
        yield Static("", id="selected-files", markup=False)
        with Container(id="loading-screen"):
            yield LoadingIndicator(id="init-loader")
            yield Label("R4Agent başlatılıyor...", id="init-status")
        with Container(id="view-switcher"):
            yield Button("Chat", id="chat-view-button", variant="primary", disabled=True)
            yield Button("Logs", id="logs-view-button", variant="default", disabled=True)
        with Container(id="chat-screen"):
            yield VerticalScroll(id="chat-log")
            yield Label("...", id="query-status")
            yield OptionList(id="command-list")
            with Container(id="prompt-row"):
                yield PromptTextArea(id="prompt-area")
                yield Button(
                    "REC",
                    id="record-button",
                    variant="default",
                    disabled=True,
                )
        with Container(id="logs-screen"):
            yield Label("LOGS", id="logs-title")
            yield VerticalScroll(id="logs-view")
        with Container(id="farewell-screen"):
            yield Label(":-)", id="farewell-face")
            
    def on_mount(self) -> None:
        self.log_controller.capture()
        self.commands: dict[str, Callable[[], None]] = {
            "/reset": self.reset_command,
            "/save": self.save_command,
            "/load": self.show_chat_list,
            "/change": self.show_change_list,
            "/select": self.show_select_list,
            "/exit": self.exit_command,
        }
        self.call_after_refresh(self._update_model_banner)
        self.set_interval(0.22, self._update_model_banner)
        self.set_interval(0.18, self._update_query_status)
        self.run_worker(
            self._initialize_agent,
            name="agent-init",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def append_log(self, message: str) -> None:
        self.log_controller.append(message)

    def _render_logs(self) -> None:
        self.log_controller.render()

    @on(Button.Pressed, "#chat-view-button")
    def show_chat_view(self) -> None:
        self._active_view = "chat"
        self.query_one("#chat-screen").display = True
        self.query_one("#logs-screen").display = False
        self.query_one("#chat-view-button", Button).variant = "primary"
        self.query_one("#logs-view-button", Button).variant = "default"

    @on(Button.Pressed, "#logs-view-button")
    def show_logs_view(self) -> None:
        self._active_view = "logs"
        self._render_logs()
        self.query_one("#chat-screen").display = False
        self.query_one("#logs-screen").display = True
        self.query_one("#chat-view-button", Button).variant = "default"
        self.query_one("#logs-view-button", Button).variant = "primary"

    def on_unmount(self) -> None:
        self.log_controller.restore()

    def action_cancel_query(self) -> None:
        self.cancel_query()

    def _finish_voice_processing(self, message: str) -> None:
        self._query_waiting = False
        self.query_one("#prompt-area", PromptTextArea).disabled = False
        status = self.query_one("#query-status", Label)
        status.update(message)
        status.display = True
        self.set_timer(1.5, lambda: setattr(status, "display", False))

    @on(Button.Pressed, "#record-button")
    def on_record_button_pressed(self) -> None:
        if self.handler is None or (self._query_waiting and not self._voice_recording):
            return

        button = self.query_one("#record-button", Button)
        promptarea = self.query_one("#prompt-area", PromptTextArea)
        if not self._voice_recording:
            self._query_waiting = True
            self._query_status_offset = 0
            promptarea.disabled = True
            button.disabled = True
            self.query_one("#query-status", Label).display = True
            self.run_worker(
                lambda: provs.WHISPER.start_recording(),
                name="voice-start",
                exclusive=True,
                thread=True,
                exit_on_error=False,
            )
            return

        self._voice_recording = False
        button.disabled = True
        self.query_one("#query-status", Label).update("Ses işleniyor")
        self._voice_worker = self.run_worker(
            lambda: provs.WHISPER.stop_recording(),
            name="voice-stop",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _initialize_agent(self) -> Handler:
        return Handler(R4Agent())

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        """Finalize each worker only in its owning UI state transition."""
        if event.worker.name == "agent-init":
            loader = self.query_one("#init-loader", LoadingIndicator)
            status = self.query_one("#init-status", Label)
            if event.state is WorkerState.SUCCESS:
                self.handler = event.worker.result
                self.session_controller = SessionController(self.handler)
                self.query_one("#record-button", Button).disabled = False
                self.query_one("#chat-view-button", Button).disabled = False
                self.query_one("#logs-view-button", Button).disabled = False
                self.query_one("#view-switcher").display = True
                self._update_model_banner()
                self.query_one("#loading-screen").display = False
                self.query_one("#chat-screen").display = True
                self.update_log()
                self._render_logs()
            elif event.state is WorkerState.ERROR:
                loader.display = False
                status.update(f"Başlatma başarısız: {event.worker.error}")
            return

        if event.worker.name == "load-chat":
            promptarea = self.query_one("#prompt-area", PromptTextArea)
            promptarea.disabled = False
            if event.state is WorkerState.SUCCESS:
                self.query_one("#query-status", Label).update("Sohbet yüklendi")
                self.update_log()
            elif event.state is WorkerState.ERROR:
                self.query_one("#query-status", Label).update(
                    f"Yükleme hatası: {event.worker.error}"
                )
            self.query_one("#query-status", Label).display = True
            return

        if event.worker.name == "provider-rebuild":
            promptarea = self.query_one("#prompt-area", PromptTextArea)
            promptarea.disabled = False
            if event.state is WorkerState.SUCCESS:
                self._update_model_banner()
                self.query_one("#query-status", Label).update("Provider değiştirildi")
                self.update_log()
            elif event.state is WorkerState.ERROR:
                self.query_one("#query-status", Label).update(
                    f"Provider değiştirme hatası: {event.worker.error}"
                )
            self.query_one("#query-status", Label).display = True
            return

        if event.worker.name == "voice-start":
            button = self.query_one("#record-button", Button)
            if event.state is WorkerState.SUCCESS:
                self._voice_recording = True
                button.label = "STOP"
                button.disabled = False
                self.query_one("#query-status", Label).update("Kayıt yapılıyor")
            elif event.state is WorkerState.ERROR:
                self._voice_recording = False
                button.label = "REC"
                button.disabled = False
                self._finish_voice_processing(
                    f"Ses kaydı başlatılamadı: {event.worker.error}"
                )
            elif event.state is WorkerState.CANCELLED:
                self._voice_recording = False
                button.label = "REC"
                button.disabled = False
                self._finish_voice_processing("Ses kaydı iptal edildi")
            return

        if event.worker.name == "voice-stop":
            if event.state not in (
                WorkerState.SUCCESS,
                WorkerState.CANCELLED,
                WorkerState.ERROR,
            ):
                return

            self._voice_worker = None
            button = self.query_one("#record-button", Button)
            promptarea = self.query_one("#prompt-area", PromptTextArea)
            button.label = "REC"
            button.disabled = False
            promptarea.disabled = False
            if event.state is WorkerState.SUCCESS:
                segments = event.worker.result or []
                transcription = " ".join(
                    segment_text.strip()
                    for segment in segments
                    if (segment_text := (getattr(segment, "text", "") or "")).strip()
                )
                if transcription:
                    current = promptarea.text.strip()
                    promptarea.load_text_at_end(
                        f"{current} {transcription}".strip()
                    )
                    promptarea.focus()
                    self._finish_voice_processing("Ses metne dönüştürüldü")
                else:
                    self._finish_voice_processing("Ses metni algılanamadı")
            elif event.state is WorkerState.ERROR:
                self._finish_voice_processing(
                    f"Ses işleme hatası: {event.worker.error}"
                )
            else:
                self._finish_voice_processing("Ses işleme iptal edildi")
            return

        if event.worker.name != "query":
            return

        if event.worker is not self._query_worker:
            return

        if event.state not in (
            WorkerState.SUCCESS,
            WorkerState.CANCELLED,
            WorkerState.ERROR,
        ):
            return

        promptarea = self.query_one("#prompt-area", PromptTextArea)
        promptarea.disabled = False
        self._query_worker = None
        if event.state is WorkerState.SUCCESS:
            self._query_waiting = False
            self.query_controller.finish()
            result = event.worker.result
            if result:
                self._ensure_response_message(result[-1][0])
            self.query_one("#query-status", Label).display = False
            self.update_log()
        elif event.state is WorkerState.CANCELLED:
            self._query_waiting = False
            self.query_controller.finish()
            self.query_one("#query-status", Label).update("Sorgu durduruldu")
            self.query_one("#query-status", Label).display = True
        elif event.state is WorkerState.ERROR:
            self._query_waiting = False
            self.query_controller.finish()
            self.query_one("#query-status", Label).update(
                f"Sorgu hatası: {event.worker.error}"
            )
            self.query_one("#query-status", Label).display = True

    def _looks_like_command(self, text: str) -> bool:
        return CommandParser.looks_like_command(text)

    @staticmethod
    def _command_fragment(text: str) -> str | None:
        return CommandParser.fragment(text)

    def _build_query_prompt(self, prompt: str) -> str:
        if not self._selected_files:
            return prompt

        file_context = "\n".join(f'file:"{path}"' for path in self._selected_files)
        return f"{prompt}\n{file_context}"

    def submit_prompt(self, prompt: str) -> None:
        """Route a prompt to a command handler or to the cancellable query worker."""
        if not prompt.strip() or self.handler is None or self._query_waiting:
            return

        command = self._command_fragment(prompt) or prompt.strip()
        if self._looks_like_command(command):
            if command == "/change":
                self.show_change_list()
            elif command.startswith("/change "):
                self.change_provider(command.removeprefix("/change ").strip())
            elif command == "/select":
                self.show_select_list()
            elif command.startswith("/select "):
                if self.select_path(command.removeprefix("/select ").strip()):
                    self.query_one("#prompt-area", PromptTextArea).clear()
                    self.hide_command_list()
            elif command in self.commands:
                self.commands[command]()
            elif command.startswith("/load "):
                self.load_command(command.removeprefix("/load ").strip())
            else:
                self.show_command_list(command)
            return

        promptarea = self.query_one("#prompt-area", PromptTextArea)
        promptarea.clear()
        promptarea.disabled = True
        self.hide_command_list()
        self._query_waiting = True
        self._query_status_offset = 0
        query_generation, cancel_event = self.query_controller.begin()
        query_prompt = self._build_query_prompt(prompt)
        self._selected_files.clear()
        self.update_selected_files()
        self.query_one("#query-status", Label).display = True
        self.update_log(Message(role=Roles.user, content=prompt))

        def run_query():
            return self.query_controller.run(
                self.handler,
                query_prompt,
                query_generation,
                cancel_event,
            )

        self._query_worker = self.run_worker(
            run_query,
            name="query",
            exclusive=True,
            start=False,
            thread=True,
            exit_on_error=False,
        )
        self._query_worker._start(self, self.workers._remove_worker)

    def cancel_query(self) -> bool:
        """Signal the active query stream to stop and restore prompt input."""
        if not self._query_waiting:
            return False

        if self._query_worker is not None:
            self._query_worker.cancel()
        if not self.query_controller.cancel(self._query_worker):
            return False
        self._query_waiting = False
        self._query_worker = None
        self.query_one("#prompt-area", PromptTextArea).disabled = False
        status = self.query_one("#query-status", Label)
        status.update("Mesaj iptal edildi")
        status.display = True
        self.set_timer(1.5, lambda: setattr(status, "display", False))
        return True

    def _ensure_response_message(self, content: str) -> None:
        sequence = self.handler.r4.message_sequnce.sequence
        if sequence and sequence[-1].role == Roles.assistant:
            return
        sequence.append(Message(role=Roles.assistant, content=content or ""))
    def reset_command(self) -> None:
        if self.handler is None:
            return

        self.session_controller.reset()
        self._selected_files.clear()
        self.update_selected_files()
        self.query_one("#prompt-area", PromptTextArea).clear()
        self.hide_command_list()
        self.query_one("#query-status", Label).update("Mesajlar sıfırlandı")
        self.query_one("#query-status", Label).display = True
        self.update_log()

    def exit_command(self) -> None:
        self.query_one("#loading-screen").display = False
        self.query_one("#chat-screen").display = False
        self.query_one("#farewell-screen").display = True
        self.query_one("#farewell-face", Label).update(":-)")
        self.set_timer(0.85, self._wink)
        self.set_timer(1.5, self.exit)

    def _wink(self) -> None:
        self.query_one("#farewell-face", Label).update(";-)")

    def save_command(self) -> None:
        if self.handler is None:
            return

        self.session_controller.save()
        self.query_one("#prompt-area", PromptTextArea).clear()
        self.hide_command_list()
        self.query_one("#query-status", Label).update("Sohbet kaydedildi")
        self.query_one("#query-status", Label).display = True
        self.update_log()

    def load_command(self, chat_name: str) -> None:
        if self.handler is None or not chat_name:
            return

        chat_path = next(
            (
                path
                for path in self.session_controller.list_chats()
                if path.name == chat_name
            ),
            None,
        )
        if chat_path is None:
            self.query_one("#query-status", Label).update(
                f"Sohbet bulunamadı: {chat_name}"
            )
            self.query_one("#query-status", Label).display = True
            return

        promptarea = self.query_one("#prompt-area", PromptTextArea)
        promptarea.clear()
        promptarea.disabled = True
        self.hide_command_list()
        self.query_one("#query-status", Label).update("Sohbet yükleniyor...")
        self.query_one("#query-status", Label).display = True
        self.run_worker(
            lambda: self.session_controller.load(chat_path),
            name="load-chat",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    @on(TextArea.Changed, "#prompt-area")
    def on_prompt_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        trimmed = text.strip()
        command = self._command_fragment(text)

        if command == "/load":
            self.show_chat_list()
        elif command is not None and command.startswith("/load "):
            self.show_chat_list(command.removeprefix("/load ").strip())
        elif command == "/change" or (command is not None and command.startswith("/change ")):
            self.show_change_list(command.removeprefix("/change").strip())
        elif command == "/select" or (command is not None and command.startswith("/select ")):
            self.show_select_list(command.removeprefix("/select").strip())
        elif command is not None and self._looks_like_command(command):
            self.show_command_list(command)
        else:
            self.hide_command_list()

    @on(OptionList.OptionSelected, "#command-list")
    def on_command_selected(self, event: OptionList.OptionSelected) -> None:
        command = event.option.id
        if command is None:
            return

        if command.startswith("/select "):
            promptarea = self.query_one("#prompt-area", PromptTextArea)
            promptarea.load_text_at_end(command)
            promptarea.focus()
            self.hide_command_list()
            return

        promptarea = self.query_one("#prompt-area", PromptTextArea)
        promptarea.load_text_at_end(command)
        promptarea.focus()
        self.hide_command_list()

    def complete_command_hint(self) -> bool:
        command_list = self.query_one("#command-list", OptionList)
        if not command_list.display or not command_list.option_count:
            return False

        option = command_list.highlighted_option or command_list.get_option_at_index(0)
        command = option.id
        if command is None:
            return False

        if command.startswith("/select "):
            promptarea = self.query_one("#prompt-area", PromptTextArea)
            promptarea.load_text_at_end(command)
            promptarea.focus()
            self.hide_command_list()
            return True

        promptarea = self.query_one("#prompt-area", PromptTextArea)
        promptarea.load_text_at_end(command)
        promptarea.focus()
        self.hide_command_list()
        return True

    def show_command_list(self, query: str = "/") -> None:
        command_list = self.query_one("#command-list", OptionList)
        options = [
            Option(command, id=command)
            for command in self.commands
            if command.startswith(query)
        ]
        command_list.set_options(options)
        command_list.display = bool(options)

    def show_chat_list(self, query: str = "") -> None:
        command_list = self.query_one("#command-list", OptionList)
        options = [
            Option(path.stem, id=f"/load {path.name}")
            for path in self.session_controller.list_chats()
            if path.name.startswith(query)
        ]
        command_list.set_options(options)
        command_list.display = bool(options)

    def show_change_list(self, query: str = "") -> None:
        """Build model-field or model-key suggestions for the change command."""
        fields = {
            "context_model": provs.registery.context_models,
            "embed_model": provs.registery.embed_models,
            "toolgen_model": provs.registery.toolgen_models,
        }
        command_list = self.query_one("#command-list", OptionList)
        parts = query.split()

        if not parts or (len(parts) == 1 and not query.endswith(" ") and parts[0] not in fields):
            options = [
                Option(field, id=f"/change {field}")
                for field in fields
                if not parts or field.startswith(parts[0])
            ]
        else:
            field = parts[0]
            keys = fields.get(field, {})
            key_query = parts[1] if len(parts) > 1 else ""
            options = [
                Option(key, id=f"/change {field} {key}")
                for key in keys
                if key.startswith(key_query)
            ]

        command_list.set_options(options)
        command_list.display = bool(options)

    def show_select_list(self, query: str = "") -> None:
        """Build a bounded workspace-relative file and directory picker."""
        root = Path.cwd().resolve()
        query = query.strip()
        requested = (root / query).resolve() if query else root

        if requested.is_dir():
            parent = requested
            prefix = ""
        else:
            parent = requested.parent
            prefix = requested.name

        if not parent.is_dir():
            self.query_one("#command-list", OptionList).set_options([])
            self.query_one("#command-list", OptionList).display = False
            return

        options = []
        for path in sorted(parent.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if prefix and not path.name.startswith(prefix):
                continue

            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                options.append(Option(f"{relative}/", id=f"/select {relative}/"))
            elif path.is_file() and path.suffix:
                options.append(Option(relative, id=f"/select {relative}"))

        command_list = self.query_one("#command-list", OptionList)
        command_list.set_options(options)
        command_list.display = bool(options)

    def select_path(self, path_text: str) -> bool:
        root = Path.cwd().resolve()
        selected = (root / path_text).resolve()

        try:
            relative = selected.relative_to(root).as_posix()
        except ValueError:
            return False

        if selected.is_dir():
            self.show_select_list(relative + "/")
            return False
        if not selected.is_file() or not selected.suffix:
            return False

        if selected not in self._selected_files:
            self._selected_files.append(selected)
            self.update_selected_files()
        return True

    def change_provider(self, change: str) -> None:
        parts = change.split()
        if len(parts) != 2:
            self.show_change_list(change)
            return

        field, key = parts
        fields = {
            "context_model": provs.registery.set_context_model,
            "embed_model": provs.registery.set_embed_model,
            "toolgen_model": provs.registery.set_toolgen_model,
        }
        setter = fields.get(field)
        if setter is None:
            self.show_change_list(change)
            return

        promptarea = self.query_one("#prompt-area", PromptTextArea)
        promptarea.clear()
        promptarea.disabled = True
        self.hide_command_list()
        self.query_one("#query-status", Label).update("Provider yeniden kuruluyor...")
        self.query_one("#query-status", Label).display = True

        def rebuild() -> None:
            setter(key)
            self.handler.r4.rebuild()

        self.run_worker(
            rebuild,
            name="provider-rebuild",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )
        self._update_model_banner()

    def hide_command_list(self) -> None:
        self.query_one("#command-list", OptionList).display = False

    def update_selected_files(self) -> None:
        panel = self.query_one("#selected-files", Static)
        if not self._selected_files:
            panel.update("")
            panel.display = False
            return

        panel.update(
            "\n".join(f"+{path}" for path in self._selected_files)
        )
        panel.display = True

    def update_log(self, pending_message: Message | None = None) -> None:
        """Render the current agent sequence through the message projection module."""
        if self.handler is None:
            return
        chatlog = self.query_one("#chat-log", VerticalScroll)
        render_messages(
            chatlog,
            self.handler.r4.message_sequnce.sequence,
            pending_message,
        )

    
if __name__ == "__main__":
    R4TUI().run()