from pathlib import Path
from typing import Any, Callable, TypedDict
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.widgets import Button, Header, Label, LoadingIndicator, OptionList, Static, TextArea
from textual.widgets.option_list import Option
from textual.worker import Worker, WorkerState

from r4agent import Message, R4Agent
from r4agent.providers import ModelRegistery, ProviderManager, UsageRegisteryLoader
from r4agent.struct.base import Roles

import os

from .command_parser import CommandParser
from .handler import Handler
from .log_controller import LogController
from .message_renderer import render_messages
from .query_controller import QueryController
from .session_controller import SessionController
from .widgets import ModelBanner, PromptTextArea


class CommandNode(TypedDict, total=False):
    subcommands: dict[str, "CommandNode"]
    arg_provider: Callable[["R4TUI", list[str]], list[str]] | None
    handler: Callable[["R4TUI", list[str]], None]


class CommandRunner:
    """Çok aşamalı (Tree) komut çözümleme ve otomatik tamamlama motoru."""

    def __init__(self, app: "R4TUI", tree: dict[str, CommandNode]):
        self.app = app
        self.tree = tree

    def resolve(self, full_text: str) -> tuple[CommandNode | None, list[str]]:
        parts = full_text.strip().split()
        if not parts:
            return None, []

        cmd_root = parts[0]
        if cmd_root not in self.tree:
            return None, []

        current_node = self.tree[cmd_root]
        consumed_index = 1

        while consumed_index < len(parts):
            next_word = parts[consumed_index]
            subcommands = current_node.get("subcommands", {})
            if next_word in subcommands:
                current_node = subcommands[next_word]
                consumed_index += 1
            else:
                break

        remaining_args = parts[consumed_index:]
        return current_node, remaining_args

    def get_suggestions(self, full_text: str) -> list[str]:
        if not full_text.startswith("/"):
            return []

        parts = full_text.split()
        if len(parts) == 1 and not full_text.endswith(" "):
            return [cmd for cmd in self.tree if cmd.startswith(parts[0])]

        node, remaining_args = self.resolve(full_text)
        if not node:
            return []

        subcommands = node.get("subcommands", {})
        if subcommands:
            prefix = remaining_args[0] if remaining_args else ""
            matches = [sub for sub in subcommands if sub.startswith(prefix)]
            base_cmd = " ".join(parts[: len(parts) - (1 if remaining_args else 0)])
            return [f"{base_cmd} {match}".strip() for match in matches]

        provider = node.get("arg_provider")
        if provider:
            return provider(self.app, remaining_args)

        return []

    def execute(self, full_text: str) -> bool:
        node, remaining_args = self.resolve(full_text)
        if node and "handler" in node:
            node["handler"](self.app, remaining_args)
            return True
        return False


class R4TUI(App):
    CSS_PATH = "styles.tcss"
    TITLE = "R4Agent"
    BINDINGS = [
        Binding("ctrl+b", "cancel_query", show=True, priority=True),
    ]

    def __init__(self):
        registery_sets, prompts = UsageRegisteryLoader.load()
        self.usage_registery_sets = registery_sets
        self.usage_prompts = prompts
        self.usage_chosen_registery_set = "coder"
        self.stream = True
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
        self.prompts = prompts
        
        self.current_screen_idx = 0

        # Çok Aşamalı Komut Ağacı (COMMAND TREE)
        self.command_tree: dict[str, CommandNode] = {
            "/reset": {
                "handler": lambda app, args: app.reset_command(),
            },
            "/save": {
                "handler": lambda app, args: app.save_command(),
            },
            "/exit": {
                "handler": lambda app, args: app.exit_command(),
            },
            "/load": {
                "arg_provider": lambda app, args: [
                    f"/load {p.name}"
                    for p in app.session_controller.list_chats()
                    if p.name.startswith(args[0] if args else "")
                ],
                "handler": lambda app, args: app.load_command(args[0] if args else ""),
            },
            "/select": {
                "arg_provider": lambda app, args: app._get_select_suggestions(args[0] if args else ""),
                "handler": lambda app, args: app._handle_select_command(" ".join(args)),
            },
            "/rag" : {
                "subcommands" : {
                    "add_document": {
                        "arg_provider": lambda app, args: app._get_rag_add_documents_suggestions(args[0] if args else ""),
                        "handler": lambda app,args: app._handle_rag_add_documents_command(" ".join(args))
                    }
                }
            },
            "/prompt" : {
                "arg_provider": lambda app, args: app._get_prompt_suggestions(args),
                "handler": lambda app, args: app._handle_prompt_command(" ".join(args)),
            },
            "/providerset" : {
                "arg_provider" : lambda app, args: app._get_providerset_suggestions(args),
                "handler" : lambda app, args : app._handle_providerset_command(" ".join(args)),   
            },
            "/change": {
                # 2 Aşamalı & 3 Aşamalı Hibrit Yapı
                "subcommands": {
                    "context_model": {
                        "arg_provider": lambda app, args: [
                            f"/change context_model {k}"
                            for k in ModelRegistery.context_models
                            if k.startswith(args[0] if args else "")
                        ],
                        "handler": lambda app, args: app.change_provider(f"context_model {args[0]}") if args else None,
                    },
                    "embed_model": {
                        "arg_provider": lambda app, args: [
                            f"/change embed_model {k}"
                            for k in ModelRegistery.embed_models
                            if k.startswith(args[0] if args else "")
                        ],
                        "handler": lambda app, args: app.change_provider(f"embed_model {args[0]}") if args else None,
                    },
                    "toolgen_model": {
                        "arg_provider": lambda app, args: [
                            f"/change toolgen_model {k}"
                            for k in ModelRegistery.toolgen_models
                            if k.startswith(args[0] if args else "")
                        ],
                        "handler": lambda app, args: app.change_provider(f"toolgen_model {args[0]}") if args else None,
                    },
                    "system_prompt" : {
                        "handler": lambda app, args: app.change_provider(f"system_prompt {args[0]}")
                    }
                    # # 3 Aşamalı Örnek: /change provider context_model <model>
                    # "provider": {
                    #     "subcommands": {
                    #         "context_model": {
                    #             "arg_provider": lambda app, args: [
                    #                 f"/change provider context_model {k}"
                    #                 for k in ModelRegistery.context_models
                    #                 if k.startswith(args[0] if args else "")
                    #             ],
                    #             "handler": lambda app, args: app.change_provider(f"context_model {args[0]}") if args else None,
                    #         }
                    #     }
                    # },
                }
            },
        }
        self.command_runner = CommandRunner(self, self.command_tree)
        super().__init__()

    def _build_model_banner_text(self) -> str:
        if self.handler is None:
            return "CG: Yükleniyor... TG: Yükleniyor... EG: Yükleniyor..."
        context = self.handler.r4.provider.registery_set.context_model
        tool = self.handler.r4.provider.registery_set.toolgen_model
        embed = self.handler.r4.provider.registery_set.embed_model
        whisper = self.handler.r4.provider.registery_set.whisper_model
        base = f"CG: {context}   TG: {tool}   EG: {embed}    W: {whisper}"
        repeat = base + "   " + base + "   " + base
        window = 80
        start = self._banner_offset % len(base)
        return repeat[start : start + window]

    def _update_model_banner(self) -> None:
        try:
            banner = self.model_banner
        except Exception:
            return

        banner.set_text(self._build_model_banner_text())
        self._banner_offset = (self._banner_offset + 1) % 200

    def _update_query_status(self) -> None:
        if not self._query_waiting:
            return

        frames = ("  ...", " ... ", "...  ", " ... ")
        status = self.query_status_label
        status.update(frames[self._query_status_offset])
        self._query_status_offset = (self._query_status_offset + 1) % len(frames)

    def compose(self) -> ComposeResult:
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
                yield Button("REC", id="record-button", variant="default", disabled=True)
        with Container(id="logs-screen"):
            yield Label("LOGS", id="logs-title")
            yield VerticalScroll(id="logs-view")
        with Container(id="farewell-screen"):
            yield Label(":-)", id="farewell-face")

    def on_mount(self) -> None:
        self.model_banner = self.query_one("#model-banner", ModelBanner)
        self.query_status_label = self.query_one("#query-status", Label)
        self.chat_view_button = self.query_one("#chat-view-button", Button)
        self.logs_view_button = self.query_one("#logs-view-button", Button)
        self.chat_screen = self.query_one("#chat-screen", Container)
        self.logs_screen = self.query_one("#logs-screen", Container)
        self.prompt_area = self.query_one("#prompt-area", PromptTextArea)
        self.init_loader = self.query_one("#init-loader", LoadingIndicator)
        self.init_status_label = self.query_one("#init-status", Label)
        self.record_button = self.query_one("#record-button", Button)
        self.view_switcher = self.query_one("#view-switcher", Container)
        self.loading_screen = self.query_one("#loading-screen", Container)
        self.farewell_screen = self.query_one("#farewell-screen", Container)
        self.farewell_face_label = self.query_one("#farewell-face", Label)
        self.command_list = self.query_one("#command-list", OptionList)
        self.selected_files_panel = self.query_one("#selected-files", Static)
        self.chat_log = self.query_one("#chat-log", VerticalScroll)

        self.log_controller.capture()
        self.run_worker(
            self._initialize_agent,
            name="agent-init",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )
        self.set_interval(0.22, self._update_model_banner)
        self.set_interval(0.18, self._update_query_status)

    def append_log(self, message: str) -> None:
        self.log_controller.append(message)

    def _render_logs(self) -> None:
        self.log_controller.render()

    @on(Button.Pressed, "#chat-view-button")
    def show_chat_view(self) -> None:
        self._active_view = "chat"
        self.chat_screen.display = True
        self.logs_screen.display = False
        self.chat_view_button.variant = "primary"
        self.logs_view_button.variant = "default"
        self.current_screen_idx = 0

    @on(Button.Pressed, "#logs-view-button")
    def show_logs_view(self) -> None:
        self._active_view = "logs"
        self._render_logs()
        self.chat_screen.display = False
        self.logs_screen.display = True
        self.chat_view_button.variant = "default"
        self.logs_view_button.variant = "primary"
        self.current_screen_idx = 1

    def on_unmount(self) -> None:
        self.log_controller.restore()

    def action_cancel_query(self) -> None:
        self.cancel_query()

    def _finish_voice_processing(self, message: str) -> None:
        self._query_waiting = False
        self.prompt_area.disabled = False
        status = self.query_status_label
        status.update(message)
        status.display = True
        self.set_timer(1.5, lambda: setattr(status, "display", False))

    @on(Button.Pressed, "#record-button")
    def on_record_button_pressed(self) -> None:
        if self.handler is None or (self._query_waiting and not self._voice_recording):
            return

        button = self.record_button
        promptarea = self.prompt_area
        if not self._voice_recording:
            self._query_waiting = True
            self._query_status_offset = 0
            promptarea.disabled = True
            button.disabled = True
            self.query_status_label.display = True
            self.run_worker(
                lambda: self.handler.r4.provider.whisper.start_recording(),
                name="voice-start",
                exclusive=True,
                thread=True,
                exit_on_error=False,
            )
            return

        self._voice_recording = False
        button.disabled = True
        self.query_status_label.update("Ses işleniyor")
        self._voice_worker = self.run_worker(
            lambda: self.handler.r4.provider.whisper.stop_recording(),
            name="voice-stop",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _initialize_agent(self) -> Handler:
        return Handler(
            R4Agent(
                ProviderManager(
                    self.usage_registery_sets[self.usage_chosen_registery_set]), 
                self.stream))

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.name == "agent-init":
            loader = self.init_loader
            status = self.init_status_label
            if event.state is WorkerState.SUCCESS:
                self.handler = event.worker.result
                self.session_controller = SessionController(self.handler)
                self.record_button.disabled = False
                self.chat_view_button.disabled = False
                self.logs_view_button.disabled = False
                self.view_switcher.display = True
                self.call_after_refresh(self._update_model_banner)
                self.loading_screen.display = False
                self.chat_screen.display = True
                self.update_log()
                self._render_logs()
            elif event.state is WorkerState.ERROR:
                loader.display = False
                status.update(f"Başlatma başarısız: {event.worker.error}")
            return

        if event.worker.name == "load-chat":
            promptarea = self.prompt_area
            promptarea.disabled = False
            if event.state is WorkerState.SUCCESS:
                self.query_status_label.update("Sohbet yüklendi")
                self.update_log()
            elif event.state is WorkerState.ERROR:
                self.query_status_label.update(f"Yükleme hatası: {event.worker.error}")
            self.query_status_label.display = True
            return

        if event.worker.name == "provider-rebuild":
            promptarea = self.prompt_area
            promptarea.disabled = False
            if event.state is WorkerState.SUCCESS:
                self._update_model_banner()
                self.query_status_label.update("Provider değiştirildi")
                self.update_log()
            elif event.state is WorkerState.ERROR:
                self.query_status_label.update(f"Provider değiştirme hatası: {event.worker.error}")
            self.query_status_label.display = True
            return

        if event.worker.name == "voice-start":
            button = self.record_button
            if event.state is WorkerState.SUCCESS:
                self._voice_recording = True
                button.label = "STOP"
                button.disabled = False
                self.query_status_label.update("Kayıt yapılıyor")
            elif event.state is WorkerState.ERROR:
                self._voice_recording = False
                button.label = "REC"
                button.disabled = False
                self._finish_voice_processing(f"Ses kaydı başlatılamadı: {event.worker.error}")
            elif event.state is WorkerState.CANCELLED:
                self._voice_recording = False
                button.label = "REC"
                button.disabled = False
                self._finish_voice_processing("Ses kaydı iptal edildi")
            return
        
        if event.worker.name == "rag-add-document":
            if event.state not in (WorkerState.SUCCESS, WorkerState.CANCELLED, WorkerState.ERROR):
                return
            self.prompt_area.disabled = False
            if event.state is WorkerState.SUCCESS:
                self._update_model_banner()
                self.query_status_label.update("Döküman başarıyla eklendi")
            elif event.state is WorkerState.ERROR:
                self.query_status_label.update(f"Döküman eklerken hata: {event.worker.error}")
            self.query_status_label.display = True
            return

            
        if event.worker.name == "voice-stop":
            if event.state not in (WorkerState.SUCCESS, WorkerState.CANCELLED, WorkerState.ERROR):
                return

            self._voice_worker = None
            button = self.record_button
            promptarea = self.prompt_area
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
                    promptarea.load_text_at_end(f"{current} {transcription}".strip())
                    promptarea.focus()
                    self._finish_voice_processing("Ses metne dönüştürüldü")
                else:
                    self._finish_voice_processing("Ses metni algılanamadı")
            elif event.state is WorkerState.ERROR:
                self._finish_voice_processing(f"Ses işleme hatası: {event.worker.error}")
            else:
                self._finish_voice_processing("Ses işleme iptal edildi")
            return

        if event.worker.name != "query" or event.worker is not self._query_worker:
            return

        if event.state not in (WorkerState.SUCCESS, WorkerState.CANCELLED, WorkerState.ERROR):
            return

        promptarea = self.prompt_area
        promptarea.disabled = False
        self._query_worker = None
        if event.state is WorkerState.SUCCESS:
            self._query_waiting = False
            self.query_controller.finish()
            result = event.worker.result
            if result:
                self._ensure_response_message(result[-1][0])
            self.query_status_label.display = False
            self.update_log()
        elif event.state is WorkerState.CANCELLED:
            self._query_waiting = False
            self.query_controller.finish()
            self.query_status_label.update("Sorgu durduruldu")
            self.query_status_label.display = True
        elif event.state is WorkerState.ERROR:
            self._query_waiting = False
            self.query_controller.finish()
            self.query_status_label.update(f"Sorgu hatası: {event.worker.error}")
            self.query_status_label.display = True

    def _build_query_prompt(self, prompt: str) -> str:
        if not self._selected_files:
            return prompt
        file_context = "\n".join(f'file:"{path}"' for path in self._selected_files)
        return f"{prompt}\n{file_context}"

    def submit_prompt(self, prompt: str) -> None:
        if not prompt.strip() or self.handler is None or self._query_waiting:
            return

        full_prompt = prompt.strip()
        if full_prompt.startswith("/"):
            if self.command_runner.execute(full_prompt):
                # self.prompt_area.clear()
                self.hide_command_list()
                return

        promptarea = self.prompt_area
        promptarea.clear()
        promptarea.disabled = True
        self.hide_command_list()
        self._query_waiting = True
        self._query_status_offset = 0
        query_generation, cancel_event = self.query_controller.begin()
        query_prompt = self._build_query_prompt(prompt)
        self._selected_files.clear()
        self.update_selected_files()
        self.query_status_label.display = True
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
        if not self._query_waiting:
            return False

        if self._query_worker is not None:
            self._query_worker.cancel()
        if not self.query_controller.cancel(self._query_worker):
            return False
        self._query_waiting = False
        self._query_worker = None
        self.prompt_area.disabled = False
        status = self.query_status_label
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
        self.prompt_area.clear()
        self.hide_command_list()
        self.query_status_label.update("Mesajlar sıfırlandı")
        self.query_status_label.display = True
        self.update_log()

    def exit_command(self) -> None:
        self.loading_screen.display = False
        self.chat_screen.display = False
        self.farewell_screen.display = True
        self.farewell_face_label.update(":-)")
        self.set_timer(0.85, self._wink)
        self.set_timer(1.5, self.exit)

    def _wink(self) -> None:
        self.farewell_face_label.update(";-)")

    def save_command(self) -> None:
        if self.handler is None:
            return
        self.session_controller.save()
        self.prompt_area.clear()
        self.hide_command_list()
        self.query_status_label.update("Sohbet kaydedildi")
        self.query_status_label.display = True
        self.update_log()

    def load_command(self, chat_name: str) -> None:
        if self.handler is None or not chat_name:
            return

        chat_path = next(
            (path for path in self.session_controller.list_chats() if path.name == chat_name),
            None,
        )
        if chat_path is None:
            self.query_status_label.update(f"Sohbet bulunamadı: {chat_name}")
            self.query_status_label.display = True
            return

        promptarea = self.prompt_area
        promptarea.clear()
        promptarea.disabled = True
        self.hide_command_list()
        self.query_status_label.update("Sohbet yükleniyor...")
        self.query_status_label.display = True
        self.run_worker(
            lambda: self.session_controller.load(chat_path),
            name="load-chat",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )

    def _get_select_suggestions(self, query: str = "") -> list[str]:
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
            return []

        options = []
        for path in sorted(parent.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if prefix and not path.name.startswith(prefix):
                continue
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                options.append(f"/select {relative}/")
            elif path.is_file() and path.suffix:
                options.append(f"/select {relative}")
        return options
    
    def _handle_select_command(self, path_text: str) -> None:
        if self.select_path(path_text):
            self.prompt_area.clear()
            self.hide_command_list()
            
    def _get_rag_add_documents_suggestions(self, query: str = "") -> list[str]:
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
            return []

        options = []
        for path in sorted(parent.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if prefix and not path.name.startswith(prefix):
                continue
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                options.append(f"/rag add_document {relative}/")
            elif path.is_file() and path.suffix:
                options.append(f"/rag add_document {relative}")
        return options

    def _handle_rag_add_documents_command(self, path_text:str)->None:
        self.run_worker(
            lambda: self.app.handler.r4.provider.dbclient.add_document(str(path_text)),
            name="rag-add-document",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )
        self.prompt_area.clear()
        self.hide_command_list()
        self.prompt_area.disabled = True

    def select_path(self, path_text: str) -> bool:
        root = Path.cwd().resolve()
        selected = (root / path_text).resolve()
        try:
            relative = selected.relative_to(root).as_posix()
        except ValueError:
            return False

        if selected.is_dir():
            options = self._get_select_suggestions(relative + "/")
            self.command_list.set_options([Option(opt, id=opt) for opt in options])
            self.command_list.display = bool(options)
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
            return

        field, key = parts
        fields = {
            "context_model": self.handler.r4.provider.change_context_model,
            "embed_model": self.handler.r4.provider.change_embed_model,
            "toolgen_model": self.handler.r4.provider.change_tool_model,
            "system_prompt": self.handler.r4.provider.change_system_prompt
        }
        setter = fields.get(field)
        if setter is None:
            return

        promptarea = self.prompt_area
        promptarea.clear()
        promptarea.disabled = True
        self.hide_command_list()
        self.query_status_label.update("Provider yeniden kuruluyor...")
        self.query_status_label.display = True

        def rebuild() -> None:
            setter(key)

        self.run_worker(
            rebuild,
            name="provider-rebuild",
            exclusive=True,
            thread=True,
            exit_on_error=False,
        )
        self._update_model_banner()
        if field == "system_prompt":
            self.update_log()
           
    def _get_prompt_suggestions(self, args: list[str]) -> list[str]:
        # args içerisinden sadece yazılan son kelimeyi al
        search_key = args[0].strip() if args else ""
        return [
            f"/prompt {key}" 
            for key in self.prompts.keys() 
            if key.startswith(search_key)
        ]
    
    def _handle_prompt_command(self, key_text: str) -> None:
        # 1. Gelen metni temizle (/prompt kalıntısı veya fazla boşluklar varsa sök)
        clean_key = key_text.replace("/prompt", "").strip()
        
        # 2. Sözlükten değeri çek
        value = self.prompts.get(clean_key)
        if value:
            # 3. Önce kutudaki komut yazısını sil, sonra yeni prompt metnini yaz
            self.prompt_area.clear()
            self.prompt_area.text = value
            self.prompt_area.focus()
            
    def _get_providerset_suggestions(self, args: list[str]) -> list[str]:
        search_key = args[0].strip() if args else ""
        return [
            f"/providerset {key}" 
            for key in list(self.usage_registery_sets.keys())
            if key.startswith(search_key)
        ]
    
    def _handle_providerset_command(self, key_text:str )-> None:
        key = key_text.replace("/providerset", "").strip()
        self.usage_chosen_registery_set = self.usage_registery_sets.get(key)
        self.handler.r4.change_provider_manager(
            ProviderManager(self.usage_chosen_registery_set)
        )
        self._update_model_banner()
        self.prompt_area.clear()
        self.prompt_area.focus()     
        
    @on(TextArea.Changed, "#prompt-area")
    def on_prompt_changed(self, event: TextArea.Changed) -> None:
        text = event.text_area.text
        if not text.startswith("/"):
            self.hide_command_list()
            return

        suggestions = self.command_runner.get_suggestions(text)
        options = [Option(item, id=item) for item in suggestions]
        self.command_list.set_options(options)
        self.command_list.display = bool(options)

    @on(OptionList.OptionSelected, "#command-list")
    def on_command_selected(self, event: OptionList.OptionSelected) -> None:
        command = event.option.id
        if command is None:
            return

        promptarea = self.prompt_area
        promptarea.load_text_at_end(command)
        promptarea.focus()

    def complete_command_hint(self) -> bool:
        command_list = self.command_list
        if not command_list.display or not command_list.option_count:
            return False

        option = command_list.highlighted_option or command_list.get_option_at_index(0)
        command = option.id
        if command is None:
            return False

        promptarea = self.prompt_area
        promptarea.load_text_at_end(command)
        promptarea.focus()
        return True

    def hide_command_list(self) -> None:
        self.command_list.display = False

    def update_selected_files(self) -> None:
        panel = self.selected_files_panel
        if not self._selected_files:
            panel.update("")
            panel.display = False
            return

        panel.update("\n".join(f"+{path}" for path in self._selected_files))
        panel.display = True

    def update_log(self, pending_message: Message | None = None) -> None:
        if self.handler is None:
            return
        chatlog = self.chat_log
        render_messages(
            chatlog,
            self.handler.r4.message_sequnce.sequence,
            pending_message,
        )
        
    def on_key(self, event) -> None:
        if event.key == "enter":
            self.prompt_area.focus()
        elif event.key == "right" or event.key == "left":
            if self.current_screen_idx == 0:
                self.show_logs_view()
            elif self.current_screen_idx == 1:
                self.show_chat_view()
            


if __name__ == "__main__":
    R4TUI().run()