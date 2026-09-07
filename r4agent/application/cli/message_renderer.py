from __future__ import annotations

from textual.containers import VerticalScroll

from r4agent import Message
from r4agent.struct.base import Roles

from .widgets import MessageBlock, ToolChain


def render_messages(chatlog: VerticalScroll, sequence: list[Message], pending_message: Message | None = None) -> None:
    """Project a message sequence into Textual widgets in one ordered pass."""
    chatlog.remove_children()
    tool_chain: list[MessageBlock] = []

    def mount_tool_chain() -> None:
        """Mount the pending tool blocks as one chain when an assistant follows."""
        nonlocal tool_chain
        if not tool_chain:
            return
        if len(tool_chain) > 1 and tool_chain[-1].message.role == Roles.assistant:
            chatlog.mount(ToolChain(tool_chain))
        else:
            for block in tool_chain:
                chatlog.mount(block)
        tool_chain = []

    for index, message in enumerate(sequence):
        if message.role == Roles.assistant and not message.content and message.tool_calls:
            continue

        tool_call = None
        if message.role == Roles.tool:
            for previous in reversed(sequence[:index]):
                if previous.role == Roles.assistant and previous.tool_calls:
                    tool_call = previous.tool_calls[-1]
                    break

        block = MessageBlock(message, tool_call=tool_call)
        if message.role == Roles.tool:
            tool_chain.append(block)
        elif message.role == Roles.assistant and tool_chain:
            tool_chain.append(block)
            mount_tool_chain()
        else:
            mount_tool_chain()
            chatlog.mount(block)

    mount_tool_chain()
    if pending_message is not None:
        chatlog.mount(MessageBlock(pending_message))
    chatlog.scroll_end(animate=False)
