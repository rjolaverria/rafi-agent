import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai.types.chat import ChatCompletionMessage

from tools import ToolResult

logger = logging.getLogger(__name__)


class ResponseHook(Protocol):
    def __call__(self, message: ChatCompletionMessage) -> None: ...


class ToolCallHook(Protocol):
    def __call__(self, tool_name: str, args: Any) -> None: ...


class ToolResultHook(Protocol):
    def __call__(self, tool_result: ToolResult) -> None: ...


@dataclass
class Hooks:
    after_response: list[ResponseHook] = field(default_factory=list)
    before_tool_call: list[ToolCallHook] = field(default_factory=list)
    after_tool_call: list[ToolResultHook] = field(default_factory=list)

    def on_response(self, hook: ResponseHook) -> None:
        self.after_response.append(hook)

    def on_tool_call(self, hook: ToolCallHook) -> None:
        self.before_tool_call.append(hook)

    def on_tool_result(self, hook: ToolResultHook) -> None:
        self.after_tool_call.append(hook)

    def trigger_response(self, message: ChatCompletionMessage) -> None:
        for hook in self.after_response:
            try:
                hook(message)
            except Exception:
                logger.exception("Hook %s failed during %s", hook, "after_response")

    def trigger_tool_call(self, tool_name: str, args: Any) -> None:
        for hook in self.before_tool_call:
            try:
                hook(tool_name, args)
            except Exception:
                logger.exception("Hook %s failed during %s", hook, "before_tool_call")

    def trigger_tool_result(self, tool_result: ToolResult) -> None:
        for hook in self.after_tool_call:
            try:
                hook(tool_result)
            except Exception:
                logger.exception("Hook %s failed during %s", hook, "after_tool_call")
