import logging
from typing import Any

import pytest
from openai.types.chat import ChatCompletionMessage

from hooks import Hooks
from tools import ToolResult


def _noop_response(message: ChatCompletionMessage) -> None:
    pass


def _noop_tool_call(tool_name: str, args: Any) -> None:
    pass


def _noop_tool_result(tool_result: ToolResult) -> None:
    pass


class TestHooksRegistration:
    def test_on_response_registers_hook(self):
        hooks = Hooks()
        hooks.on_response(_noop_response)
        assert _noop_response in hooks.after_response

    def test_on_tool_call_registers_hook(self):
        hooks = Hooks()
        hooks.on_tool_call(_noop_tool_call)
        assert _noop_tool_call in hooks.before_tool_call

    def test_on_tool_result_registers_hook(self):
        hooks = Hooks()
        hooks.on_tool_result(_noop_tool_result)
        assert _noop_tool_result in hooks.after_tool_call

    def test_multiple_hooks_registered(self):
        hooks = Hooks()
        hooks.on_response(_noop_response)
        hooks.on_response(_noop_response)
        assert len(hooks.after_response) == 2

    def test_default_hooks_are_empty(self):
        hooks = Hooks()
        assert hooks.after_response == []
        assert hooks.before_tool_call == []
        assert hooks.after_tool_call == []

    def test_hooks_instances_have_independent_lists(self):
        a = Hooks()
        b = Hooks()
        a.on_response(_noop_response)
        assert len(b.after_response) == 0


def _make_message(content: str = "hello") -> ChatCompletionMessage:
    return ChatCompletionMessage(role="assistant", content=content)


def _make_tool_result(name: str = "test") -> ToolResult:
    return ToolResult(error=False, name=name, result="ok")


class TestTriggerResponse:
    def test_calls_hook_with_message(self):
        hooks = Hooks()
        received: list[ChatCompletionMessage] = []

        def capture(message: ChatCompletionMessage) -> None:
            received.append(message)

        hooks.on_response(capture)
        msg = _make_message()
        hooks.trigger_response(msg)
        assert received == [msg]

    def test_calls_multiple_hooks_in_order(self):
        hooks = Hooks()
        order: list[int] = []

        def first(message: ChatCompletionMessage) -> None:
            order.append(1)

        def second(message: ChatCompletionMessage) -> None:
            order.append(2)

        hooks.on_response(first)
        hooks.on_response(second)
        hooks.trigger_response(_make_message())
        assert order == [1, 2]

    def test_no_hooks_does_not_raise(self):
        hooks = Hooks()
        hooks.trigger_response(_make_message())

    def test_failing_hook_does_not_propagate(self):
        hooks = Hooks()

        def bad_hook(message: ChatCompletionMessage) -> None:
            raise ValueError("boom")

        hooks.on_response(bad_hook)
        hooks.trigger_response(_make_message())  # should not raise

    def test_failing_hook_logs_exception(self, caplog: pytest.LogCaptureFixture):
        hooks = Hooks()

        def bad_hook(message: ChatCompletionMessage) -> None:
            raise ValueError("boom")

        hooks.on_response(bad_hook)
        with caplog.at_level(logging.ERROR, logger="hooks"):
            hooks.trigger_response(_make_message())
        assert "boom" in caplog.text
        assert "after_response" in caplog.text

    def test_failure_does_not_stop_subsequent_hooks(self):
        hooks = Hooks()
        called: list[bool] = []

        def bad_hook(message: ChatCompletionMessage) -> None:
            raise RuntimeError("fail")

        def good_hook(message: ChatCompletionMessage) -> None:
            called.append(True)

        hooks.on_response(bad_hook)
        hooks.on_response(good_hook)
        hooks.trigger_response(_make_message())
        assert called == [True]


class TestTriggerToolCall:
    def test_calls_hook_with_args(self):
        hooks = Hooks()
        received: list[tuple[str, Any]] = []

        def capture(tool_name: str, args: Any) -> None:
            received.append((tool_name, args))

        hooks.on_tool_call(capture)
        hooks.trigger_tool_call("readfile", {"file_path": "/tmp/x"})
        assert received == [("readfile", {"file_path": "/tmp/x"})]

    def test_calls_multiple_hooks_in_order(self):
        hooks = Hooks()
        order: list[int] = []

        def first(tool_name: str, args: Any) -> None:
            order.append(1)

        def second(tool_name: str, args: Any) -> None:
            order.append(2)

        hooks.on_tool_call(first)
        hooks.on_tool_call(second)
        hooks.trigger_tool_call("bash", "ls")
        assert order == [1, 2]

    def test_no_hooks_does_not_raise(self):
        hooks = Hooks()
        hooks.trigger_tool_call("bash", "ls")

    def test_failing_hook_does_not_propagate(self):
        hooks = Hooks()

        def bad_hook(tool_name: str, args: Any) -> None:
            raise ZeroDivisionError

        hooks.on_tool_call(bad_hook)
        hooks.trigger_tool_call("bash", "ls")  # should not raise

    def test_failing_hook_logs_exception(self, caplog: pytest.LogCaptureFixture):
        hooks = Hooks()

        def bad_hook(tool_name: str, args: Any) -> None:
            raise ZeroDivisionError

        hooks.on_tool_call(bad_hook)
        with caplog.at_level(logging.ERROR, logger="hooks"):
            hooks.trigger_tool_call("bash", "ls")
        assert "before_tool_call" in caplog.text

    def test_failure_does_not_stop_subsequent_hooks(self):
        hooks = Hooks()
        called: list[bool] = []

        def bad_hook(tool_name: str, args: Any) -> None:
            raise ZeroDivisionError

        def good_hook(tool_name: str, args: Any) -> None:
            called.append(True)

        hooks.on_tool_call(bad_hook)
        hooks.on_tool_call(good_hook)
        hooks.trigger_tool_call("bash", "ls")
        assert called == [True]


class TestTriggerToolResult:
    def test_calls_hook_with_result(self):
        hooks = Hooks()
        received: list[ToolResult] = []

        def capture(tool_result: ToolResult) -> None:
            received.append(tool_result)

        hooks.on_tool_result(capture)
        result = _make_tool_result()
        hooks.trigger_tool_result(result)
        assert received == [result]

    def test_calls_multiple_hooks_in_order(self):
        hooks = Hooks()
        order: list[int] = []

        def first(tool_result: ToolResult) -> None:
            order.append(1)

        def second(tool_result: ToolResult) -> None:
            order.append(2)

        hooks.on_tool_result(first)
        hooks.on_tool_result(second)
        hooks.trigger_tool_result(_make_tool_result())
        assert order == [1, 2]

    def test_no_hooks_does_not_raise(self):
        hooks = Hooks()
        hooks.trigger_tool_result(_make_tool_result())

    def test_failing_hook_does_not_propagate(self):
        hooks = Hooks()

        def bad_hook(tool_result: ToolResult) -> None:
            raise ZeroDivisionError

        hooks.on_tool_result(bad_hook)
        hooks.trigger_tool_result(_make_tool_result())  # should not raise

    def test_failing_hook_logs_exception(self, caplog: pytest.LogCaptureFixture):
        hooks = Hooks()

        def bad_hook(tool_result: ToolResult) -> None:
            raise ZeroDivisionError

        hooks.on_tool_result(bad_hook)
        with caplog.at_level(logging.ERROR, logger="hooks"):
            hooks.trigger_tool_result(_make_tool_result())
        assert "after_tool_call" in caplog.text

    def test_failure_does_not_stop_subsequent_hooks(self):
        hooks = Hooks()
        called: list[bool] = []

        def bad_hook(tool_result: ToolResult) -> None:
            raise ZeroDivisionError

        def good_hook(tool_result: ToolResult) -> None:
            called.append(True)

        hooks.on_tool_result(bad_hook)
        hooks.on_tool_result(good_hook)
        hooks.trigger_tool_result(_make_tool_result())
        assert called == [True]
