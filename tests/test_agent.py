import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function

from agent import Agent, _ensure_list
from model import Model
from tools import AgentTool, ToolResult


# ── Helpers ──────────────────────────────────────────────────────────────────


class FakeTool(AgentTool):
    """A fake tool for testing."""

    value: str = "default"

    def execute(self) -> ToolResult:
        return ToolResult(
            error=False, name=self.tool_name(), result=f"got:{self.value}"
        )


class FailingTool(AgentTool):
    """A tool that always raises."""

    def execute(self) -> ToolResult:
        raise RuntimeError("tool blew up")


@pytest.fixture()
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def model(mock_client: MagicMock) -> Model:
    return Model(name="test-model", client=mock_client)


def _make_completion(
    content: str = "done",
    tool_calls: list[ChatCompletionMessageToolCallUnion] | None = None,
) -> ChatCompletion:
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    choice = Choice(finish_reason="stop", index=0, message=message)
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[choice],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def _make_fn_tool_call(
    name: str, args: dict[str, Any], call_id: str = "call_1"
) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=json.dumps(args)),
    )


# ── _ensure_list ─────────────────────────────────────────────────────────────


class TestEnsureList:
    def test_none_returns_empty_list(self):
        assert _ensure_list(None) == []

    def test_single_value_returns_list(self):
        assert _ensure_list(42) == [42]

    def test_list_returned_as_is(self):
        original = [1, 2, 3]
        assert _ensure_list(original) is original

    def test_empty_list_returned_as_is(self):
        original: list = []
        assert _ensure_list(original) is original

    def test_single_string(self):
        assert _ensure_list("hello") == ["hello"]


# ── Agent.__init__ ───────────────────────────────────────────────────────────


class TestAgentInit:
    def test_tools_map_built_from_classes(self, model: Model):
        agent = Agent(model, agent_tools=[FakeTool])
        assert "faketool" in agent._tools_map
        assert agent._tools_map["faketool"] is FakeTool

    def test_tools_schema_generated(self, model: Model):
        agent = Agent(model, agent_tools=[FakeTool])
        names = [s["function"]["name"] for s in agent._tools_schema]
        assert "faketool" in names
        assert "readtodos" in names
        assert "modifytodos" in names
        assert len(agent._tools_schema) == 3

    def test_multiple_tools(self, model: Model):
        agent = Agent(model, agent_tools=[FakeTool, FailingTool])
        assert (
            len(agent._tools_map) == 4
        )  # FakeTool + FailingTool + ReadTodos + ModifyTodos
        assert len(agent._tools_schema) == 4

    def test_system_prompt_stored(self, model: Model):
        agent = Agent(model, agent_tools=[], system_prompt="Be helpful")
        assert agent.system_prompt == "Be helpful"

    def test_no_system_prompt(self, model: Model):
        agent = Agent(model, agent_tools=[])
        assert agent.system_prompt is None

    def test_hooks_initialized_empty_by_default(self, model: Model):
        agent = Agent(model, agent_tools=[])
        assert agent.hooks.after_response == []
        assert agent.hooks.before_tool_call == []
        assert agent.hooks.after_tool_call == []

    def test_on_response_single_hook(self, model: Model):
        hook = MagicMock()
        agent = Agent(model, agent_tools=[], on_response=hook)
        assert hook in agent.hooks.after_response

    def test_on_response_list_of_hooks(self, model: Model):
        h1, h2 = MagicMock(), MagicMock()
        agent = Agent(model, agent_tools=[], on_response=[h1, h2])
        assert agent.hooks.after_response == [h1, h2]

    def test_on_tool_call_single_hook(self, model: Model):
        hook = MagicMock()
        agent = Agent(model, agent_tools=[], on_tool_call=hook)
        assert hook in agent.hooks.before_tool_call

    def test_on_tool_result_single_hook(self, model: Model):
        hook = MagicMock()
        agent = Agent(model, agent_tools=[], on_tool_result=hook)
        assert hook in agent.hooks.after_tool_call


# ── Agent.run ────────────────────────────────────────────────────────────────


class TestAgentRun:
    def test_returns_on_no_tool_calls(self, model: Model, mock_client: MagicMock):
        mock_client.chat.completions.create.return_value = _make_completion(
            content="hello"
        )
        agent = Agent(model, agent_tools=[FakeTool])
        agent.run([{"role": "user", "content": "hi"}])
        mock_client.chat.completions.create.assert_called_once()

    def test_raises_on_empty_choices(self, model: Model, mock_client: MagicMock):
        completion = _make_completion()
        completion.choices = []
        mock_client.chat.completions.create.return_value = completion
        agent = Agent(model, agent_tools=[])
        with pytest.raises(RuntimeError, match="no choices"):
            agent.run([{"role": "user", "content": "hi"}])

    def test_system_prompt_prepended(self, model: Model, mock_client: MagicMock):
        mock_client.chat.completions.create.return_value = _make_completion()
        agent = Agent(model, agent_tools=[], system_prompt="You are helpful")
        agent.run([{"role": "user", "content": "hi"}])
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "You are helpful" in messages[0]["content"]
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_no_system_prompt_not_prepended(self, model: Model, mock_client: MagicMock):
        mock_client.chat.completions.create.return_value = _make_completion()
        agent = Agent(model, agent_tools=[])
        agent.run([{"role": "user", "content": "hi"}])
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "You are helpful" not in messages[0]["content"]
        assert messages[1] == {"role": "user", "content": "hi"}

    def test_executes_tool_and_loops(self, model: Model, mock_client: MagicMock):
        tc = _make_fn_tool_call("faketool", {"value": "abc"})
        first_response = _make_completion(content="calling tool", tool_calls=[tc])
        second_response = _make_completion(content="all done")
        mock_client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]
        agent = Agent(model, agent_tools=[FakeTool])
        agent.run([{"role": "user", "content": "do it"}])
        assert mock_client.chat.completions.create.call_count == 2

    def test_tool_result_appended_to_messages(
        self, model: Model, mock_client: MagicMock
    ):
        tc = _make_fn_tool_call("faketool", {"value": "test"})
        first_response = _make_completion(content="calling", tool_calls=[tc])
        second_response = _make_completion(content="done")
        mock_client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]
        agent = Agent(model, agent_tools=[FakeTool])
        agent.run([{"role": "user", "content": "go"}])
        second_call_messages = mock_client.chat.completions.create.call_args_list[
            1
        ].kwargs["messages"]
        tool_msg = second_call_messages[-1]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"
        assert tool_msg["content"] == "got:test"

    def test_assistant_message_appended_with_tool_calls(
        self, model: Model, mock_client: MagicMock
    ):
        tc = _make_fn_tool_call("faketool", {"value": "x"})
        first_response = _make_completion(content="thinking", tool_calls=[tc])
        second_response = _make_completion(content="done")
        mock_client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]
        agent = Agent(model, agent_tools=[FakeTool])
        agent.run([{"role": "user", "content": "go"}])
        second_call_messages = mock_client.chat.completions.create.call_args_list[
            1
        ].kwargs["messages"]
        assistant_msg = second_call_messages[2]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] == "thinking"
        assert len(assistant_msg["tool_calls"]) == 1
        assert assistant_msg["tool_calls"][0]["id"] == "call_1"
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "faketool"

    def test_multiple_tool_calls_in_single_response(
        self, model: Model, mock_client: MagicMock
    ):
        tc1 = _make_fn_tool_call("faketool", {"value": "a"}, call_id="call_1")
        tc2 = _make_fn_tool_call("faketool", {"value": "b"}, call_id="call_2")
        first_response = _make_completion(content="both", tool_calls=[tc1, tc2])
        second_response = _make_completion(content="done")
        mock_client.chat.completions.create.side_effect = [
            first_response,
            second_response,
        ]
        agent = Agent(model, agent_tools=[FakeTool])
        agent.run([{"role": "user", "content": "go"}])
        second_call_messages = mock_client.chat.completions.create.call_args_list[
            1
        ].kwargs["messages"]
        tool_msgs = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["content"] == "got:a"
        assert tool_msgs[1]["content"] == "got:b"

    def test_tools_schema_passed_to_api(self, model: Model, mock_client: MagicMock):
        mock_client.chat.completions.create.return_value = _make_completion()
        agent = Agent(model, agent_tools=[FakeTool])
        agent.run([{"role": "user", "content": "hi"}])
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["tools"] == agent._tools_schema

    def test_model_name_passed_to_api(self, model: Model, mock_client: MagicMock):
        mock_client.chat.completions.create.return_value = _make_completion()
        agent = Agent(model, agent_tools=[])
        agent.run([{"role": "user", "content": "hi"}])
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs["model"] == "test-model"

    def test_iterations_reset_between_runs(self, model: Model, mock_client: MagicMock):
        tc = _make_fn_tool_call("faketool", {"value": "a"})
        mock_client.chat.completions.create.side_effect = [
            _make_completion(content="call", tool_calls=[tc]),
            _make_completion(content="done"),
        ]
        agent = Agent(model, agent_tools=[FakeTool])
        agent.run([{"role": "user", "content": "first"}])
        assert agent.state.iterations == 2

        mock_client.chat.completions.create.side_effect = [
            _make_completion(content="done"),
        ]
        agent.run([{"role": "user", "content": "second"}])
        assert agent.state.iterations == 1


# ── Agent._execute_tool ─────────────────────────────────────────────────────


class TestExecuteTool:
    def test_executes_known_tool(self, model: Model):
        agent = Agent(model, agent_tools=[FakeTool])
        result = agent._execute_tool("faketool", json.dumps({"value": "hi"}))
        assert result == "got:hi"

    def test_unknown_tool_returns_error(self, model: Model):
        agent = Agent(model, agent_tools=[FakeTool])
        result = agent._execute_tool("nope", "{}")
        assert "Unknown tool" in result
        assert "nope" in result

    def test_invalid_json_returns_error(self, model: Model):
        agent = Agent(model, agent_tools=[FakeTool])
        result = agent._execute_tool("faketool", "not json")
        assert "not valid JSON" in result

    def test_tool_execution_error_returns_error(self, model: Model):
        agent = Agent(model, agent_tools=[FailingTool])
        result = agent._execute_tool("failingtool", "{}")
        assert "error" in result.lower()
        assert "tool blew up" in result

    def test_triggers_tool_call_hook(self, model: Model):
        hook = MagicMock()
        agent = Agent(model, agent_tools=[FakeTool], on_tool_call=hook)
        agent._execute_tool("faketool", json.dumps({"value": "x"}))
        hook.assert_called_once_with("faketool", {"value": "x"})

    def test_triggers_tool_result_hook(self, model: Model):
        hook = MagicMock()
        agent = Agent(model, agent_tools=[FakeTool], on_tool_result=hook)
        agent._execute_tool("faketool", json.dumps({"value": "x"}))
        hook.assert_called_once()
        result_arg = hook.call_args[0][0]
        assert isinstance(result_arg, ToolResult)
        assert result_arg.result == "got:x"

    def test_hooks_not_called_for_unknown_tool(self, model: Model):
        tool_call_hook = MagicMock()
        tool_result_hook = MagicMock()
        agent = Agent(
            model,
            agent_tools=[FakeTool],
            on_tool_call=tool_call_hook,
            on_tool_result=tool_result_hook,
        )
        agent._execute_tool("unknown", "{}")
        tool_call_hook.assert_not_called()
        tool_result_hook.assert_not_called()

    def test_uses_default_field_value(self, model: Model):
        agent = Agent(model, agent_tools=[FakeTool])
        result = agent._execute_tool("faketool", "{}")
        assert result == "got:default"


# ── Agent.run hooks integration ──────────────────────────────────────────────


class TestAgentRunHooks:
    def test_response_hook_called(self, model: Model, mock_client: MagicMock):
        mock_client.chat.completions.create.return_value = _make_completion(
            content="hi"
        )
        hook = MagicMock()
        agent = Agent(model, agent_tools=[], on_response=hook)
        agent.run([{"role": "user", "content": "hello"}])
        hook.assert_called_once()
        msg = hook.call_args[0][0]
        assert msg.content == "hi"

    def test_response_hook_called_each_iteration(
        self, model: Model, mock_client: MagicMock
    ):
        tc = _make_fn_tool_call("faketool", {"value": "a"})
        mock_client.chat.completions.create.side_effect = [
            _make_completion(content="step1", tool_calls=[tc]),
            _make_completion(content="step2"),
        ]
        hook = MagicMock()
        agent = Agent(model, agent_tools=[FakeTool], on_response=hook)
        agent.run([{"role": "user", "content": "go"}])
        assert hook.call_count == 2
