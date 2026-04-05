import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from state import AgentState
from sub_agents import RunSubAgents, SUB_AGENT_SYSTEM_PROMPT
from tools import AgentTool, ToolResult


@pytest.fixture()
def state():
    return AgentState()


# ── Schema & metadata ──────────────────────────────────────────────────────


class TestRunSubAgentsSchema:
    def test_tool_name(self):
        assert RunSubAgents.tool_name() == "runsubagents"

    def test_is_agent_tool(self):
        assert issubclass(RunSubAgents, AgentTool)

    def test_schema_has_instructions_field(self):
        schema = json.loads(json.dumps(RunSubAgents.to_json_schema()))
        params = schema["function"]["parameters"]
        assert "instructions" in params["properties"]
        assert params["properties"]["instructions"]["type"] == "array"

    def test_schema_description(self):
        schema = json.loads(json.dumps(RunSubAgents.to_json_schema()))
        assert "parallel" in schema["function"]["description"].lower()


# ── run_sub_agent ──────────────────────────────────────────────────────────


class TestRunSubAgent:
    @pytest.mark.asyncio
    async def test_returns_query_and_results(self, state):
        tool = RunSubAgents(state=state, instructions=["test query"])

        mock_agent = AsyncMock()

        async def fake_run(messages):
            pass

        mock_agent.run = fake_run

        with patch("sub_agents.Agent") as MockAgent:
            MockAgent.return_value = mock_agent
            result = await tool.run_sub_agent(0, "test query")

        assert result["query"] == "test query"
        assert "results" in result

    @pytest.mark.asyncio
    async def test_passes_query_as_user_message(self, state):
        tool = RunSubAgents(state=state, instructions=["find files"])

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock()

        with patch("sub_agents.Agent", return_value=mock_agent):
            await tool.run_sub_agent(0, "find files")

        mock_agent.run.assert_called_once_with(
            [{"role": "user", "content": "find files"}]
        )

    @pytest.mark.asyncio
    async def test_sub_agent_uses_correct_system_prompt(self, state):
        tool = RunSubAgents(state=state, instructions=["q"])

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock()

        with patch("sub_agents.Agent", return_value=mock_agent) as MockAgent:
            await tool.run_sub_agent(0, "q")

        _, kwargs = MockAgent.call_args
        assert kwargs["system_prompt"] == SUB_AGENT_SYSTEM_PROMPT

    @pytest.mark.asyncio
    async def test_collects_response_content(self, state):
        tool = RunSubAgents(state=state, instructions=["q"])

        captured_callback = None

        def capture_agent(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs["on_response"]
            agent = AsyncMock()
            agent.run = AsyncMock()
            return agent

        with patch("sub_agents.Agent", side_effect=capture_agent):
            result = await tool.run_sub_agent(0, "q")

        assert result["results"] == ""

    @pytest.mark.asyncio
    async def test_on_response_callback_appends_content(self, state):
        tool = RunSubAgents(state=state, instructions=["q"])

        captured_callback = None

        def capture_agent(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs["on_response"]
            agent = AsyncMock()

            async def fake_run(messages):
                assert captured_callback is not None
                msg1 = MagicMock()
                msg1.content = "hello "
                captured_callback(msg1)
                msg2 = MagicMock()
                msg2.content = "world"
                captured_callback(msg2)

            agent.run = fake_run
            return agent

        with patch("sub_agents.Agent", side_effect=capture_agent):
            result = await tool.run_sub_agent(0, "q")

        assert result["results"] == "hello world"

    @pytest.mark.asyncio
    async def test_on_response_handles_none_content(self, state):
        tool = RunSubAgents(state=state, instructions=["q"])

        captured_callback = None

        def capture_agent(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs["on_response"]
            agent = AsyncMock()

            async def fake_run(messages):
                assert captured_callback is not None
                msg = MagicMock()
                msg.content = None
                captured_callback(msg)

            agent.run = fake_run
            return agent

        with patch("sub_agents.Agent", side_effect=capture_agent):
            result = await tool.run_sub_agent(0, "q")

        assert result["results"] == ""


# ── run_sub_agents (parallel) ──────────────────────────────────────────────


class TestRunSubAgentsParallel:
    @pytest.mark.asyncio
    async def test_runs_multiple_queries(self, state):
        tool = RunSubAgents(state=state, instructions=["q1", "q2", "q3"])

        def make_agent(*args, **kwargs):
            agent = AsyncMock()
            agent.run = AsyncMock()
            return agent

        with patch("sub_agents.Agent", side_effect=make_agent):
            results = await tool.run_sub_agents(["q1", "q2", "q3"])

        assert len(results) == 3
        assert {r["query"] for r in results} == {"q1", "q2", "q3"}

    @pytest.mark.asyncio
    async def test_empty_instructions(self, state):
        tool = RunSubAgents(state=state, instructions=[])
        results = await tool.run_sub_agents([])
        assert results == []


# ── execute ────────────────────────────────────────────────────────────────


class TestRunSubAgentsExecute:
    @pytest.mark.asyncio
    async def test_returns_tool_result(self, state):
        tool = RunSubAgents(state=state, instructions=["q1"])

        def make_agent(*args, **kwargs):
            agent = AsyncMock()
            agent.run = AsyncMock()
            return agent

        with (
            patch("sub_agents.Agent", side_effect=make_agent),
            patch("sub_agents.SubAgentProgressDisplay") as MockDisplay,
        ):
            MockDisplay.return_value = MagicMock()
            result = await tool.execute()

        assert isinstance(result, ToolResult)
        assert not result.error
        assert result.name == "runsubagents"

    @pytest.mark.asyncio
    async def test_formats_multiple_results(self, state):
        tool = RunSubAgents(state=state, instructions=["q1", "q2"])

        call_count = 0

        def make_agent(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            cb = kwargs["on_response"]
            agent = AsyncMock()

            idx = call_count

            async def fake_run(messages):
                msg = MagicMock()
                msg.content = f"answer {idx}"
                cb(msg)

            agent.run = fake_run
            return agent

        with (
            patch("sub_agents.Agent", side_effect=make_agent),
            patch("sub_agents.SubAgentProgressDisplay") as MockDisplay,
        ):
            mock_display = MagicMock()
            mock_display.make_hooks.return_value = (
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )
            mock_display._statuses = [MagicMock(), MagicMock()]
            MockDisplay.return_value = mock_display
            result = await tool.execute()

        assert "---" in result.result
        assert "**q1**" in result.result
        assert "**q2**" in result.result

    @pytest.mark.asyncio
    async def test_single_instruction_no_separator(self, state):
        tool = RunSubAgents(state=state, instructions=["only one"])

        def make_agent(*args, **kwargs):
            agent = AsyncMock()
            agent.run = AsyncMock()
            return agent

        with (
            patch("sub_agents.Agent", side_effect=make_agent),
            patch("sub_agents.SubAgentProgressDisplay") as MockDisplay,
        ):
            mock_display = MagicMock()
            mock_display.make_hooks.return_value = (
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )
            mock_display._statuses = [MagicMock()]
            MockDisplay.return_value = mock_display
            result = await tool.execute()

        assert "---" not in result.result
        assert "**only one**" in result.result

    @pytest.mark.asyncio
    async def test_display_lifecycle(self, state):
        """Verify display.start() and display.stop() are called around execution."""
        tool = RunSubAgents(state=state, instructions=["q1"])

        def make_agent(*args, **kwargs):
            agent = AsyncMock()
            agent.run = AsyncMock()
            return agent

        with (
            patch("sub_agents.Agent", side_effect=make_agent),
            patch("sub_agents.SubAgentProgressDisplay") as MockDisplay,
        ):
            mock_display = MagicMock()
            mock_display.make_hooks.return_value = (
                MagicMock(),
                MagicMock(),
                MagicMock(),
            )
            mock_display._statuses = [MagicMock()]
            MockDisplay.return_value = mock_display
            await tool.execute()

        mock_display.start.assert_called_once()
        mock_display.stop.assert_called_once()
