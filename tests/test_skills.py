import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallUnion,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function

from agent import Agent
from model import Model
from skills import SKILLS, ListSkills, Skill, UseSkill
from state import AgentState
from tools import Bash, ReadFile, ToolResult, WriteFile


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_state() -> AgentState:
    return AgentState()


def _make_completion(
    content: str = "done",
    tool_calls: list[ChatCompletionMessageToolCallUnion] | None = None,
) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls)
    choice = Choice(finish_reason="stop", index=0, message=message)
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[choice],
        created=0,
        model="test-model",
        object="chat.completion",
    )


def _make_fn_tool_call(
    name: str, args: dict, call_id: str = "call_1"
) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=json.dumps(args)),
    )


@pytest.fixture()
def mock_client() -> MagicMock:
    c = MagicMock()
    c.chat.completions.create = AsyncMock()
    return c


@pytest.fixture()
def model(mock_client: MagicMock) -> Model:
    return Model(name="test-model", client=mock_client)


# ── Skill dataclass ───────────────────────────────────────────────────────────


class TestSkill:
    def test_skill_has_name_description_tools(self):
        skill = Skill(name="fs", description="File ops", tools=[ReadFile, WriteFile])
        assert skill.name == "fs"
        assert skill.description == "File ops"
        assert skill.tools == [ReadFile, WriteFile]


# ── SKILLS registry ───────────────────────────────────────────────────────────


class TestSkillsRegistry:
    def test_default_skills_present(self):
        assert "filesystem" in SKILLS
        assert "shell" in SKILLS
        assert "web" in SKILLS

    def test_filesystem_skill_tools(self):
        assert ReadFile in SKILLS["filesystem"].tools
        assert WriteFile in SKILLS["filesystem"].tools

    def test_shell_skill_tools(self):
        assert Bash in SKILLS["shell"].tools


# ── ListSkills tool ───────────────────────────────────────────────────────────


class TestListSkills:
    def _make_tool(
        self,
        activated: set[str] | None = None,
        registry: dict | None = None,
    ) -> ListSkills:
        state = _make_state()
        if activated:
            state.activated_skills = activated
        state.skills_registry = registry if registry is not None else dict(SKILLS)
        return ListSkills.model_validate({"state": state})

    def test_returns_available_status_when_no_skills_active(self):
        tool = self._make_tool()
        result = tool.execute()
        assert result.error is False
        assert "[available]" in result.result

    def test_active_skill_shows_active_status(self):
        tool = self._make_tool(activated={"filesystem"})
        result = tool.execute()
        assert "[active]" in result.result
        assert "[available]" in result.result

    def test_lists_tool_names_per_skill(self):
        tool = self._make_tool()
        result = tool.execute()
        assert "readfile" in result.result
        assert "writefile" in result.result
        assert "bash" in result.result

    def test_empty_skills_message(self):
        tool = self._make_tool(registry={})
        result = tool.execute()
        assert result.result == "No skills available."


# ── UseSkill tool ─────────────────────────────────────────────────────────────


class TestUseSkill:
    def _make_tool(self, skill_name: str, activated: set[str] | None = None) -> UseSkill:
        state = _make_state()
        state.skills_registry = dict(SKILLS)
        if activated:
            state.activated_skills = set(activated)
        return UseSkill.model_validate({"skill_name": skill_name, "state": state})

    def test_activates_known_skill(self):
        tool = self._make_tool("filesystem")
        result = tool.execute()
        assert result.error is False
        assert "filesystem" in tool.state.activated_skills

    def test_returns_tool_names_on_activation(self):
        tool = self._make_tool("shell")
        result = tool.execute()
        assert "bash" in result.result

    def test_unknown_skill_returns_error(self):
        tool = self._make_tool("nonexistent")
        result = tool.execute()
        assert result.error is True
        assert "nonexistent" in result.result

    def test_already_active_skill_returns_info_not_error(self):
        tool = self._make_tool("filesystem", activated={"filesystem"})
        result = tool.execute()
        assert result.error is False
        assert "already active" in result.result

    def test_already_active_skill_does_not_duplicate(self):
        tool = self._make_tool("filesystem", activated={"filesystem"})
        tool.execute()
        assert tool.state.activated_skills == {"filesystem"}


# ── Agent with skills ─────────────────────────────────────────────────────────


class TestAgentWithSkills:
    def _make_skills(self) -> dict[str, Skill]:
        from tools import Bash, ReadFile

        return {
            "fs": Skill(name="fs", description="Filesystem", tools=[ReadFile]),
            "sh": Skill(name="sh", description="Shell", tools=[Bash]),
        }

    def test_skill_tools_not_in_initial_schema(self, model: Model):
        skills = self._make_skills()
        agent = Agent(model, agent_tools=[], skills=skills)
        names = [s["function"]["name"] for s in agent._tools_schema]
        assert "readfile" not in names
        assert "bash" not in names

    def test_listskills_and_useskill_always_in_schema(self, model: Model):
        skills = self._make_skills()
        agent = Agent(model, agent_tools=[], skills=skills)
        names = [s["function"]["name"] for s in agent._tools_schema]
        assert "listskills" in names
        assert "useskill" in names

    def test_activated_skill_tools_appear_in_schema(self, model: Model):
        skills = self._make_skills()
        agent = Agent(model, agent_tools=[], skills=skills)
        agent.state.activated_skills.add("fs")
        names = [s["function"]["name"] for s in agent._tools_schema]
        assert "readfile" in names
        assert "bash" not in names

    def test_multiple_activated_skills_all_appear(self, model: Model):
        skills = self._make_skills()
        agent = Agent(model, agent_tools=[], skills=skills)
        agent.state.activated_skills.update({"fs", "sh"})
        names = [s["function"]["name"] for s in agent._tools_schema]
        assert "readfile" in names
        assert "bash" in names

    def test_all_skill_tools_executable_via_tools_map(self, model: Model):
        skills = self._make_skills()
        agent = Agent(model, agent_tools=[], skills=skills)
        assert "readfile" in agent._tools_map
        assert "bash" in agent._tools_map

    def test_no_skills_behaves_like_before(self, model: Model):
        from tests.test_agent import FakeTool

        agent = Agent(model, agent_tools=[FakeTool])
        names = [s["function"]["name"] for s in agent._tools_schema]
        assert "faketool" in names
        assert "listskills" not in names
        assert "useskill" not in names

    def test_skills_system_instructions_in_system_prompt(self, model: Model):
        skills = self._make_skills()
        agent = Agent(model, agent_tools=[], skills=skills)
        prompt = agent._build_system_prompt()
        assert "listskills" in prompt
        assert "useskill" in prompt

    def test_no_skills_instructions_without_skills(self, model: Model):
        agent = Agent(model, agent_tools=[])
        prompt = agent._build_system_prompt()
        assert "listskills" not in prompt

    @pytest.mark.asyncio
    async def test_useskill_during_run_unlocks_tools(
        self, model: Model, mock_client: MagicMock
    ):
        skills = self._make_skills()
        agent = Agent(model, agent_tools=[], skills=skills)

        use_skill_call = _make_fn_tool_call("useskill", {"skill_name": "fs"}, "call_1")
        first = _make_completion(content="activating", tool_calls=[use_skill_call])
        second = _make_completion(content="done")
        mock_client.chat.completions.create.side_effect = [first, second]

        await agent.run([{"role": "user", "content": "read a file"}])

        assert "fs" in agent.state.activated_skills
        second_call_tools = mock_client.chat.completions.create.call_args_list[1].kwargs[
            "tools"
        ]
        tool_names = [t["function"]["name"] for t in second_call_tools]
        assert "readfile" in tool_names
