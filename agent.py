import inspect
import json
from typing import TYPE_CHECKING

from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolParam,
)
from pydantic import ValidationError

from hooks import Hooks, ResponseHook, ToolCallHook, ToolResultHook
from model import Model
from state import AgentState
from tools import AgentTool, ModifyTodos, ReadTodos, ToolResult

if TYPE_CHECKING:
    from skills import Skill

INTERNAL_SYSTEM_INSTRUCTIONS = """
**IMPORTANT**: Always use todos to track progress and they must add todos before performing any actions. Do not make any assumptions about the state of the todos - always read the current list before modifying it. Each todo should be checked off when completed, and the final response should include a summary of what was accomplished and the state of the todos.
""".strip()

SKILLS_INSTRUCTIONS = "**SKILLS**: Call `listskills` to discover available tools grouped as skills. Use `useskill` to activate a skill before calling its tools. Only activate the skills you actually need."


def _ensure_list[T](value: list[T] | T | None) -> list[T]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


class Agent:
    def __init__(
        self,
        model: Model,
        *,
        agent_tools: list[type[AgentTool]],
        skills: "dict[str, Skill] | None" = None,
        system_prompt: str | None = None,
        on_response: list[ResponseHook] | ResponseHook | None = None,
        on_tool_call: list[ToolCallHook] | ToolCallHook | None = None,
        on_tool_result: list[ToolResultHook] | ToolResultHook | None = None,
        max_iterations: int = 15,
    ):
        self.model: Model = model
        self.system_prompt = system_prompt
        self._skills: dict[str, Skill] = skills or {}

        skill_meta_tools: list[type[AgentTool]] = []
        if self._skills:
            from skills import ListSkills, UseSkill

            skill_meta_tools = [ListSkills, UseSkill]

        self._base_tools = agent_tools + [ReadTodos, ModifyTodos] + skill_meta_tools

        all_skill_tools = [t for s in self._skills.values() for t in s.tools]
        self._tools_map = {
            cls.tool_name(): cls for cls in self._base_tools + all_skill_tools
        }
        self._base_tools_schema = [cls.to_json_schema() for cls in self._base_tools]
        self.max_iterations = max_iterations
        self.state = AgentState(skills_registry=self._skills)
        self.hooks = Hooks(
            after_response=_ensure_list(on_response),
            before_tool_call=_ensure_list(on_tool_call),
            after_tool_call=_ensure_list(on_tool_result),
        )

    @property
    def _tools_schema(self) -> list[ChatCompletionToolParam]:
        active_skill_schemas = [
            t.to_json_schema()
            for skill_name, skill in self._skills.items()
            if skill_name in self.state.activated_skills
            for t in skill.tools
        ]
        return self._base_tools_schema + active_skill_schemas

    def _build_system_prompt(self) -> str:
        parts = [INTERNAL_SYSTEM_INSTRUCTIONS]
        if self._skills:
            parts.append(SKILLS_INSTRUCTIONS)
        if self.system_prompt:
            parts.append(self.system_prompt)
        return "\n".join(parts)

    async def run(self, messages: list[ChatCompletionMessageParam]):
        messages = [
            ChatCompletionSystemMessageParam(
                role="system",
                content=self._build_system_prompt(),
            ),
            *messages,
        ]
        self.state.iterations = 0
        while True:
            self.state.iterations += 1
            tools = self._tools_schema

            if self.state.iterations >= self.max_iterations:
                tools = []
                messages.append(
                    {
                        "role": "user",
                        "content": f"Maximum iterations of {self.max_iterations} reached. Stopping.",
                    }
                )

            chat = await self.model.client.chat.completions.create(
                model=self.model.name,
                messages=messages,
                tools=tools,
            )

            if not chat.choices:
                raise RuntimeError("no choices in response")

            message = chat.choices[0].message
            self.hooks.trigger_response(message)

            fn_tool_calls = [
                tc
                for tc in (message.tool_calls or [])
                if isinstance(tc, ChatCompletionMessageFunctionToolCall)
            ]

            if not fn_tool_calls:
                return

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in fn_tool_calls
                    ],
                }
            )

            for tool_call in fn_tool_calls:
                result = await self._execute_tool(
                    tool_call.function.name, tool_call.function.arguments
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                )

    async def _execute_tool(self, tool_name: str, args_str: str) -> str:
        try:
            args = json.loads(args_str)
            tool_cls = self._tools_map.get(tool_name)

            if not tool_cls:
                return f"Unknown tool '{tool_name}'. Available: {list(self._tools_map)}"

            tool = tool_cls.model_validate({**args, "state": self.state})
            self.hooks.trigger_tool_call(tool_name, args)
            raw = tool.execute()
            result: ToolResult = await raw if inspect.isawaitable(raw) else raw  # type: ignore[assignment]
            self.hooks.trigger_tool_result(result)
            return result.result
        except json.JSONDecodeError:
            return f"Invalid arguments format for tool '{tool_name}' - not valid JSON: {args_str}"
        except ValidationError as ve:
            return f"Invalid arguments for tool '{tool_name}': {ve}"
        except Exception as e:
            return f"There was an error calling the requested tool: {e}"
