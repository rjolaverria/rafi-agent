import inspect
import json
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)
from pydantic import ValidationError

from hooks import Hooks, ResponseHook, ToolCallHook, ToolResultHook
from model import Model
from skills import format_skills_for_prompt
from state import AgentState, Skill
from tools import AgentTool, ModifyTodos, ReadTodos, ToolResult, UseSkill

INTERNAL_SYSTEM_INSTRUCTIONS = """
**IMPORTANT**: Always use todos to track progress and they must add todos before performing any actions. Do not make any assumptions about the state of the todos - always read the current list before modifying it. Each todo should be checked off when completed, and the final response should include a summary of what was accomplished and the state of the todos.
""".strip()


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
        skills: list[Skill] | None = None,
        system_prompt: str | None = None,
        on_response: list[ResponseHook] | ResponseHook | None = None,
        on_tool_call: list[ToolCallHook] | ToolCallHook | None = None,
        on_tool_result: list[ToolResultHook] | ToolResultHook | None = None,
        max_iterations: int = 15,
    ):
        self.model: Model = model
        self.system_prompt = system_prompt
        self._skills: list[Skill] = skills or []
        skill_tools: list[type[AgentTool]] = [UseSkill] if self._skills else []
        self._tools = agent_tools + [ReadTodos, ModifyTodos] + skill_tools
        self._tools_map = {cls.tool_name(): cls for cls in self._tools}
        self._tools_schema = [cls.to_json_schema() for cls in self._tools]
        self.max_iterations = max_iterations
        self.state = AgentState(skills_registry={s.name: s for s in self._skills})
        self.hooks = Hooks(
            after_response=_ensure_list(on_response),
            before_tool_call=_ensure_list(on_tool_call),
            after_tool_call=_ensure_list(on_tool_result),
        )

    def _build_system_prompt(self) -> str:
        parts = [INTERNAL_SYSTEM_INSTRUCTIONS]
        skills_section = format_skills_for_prompt(self._skills)
        if skills_section:
            parts.append(skills_section)
        if self.system_prompt:
            parts.append(self.system_prompt)
        return "\n\n".join(parts)

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
