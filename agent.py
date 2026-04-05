from pydantic import ValidationError
from state import State
import json

from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)

from hooks import Hooks, ResponseHook, ToolCallHook, ToolResultHook
from model import Model
from tools import AgentTool


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
        system_prompt: str | None = None,
        on_response: list[ResponseHook] | ResponseHook | None = None,
        on_tool_call: list[ToolCallHook] | ToolCallHook | None = None,
        on_tool_result: list[ToolResultHook] | ToolResultHook | None = None,
        max_iterations: int = 15,
    ):
        self.model: Model = model
        self.system_prompt = system_prompt
        self._tools_map = {cls.tool_name(): cls for cls in agent_tools}
        self._tools_schema = [cls.to_json_schema() for cls in agent_tools]
        self.max_iterations = max_iterations
        self.state = State()
        self.hooks = Hooks(
            after_response=_ensure_list(on_response),
            before_tool_call=_ensure_list(on_tool_call),
            after_tool_call=_ensure_list(on_tool_result),
        )

    def run(self, messages: list[ChatCompletionMessageParam]):
        if self.system_prompt:
            messages = [
                ChatCompletionSystemMessageParam(
                    role="system", content=self.system_prompt
                ),
                *messages,
            ]
        while True:
            self.state.iterations += 1
            tools = self._tools_schema

            if self.state.iterations >= self.max_iterations:
                tools = []
                messages.append(
                    {
                        "role": "system",
                        "content": f"Maximum iterations of {self.max_iterations} reached. Stopping.",
                    }
                )

            chat = self.model.client.chat.completions.create(
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
                result = self._execute_tool(
                    tool_call.function.name, tool_call.function.arguments
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tool_call.id, "content": result}
                )

    def _execute_tool(self, tool_name: str, args_str: str) -> str:
        try:
            args = json.loads(args_str)
            tool_cls = self._tools_map.get(tool_name)

            if not tool_cls:
                return f"Unknown tool '{tool_name}'. Available: {list(self._tools_map)}"

            tool = tool_cls.model_validate(args)
            self.hooks.trigger_tool_call(tool_name, args)
            result = tool.execute()
            self.hooks.trigger_tool_result(result)
            return result.result
        except json.JSONDecodeError:
            return f"Invalid arguments format for tool '{tool_name}' - not valid JSON: {args_str}"
        except ValidationError as ve:
            return f"Invalid arguments for tool '{tool_name}': {ve}"
        except Exception as e:
            return f"There was an error calling the requested tool: {e}"
