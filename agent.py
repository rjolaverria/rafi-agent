import json
import os

from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
)

from tools import AgentTool

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

_client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


class Agent:
    def __init__(
        self,
        model: str,
        agent_tools: list[type[AgentTool]],
        *,
        system_prompt: str | None = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self._tools_map = {cls.tool_name(): cls for cls in agent_tools}
        self._tools_schema = [cls.to_json_schema() for cls in agent_tools]

    def run(self, messages: list[ChatCompletionMessageParam]):
        if self.system_prompt:
            messages = [
                ChatCompletionSystemMessageParam(
                    role="system", content=self.system_prompt
                ),
                *messages,
            ]
        while True:
            chat = _client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._tools_schema,
            )

            if not chat.choices or len(chat.choices) == 0:
                raise RuntimeError("no choices in response")

            choice = chat.choices[0]
            message = choice.message

            fn_tool_calls = [
                tc
                for tc in (message.tool_calls or [])
                if isinstance(tc, ChatCompletionMessageFunctionToolCall)
            ]

            if not fn_tool_calls:
                print(message.content)
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
                return "Invalid tool"
            tool = tool_cls(**args)
            result = tool.execute()
            return result.result
        except json.JSONDecodeError:
            return "An error occurred while parsing the arguments"
        except Exception as e:
            return f"There was an error calling the requested tool: {e}"
