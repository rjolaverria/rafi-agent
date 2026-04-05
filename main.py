from model import Model
import os
from openai import OpenAI
import argparse

from agent import Agent
from display import print_response, print_tool_call, print_tool_result
from tools import Bash, ReadFile, WriteFile, SearchWeb

SYSTEM_PROMPT = (
    "You are a helpful computer use agent. You can read and write files, "
    "and execute bash commands to accomplish tasks on the user's behalf."
)

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY is not set")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def main():
    parser = argparse.ArgumentParser(description="Run the agent with a message.")
    parser.add_argument("--model", default="openai/gpt-5.4", help="Model to use")
    args = parser.parse_args()

    agent = Agent(
        Model(name=args.model, client=client),
        agent_tools=[ReadFile, WriteFile, Bash, SearchWeb],
        system_prompt=SYSTEM_PROMPT,
        on_response=print_response,
        on_tool_call=print_tool_call,
        on_tool_result=print_tool_result,
    )

    messages = []
    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        agent.run(messages)


if __name__ == "__main__":
    main()
