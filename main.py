import argparse

from agent import Agent
from display import print_response, print_tool_call, print_tool_result
from tools import Bash, ReadFile, WriteFile

SYSTEM_PROMPT = (
    "You are a helpful computer use agent. You can read and write files, "
    "and execute bash commands to accomplish tasks on the user's behalf."
)


def main():
    parser = argparse.ArgumentParser(description="Run the agent with a message.")
    parser.add_argument("--model", default="openai/gpt-4.1", help="Model to use")
    args = parser.parse_args()

    agent = Agent(
        args.model,
        agent_tools=[ReadFile, WriteFile, Bash],
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
