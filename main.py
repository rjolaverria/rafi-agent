import asyncio
import argparse

from agent import Agent
from client import client
from display import print_response, print_tool_call, print_tool_result
from model import Model
from skills import SKILLS, Skill
from sub_agents import RunSubAgents

SKILLS["parallel"] = Skill(
    name="parallel",
    description="Delegate independent subtasks to sub-agents running in parallel",
    tools=[RunSubAgents],
)

SYSTEM_PROMPT = """\
You are an autonomous software engineering agent that operates on the user's local machine.

## Skills
Skills are bundles of tools activated on demand. Start each task by calling `listskills`, then `useskill` to unlock what you need.

- **filesystem**: Read and write files. Always read before overwriting.
- **shell**: Run shell commands (git, build tools, tests, etc.).
- **web**: Search for documentation, error messages, or unfamiliar concepts.
- **parallel**: Delegate independent subtasks to sub-agents running simultaneously.

## Approach
1. Understand the request — ask clarifying questions if the task is ambiguous.
2. List and activate the skills you'll need.
3. Explore first — read files and run commands to understand the current state.
4. Plan with todos — break non-trivial tasks into steps and track them.
5. Make changes — edit files, run commands, and verify each step.
6. Verify — run tests or check output to confirm changes work.

Be direct and concise. Show your work through actions, not narration.\
"""


async def main():
    parser = argparse.ArgumentParser(description="Run the agent with a message.")
    parser.add_argument("--model", default="openai/gpt-5.4", help="Model to use")
    args = parser.parse_args()

    agent = Agent(
        Model(name=args.model, client=client),
        agent_tools=[],
        skills=SKILLS,
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
        await agent.run(messages)


if __name__ == "__main__":
    asyncio.run(main())
