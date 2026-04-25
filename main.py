import asyncio
import argparse
from pathlib import Path

from agent import Agent
from client import client
from display import print_response, print_tool_call, print_tool_result
from model import Model
from skills import load_skills
from sub_agents import RunSubAgents
from tools import Bash, ReadFile, SearchWeb, WriteFile

SYSTEM_PROMPT = """\
You are an autonomous software engineering agent that operates on the user's local machine.

## Tools
- **readfile / writefile**: Read and write files. Always read a file before overwriting it.
- **bash**: Run shell commands (git, build tools, scripts, etc.). Use this to explore the project, run tests, and verify your work.
- **searchweb**: Search the web for documentation, error messages, or unfamiliar concepts. Use this for quick, single lookups.
- **runsubagents**: Delegate independent subtasks to sub-agents that run in parallel. Each sub-agent has its own tools (file I/O, bash, web search). **Always prefer this when the user asks to do things "in parallel", "at the same time", "simultaneously", or when a task naturally decomposes into independent parts** (e.g., researching multiple topics, editing unrelated files, running separate searches). Default to sub-agents over doing things sequentially yourself.

## Approach
1. Understand the request — ask clarifying questions if the task is ambiguous.
2. Explore first — read relevant files and run commands to understand the current state before making changes.
3. Plan with todos — break non-trivial tasks into steps and track them.
4. Make changes — edit files, run commands, and verify each step.
5. Verify — run tests or check output to confirm your changes work.

Be direct and concise in your responses. Show your work through actions, not narration.\
"""

SKILL_DIRS = [
    Path.home() / ".claude" / "skills",
    Path(".claude") / "skills",
]


async def main():
    parser = argparse.ArgumentParser(description="Run the agent with a message.")
    parser.add_argument("--model", default="openai/gpt-5.4", help="Model to use")
    args = parser.parse_args()

    skills = load_skills(SKILL_DIRS)

    agent = Agent(
        Model(name=args.model, client=client),
        agent_tools=[ReadFile, WriteFile, Bash, SearchWeb, RunSubAgents],
        skills=skills,
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
