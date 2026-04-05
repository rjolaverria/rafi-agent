import asyncio
from typing import Any

from model import Model
from agent import Agent
from client import client
from pydantic import Field
from tools import AgentTool, ReadFile, WriteFile, Bash, SearchWeb, ToolResult
from sub_agent_display import SubAgentProgressDisplay

SUB_AGENT_SYSTEM_PROMPT = (
    "You are a helpful sub-agent that can perform specific tasks to assist the main agent. "
    "You can read and write files, execute bash commands, and search the web to accomplish your assigned task."
)


class RunSubAgents(AgentTool):
    """Runs a list of sub-agents in parallel with the provided messages"""

    instructions: list[str] = Field(
        description="The list of instructions for each sub-agent. Each instruction will be given to a separate sub-agent as the user message."
    )

    async def run_sub_agent(
        self,
        index: int,
        query: str,
        display: SubAgentProgressDisplay | None = None,
    ) -> dict[str, str]:
        results: list[str] = []

        hooks_kwargs: dict[str, Any] = {
            "on_response": lambda message: results.append(message.content or ""),
        }

        if display:
            resp_hook, tc_hook, tr_hook = display.make_hooks(index)
            hooks_kwargs["on_response"] = [
                lambda message: results.append(message.content or ""),
                resp_hook,
            ]
            hooks_kwargs["on_tool_call"] = tc_hook
            hooks_kwargs["on_tool_result"] = tr_hook

        sub_agent = Agent(
            Model(name="openai/gpt-5.4-mini", client=client),
            agent_tools=[ReadFile, WriteFile, Bash, SearchWeb],
            system_prompt=SUB_AGENT_SYSTEM_PROMPT,
            **hooks_kwargs,
        )

        await sub_agent.run([{"role": "user", "content": query}])

        if display:
            display._statuses[index].status = "done"
            display._statuses[index].tool_name = None
            display._statuses[index].tool_detail = None

        return {
            "query": query,
            "results": "".join(results),
        }

    async def run_sub_agents(
        self,
        queries: list[str],
        display: SubAgentProgressDisplay | None = None,
    ):
        tasks = [
            asyncio.create_task(self.run_sub_agent(i, query, display))
            for i, query in enumerate(queries)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                if display:
                    display._statuses[i].status = "error"
                processed.append({"query": queries[i], "results": f"Error: {r}"})
            else:
                processed.append(r)
        return processed

    async def execute(self) -> ToolResult:
        from display import console

        display = SubAgentProgressDisplay(self.instructions, console)
        display.start()
        try:
            results = await self.run_sub_agents(self.instructions, display)
        finally:
            display.stop()

        formatted = "\n\n---\n\n".join(
            f"**{r['query']}**\n{r['results']}" for r in results
        )
        return ToolResult(error=False, name=self.tool_name(), result=formatted)
