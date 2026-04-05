import asyncio
from model import Model
from agent import Agent
from client import client
from pydantic import Field
from tools import AgentTool, ReadFile, WriteFile, Bash, SearchWeb, ToolResult

SUB_AGENT_SYSTEM_PROMPT = (
    "You are a helpful sub-agent that can perform specific tasks to assist the main agent. "
    "You can read and write files, execute bash commands, and search the web to accomplish your assigned task."
)


class RunSubAgents(AgentTool):
    """Runs a list of sub-agents in parallel with the provided messages"""

    instructions: list[str] = Field(
        description="The list of instructions for each sub-agent. Each instruction will be given to a separate sub-agent as the user message."
    )

    async def run_sub_agent(self, query: str) -> dict[str, str]:
        results = []
        sub_agent = Agent(
            Model(name="openai/gpt-5.4-mini", client=client),
            agent_tools=[ReadFile, WriteFile, Bash, SearchWeb],
            system_prompt=SUB_AGENT_SYSTEM_PROMPT,
            on_response=lambda message: results.append(message.content or ""),
        )

        await sub_agent.run([{"role": "user", "content": query}])

        return {
            "query": query,
            "results": "".join(results),
        }

    async def run_sub_agents(self, queries: list[str]):
        tasks = [asyncio.create_task((self.run_sub_agent(query))) for query in queries]
        results = await asyncio.gather(*tasks)
        return results

    async def execute(self) -> ToolResult:
        results = await self.run_sub_agents(self.instructions)
        formatted = "\n\n---\n\n".join(
            f"**{r['query']}**\n{r['results']}" for r in results
        )
        return ToolResult(error=False, name=self.tool_name(), result=formatted)
