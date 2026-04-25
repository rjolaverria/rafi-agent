import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from collections.abc import Awaitable
from typing import Any

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

import web_search
from state import AgentState, Skill


class ToolResult(BaseModel):
    error: bool
    name: str
    result: str
    raw: SkipJsonSchema[Any] = None

    def to_message(self, tool_call_id: str) -> dict[str, str]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": self.result}


class AgentTool(BaseModel, ABC):
    state: SkipJsonSchema[AgentState]

    @classmethod
    def tool_name(cls) -> str:
        return cls.__name__.lower()

    @classmethod
    def to_json_schema(cls) -> ChatCompletionToolParam:
        schema = cls.model_json_schema()
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": cls.tool_name(),
                "description": cls.__doc__ or "",
                "parameters": schema,
            },
        }

    @abstractmethod
    def execute(self) -> ToolResult | Awaitable[ToolResult]: ...


class ReadFile(AgentTool):
    """Reads and returns the contents of a file."""

    file_path: str = Field(description="The path to the file to read")

    def execute(self) -> ToolResult:
        path = Path(self.file_path)
        if not path.exists():
            return ToolResult(
                error=True,
                name=self.tool_name(),
                result=f"File not found: {self.file_path}",
            )
        return ToolResult(error=False, name=self.tool_name(), result=path.read_text())


class WriteFile(AgentTool):
    """Writes to a file. Creates parent directories and the file if they don't exist."""

    file_path: str = Field(description="The path to the file to write to")
    content: str = Field(description="The content of the file")

    def execute(self) -> ToolResult:
        path = Path(self.file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.content)
        return ToolResult(
            error=False,
            name=self.tool_name(),
            result=f"Successfully wrote to {self.file_path}",
        )


class Bash(AgentTool):
    """Executes a bash command and returns both stdout and stderr if any."""

    command: str = Field(description="The bash command to execute")

    def execute(self) -> ToolResult:
        try:
            result = subprocess.run(
                self.command, shell=True, capture_output=True, text=True, timeout=30
            )

            parts: list[str] = []
            if result.stdout:
                parts.append(result.stdout)
            if result.stderr:
                parts.append(result.stderr)
            output = "\n".join(parts) if parts else "(no output)"
            if result.returncode != 0:
                output = f"exit code: {result.returncode}\n{output}"
            return ToolResult(
                error=result.returncode != 0,
                name=self.tool_name(),
                result=output.strip(),
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                error=True,
                name=self.tool_name(),
                result="Command timed out after 30 seconds",
            )


class ReadTodos(AgentTool):
    """Reads the list of todos."""

    def execute(self) -> ToolResult:
        return ToolResult(
            error=False,
            name=self.tool_name(),
            result=str(self.state.todos),
            raw=self.state.todos,
        )


class ModifyTodos(AgentTool):
    """Modifies the todo list. Use action 'add' to create new todos, 'remove' to delete todos, or 'complete' to mark todos as done. Always returns the updated todo list."""

    action: str = Field(
        description="The action to perform: 'add', 'remove', or 'complete'"
    )
    items: list[str] = Field(description="The todo items to apply the action to")

    def execute(self) -> ToolResult:
        todos = self.state.todos
        actions = {"add": todos.add, "remove": todos.remove, "complete": todos.complete}

        if self.action not in actions:
            return ToolResult(
                error=True,
                name=self.tool_name(),
                result=f"Unknown action '{self.action}'. Must be one of: add, remove, complete.",
            )

        result = actions[self.action](self.items)

        if self.action == "add":
            msg = f"Added: {result}" if result else "No new items added (duplicates)."
        else:
            changed, not_found = result
            parts = []
            if changed:
                label = "Removed" if self.action == "remove" else "Completed"
                parts.append(f"{label}: {changed}")
            if not_found:
                parts.append(f"Not found: {not_found}")
            msg = " | ".join(parts) if parts else "No changes."

        return ToolResult(
            error=False,
            name=self.tool_name(),
            result=f"{msg}\n\n{todos}",
            raw=todos,
        )


class SearchWeb(AgentTool):
    """Searches the web and returns the results with citations."""

    query: str = Field(description="The search query")

    def execute(self) -> ToolResult:
        try:
            results = web_search.search(self.query)

            formatted: list[str] = []
            for r in results:
                parts = [f"**{r['title']}**", r["url"]]
                if r["highlights"]:
                    parts.append(r["highlights"])
                formatted.append("\n".join(parts))

            return ToolResult(
                error=False,
                name=self.tool_name(),
                result="\n\n---\n\n".join(formatted)
                if formatted
                else "No results found.",
            )
        except Exception as e:
            return ToolResult(
                error=True,
                name=self.tool_name(),
                result=f"Search failed: {e}",
            )


class UseSkill(AgentTool):
    """Loads the full instructions for a named skill. Call this when the task matches one of the available skills listed in the system prompt."""

    skill_name: str = Field(description="The name of the skill to load")

    def execute(self) -> ToolResult:
        skill: Skill | None = self.state.skills_registry.get(self.skill_name)
        if not skill:
            available = list(self.state.skills_registry)
            return ToolResult(
                error=True,
                name=self.tool_name(),
                result=f"Unknown skill '{self.skill_name}'. Available: {available}",
            )

        content = skill.path.read_text()

        skill_dir = skill.path.parent
        siblings = sorted(p for p in skill_dir.iterdir() if p.name != "SKILL.md")
        if siblings:
            names = "\n".join(f"  - {p.name}" for p in siblings)
            content += (
                f"\n\n---\nThe above is SKILL.md. Additional reference files are available"
                f" in `{skill_dir}/`:\n{names}"
            )

        return ToolResult(error=False, name=self.tool_name(), result=content)
