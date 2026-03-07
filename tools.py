import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from openai.types.chat import ChatCompletionToolParam
from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    error: bool
    name: str
    result: str

    def to_message(self, tool_call_id: str) -> dict[str, str]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": self.result}


class AgentTool(BaseModel, ABC):
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
    def execute(self) -> ToolResult: ...


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
                parts.append("stdout:\n" + result.stdout)
            if result.stderr:
                parts.append("stderr:\n" + result.stderr)
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
