import subprocess
from collections.abc import Callable
from pathlib import Path

from openai.types.chat import ChatCompletionToolParam


def read_file(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"
    return path.read_text()


def write_file(file_path: str, content: str) -> str:
    path = Path(file_path)
    path.write_text(content)
    return f"Successfully wrote to {file_path}"


def bash(command: str) -> str:
    try:
        result = subprocess.run(command, shell=True, capture_output=True, timeout=30)
        stdout = result.stdout.decode()
        stderr = result.stderr.decode()

        output = "stdout:\n" + stdout if stdout else "(no output)"
        if stderr:
            output += "\nstderr:\n" + stderr
        if result.returncode != 0:
            output = f"exit code: {result.returncode}\n{output}"
        return output.strip()
    except subprocess.TimeoutExpired:
        return "Command timed out after 30 seconds"


_read_file_params: dict[str, object] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "The path to the file to read",
        }
    },
    "required": ["file_path"],
}

_write_file_params: dict[str, object] = {
    "type": "object",
    "properties": {
        "file_path": {
            "type": "string",
            "description": "The path to the file to write to",
        },
        "content": {
            "type": "string",
            "description": "The content of the file",
        },
    },
    "required": ["file_path", "content"],
}

_bash_params: dict[str, object] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "The bash command to execute",
        },
    },
    "required": ["command"],
}

tools: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads and return the contents of a file",
            "parameters": _read_file_params,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Writes to a file if it exists. If it doesn't it will be created with the content",
            "parameters": _write_file_params,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Executes a bash command and returns both stdout and stderr if any.",
            "parameters": _bash_params,
        },
    },
]

tools_map: dict[str, Callable] = {
    "read_file": read_file,
    "write_file": write_file,
    "bash": bash,
}


def get_tool(tool_name: str) -> Callable | None:
    return tools_map.get(tool_name)
