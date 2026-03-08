from typing import Any

from openai.types.chat import ChatCompletionMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from tools import ToolResult

console = Console()

TOOL_ARGS_DISPLAY: dict[str, list[str]] = {
    "bash": ["command"],
    "readfile": ["file_path"],
    "writefile": ["file_path"],
}


def print_response(message: ChatCompletionMessage) -> None:
    if not message.content:
        return
    console.print()
    console.print(Text(" agent ", style="bold white on magenta"))
    console.print(Markdown(message.content))
    console.print()


def print_tool_call(tool_name: str, args: Any) -> None:
    label = Text()
    label.append(" " + tool_name + " ", style="bold white on blue")

    display_keys = TOOL_ARGS_DISPLAY.get(tool_name, [])
    summary_parts = [str(args[k]) for k in display_keys if k in args]
    if summary_parts:
        label.append(" " + " ".join(summary_parts), style="dim")

    console.print()
    console.print(label)

    hidden_keys = set(display_keys)
    remaining = {k: v for k, v in args.items() if k not in hidden_keys}

    if remaining:
        for key, value in remaining.items():
            content = str(value)
            if "\n" in content or len(content) > 80:
                console.print(
                    Panel(
                        Syntax(content, "text", theme="ansi_dark", word_wrap=True),
                        border_style="blue",
                        expand=True,
                    )
                )
            else:
                console.print(Text(f"  {key}: {content}", style="dim"))


def print_tool_result(tool_result: ToolResult) -> None:
    if not tool_result.result:
        return

    style = "red" if tool_result.error else "green"
    content = tool_result.result

    if len(content) > 2000:
        content = content[:2000] + f"\n... ({len(tool_result.result)} chars total)"

    console.print(
        Panel(
            Syntax(content, "text", theme="ansi_dark", word_wrap=True),
            border_style=style,
            expand=True,
        )
    )
