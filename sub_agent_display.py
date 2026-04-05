from dataclasses import dataclass
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from display import TOOL_ARGS_DISPLAY, HIDDEN_TOOL_CALLS
from tools import ToolResult


@dataclass
class SubAgentStatus:
    index: int
    instruction: str
    status: str = "pending"
    tool_name: str | None = None
    tool_detail: str | None = None
    steps_completed: int = 0


def _truncate(text: str, max_len: int = 40) -> str:
    return text[: max_len - 1] + "\u2026" if len(text) > max_len else text


class SubAgentProgressDisplay:
    def __init__(self, instructions: list[str], console: Console) -> None:
        self._statuses = [
            SubAgentStatus(index=i, instruction=instr)
            for i, instr in enumerate(instructions)
        ]
        self._console = console
        self._live = Live(
            get_renderable=self._build_table,
            console=console,
            transient=True,
            refresh_per_second=8,
        )

    def start(self) -> None:
        self._live.start(refresh=True)

    def stop(self) -> None:
        self._live.stop()
        self._print_summary()

    def _build_table(self) -> Table:
        table = Table(
            show_header=True,
            show_edge=True,
            border_style="blue",
            title="sub-agents",
            title_style="bold white on blue",
            padding=(0, 1),
        )
        table.add_column("#", width=3, justify="right")
        table.add_column("Task", ratio=2)
        table.add_column("Status", ratio=2)
        table.add_column("Steps", width=5, justify="right")

        for s in self._statuses:
            task = _truncate(s.instruction)
            steps = str(s.steps_completed)

            if s.status == "done":
                status_cell: Text | Spinner = Text("done", style="bold green")
                marker = Text("\u2713", style="green")
            elif s.status == "error":
                status_cell = Text("error", style="bold red")
                marker = Text("\u2717", style="red")
            elif s.tool_name:
                detail = ""
                if s.tool_detail:
                    detail = f" {_truncate(s.tool_detail, 25)}"
                status_cell = Spinner("dots", text=Text(f"{s.tool_name}{detail}"))
                marker = Text(str(s.index + 1), style="dim")
            elif s.status == "running":
                status_cell = Spinner("dots", text=Text("thinking"))
                marker = Text(str(s.index + 1), style="dim")
            else:
                status_cell = Text("waiting", style="dim")
                marker = Text(str(s.index + 1), style="dim")

            table.add_row(marker, task, status_cell, steps)

        return table

    def _print_summary(self) -> None:
        done_count = sum(1 for s in self._statuses if s.status == "done")
        error_count = sum(1 for s in self._statuses if s.status == "error")
        total_steps = sum(s.steps_completed for s in self._statuses)

        parts = [f"{done_count}/{len(self._statuses)} completed"]
        if error_count:
            parts.append(f"{error_count} errors")
        parts.append(f"{total_steps} total steps")

        summary = Text()
        summary.append(" sub-agents ", style="bold white on blue")
        summary.append(" " + ", ".join(parts), style="dim")
        self._console.print()
        self._console.print(summary)

    def make_hooks(self, index: int) -> tuple:
        status = self._statuses[index]

        def on_response(message: Any) -> None:
            if status.status != "done":
                status.status = "running"
                status.tool_name = None
                status.tool_detail = None

        def on_tool_call(tool_name: str, args: Any) -> None:
            if tool_name in HIDDEN_TOOL_CALLS:
                return
            status.status = "running"
            status.tool_name = tool_name
            display_keys = TOOL_ARGS_DISPLAY.get(tool_name, [])
            detail_parts = [str(args[k]) for k in display_keys if k in args]
            status.tool_detail = " ".join(detail_parts) if detail_parts else None

        def on_tool_result(tool_result: ToolResult) -> None:
            status.steps_completed += 1
            status.tool_name = None
            status.tool_detail = None

        return on_response, on_tool_call, on_tool_result
