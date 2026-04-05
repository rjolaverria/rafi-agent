from unittest.mock import MagicMock

import pytest
from rich.console import Console
from rich.table import Table

from sub_agent_display import SubAgentProgressDisplay, SubAgentStatus, _truncate
from tools import ToolResult


@pytest.fixture()
def console():
    return Console(force_terminal=True, width=120)


# ── SubAgentStatus ────────────────────────────────────────────────────────


class TestSubAgentStatus:
    def test_defaults(self):
        s = SubAgentStatus(index=0, instruction="do stuff")
        assert s.status == "pending"
        assert s.tool_name is None
        assert s.tool_detail is None
        assert s.steps_completed == 0

    def test_custom_values(self):
        s = SubAgentStatus(
            index=1,
            instruction="write code",
            status="running",
            tool_name="bash",
            tool_detail="ls -la",
            steps_completed=3,
        )
        assert s.index == 1
        assert s.status == "running"
        assert s.tool_name == "bash"
        assert s.steps_completed == 3


# ── _truncate ─────────────────────────────────────────────────────────────


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_exact_length_unchanged(self):
        assert _truncate("hello", 5) == "hello"

    def test_long_text_truncated(self):
        result = _truncate("hello world", 8)
        assert len(result) == 8
        assert result.endswith("\u2026")

    def test_default_max_len(self):
        short = "short"
        assert _truncate(short) == short
        long = "a" * 50
        assert len(_truncate(long)) == 40


# ── SubAgentProgressDisplay ───────────────────────────────────────────────


class TestSubAgentProgressDisplay:
    def test_creates_statuses_for_each_instruction(self, console):
        display = SubAgentProgressDisplay(["q1", "q2", "q3"], console)
        assert len(display._statuses) == 3
        assert display._statuses[0].instruction == "q1"
        assert display._statuses[1].instruction == "q2"
        assert display._statuses[2].instruction == "q3"

    def test_statuses_start_as_pending(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        assert display._statuses[0].status == "pending"

    def test_build_table_returns_table(self, console):
        display = SubAgentProgressDisplay(["q1", "q2"], console)
        table = display._build_table()
        assert isinstance(table, Table)
        assert table.row_count == 2

    def test_build_table_with_various_statuses(self, console):
        display = SubAgentProgressDisplay(["q1", "q2", "q3", "q4"], console)
        display._statuses[0].status = "done"
        display._statuses[1].status = "running"
        display._statuses[1].tool_name = "bash"
        display._statuses[1].tool_detail = "ls"
        display._statuses[2].status = "running"
        display._statuses[3].status = "error"

        table = display._build_table()
        assert table.row_count == 4


# ── make_hooks ────────────────────────────────────────────────────────────


class TestMakeHooks:
    def test_returns_three_callables(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        hooks = display.make_hooks(0)
        assert len(hooks) == 3
        assert all(callable(h) for h in hooks)

    def test_response_hook_sets_running(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        on_response, _, _ = display.make_hooks(0)

        msg = MagicMock()
        msg.content = "hello"
        on_response(msg)

        assert display._statuses[0].status == "running"

    def test_response_hook_clears_tool_info(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        display._statuses[0].tool_name = "bash"
        display._statuses[0].tool_detail = "ls"
        on_response, _, _ = display.make_hooks(0)

        on_response(MagicMock())

        assert display._statuses[0].tool_name is None
        assert display._statuses[0].tool_detail is None

    def test_response_hook_skips_done_status(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        display._statuses[0].status = "done"
        on_response, _, _ = display.make_hooks(0)

        on_response(MagicMock())

        assert display._statuses[0].status == "done"

    def test_tool_call_hook_sets_tool_name(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        _, on_tool_call, _ = display.make_hooks(0)

        on_tool_call("bash", {"command": "ls -la"})

        assert display._statuses[0].tool_name == "bash"
        assert display._statuses[0].tool_detail == "ls -la"
        assert display._statuses[0].status == "running"

    def test_tool_call_hook_uses_tool_args_display(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        _, on_tool_call, _ = display.make_hooks(0)

        on_tool_call("readfile", {"file_path": "/tmp/foo.py"})

        assert display._statuses[0].tool_name == "readfile"
        assert display._statuses[0].tool_detail == "/tmp/foo.py"

    def test_tool_call_hook_no_detail_for_unknown_keys(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        _, on_tool_call, _ = display.make_hooks(0)

        on_tool_call("searchweb", {"query": "python async"})

        assert display._statuses[0].tool_name == "searchweb"
        assert display._statuses[0].tool_detail is None

    def test_tool_call_hook_ignores_hidden_tools(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        _, on_tool_call, _ = display.make_hooks(0)

        on_tool_call("readtodos", {})

        assert display._statuses[0].tool_name is None
        assert display._statuses[0].status == "pending"

    def test_tool_result_hook_increments_steps(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        _, _, on_tool_result = display.make_hooks(0)

        result = ToolResult(error=False, name="bash", result="ok")
        on_tool_result(result)

        assert display._statuses[0].steps_completed == 1

    def test_tool_result_hook_clears_tool_info(self, console):
        display = SubAgentProgressDisplay(["q1"], console)
        display._statuses[0].tool_name = "bash"
        display._statuses[0].tool_detail = "ls"
        _, _, on_tool_result = display.make_hooks(0)

        result = ToolResult(error=False, name="bash", result="ok")
        on_tool_result(result)

        assert display._statuses[0].tool_name is None
        assert display._statuses[0].tool_detail is None

    def test_multiple_hooks_are_independent(self, console):
        display = SubAgentProgressDisplay(["q1", "q2"], console)
        _, tc0, tr0 = display.make_hooks(0)
        _, tc1, tr1 = display.make_hooks(1)

        tc0("bash", {"command": "echo hi"})
        tc1("readfile", {"file_path": "/tmp/x"})

        assert display._statuses[0].tool_name == "bash"
        assert display._statuses[1].tool_name == "readfile"

        tr0(ToolResult(error=False, name="bash", result="ok"))
        assert display._statuses[0].steps_completed == 1
        assert display._statuses[1].steps_completed == 0
