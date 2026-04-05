import json
from typing import Any

import pytest

from state import State
from tools import AgentTool, Bash, ReadFile, ToolResult, WriteFile


@pytest.fixture()
def state():
    return State()


class TestToolResult:
    def test_to_message(self):
        result = ToolResult(error=False, name="test", result="ok")
        msg = result.to_message("call_123")
        assert msg == {"role": "tool", "tool_call_id": "call_123", "content": "ok"}


class TestReadFile:
    def test_reads_existing_file(self, tmp_path, state):
        f = tmp_path / "hello.txt"
        f.write_text("hello world")
        result = ReadFile(state=state, file_path=str(f)).execute()
        assert not result.error
        assert result.result == "hello world"

    def test_error_on_missing_file(self, state):
        result = ReadFile(state=state, file_path="/nonexistent/path/file.txt").execute()
        assert result.error
        assert "File not found" in result.result

    def test_reads_empty_file(self, tmp_path, state):
        f = tmp_path / "empty.txt"
        f.write_text("")
        result = ReadFile(state=state, file_path=str(f)).execute()
        assert not result.error
        assert result.result == ""


class TestWriteFile:
    def test_writes_to_new_file(self, tmp_path, state):
        f = tmp_path / "out.txt"
        result = WriteFile(state=state, file_path=str(f), content="content").execute()
        assert not result.error
        assert "Successfully wrote" in result.result
        assert f.read_text() == "content"

    def test_overwrites_existing_file(self, tmp_path, state):
        f = tmp_path / "out.txt"
        f.write_text("old")
        WriteFile(state=state, file_path=str(f), content="new").execute()
        assert f.read_text() == "new"

    def test_creates_parent_directories(self, tmp_path, state):
        f = tmp_path / "a" / "b" / "c.txt"
        result = WriteFile(state=state, file_path=str(f), content="nested").execute()
        assert not result.error
        assert f.read_text() == "nested"

    def test_writes_empty_content(self, tmp_path, state):
        f = tmp_path / "empty.txt"
        WriteFile(state=state, file_path=str(f), content="").execute()
        assert f.read_text() == ""


class TestBash:
    def test_stdout(self, state):
        result = Bash(state=state, command="echo hello").execute()
        assert not result.error
        assert result.result == "hello"

    def test_stderr(self, state):
        result = Bash(state=state, command="echo err >&2").execute()
        assert result.result == "err"

    def test_stdout_and_stderr(self, state):
        result = Bash(state=state, command="echo out && echo err >&2").execute()
        assert "out" in result.result
        assert "err" in result.result

    def test_no_output(self, state):
        result = Bash(state=state, command="true").execute()
        assert result.result == "(no output)"

    def test_nonzero_exit_code(self, state):
        result = Bash(state=state, command="false").execute()
        assert result.error
        assert result.result.startswith("exit code: 1")

    def test_nonzero_exit_with_stderr(self, state):
        result = Bash(state=state, command="echo fail >&2 && exit 2").execute()
        assert result.error
        assert "exit code: 2" in result.result
        assert "fail" in result.result

    def test_pipes(self, state):
        result = Bash(state=state, command="echo 'a b c' | wc -w").execute()
        assert "3" in result.result

    def test_timeout(self, monkeypatch, state):
        import subprocess as sp

        def mock_run(*args, **kwargs):
            raise sp.TimeoutExpired(cmd="sleep 60", timeout=30)

        monkeypatch.setattr(sp, "run", mock_run)
        result = Bash(state=state, command="sleep 60").execute()
        assert result.error
        assert "timed out" in result.result


def _schema_to_dict(cls: type[AgentTool]) -> dict[str, Any]:
    return json.loads(json.dumps(cls.to_json_schema()))


class TestAgentToolSchema:
    def test_to_json_schema_structure(self):
        schema = _schema_to_dict(ReadFile)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "readfile"
        assert "parameters" in schema["function"]
        assert "description" in schema["function"]

    def test_schema_has_properties_and_required(self):
        schema = _schema_to_dict(WriteFile)
        params = schema["function"]["parameters"]
        assert "file_path" in params["properties"]
        assert "content" in params["properties"]
        assert "file_path" in params["required"]
        assert "content" in params["required"]

    def test_schema_includes_field_descriptions(self):
        schema = _schema_to_dict(Bash)
        cmd_prop = schema["function"]["parameters"]["properties"]["command"]
        assert "description" in cmd_prop

    def test_schema_uses_docstring_as_description(self):
        schema = _schema_to_dict(ReadFile)
        assert "Reads and returns" in schema["function"]["description"]

    def test_no_title_in_parameters(self):
        schema = _schema_to_dict(ReadFile)
        assert "title" not in schema["function"]["parameters"]


class TestAgentToolSubclasses:
    def test_all_concrete_tools_are_agent_tools(self):
        for cls in [ReadFile, WriteFile, Bash]:
            assert issubclass(cls, AgentTool)

    def test_tool_names(self):
        assert ReadFile.tool_name() == "readfile"
        assert WriteFile.tool_name() == "writefile"
        assert Bash.tool_name() == "bash"
