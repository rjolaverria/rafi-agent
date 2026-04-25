from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import Agent
from model import Model
from skill_types import Skill
from skills import UseSkill, _parse_frontmatter, format_skills_for_prompt, load_skills
from state import AgentState


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_state(registry: dict | None = None) -> AgentState:
    state = AgentState()
    state.skills_registry = registry or {}
    return state


def _write_skill(
    base: Path,
    dir_name: str,
    name: str,
    description: str,
    extra_files: list[str] | None = None,
) -> Path:
    skill_dir = base / dir_name
    skill_dir.mkdir(parents=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# Instructions\nDo stuff."
    )
    for filename in extra_files or []:
        (skill_dir / filename).write_text(f"Content of {filename}")
    return skill_md


@pytest.fixture()
def mock_client() -> MagicMock:
    c = MagicMock()
    c.chat.completions.create = AsyncMock()
    return c


@pytest.fixture()
def model(mock_client: MagicMock) -> Model:
    return Model(name="test-model", client=mock_client)


# ── _parse_frontmatter ────────────────────────────────────────────────────────


class TestParseFrontmatter:
    def test_parses_name_and_description(self):
        content = "---\nname: my-skill\ndescription: Does something useful.\n---\n# Body"
        result = _parse_frontmatter(content)
        assert result["name"] == "my-skill"
        assert result["description"] == "Does something useful."

    def test_returns_empty_dict_when_no_frontmatter(self):
        assert _parse_frontmatter("# Just a heading\nNo frontmatter.") == {}

    def test_returns_empty_dict_on_malformed_delimiter(self):
        assert _parse_frontmatter("--\nname: x\n--\n") == {}

    def test_ignores_body_after_closing_delimiter(self):
        content = "---\nname: skill\n---\ndescription: not-frontmatter"
        result = _parse_frontmatter(content)
        assert "description" not in result

    def test_strips_whitespace_from_keys_and_values(self):
        content = "---\n  name :  padded  \n---\n"
        result = _parse_frontmatter(content)
        assert result["name"] == "padded"


# ── load_skills ───────────────────────────────────────────────────────────────


class TestLoadSkills:
    def test_loads_skill_from_directory(self, tmp_path: Path):
        skill_md = _write_skill(tmp_path, "my-skill", "my-skill", "Does something.")
        skills = load_skills([tmp_path])
        assert len(skills) == 1
        assert skills[0].name == "my-skill"
        assert skills[0].description == "Does something."
        assert skills[0].path == skill_md

    def test_skips_missing_directory(self, tmp_path: Path):
        assert load_skills([tmp_path / "nonexistent"]) == []

    def test_skips_files_in_skills_dir(self, tmp_path: Path):
        (tmp_path / "not-a-dir.md").write_text("just a file")
        assert load_skills([tmp_path]) == []

    def test_skips_dirs_without_skill_md(self, tmp_path: Path):
        (tmp_path / "empty-dir").mkdir()
        assert load_skills([tmp_path]) == []

    def test_skips_skill_with_missing_name(self, tmp_path: Path):
        (tmp_path / "bad-skill").mkdir()
        (tmp_path / "bad-skill" / "SKILL.md").write_text(
            "---\ndescription: No name here.\n---\n"
        )
        assert load_skills([tmp_path]) == []

    def test_skips_skill_with_missing_description(self, tmp_path: Path):
        (tmp_path / "bad-skill").mkdir()
        (tmp_path / "bad-skill" / "SKILL.md").write_text("---\nname: no-desc\n---\n")
        assert load_skills([tmp_path]) == []

    def test_loads_from_multiple_dirs(self, tmp_path: Path):
        dir_a, dir_b = tmp_path / "a", tmp_path / "b"
        _write_skill(dir_a, "skill-a", "skill-a", "Skill A.")
        _write_skill(dir_b, "skill-b", "skill-b", "Skill B.")
        names = [s.name for s in load_skills([dir_a, dir_b])]
        assert "skill-a" in names
        assert "skill-b" in names

    def test_returns_skills_sorted_by_directory_name(self, tmp_path: Path):
        _write_skill(tmp_path, "z-skill", "z-skill", "Last.")
        _write_skill(tmp_path, "a-skill", "a-skill", "First.")
        skills = load_skills([tmp_path])
        assert skills[0].name == "a-skill"
        assert skills[1].name == "z-skill"


# ── format_skills_for_prompt ──────────────────────────────────────────────────


class TestFormatSkillsForPrompt:
    def test_returns_empty_string_for_no_skills(self):
        assert format_skills_for_prompt([]) == ""

    def test_includes_skill_name_and_description(self):
        skills = [Skill(name="pdf", description="Work with PDFs.", path=Path("/s/SKILL.md"))]
        result = format_skills_for_prompt(skills)
        assert "pdf" in result
        assert "Work with PDFs." in result

    def test_does_not_include_path(self):
        skills = [Skill(name="x", description="desc", path=Path("/skills/x/SKILL.md"))]
        result = format_skills_for_prompt(skills)
        assert "/skills/x/SKILL.md" not in result

    def test_mentions_useskill_tool(self):
        skills = [Skill(name="x", description="desc", path=Path("/x/SKILL.md"))]
        assert "useskill" in format_skills_for_prompt(skills)

    def test_includes_all_skills(self):
        skills = [
            Skill(name="a", description="Skill A.", path=Path("/a/SKILL.md")),
            Skill(name="b", description="Skill B.", path=Path("/b/SKILL.md")),
        ]
        result = format_skills_for_prompt(skills)
        assert "Skill A." in result
        assert "Skill B." in result


# ── UseSkill tool ─────────────────────────────────────────────────────────────


class TestUseSkill:
    def _make_tool(self, skill_name: str, registry: dict | None = None) -> UseSkill:
        return UseSkill.model_validate(
            {"skill_name": skill_name, "state": _make_state(registry)}
        )

    def test_returns_skill_md_contents(self, tmp_path: Path):
        skill_md = _write_skill(tmp_path, "my-skill", "my-skill", "Does stuff.")
        skill = Skill(name="my-skill", description="Does stuff.", path=skill_md)
        tool = self._make_tool("my-skill", {"my-skill": skill})
        result = tool.execute()
        assert result.error is False
        assert "Do stuff." in result.result

    def test_unknown_skill_returns_error(self, tmp_path: Path):
        tool = self._make_tool("nonexistent", {})
        result = tool.execute()
        assert result.error is True
        assert "nonexistent" in result.result

    def test_no_siblings_no_extra_section(self, tmp_path: Path):
        skill_md = _write_skill(tmp_path, "solo", "solo", "Solo skill.")
        skill = Skill(name="solo", description="Solo skill.", path=skill_md)
        tool = self._make_tool("solo", {"solo": skill})
        result = tool.execute()
        assert "Additional reference files" not in result.result

    def test_siblings_reported_with_directory_path(self, tmp_path: Path):
        skill_md = _write_skill(
            tmp_path, "rich", "rich", "Rich skill.", extra_files=["ADVANCED.md", "helpers.py"]
        )
        skill = Skill(name="rich", description="Rich skill.", path=skill_md)
        tool = self._make_tool("rich", {"rich": skill})
        result = tool.execute()
        assert "ADVANCED.md" in result.result
        assert "helpers.py" in result.result
        assert str(tmp_path / "rich") in result.result

    def test_siblings_section_clarifies_content_is_skill_md(self, tmp_path: Path):
        skill_md = _write_skill(tmp_path, "s", "s", "s.", extra_files=["REF.md"])
        skill = Skill(name="s", description="s.", path=skill_md)
        tool = self._make_tool("s", {"s": skill})
        result = tool.execute()
        assert "SKILL.md" in result.result


# ── Agent integration ─────────────────────────────────────────────────────────


class TestAgentWithSkills:
    def _make_skills(self, tmp_path: Path) -> list[Skill]:
        _write_skill(tmp_path, "my-skill", "my-skill", "Helps with my tasks.")
        return load_skills([tmp_path])

    def test_useskill_in_tools_when_skills_provided(self, model: Model, tmp_path: Path):
        skills = self._make_skills(tmp_path)
        agent = Agent(model, agent_tools=[], skills=skills)
        names = [s["function"]["name"] for s in agent._tools_schema]
        assert "useskill" in names

    def test_useskill_not_in_tools_without_skills(self, model: Model):
        agent = Agent(model, agent_tools=[])
        names = [s["function"]["name"] for s in agent._tools_schema]
        assert "useskill" not in names

    def test_skill_metadata_in_system_prompt(self, model: Model, tmp_path: Path):
        skills = self._make_skills(tmp_path)
        agent = Agent(model, agent_tools=[], skills=skills)
        prompt = agent._build_system_prompt()
        assert "my-skill" in prompt
        assert "Helps with my tasks." in prompt

    def test_skill_path_not_in_system_prompt(self, model: Model, tmp_path: Path):
        skills = self._make_skills(tmp_path)
        agent = Agent(model, agent_tools=[], skills=skills)
        assert "SKILL.md" not in agent._build_system_prompt()

    def test_skills_registry_populated_in_state(self, model: Model, tmp_path: Path):
        skills = self._make_skills(tmp_path)
        agent = Agent(model, agent_tools=[], skills=skills)
        assert "my-skill" in agent.state.skills_registry

    def test_tool_schema_count_includes_useskill(self, model: Model, tmp_path: Path):
        from tests.test_agent import FakeTool

        skills = self._make_skills(tmp_path)
        agent = Agent(model, agent_tools=[FakeTool], skills=skills)
        names = [s["function"]["name"] for s in agent._tools_schema]
        # FakeTool + ReadTodos + ModifyTodos + UseSkill
        assert len(names) == 4
