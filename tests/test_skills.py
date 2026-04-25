from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent import Agent
from model import Model
from skills import Skill, _parse_frontmatter, format_skills_for_prompt, load_skills


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
    def _write_skill(self, base: Path, dir_name: str, name: str, description: str) -> Path:
        skill_dir = base / dir_name
        skill_dir.mkdir(parents=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n# Instructions\nDo stuff."
        )
        return skill_md

    def test_loads_skill_from_directory(self, tmp_path: Path):
        skill_md = self._write_skill(tmp_path, "my-skill", "my-skill", "Does something.")
        skills = load_skills([tmp_path])
        assert len(skills) == 1
        assert skills[0].name == "my-skill"
        assert skills[0].description == "Does something."
        assert skills[0].path == skill_md

    def test_skips_missing_directory(self, tmp_path: Path):
        skills = load_skills([tmp_path / "nonexistent"])
        assert skills == []

    def test_skips_files_in_skills_dir(self, tmp_path: Path):
        (tmp_path / "not-a-dir.md").write_text("just a file")
        skills = load_skills([tmp_path])
        assert skills == []

    def test_skips_dirs_without_skill_md(self, tmp_path: Path):
        (tmp_path / "empty-dir").mkdir()
        skills = load_skills([tmp_path])
        assert skills == []

    def test_skips_skill_with_missing_name(self, tmp_path: Path):
        (tmp_path / "bad-skill").mkdir()
        (tmp_path / "bad-skill" / "SKILL.md").write_text(
            "---\ndescription: No name here.\n---\n"
        )
        skills = load_skills([tmp_path])
        assert skills == []

    def test_skips_skill_with_missing_description(self, tmp_path: Path):
        (tmp_path / "bad-skill").mkdir()
        (tmp_path / "bad-skill" / "SKILL.md").write_text("---\nname: no-desc\n---\n")
        skills = load_skills([tmp_path])
        assert skills == []

    def test_loads_from_multiple_dirs(self, tmp_path: Path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        self._write_skill(dir_a, "skill-a", "skill-a", "Skill A.")
        self._write_skill(dir_b, "skill-b", "skill-b", "Skill B.")
        skills = load_skills([dir_a, dir_b])
        names = [s.name for s in skills]
        assert "skill-a" in names
        assert "skill-b" in names

    def test_returns_skills_sorted_by_directory_name(self, tmp_path: Path):
        self._write_skill(tmp_path, "z-skill", "z-skill", "Last.")
        self._write_skill(tmp_path, "a-skill", "a-skill", "First.")
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

    def test_includes_path_to_skill_file(self):
        skills = [Skill(name="x", description="desc", path=Path("/skills/x/SKILL.md"))]
        result = format_skills_for_prompt(skills)
        assert "/skills/x/SKILL.md" in result

    def test_includes_all_skills(self):
        skills = [
            Skill(name="a", description="Skill A.", path=Path("/a/SKILL.md")),
            Skill(name="b", description="Skill B.", path=Path("/b/SKILL.md")),
        ]
        result = format_skills_for_prompt(skills)
        assert "Skill A." in result
        assert "Skill B." in result


# ── Agent with skills ─────────────────────────────────────────────────────────


@pytest.fixture()
def mock_client() -> MagicMock:
    c = MagicMock()
    c.chat.completions.create = AsyncMock()
    return c


@pytest.fixture()
def model(mock_client: MagicMock) -> Model:
    return Model(name="test-model", client=mock_client)


class TestAgentWithSkills:
    def _make_skills(self, tmp_path: Path) -> list[Skill]:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "---\nname: my-skill\ndescription: Helps with my tasks.\n---\n# Instructions"
        )
        return load_skills([tmp_path])

    def test_skill_metadata_in_system_prompt(self, model: Model, tmp_path: Path):
        skills = self._make_skills(tmp_path)
        agent = Agent(model, agent_tools=[], skills=skills)
        prompt = agent._build_system_prompt()
        assert "my-skill" in prompt
        assert "Helps with my tasks." in prompt

    def test_skill_path_in_system_prompt(self, model: Model, tmp_path: Path):
        skills = self._make_skills(tmp_path)
        agent = Agent(model, agent_tools=[], skills=skills)
        prompt = agent._build_system_prompt()
        assert "SKILL.md" in prompt

    def test_no_skills_no_skills_section(self, model: Model):
        agent = Agent(model, agent_tools=[])
        prompt = agent._build_system_prompt()
        assert "Available Skills" not in prompt

    def test_tool_schema_unchanged_by_skills(self, model: Model, tmp_path: Path):
        from tests.test_agent import FakeTool

        skills = self._make_skills(tmp_path)
        with_skills = Agent(model, agent_tools=[FakeTool], skills=skills)
        without_skills = Agent(model, agent_tools=[FakeTool])
        assert with_skills._tools_schema == without_skills._tools_schema
