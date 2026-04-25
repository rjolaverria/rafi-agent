import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    path: Path


def _parse_frontmatter(content: str) -> dict[str, str]:
    """Parse simple key: value YAML frontmatter delimited by ---."""
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}
    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def load_skills(dirs: list[Path]) -> list[Skill]:
    """Scan directories for skill subdirectories containing SKILL.md and return their metadata."""
    skills: list[Skill] = []
    for d in dirs:
        if not d.exists():
            continue
        for skill_dir in sorted(d.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            meta = _parse_frontmatter(skill_md.read_text())
            name = meta.get("name", "").strip()
            description = meta.get("description", "").strip()
            if name and description:
                skills.append(Skill(name=name, description=description, path=skill_md))
    return skills


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """Format skill metadata for injection into the system prompt (Level 1 loading)."""
    if not skills:
        return ""
    lines = [
        "## Available Skills",
        "When a task matches a skill below, read its SKILL.md with the readfile tool to load full instructions before proceeding.",
        "",
    ]
    for skill in skills:
        lines.append(f"- **{skill.name}** (`{skill.path}`): {skill.description}")
    return "\n".join(lines)
