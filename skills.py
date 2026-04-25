from dataclasses import dataclass

from pydantic import Field

from tools import AgentTool, Bash, ReadFile, SearchWeb, ToolResult, WriteFile


@dataclass
class Skill:
    name: str
    description: str
    tools: list[type[AgentTool]]


SKILLS: dict[str, Skill] = {
    "filesystem": Skill(
        name="filesystem",
        description="Read and write files on the local filesystem",
        tools=[ReadFile, WriteFile],
    ),
    "shell": Skill(
        name="shell",
        description="Execute bash commands and shell scripts",
        tools=[Bash],
    ),
    "web": Skill(
        name="web",
        description="Search the web for documentation and information",
        tools=[SearchWeb],
    ),
}


class ListSkills(AgentTool):
    """Lists all available skills and their activation status. Call this to discover what tools you can unlock."""

    def execute(self) -> ToolResult:
        registry: dict[str, Skill] = self.state.skills_registry  # type: ignore[assignment]
        active = self.state.activated_skills
        lines = []
        for name, skill in registry.items():
            status = "[active]" if name in active else "[available]"
            tool_names = [t.tool_name() for t in skill.tools]
            lines.append(
                f"**{name}** {status}: {skill.description} (tools: {', '.join(tool_names)})"
            )
        return ToolResult(
            error=False,
            name=self.tool_name(),
            result="\n".join(lines) if lines else "No skills available.",
        )


class UseSkill(AgentTool):
    """Activates a skill to unlock its tools for use. Call listskills first to see what's available."""

    skill_name: str = Field(description="The name of the skill to activate")

    def execute(self) -> ToolResult:
        registry: dict[str, Skill] = self.state.skills_registry  # type: ignore[assignment]
        skill = registry.get(self.skill_name)
        if not skill:
            return ToolResult(
                error=True,
                name=self.tool_name(),
                result=f"Unknown skill '{self.skill_name}'. Available: {list(registry)}",
            )
        if self.skill_name in self.state.activated_skills:
            tool_names = [t.tool_name() for t in skill.tools]
            return ToolResult(
                error=False,
                name=self.tool_name(),
                result=f"Skill '{self.skill_name}' is already active. Tools: {tool_names}",
            )
        self.state.activated_skills.add(self.skill_name)
        tool_names = [t.tool_name() for t in skill.tools]
        return ToolResult(
            error=False,
            name=self.tool_name(),
            result=f"Skill '{self.skill_name}' activated. You now have access to: {tool_names}",
        )
