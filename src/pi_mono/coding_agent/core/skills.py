from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pi_mono.coding_agent.config import SKILLS_DIR

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class Skill:
    """A loaded skill."""
    name: str
    description: str
    prompt: str
    source_path: Path | None = None


class SkillLoader:
    """Loads skills from the skills directory."""

    def __init__(self, skills_dir: Path | None = None) -> None:
        self._skills_dir = skills_dir or SKILLS_DIR

    def load_all(self) -> list[Skill]:
        skills: list[Skill] = []
        if not self._skills_dir.exists():
            return skills

        for path in sorted(self._skills_dir.glob("*.md")):
            try:
                content = path.read_text(encoding="utf-8")
                name = path.stem
                # First line is description
                lines = content.strip().splitlines()
                description = lines[0].lstrip("# ").strip() if lines else name
                prompt = content

                skills.append(Skill(
                    name=name,
                    description=description,
                    prompt=prompt,
                    source_path=path,
                ))
            except Exception:
                continue

        return skills

    def get_skill(self, name: str) -> Skill | None:
        for skill in self.load_all():
            if skill.name == name:
                return skill
        return None
