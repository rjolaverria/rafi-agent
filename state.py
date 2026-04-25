from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    path: Path


@dataclass(slots=True)
class Todos:
    _items: dict[str, bool] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self._items:
            return "No todos."
        return "\n".join(
            f"- [{'x' if done else ' '}] {item}" for item, done in self._items.items()
        )

    def add(self, items: list[str]) -> list[str]:
        """Adds items to the list of todos. Returns the list of added items (without duplicates and stripped)"""
        to_add = [i for item in items if (i := item.strip()) not in self._items]
        self._items.update(dict.fromkeys(to_add, False))
        return to_add

    def remove(self, items: list[str]) -> tuple[list[str], list[str]]:
        """Removes items from the list of todos. Returns a tuple of (removed_items, not_exists_items)"""
        normalized = {item.strip() for item in items}
        to_remove = [item for item in normalized if item in self._items]
        not_exists = [item for item in normalized if item not in self._items]
        for item in to_remove:
            del self._items[item]
        return to_remove, not_exists

    def complete(self, items: list[str]) -> tuple[list[str], list[str]]:
        """Marks items as completed. Returns a tuple of (completed_items, not_exists_items). It is idempotent, so marking an already completed item as completed again will not cause an error and the item will not be returned in the completed_items list."""
        normalized = {item.strip() for item in items}
        to_complete = [
            item for item in normalized if item in self._items and not self._items[item]
        ]
        not_exists = [item for item in normalized if item not in self._items]
        for item in to_complete:
            self._items[item] = True
        return to_complete, not_exists


@dataclass(slots=True)
class AgentState:
    iterations: int = 0
    todos: Todos = field(default_factory=Todos)
    skills_registry: dict[str, Skill] = field(default_factory=dict)
