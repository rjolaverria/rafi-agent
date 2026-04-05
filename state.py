from dataclasses import dataclass


@dataclass(slots=True)
class State:
    iterations: int = 0
