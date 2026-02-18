from __future__ import annotations


class FreshNameGenerator:
    def __init__(self, prefix: str = "gen_var"):
        self.prefix = prefix
        self.counter = 0

    def generate(self, prefix: str | None = None) -> str:
        name = f"{prefix or self.prefix}{self.counter}"
        self.counter += 1
        return name

    def reset(self) -> None:
        self.counter = 0
