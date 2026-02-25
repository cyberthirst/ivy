from __future__ import annotations


class FreshNameGenerator:
    def __init__(self):
        self._counters: dict[str, int] = {}
        self._existing_names: set[str] = set()

    def generate(self, prefix: str = "gen_var") -> str:
        counter = self._counters.get(prefix, 0)
        while True:
            name = f"{prefix}{counter}"
            counter += 1
            if name not in self._existing_names:
                self._counters[prefix] = counter
                return name

    def reset(self, existing_names: set[str] | None = None) -> None:
        self._counters.clear()
        self._existing_names = existing_names or set()
