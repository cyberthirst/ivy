from typing import Any


class Ctx:
    scopes: list[dict[str, Any]]

    def __init__(self):
        self.scopes = [{}]

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def __getitem__(self, key):
        for scope in reversed(self.scopes):
            if key in scope:
                return scope[key]
        raise KeyError(key)

    def __setitem__(self, key, value):
        self.scopes[-1][key] = value

    def __contains__(self, key):
        for scope in reversed(self.scopes):
            if key in scope:
                return True
        return False

    def __repr__(self):
        return f"Ctx({self.scopes})"
