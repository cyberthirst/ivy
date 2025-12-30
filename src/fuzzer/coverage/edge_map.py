"""
Stable arc hashing to a fixed-size edge counter map (AFL-style).
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional, Set, Tuple

from .collector import Arc


def _canonical_vyper_filename(filename: str) -> str:
    norm = filename.replace("\\", "/")
    idx = norm.rfind("vyper/")
    return norm[idx:] if idx != -1 else norm


class EdgeMap:
    def __init__(self, map_size: int, *, tag_with_config: bool = False):
        if map_size <= 0 or (map_size & (map_size - 1)) != 0:
            raise ValueError("map_size must be a positive power of two")

        self.map_size = map_size
        self._mask = map_size - 1
        self.tag_with_config = tag_with_config

    def hash_arc(
        self,
        arc: Arc,
        *,
        config_name: Optional[str] = None,
    ) -> int:
        filename, from_line, to_line = arc
        key = f"{_canonical_vyper_filename(filename)}:{from_line}->{to_line}"
        if self.tag_with_config:
            key = f"{config_name or ''}|{key}"

        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") & self._mask

    def hash_arcs(
        self,
        arcs: Set[Arc],
        *,
        config_name: Optional[str] = None,
    ) -> Set[int]:
        return {self.hash_arc(a, config_name=config_name) for a in arcs}

    def hash_arcs_by_config(self, arcs_by_config: Dict[str, Set[Arc]]) -> Set[int]:
        edge_ids: Set[int] = set()
        for config_name, arcs in arcs_by_config.items():
            edge_ids.update(self.hash_arcs(arcs, config_name=config_name))
        return edge_ids
