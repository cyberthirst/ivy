from __future__ import annotations

import time
from pathlib import Path

from fuzzer.coverage.disk_corpus import (
    DiskEntryMeta,
    delete_corpus_entry,
    read_all_meta,
)


def compact_corpus(corpus_dir: Path, max_corpus_size: int) -> None:
    """Run corpus compaction in an isolated process.

    This is a top-level function so it can be used as a ``spawn`` target.
    If it crashes (including OOM-kill), the main monitor is unaffected.
    """
    queue_dir = corpus_dir / "queue"

    all_entries = read_all_meta(corpus_dir)
    if not all_entries:
        _cleanup_orphan_files(queue_dir)
        return

    by_fp: dict[str, list[DiskEntryMeta]] = {}
    for entry in all_entries:
        by_fp.setdefault(entry.coverage_fp, []).append(entry)

    to_delete: set[str] = set()
    for entries in by_fp.values():
        keep_ids: set[str] = set()
        smallest = min(entries, key=lambda e: e.source_size)
        fastest = min(entries, key=lambda e: e.cycle_time_s)
        keep_ids.add(smallest.entry_id)
        keep_ids.add(fastest.entry_id)

        for entry in entries:
            if entry.entry_id not in keep_ids:
                to_delete.add(entry.entry_id)

    remaining = [e for e in all_entries if e.entry_id not in to_delete]
    if len(remaining) > max_corpus_size:
        evictable = sorted(remaining, key=lambda e: e.timestamp)
        excess = len(remaining) - max_corpus_size
        for entry in evictable[:excess]:
            to_delete.add(entry.entry_id)

    for entry_id in to_delete:
        delete_corpus_entry(corpus_dir, entry_id)

    _cleanup_orphan_files(queue_dir)


def _cleanup_orphan_files(queue_dir: Path) -> None:
    cutoff = time.time() - 60.0
    for suffix in (".scenario.json", ".edges.bin"):
        for path in queue_dir.glob(f"*{suffix}"):
            if path.stat().st_mtime >= cutoff:
                continue
            entry_id = path.name[: -len(suffix)]
            if (queue_dir / f"{entry_id}.meta.json").exists():
                continue
            path.unlink(missing_ok=True)
