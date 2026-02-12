#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "triage_config.json"
MAX_PARALLEL_AGENTS = 4
CLAUDE_MODEL = "claude-opus-4-6"
CODEX_MODEL = "gpt-5.3-codex"
MAX_INDEX_CHARS = 180_000


@dataclass(frozen=True)
class TriagePaths:
    run_dir: Path
    filtered_divergences: Path
    filtered_crashes: Path
    unverified_root: Path
    unverified_divergences: Path
    unverified_crashes: Path
    verified_dir: Path


@dataclass(frozen=True)
class TriageConfig:
    output_root: Path


def _print_step(message: str) -> None:
    print(message, flush=True)


def _read_text(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        return f.read()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def _sorted_md_files(directory: Path, *, recursive: bool) -> list[Path]:
    if not directory.exists():
        return []
    pattern = "**/*.md" if recursive else "*.md"
    return sorted(path for path in directory.glob(pattern) if path.is_file())


def _clear_md_files(directory: Path) -> None:
    if not directory.exists():
        return
    for path in directory.glob("*.md"):
        if path.is_file():
            path.unlink()


def _parse_config(config_path: Path) -> TriageConfig:
    with config_path.open(encoding="utf-8") as f:
        payload = json.load(f)

    output_root = payload.get("output_root", "./issue-archives")
    if not isinstance(output_root, str):
        output_root = str(output_root)
    resolved_output_root = _resolve_project_path(output_root)
    return TriageConfig(output_root=resolved_output_root)


def _prepare_paths(run_dir: Path, skip_verify: bool) -> TriagePaths:
    filtered = run_dir / "filtered"
    paths = TriagePaths(
        run_dir=run_dir,
        filtered_divergences=filtered / "divergences",
        filtered_crashes=filtered / "compiler_crashes",
        unverified_root=run_dir / "unverified",
        unverified_divergences=run_dir / "unverified" / "divergences",
        unverified_crashes=run_dir / "unverified" / "compiler_crashes",
        verified_dir=run_dir / "verified",
    )

    paths.unverified_divergences.mkdir(parents=True, exist_ok=True)
    paths.unverified_crashes.mkdir(parents=True, exist_ok=True)
    paths.verified_dir.mkdir(parents=True, exist_ok=True)

    _clear_md_files(paths.unverified_divergences)
    _clear_md_files(paths.unverified_crashes)
    if not skip_verify:
        _clear_md_files(paths.verified_dir)

    return paths


def _run_checked(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    )


def _run_unchecked(cmd: list[str], *, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        input=input_text,
        text=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
    )


def sync_issues(config_path: Path) -> None:
    _print_step("Step 1/5: Syncing GitHub issue archive")
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "gh_issue_archive.py"),
        "--config",
        str(config_path),
    ]
    result = _run_checked(cmd)
    if result.stdout.strip():
        print(result.stdout, end="")
    if result.stderr.strip():
        print(result.stderr, file=sys.stderr, end="")


def get_vyper_version() -> str:
    cmd = ["uv", "run", "vyper", "--version"]
    result = _run_checked(cmd)
    version = result.stdout.strip()
    return version or "unknown"


def _run_claude(prompt: str, label: str) -> tuple[bool, str]:
    cmd = [
        "claude",
        "-p",
        "--dangerously-skip-permissions",
        "--model",
        CLAUDE_MODEL,
        "--output-format",
        "text",
    ]
    result = _run_unchecked(cmd, input_text=prompt)
    if result.returncode != 0:
        err = result.stderr.strip() or f"{label} failed with exit code {result.returncode}"
        return False, err
    return True, result.stdout.strip()


def dedup_divergences(paths: TriagePaths, vyper_version: str) -> list[Path]:
    _print_step("Step 2/5: Deduplicating divergence reports with Claude")
    if not paths.filtered_divergences.exists():
        _print_step(f"  Skipping: missing directory {paths.filtered_divergences}")
        return []

    prompt = f"""You are triaging fuzzer divergences for the Vyper compiler.
Vyper version: {vyper_version}

Divergence JSON files are in: {paths.filtered_divergences}
Write deduplicated bug reports to: {paths.unverified_divergences}

Instructions:
1. Read all divergence files in {paths.filtered_divergences}.
2. Use up to 4 subagents for parallel analysis.
3. Group divergences by root cause.
4. For each unique root cause, write one markdown report to {paths.unverified_divergences}/<root-cause-slug>.md.
5. Each report must contain:
   - Title descriptive of root cause
   - Vyper version
   - PoC: Python test using titanoboa that demonstrates the bug
   - Observed vs Expected behavior
   - Root cause analysis
   - List of divergence files sharing this root cause

Constraints:
- Treat files under filtered/ as read-only.
- Do not write outside {paths.unverified_divergences}.
- Do not minimize PoCs at this stage; minimization is handled during verification.
"""

    ok, output_or_error = _run_claude(prompt, "Divergence deduplication")
    if not ok:
        print(f"  Warning: {output_or_error}", file=sys.stderr)
    elif output_or_error:
        _print_step(f"  Claude output: {output_or_error.splitlines()[0]}")

    reports = _sorted_md_files(paths.unverified_divergences, recursive=False)
    _print_step(f"  Generated {len(reports)} divergence report(s)")
    return reports


def dedup_crashes(paths: TriagePaths, vyper_version: str) -> list[Path]:
    _print_step("Step 3/5: Deduplicating compiler crash reports with Claude")
    if not paths.filtered_crashes.exists():
        _print_step(f"  Skipping: missing directory {paths.filtered_crashes}")
        return []

    prompt = f"""You are triaging Vyper compiler crash reports from a fuzzer run.
Vyper version: {vyper_version}

Crash JSON files are in: {paths.filtered_crashes}
Write deduplicated bug reports to: {paths.unverified_crashes}

Instructions:
1. Read all crash files in {paths.filtered_crashes}.
2. Use up to 4 subagents for parallel analysis.
3. Group crashes by root cause (same traceback/cause should be grouped).
4. For each unique root cause, write one markdown report to {paths.unverified_crashes}/<root-cause-slug>.md.
5. Each report must contain:
   - Title descriptive of root cause
   - Vyper version
   - Vyper source PoC that triggers the crash
   - Observed vs Expected behavior
   - Root cause analysis referencing traceback details
   - List of crash files sharing this root cause

Constraints:
- Treat files under filtered/ as read-only.
- Do not write outside {paths.unverified_crashes}.
- Do not minimize PoCs at this stage; minimization is handled during verification.
"""

    ok, output_or_error = _run_claude(prompt, "Crash deduplication")
    if not ok:
        print(f"  Warning: {output_or_error}", file=sys.stderr)
    elif output_or_error:
        _print_step(f"  Claude output: {output_or_error.splitlines()[0]}")

    reports = _sorted_md_files(paths.unverified_crashes, recursive=False)
    _print_step(f"  Generated {len(reports)} crash report(s)")
    return reports


def _verify_output_name(unverified_root: Path, report_path: Path) -> str:
    relative = report_path.relative_to(unverified_root)
    return "__".join(relative.parts)


def _verify_report_worker(
    report_path_raw: str,
    output_path_raw: str,
    vyper_skill_content: str,
) -> dict[str, Any]:
    report_path = Path(report_path_raw)
    output_path = Path(output_path_raw)
    temp_output_path = output_path.with_suffix(".tmp.md")

    report_content = _read_text(report_path)
    prompt = f"""You are verifying a Vyper compiler bug report in this repository.

Report path: {report_path}

Task:
1. Reproduce the PoC in the report.
2. If reproducible:
   - Minimize the contract/source while preserving the bug.
   - Return only an updated markdown report.
3. If not reproducible, output exactly:
NOT_REPRODUCIBLE
4. Use ivy and boa tests/experiments as needed to validate behavior.

Rules:
- Do not modify the input report.
- Keep output concise and executable.
- Return markdown only (or NOT_REPRODUCIBLE).

Reference context (skills/vyper.md):
{vyper_skill_content}

Original report:
{report_content}
"""

    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--model",
        CODEX_MODEL,
        "-c",
        "reasoning_effort=xhigh",
        "-o",
        str(temp_output_path),
    ]

    result = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    if result.returncode != 0:
        return {
            "status": "error",
            "report_path": report_path_raw,
            "error": result.stderr.strip() or f"exit code {result.returncode}",
        }

    if not temp_output_path.exists():
        return {
            "status": "skipped",
            "report_path": report_path_raw,
        }

    output_content = _read_text(temp_output_path).strip()
    if not output_content or output_content == "NOT_REPRODUCIBLE":
        temp_output_path.unlink(missing_ok=True)
        return {
            "status": "not_reproducible",
            "report_path": report_path_raw,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    temp_output_path.replace(output_path)
    return {
        "status": "verified",
        "report_path": report_path_raw,
        "output_path": str(output_path),
    }


def verify_reports(paths: TriagePaths, vyper_skill_content: str) -> tuple[list[Path], int]:
    _print_step("Step 4/5: Verifying reports with Codex")
    snapshot = _sorted_md_files(paths.unverified_root, recursive=True)
    _print_step(f"  Snapshot captured: {len(snapshot)} unverified report(s)")
    if not snapshot:
        return [], 0

    failures = 0
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_AGENTS) as executor:
        futures = []
        for report_path in snapshot:
            output_name = _verify_output_name(paths.unverified_root, report_path)
            output_path = paths.verified_dir / output_name
            futures.append(
                executor.submit(
                    _verify_report_worker,
                    str(report_path),
                    str(output_path),
                    vyper_skill_content,
                )
            )

        for future in as_completed(futures):
            result = future.result()
            status = result.get("status")
            report_path = result.get("report_path")
            if status == "error":
                failures += 1
                print(f"  Warning: verify failed for {report_path}: {result.get('error')}", file=sys.stderr)

    verified = _sorted_md_files(paths.verified_dir, recursive=False)
    _print_step(f"  Verified {len(verified)} report(s)")
    return verified, failures


def _extract_title(content: str) -> str:
    lines = content.splitlines()
    if lines and lines[0].strip() == "---":
        for line in lines[1:]:
            if line.strip() == "---":
                break
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
                if title:
                    return title

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title

    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped[:120]
    return "Untitled"


def build_issue_index(issue_archives_root: Path) -> str:
    if not issue_archives_root.exists():
        return "No issue archives found."

    entries: list[str] = []
    total_chars = 0
    md_files = sorted(path for path in issue_archives_root.rglob("*.md") if path.is_file())
    for md_file in md_files:
        content = _read_text(md_file)
        title = _extract_title(content)
        preview = content[:500].replace("\x00", " ")
        relative_path = md_file.relative_to(issue_archives_root)
        entry = f"[{relative_path}] {title}\n{preview}\n---\n"
        next_size = total_chars + len(entry)
        if next_size > MAX_INDEX_CHARS:
            entries.append(
                f"[TRUNCATED] Index exceeded {MAX_INDEX_CHARS} characters; remaining files omitted."
            )
            break
        entries.append(entry)
        total_chars = next_size

    return "\n".join(entries) if entries else "No issue markdown files found."


def _dedup_report_worker(
    report_path_raw: str,
    bug_reports_dir_raw: str,
    issue_index: str,
    dedup_log_path_raw: str,
) -> dict[str, Any]:
    report_path = Path(report_path_raw)
    bug_reports_dir = Path(bug_reports_dir_raw)
    dedup_log_path = Path(dedup_log_path_raw)
    target_path = bug_reports_dir / report_path.name
    report_content = _read_text(report_path)

    prompt = f"""You are deduplicating a verified Vyper bug report against archived issues.

Verified report path: {report_path}
Novel report destination: {target_path}

Task:
1. Compare the verified report against the issue index below.
2. If the report is novel, copy {report_path} to {target_path}.
3. If it is a duplicate, do not copy anything.
4. Never modify the source report.

Output:
- First line exactly one of: NOVEL or DUPLICATE
- Second line: short reason

Issue index:
{issue_index}

Verified report:
{report_content}
"""

    cmd = [
        "codex",
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--model",
        CODEX_MODEL,
        "-c",
        "reasoning_effort=xhigh",
        "-o",
        str(dedup_log_path),
    ]
    result = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        cwd=PROJECT_ROOT,
        capture_output=True,
    )

    if result.returncode != 0:
        return {
            "status": "error",
            "report_path": report_path_raw,
            "error": result.stderr.strip() or f"exit code {result.returncode}",
        }
    return {"status": "ok", "report_path": report_path_raw}


def dedup_against_issues(
    verified_reports: list[Path],
    config: TriageConfig,
    verified_dir: Path,
) -> tuple[list[Path], int]:
    _print_step("Step 5/5: Deduplicating verified reports against issue archive")
    snapshot = sorted(verified_reports)
    _print_step(f"  Snapshot captured: {len(snapshot)} verified report(s)")
    if not snapshot:
        return [], 0

    issue_root = config.output_root
    bug_reports_dir = issue_root / "bug_reports"
    bug_reports_dir.mkdir(parents=True, exist_ok=True)

    issue_index = build_issue_index(issue_root)

    before = {path.resolve() for path in _sorted_md_files(bug_reports_dir, recursive=False)}
    dedup_logs_dir = verified_dir / ".dedup_logs"
    dedup_logs_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_AGENTS) as executor:
        futures = []
        for report_path in snapshot:
            log_path = dedup_logs_dir / report_path.name
            futures.append(
                executor.submit(
                    _dedup_report_worker,
                    str(report_path),
                    str(bug_reports_dir),
                    issue_index,
                    str(log_path),
                )
            )

        for future in as_completed(futures):
            result = future.result()
            if result.get("status") == "error":
                failures += 1
                print(
                    f"  Warning: issue dedup failed for {result.get('report_path')}: {result.get('error')}",
                    file=sys.stderr,
                )

    after = {path.resolve() for path in _sorted_md_files(bug_reports_dir, recursive=False)}
    novel = sorted(Path(path) for path in (after - before))
    _print_step(f"  Novel reports copied: {len(novel)}")
    return novel, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Triages fuzzer output by deduplicating, verifying, and archiving bug reports."
    )
    parser.add_argument(
        "report_dir",
        help="Path to fuzzer report directory",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to archive config JSON (default: scripts/triage_config.json)",
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Skip GitHub issue sync (step 1)",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip report verification (step 4)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.report_dir).expanduser().resolve()
    config_path = _resolve_project_path(args.config)

    if not run_dir.exists() or not run_dir.is_dir():
        print(f"Error: report_dir not found or not a directory: {run_dir}", file=sys.stderr)
        return 1

    if not config_path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        config = _parse_config(config_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Error: failed to parse config {config_path}: {exc}", file=sys.stderr)
        return 1

    paths = _prepare_paths(run_dir, skip_verify=args.skip_verify)

    if not args.skip_sync:
        try:
            sync_issues(config_path)
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                print(exc.stdout, end="")
            if exc.stderr:
                print(exc.stderr, file=sys.stderr, end="")
            print("Error: issue sync failed; aborting.", file=sys.stderr)
            return exc.returncode or 1
    else:
        _print_step("Step 1/5: Skipped issue sync")

    try:
        vyper_version = get_vyper_version()
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, file=sys.stderr, end="")
        print("Error: could not determine Vyper version.", file=sys.stderr)
        return exc.returncode or 1

    divergence_reports = dedup_divergences(paths, vyper_version)
    crash_reports = dedup_crashes(paths, vyper_version)
    total_unverified = len(divergence_reports) + len(crash_reports)

    verify_failures = 0
    if args.skip_verify:
        _print_step("Step 4/5: Skipped verification")
        verified_reports = _sorted_md_files(paths.verified_dir, recursive=False)
        _print_step(f"  Existing verified reports: {len(verified_reports)}")
    else:
        vyper_skill_path = PROJECT_ROOT / "skills" / "vyper.md"
        vyper_skill_content = _read_text(vyper_skill_path) if vyper_skill_path.exists() else ""
        verified_reports, verify_failures = verify_reports(paths, vyper_skill_content)

    novel_reports, dedup_failures = dedup_against_issues(
        verified_reports,
        config,
        paths.verified_dir,
    )

    duplicates = max(0, len(verified_reports) - len(novel_reports))
    print()
    print("Summary")
    print(f"  Unverified generated: {total_unverified}")
    print(f"    - Divergence reports: {len(divergence_reports)}")
    print(f"    - Crash reports: {len(crash_reports)}")
    print(f"  Verified reports: {len(verified_reports)}")
    print(f"  Novel reports: {len(novel_reports)}")
    print(f"  Duplicates: {duplicates}")
    print(f"  Verification failures: {verify_failures}")
    print(f"  Issue-dedup failures: {dedup_failures}")
    print(f"  Issue archive root: {config.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
