#!/usr/bin/env python3
"""
Archive GitHub issues locally with incremental sync.

Source of truth is per-issue JSON; Markdown is generated from those JSON docs.

Usage:
    python gh_issue_archive.py owner/repo [--format both] [--force] [--dry-run]

Requires: `gh` CLI authenticated and on PATH. Zero external Python dependencies.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_NAME = ".archive_manifest.json"
JSON_DIR_NAME = "json"
JSON_SCHEMA_VERSION = 1

MARKDOWN_FORMATS = {"md", "both"}
JSON_FORMATS = {"json", "both"}

REST_ACCEPT_HEADER = "Accept: application/vnd.github+json"
REST_PAGE_SIZE = 100

METADATA_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    issues(
      first: 100
      after: $cursor
      orderBy: { field: CREATED_AT, direction: ASC }
      states: [OPEN, CLOSED]
    ) {
      nodes {
        number
        title
        updatedAt
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_gh(args, *, check=True):
    """Run a `gh` CLI command and return stdout."""
    result = subprocess.run(["gh"] + args, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"gh error: {result.stderr.strip()}", file=sys.stderr)
        raise SystemExit(1)
    return result.stdout


def split_repo(repo):
    """Split owner/repo into parts."""
    owner, name = repo.split("/", 1)
    return owner, name


def sanitize_title(title, max_len=80):
    """Lowercase, replace non-alnum with hyphens, collapse, truncate."""
    s = title.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s or "untitled"


def issue_filename(number, title):
    """Zero-padded issue number + sanitized title."""
    return f"{number:05d}-{sanitize_title(title)}.md"


def default_output_dir(repo):
    """Generate default archive directory from owner/repo."""
    owner, name = split_repo(repo)
    owner = re.sub(r"[^A-Za-z0-9._-]+", "-", owner).strip("-") or "owner"
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "repo"
    return f"./{owner}-{name}-issues"


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Manifest (incremental sync state)
# ---------------------------------------------------------------------------

def load_manifest(out_dir):
    path = out_dir / MANIFEST_NAME
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(out_dir, manifest):
    path = out_dir / MANIFEST_NAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_issue_metadata(repo):
    """Fetch all issue numbers + title + updatedAt via GraphQL pagination."""
    owner, name = split_repo(repo)
    cursor = None
    results = []

    while True:
        args = [
            "api",
            "graphql",
            "-f",
            f"query={METADATA_QUERY}",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
        ]
        if cursor:
            args += ["-F", f"cursor={cursor}"]

        raw = run_gh(args)
        if not raw.strip():
            break

        payload = json.loads(raw)
        repository = payload.get("data", {}).get("repository")
        if repository is None:
            print("gh error: unexpected GraphQL response while fetching metadata.",
                  file=sys.stderr)
            raise SystemExit(1)

        issue_data = repository.get("issues", {})
        results.extend(issue_data.get("nodes", []))
        page_info = issue_data.get("pageInfo", {})
        if not page_info.get("hasNextPage"):
            break

        cursor = page_info.get("endCursor")
        if not cursor:
            break

    return results


def fetch_issue_core(repo, number):
    owner, name = split_repo(repo)
    endpoint = f"repos/{owner}/{name}/issues/{number}"
    raw = run_gh(["api", "-X", "GET", endpoint, "-H", REST_ACCEPT_HEADER])
    if not raw.strip():
        return None
    return json.loads(raw)


def fetch_rest_paginated(endpoint):
    items = []
    page = 1

    while True:
        raw = run_gh([
            "api",
            "-X",
            "GET",
            endpoint,
            "-H",
            REST_ACCEPT_HEADER,
            "-f",
            f"per_page={REST_PAGE_SIZE}",
            "-f",
            f"page={page}",
        ])
        if not raw.strip():
            break

        batch = json.loads(raw)
        if not isinstance(batch, list):
            print(f"gh error: expected a list from endpoint '{endpoint}'.",
                  file=sys.stderr)
            raise SystemExit(1)
        if not batch:
            break

        items.extend(batch)
        if len(batch) < REST_PAGE_SIZE:
            break
        page += 1

    return items


def fetch_issue_comments(repo, number):
    owner, name = split_repo(repo)
    endpoint = f"repos/{owner}/{name}/issues/{number}/comments"
    return fetch_rest_paginated(endpoint)


def fetch_issue_timeline(repo, number):
    owner, name = split_repo(repo)
    endpoint = f"repos/{owner}/{name}/issues/{number}/timeline"
    return fetch_rest_paginated(endpoint)


def fetch_issue_document(repo, number):
    """
    Fetch one issue document with raw data.

    Returns:
      {
        schema_version,
        archived_at,
        repo,
        issue: <REST issue object>,
        comments: [<REST comment objects>],
        timeline: [<REST timeline event objects>]
      }
    """
    try:
        core = fetch_issue_core(repo, number)
    except SystemExit:
        print(f"  Warning: failed to fetch issue #{number} core data.",
              file=sys.stderr)
        return None

    if not core:
        print(f"  Warning: empty issue payload for #{number}.", file=sys.stderr)
        return None

    comments = []
    try:
        comments = fetch_issue_comments(repo, number)
    except SystemExit:
        print(f"  Warning: failed to fetch comments for issue #{number}; continuing.",
              file=sys.stderr)

    timeline = []
    try:
        timeline = fetch_issue_timeline(repo, number)
    except SystemExit:
        print(f"  Warning: failed to fetch timeline for issue #{number}; continuing.",
              file=sys.stderr)

    return {
        "schema_version": JSON_SCHEMA_VERSION,
        "archived_at": utc_now_iso(),
        "repo": repo,
        "issue": core,
        "comments": comments,
        "timeline": timeline,
    }


def fetch_issue_documents(repo, numbers):
    """Fetch many issue documents, continuing on per-issue failures."""
    wanted_numbers = list(dict.fromkeys(numbers))
    print(f"  Fetching {len(wanted_numbers)} issue(s) ...")
    docs = []
    for i, num in enumerate(wanted_numbers, 1):
        print(f"    [{i}/{len(wanted_numbers)}] #{num}")
        doc = fetch_issue_document(repo, num)
        if doc is not None:
            docs.append(doc)
    return docs


# ---------------------------------------------------------------------------
# JSON source-of-truth storage
# ---------------------------------------------------------------------------

def json_dir(out_dir):
    return out_dir / JSON_DIR_NAME


def json_issue_path(out_dir, number):
    return json_dir(out_dir) / f"{number:05d}.json"


def issue_number_from_doc(issue_doc):
    if isinstance(issue_doc, dict) and "issue" in issue_doc:
        issue = issue_doc.get("issue", {})
        if isinstance(issue, dict):
            return issue.get("number")
    return issue_doc.get("number")


def issue_title_from_doc(issue_doc):
    if isinstance(issue_doc, dict) and "issue" in issue_doc:
        issue = issue_doc.get("issue", {})
        if isinstance(issue, dict):
            return issue.get("title", "")
    return issue_doc.get("title", "")


def issue_updated_at_from_doc(issue_doc):
    if isinstance(issue_doc, dict) and "issue" in issue_doc:
        issue = issue_doc.get("issue", {})
        if isinstance(issue, dict):
            return issue.get("updated_at", "")
    return issue_doc.get("updatedAt", "")


def read_issue_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f"  Warning: invalid JSON in {path}: {exc}", file=sys.stderr)
        return None


def write_issue_json(out_dir, issue_doc):
    number = issue_number_from_doc(issue_doc)
    if number is None:
        raise ValueError("issue document missing number")
    jdir = json_dir(out_dir)
    jdir.mkdir(parents=True, exist_ok=True)
    path = json_issue_path(out_dir, number)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(issue_doc, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _yaml_list(items):
    if not items:
        return "[]"
    return "[" + ", ".join(_yaml_escape(i) for i in items) + "]"


def _yaml_escape(s):
    if s is None:
        return '""'
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _extract_login(user):
    if isinstance(user, dict):
        return user.get("login", "unknown")
    return str(user) if user else "unknown"


def _extract_labels(labels):
    if not labels:
        return []
    return [l["name"] if isinstance(l, dict) else str(l) for l in labels]


def _extract_assignees(assignees):
    if not assignees:
        return []
    return [_extract_login(a) for a in assignees]


def _extract_milestone(milestone):
    if not milestone:
        return ""
    if isinstance(milestone, dict):
        return milestone.get("title", "")
    return str(milestone)


def _format_ts(ts):
    if not ts:
        return ""
    return ts.replace("T", " ").replace("Z", " UTC")


def _normalize_comment(comment):
    if not isinstance(comment, dict):
        return {
            "author": "unknown",
            "authorAssociation": "",
            "body": str(comment),
            "createdAt": "",
            "url": "",
        }

    if "createdAt" in comment or "authorAssociation" in comment:
        return {
            "author": comment.get("author"),
            "authorAssociation": comment.get("authorAssociation", ""),
            "body": comment.get("body", "") or "",
            "createdAt": comment.get("createdAt", ""),
            "url": comment.get("url", ""),
        }

    # REST comment shape
    return {
        "author": comment.get("user"),
        "authorAssociation": comment.get("author_association", ""),
        "body": comment.get("body", "") or "",
        "createdAt": comment.get("created_at", ""),
        "url": comment.get("html_url") or comment.get("url", ""),
    }


def _normalize_for_markdown(issue_doc):
    """
    Normalize issue document into markdown renderer shape.

    Supports:
      - new JSON source-of-truth docs (with top-level `issue/comments/timeline`)
      - legacy issue dicts from older script versions
    """
    if not isinstance(issue_doc, dict):
        return {}

    if "issue" not in issue_doc:
        normalized = dict(issue_doc)
        comments = normalized.get("comments", []) or []
        normalized["comments"] = [_normalize_comment(c) for c in comments]
        normalized["timeline"] = normalized.get("timeline", []) or []
        return normalized

    issue = issue_doc.get("issue", {}) or {}
    comments = issue_doc.get("comments", []) or []
    timeline = issue_doc.get("timeline", []) or []
    return {
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        "body": issue.get("body", "") or "",
        "author": issue.get("user"),
        "state": issue.get("state", ""),
        "createdAt": issue.get("created_at", ""),
        "updatedAt": issue.get("updated_at", ""),
        "labels": issue.get("labels", []) or [],
        "assignees": issue.get("assignees", []) or [],
        "milestone": issue.get("milestone"),
        "url": issue.get("html_url") or issue.get("url", ""),
        "comments": [_normalize_comment(c) for c in comments],
        "timeline": timeline,
    }


def _timeline_summary_line(event):
    if not isinstance(event, dict):
        return None

    event_type = event.get("event") or event.get("__typename") or "event"
    created = _format_ts(event.get("created_at") or event.get("createdAt") or "")
    actor = _extract_login(event.get("actor"))
    parts = [created or "unknown-time", event_type, f"by {actor}"]

    label = event.get("label")
    if isinstance(label, dict) and label.get("name"):
        parts.append(f"label `{label['name']}`")

    assignee = event.get("assignee")
    if assignee:
        parts.append(f"assignee `{_extract_login(assignee)}`")

    rename = event.get("rename")
    if isinstance(rename, dict):
        old_title = rename.get("from")
        new_title = rename.get("to")
        if old_title or new_title:
            parts.append(f"title `{old_title}` -> `{new_title}`")

    milestone = event.get("milestone")
    if isinstance(milestone, dict) and milestone.get("title"):
        parts.append(f"milestone `{milestone['title']}`")

    return " | ".join(parts)


def format_issue_md(issue_doc):
    """Render one issue markdown file from normalized issue document."""
    issue = _normalize_for_markdown(issue_doc)

    number = issue.get("number", 0)
    title = issue.get("title", "")
    body = issue.get("body", "") or ""
    author = _extract_login(issue.get("author"))
    state = str(issue.get("state", "")).upper()
    created = issue.get("createdAt", "")
    updated = issue.get("updatedAt", "")
    labels = _extract_labels(issue.get("labels"))
    assignees = _extract_assignees(issue.get("assignees"))
    milestone = _extract_milestone(issue.get("milestone"))
    url = issue.get("url", "")
    comments = issue.get("comments", []) or []
    timeline = issue.get("timeline", []) or []

    fm_lines = [
        "---",
        f"issue: {number}",
        f"title: {_yaml_escape(title)}",
        f"author: {_yaml_escape(author)}",
        f"state: {state.lower()}",
        f"created: {_yaml_escape(created)}",
        f"updated: {_yaml_escape(updated)}",
        f"labels: {_yaml_list(labels)}",
        f"assignees: {_yaml_list(assignees)}",
        f"milestone: {_yaml_escape(milestone)}",
        f"url: {_yaml_escape(url)}",
        "---",
    ]

    table_lines = [
        "",
        f"# #{number} -- {title}",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| **State** | {state} |",
        f"| **Author** | {author} |",
    ]
    if labels:
        table_lines.append(f"| **Labels** | {', '.join(f'`{l}`' for l in labels)} |")
    if assignees:
        table_lines.append(f"| **Assignees** | {', '.join(assignees)} |")
    if milestone:
        table_lines.append(f"| **Milestone** | {milestone} |")
    table_lines.append(f"| **Created** | {_format_ts(created)} |")
    table_lines.append(f"| **Updated** | {_format_ts(updated)} |")

    body_lines = [
        "",
        "---",
        "",
        body.rstrip(),
    ]

    timeline_lines = []
    if timeline:
        timeline_lines += ["", "---", "", f"## Timeline ({len(timeline)})"]
        for e in timeline:
            line = _timeline_summary_line(e)
            if line:
                timeline_lines.append(f"- {line}")

    comment_lines = []
    if comments:
        comment_lines += ["", "---", "", f"## Comments ({len(comments)})"]
        for c in comments:
            c_author = _extract_login(c.get("author"))
            c_created = _format_ts(c.get("createdAt", ""))
            c_role = c.get("authorAssociation", "")
            c_body = (c.get("body", "") or "").rstrip()
            c_url = c.get("url", "")

            comment_lines.append("")
            heading = f"### Comment by [{c_author}] on {c_created}"
            if c_url:
                heading += f" ([link]({c_url}))"
            comment_lines.append(heading)
            if c_role and c_role not in ("NONE", ""):
                comment_lines.append(f"> *{c_role}*")
            comment_lines.append("")
            comment_lines.append(c_body)

    parts = fm_lines + table_lines + body_lines + timeline_lines + comment_lines + [""]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# File management
# ---------------------------------------------------------------------------

def index_existing_md_files(out_dir):
    """Map issue number -> markdown path for existing issue files."""
    mapping = {}
    for f in out_dir.iterdir():
        if not f.is_file() or f.suffix != ".md":
            continue
        m = re.match(r"^(\d{5})-", f.name)
        if not m:
            continue
        mapping[int(m.group(1))] = f
    return mapping


def write_issue_md(out_dir, issue_doc, existing_md_files):
    """Write markdown for one issue, cleaning up title-based renames."""
    number = issue_number_from_doc(issue_doc)
    title = issue_title_from_doc(issue_doc)
    if number is None:
        raise ValueError("issue document missing number")

    new_name = issue_filename(number, title)

    old_file = existing_md_files.get(number)
    if old_file and old_file.name != new_name:
        old_file.unlink()

    new_path = out_dir / new_name
    new_path.write_text(format_issue_md(issue_doc), encoding="utf-8")
    existing_md_files[number] = new_path
    return new_name


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def archive(repo, out_dir, *, force=False, dry_run=False, output_format="both"):
    want_md = output_format in MARKDOWN_FORMATS
    want_json = output_format in JSON_FORMATS

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {} if force else load_manifest(out_dir)

    print(f"Fetching issue metadata from {repo} ...")
    meta = fetch_issue_metadata(repo)
    print(f"  Found {len(meta)} issues.")
    if not meta:
        print("No issues found.")
        return

    meta_by_number = {m["number"]: m for m in meta}
    to_fetch_remote = []
    to_render_local = []

    for m in meta:
        num = m["number"]
        title = m.get("title", "")
        updated = m.get("updatedAt", "")
        key = str(num)

        expected_json = json_issue_path(out_dir, num)
        expected_md = out_dir / issue_filename(num, title)

        needs_remote = (key not in manifest) or (manifest[key] != updated)
        if want_json and not expected_json.exists():
            needs_remote = True
        if want_md and not want_json and not expected_md.exists():
            needs_remote = True

        if needs_remote:
            to_fetch_remote.append(num)
        elif want_md and want_json and not expected_md.exists():
            # JSON exists and metadata unchanged; render markdown locally.
            to_render_local.append(num)

    if not to_fetch_remote and not to_render_local:
        print("All issues up to date. Nothing to archive.")
        return

    if to_fetch_remote:
        print(f"  {len(to_fetch_remote)} issue(s) need remote fetch.")
    if to_render_local:
        print(f"  {len(to_render_local)} issue(s) will be rebuilt from local JSON.")

    if dry_run:
        if to_fetch_remote:
            print("[dry-run] Would fetch and archive issues:", to_fetch_remote)
        if to_render_local:
            print("[dry-run] Would render markdown from local JSON for:", to_render_local)
        return

    existing_md_files = index_existing_md_files(out_dir) if want_md else {}

    remote_docs = []
    if to_fetch_remote:
        remote_docs = fetch_issue_documents(repo, to_fetch_remote)

    successful_remote = set()
    for doc in remote_docs:
        num = issue_number_from_doc(doc)
        if num is None:
            continue

        # JSON is source-of-truth.
        render_doc = doc
        if want_json:
            jpath = write_issue_json(out_dir, doc)
            print(f"  Wrote {jpath.relative_to(out_dir)}")
            loaded = read_issue_json(jpath)
            if loaded is not None:
                render_doc = loaded

        if want_md:
            md_name = write_issue_md(out_dir, render_doc, existing_md_files)
            print(f"  Wrote {md_name}")

        key = str(num)
        manifest[key] = meta_by_number.get(num, {}).get(
            "updatedAt",
            issue_updated_at_from_doc(doc),
        )
        successful_remote.add(num)

    local_md_failures = []
    if want_md and want_json and to_render_local:
        for num in to_render_local:
            doc = read_issue_json(json_issue_path(out_dir, num))
            if doc is None:
                local_md_failures.append(num)
                print(f"  Warning: missing/corrupt JSON for issue #{num}; cannot render markdown.",
                      file=sys.stderr)
                continue
            md_name = write_issue_md(out_dir, doc, existing_md_files)
            print(f"  Wrote {md_name} (from local JSON)")

    save_manifest(out_dir, manifest)

    remote_failures = [n for n in to_fetch_remote if n not in successful_remote]
    if remote_failures or local_md_failures:
        print(f"Done with warnings. Archived {len(successful_remote)} remotely fetched issue(s) to {out_dir}/")
        if remote_failures:
            print(f"  Warning: failed remote fetch for {len(remote_failures)} issue(s): {remote_failures}",
                  file=sys.stderr)
        if local_md_failures:
            print(f"  Warning: failed local markdown rebuild for {len(local_md_failures)} issue(s): {local_md_failures}",
                  file=sys.stderr)
        return

    local_only = len(to_render_local)
    if local_only:
        print(
            f"Done. Archived {len(successful_remote)} remotely fetched issue(s) and rebuilt {local_only} markdown file(s) to {out_dir}/"
        )
    else:
        print(f"Done. Archived {len(successful_remote)} issue(s) to {out_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Archive GitHub issues locally with JSON as source of truth.",
    )
    parser.add_argument(
        "repo",
        help="GitHub repository in owner/repo format (e.g. vyperlang/vyper)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output directory (default: ./<owner>-<repo>-issues)",
    )
    parser.add_argument(
        "--format",
        choices=("md", "json", "both"),
        default="both",
        help="Output format: markdown, json, or both (default: both)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-archive all issues, ignoring manifest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be archived without writing files",
    )
    args = parser.parse_args()

    if "/" not in args.repo or args.repo.count("/") != 1:
        print(f"Error: repo must be in owner/repo format, got: {args.repo}",
              file=sys.stderr)
        raise SystemExit(1)

    output_dir = args.output or default_output_dir(args.repo)
    archive(
        args.repo,
        output_dir,
        force=args.force,
        dry_run=args.dry_run,
        output_format=args.format,
    )


if __name__ == "__main__":
    main()

