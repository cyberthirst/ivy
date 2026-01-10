---
name: code-review
description: Dual-agent code review (Claude + Codex) - review uncommitted changes for bugs/architecture issues, cross-review, merge into final report
---

# Code Review

## Overview

Claude and Codex independently review uncommitted changes, then cross-review each other's findings (2 passes max). Claude merges into a final report. Agents can write tests as POCs but cannot modify source code.

## Workflow

### 1. Setup

```bash
COMMIT_MSG=$(git log -1 --pretty=format:'%s' | head -c 20 | tr ' /' '_-')
COMMIT_HASH=$(git rev-parse --short HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
REVIEW_DIR="reviews/${COMMIT_MSG}-${COMMIT_HASH}"

mkdir -p "$REVIEW_DIR"
git diff HEAD > "$REVIEW_DIR/diff.patch"
```

### 2. Claude Review

Review the diff and write findings to `$REVIEW_DIR/claude-$BRANCH-review.md`:

```markdown
# [BUG] [CRITICAL] Short description

- **Location**: file:line
- **Explanation**: What's wrong
- **Suggested Fix**: How to fix
- **POC**: tests/test_bug_poc.py (optional)

# [BUG] [MAJOR] Another bug
...
```

Order by severity: CRITICAL > MAJOR > MINOR.

### 3. Codex Review (parallel)

```bash
cat << 'EOF' | codex exec --full-auto -C /workspace/ivy -
Review the uncommitted changes (git diff HEAD) for bugs and architectural issues.

Write findings to: reviews/<dir>/codex-<branch>-review.md

Format:
# [BUG] [SEVERITY] Short description
- **Location**: file:line
- **Explanation**: What's wrong
- **Suggested Fix**: How to fix
- **POC**: (optional)

Severities: CRITICAL, MAJOR, MINOR (order by severity).
You may write tests in tests/ as POCs. Do NOT modify source code.
EOF
```

### 4. Cross-Review

Each agent reads the other's review. Add comments with `## [AGENT-comment]`:

```markdown
# [BUG] [MAJOR] Race condition in state update
- **Location**: src/ivy/env.py:89
...

## [codex-comment]
Confirmed. Also affects line 156.
```

2 passes max. Unresolved disagreements left for user.

### 5. Final Report

Claude merges both into `$REVIEW_DIR/final-$BRANCH-report.md`:
- Deduplicate findings
- Note agreements/disagreements
- Preserve unresolved debates

## Example

Reviewing changes on a feature branch:

```bash
# On branch: add-coverage-tracking
COMMIT_MSG=$(git log -1 --pretty=format:'%s' | head -c 20 | tr ' /' '_-')
COMMIT_HASH=$(git rev-parse --short HEAD)
REVIEW_DIR="reviews/${COMMIT_MSG}-${COMMIT_HASH}"
mkdir -p "$REVIEW_DIR"
git diff HEAD > "$REVIEW_DIR/diff.patch"

# Claude writes: reviews/add_coverage_trackin-a1b2c3d/claude-add-coverage-tracking-review.md
# Codex writes: reviews/add_coverage_trackin-a1b2c3d/codex-add-coverage-tracking-review.md
# Final: reviews/add_coverage_trackin-a1b2c3d/final-add-coverage-tracking-report.md
```

## Common Mistakes

- **Modifying source code**: Agents write tests only, no src/ changes
- **Skipping cross-review**: Second perspective catches false positives
- **Vague bugs**: Always include file:line and concrete fix
