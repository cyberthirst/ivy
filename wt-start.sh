#!/usr/bin/env bash
#
# wt-start - Create a worktree and port uncommitted changes
#
# Usage: wt-start <task-name>
#
# This script:
# 1. Saves any uncommitted changes (staged + unstaged) from current directory
# 2. Creates a new worktree from the current branch with `wt switch --create <task-name> --base=@`
# 3. Applies the saved changes to the new worktree
# 4. Prints the worktree path for the agent to cd into
#

set -euo pipefail

# --- Argument validation ---
if [ -z "${1:-}" ]; then
    echo "Usage: wt-start <task-name>" >&2
    echo "Example: wt-start fix-slice-bug" >&2
    exit 1
fi

TASK_NAME="$1"

# Validate task name: no spaces, no leading slashes, alphanumeric + dashes/underscores
if [[ ! "$TASK_NAME" =~ ^[a-zA-Z0-9][a-zA-Z0-9_/-]*$ ]]; then
    echo "Error: Invalid task name '$TASK_NAME'" >&2
    echo "Task name must start with alphanumeric and contain only alphanumeric, dash, underscore, or slash" >&2
    exit 1
fi

# --- Git repository checks ---
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "Error: Not in a git repository" >&2
    exit 1
fi

GIT_ROOT="$(git rev-parse --show-toplevel)"

# Check if branch/worktree already exists
if git show-ref --verify --quiet "refs/heads/$TASK_NAME"; then
    echo "Error: Branch '$TASK_NAME' already exists" >&2
    echo "Use a different name or remove the existing branch/worktree first:" >&2
    echo "  git branch -d $TASK_NAME" >&2
    echo "  wt remove $TASK_NAME" >&2
    exit 1
fi

# Check for merge conflicts
if git ls-files -u | grep -q .; then
    echo "Error: Unresolved merge conflicts detected" >&2
    echo "Resolve conflicts before creating a new worktree" >&2
    exit 1
fi

# --- Save uncommitted changes ---
PATCH_FILE="/tmp/wt-start-$$.patch"
HAS_CHANGES=false

if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    HAS_CHANGES=true
fi

if [ "$HAS_CHANGES" = true ]; then
    echo "Saving uncommitted changes..."
    # Use --binary to handle binary files
    git diff --binary HEAD > "$PATCH_FILE"
    if [ ! -s "$PATCH_FILE" ]; then
        rm -f "$PATCH_FILE"
        HAS_CHANGES=false
    fi
fi

# Track untracked files for later notification
UNTRACKED_FILES=$(git ls-files --others --exclude-standard)

# --- Create the worktree ---
echo "Creating worktree '$TASK_NAME'..."

# Capture output and exit status separately
WT_OUTPUT=$(wt switch --create "$TASK_NAME" --base=@ 2>&1) || true
WT_EXIT=$?

# Check if worktree was actually created by looking for it
WORKTREE_PATH=$(git worktree list --porcelain | grep -B2 "branch refs/heads/$TASK_NAME" | grep "^worktree " | sed 's/^worktree //' || true)

if [ -z "$WORKTREE_PATH" ]; then
    echo "Error: Failed to create worktree '$TASK_NAME'" >&2
    echo "$WT_OUTPUT" >&2
    rm -f "$PATCH_FILE"
    exit 1
fi

# Print wt output (contains useful info like post-create hooks)
echo "$WT_OUTPUT"

# --- Apply the patch ---
if [ "$HAS_CHANGES" = true ] && [ -f "$PATCH_FILE" ]; then
    echo ""
    echo "Applying uncommitted changes to worktree..."

    APPLY_SUCCESS=false

    # Try direct apply first
    if git -C "$WORKTREE_PATH" apply --binary "$PATCH_FILE" 2>/dev/null; then
        APPLY_SUCCESS=true
        echo "Changes applied successfully."
    else
        # Try 3-way merge
        echo "Direct apply failed, trying 3-way merge..."
        if git -C "$WORKTREE_PATH" apply --binary --3way "$PATCH_FILE" 2>/dev/null; then
            APPLY_SUCCESS=true
            echo "Changes applied with 3-way merge."
        fi
    fi

    if [ "$APPLY_SUCCESS" = true ]; then
        rm -f "$PATCH_FILE"
    else
        echo ""
        echo "Warning: Patch could not be applied automatically." >&2
        echo "This may happen if the base commit differs or there are conflicts." >&2
        echo "Patch saved at: $PATCH_FILE" >&2
        echo "To apply manually:" >&2
        echo "  cd $WORKTREE_PATH && git apply --binary $PATCH_FILE" >&2
    fi
fi

# --- Report untracked files ---
if [ -n "$UNTRACKED_FILES" ]; then
    echo ""
    echo "Note: Untracked files were NOT copied:"
    echo "$UNTRACKED_FILES" | head -10 | while read -r f; do
        # Create parent directory in cp command if needed
        DIR=$(dirname "$f")
        if [ "$DIR" != "." ]; then
            echo "  mkdir -p \"$WORKTREE_PATH/$DIR\" && cp \"$GIT_ROOT/$f\" \"$WORKTREE_PATH/$f\""
        else
            echo "  cp \"$GIT_ROOT/$f\" \"$WORKTREE_PATH/$f\""
        fi
    done
    UNTRACKED_COUNT=$(echo "$UNTRACKED_FILES" | wc -l)
    if [ "$UNTRACKED_COUNT" -gt 10 ]; then
        echo "  ... and $((UNTRACKED_COUNT - 10)) more"
    fi
fi

# --- Final output ---
echo ""
echo "========================================"
echo "Worktree ready: $WORKTREE_PATH"
echo "========================================"
echo ""
echo "To start working:"
echo "  cd $WORKTREE_PATH"
