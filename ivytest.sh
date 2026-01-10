#!/usr/bin/env bash
set -euo pipefail

# Agent-friendly test runner for ivy
# Usage:
#   ivytest tests/ivy/test_e2e.py              # Run all tests in module
#   ivytest tests/ivy/test_e2e.py::test_foo    # Run specific test
#   ivytest --refresh tests/                   # Refresh vyper exports first
#   ivytest --loud tests/                       # Run without suppressing output

VYPER_REPO="../vyper"
VYPER_EXPORTS="$VYPER_REPO/tests/export"

# Find ivy repo root from current directory
IVY_REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "Error: Must be run from within an ivy worktree"
    exit 1
}
IVY_EXPORTS_LINK="$IVY_REPO/tests/vyper-exports"

refresh=false
loud=false
args=()

for arg in "$@"; do
    case $arg in
        --refresh)
            refresh=true
            ;;
        --loud)
            loud=true
            ;;
        *)
            args+=("$arg")
            ;;
    esac
done

ensure_exports() {
    # Check if exports directory exists (either as symlink or real directory)
    if [[ ! -e "$IVY_EXPORTS_LINK" ]]; then
        # Try to create symlink if vyper exports exist
        if [[ -d "$VYPER_EXPORTS" ]]; then
            echo "Creating symlink: $IVY_EXPORTS_LINK -> $VYPER_EXPORTS"
            ln -s "$VYPER_EXPORTS" "$IVY_EXPORTS_LINK"
        else
            echo "Warning: No exports found at $IVY_EXPORTS_LINK or $VYPER_EXPORTS"
        fi
    fi
}

check_vyper_repo() {
    if [[ ! -d "$VYPER_REPO" ]]; then
        echo "Error: Vyper repository not found at $VYPER_REPO"
        exit 1
    fi

    local current_branch
    current_branch="$(git -C "$VYPER_REPO" branch --show-current)"
    if [[ "$current_branch" != "export-tests" ]]; then
        echo "Error: Vyper repo must be on 'export-tests' branch (currently on '$current_branch')"
        exit 1
    fi
}

run_refresh() {
    check_vyper_repo

    echo "Removing existing exports at $VYPER_EXPORTS"
    rm -rf "$VYPER_EXPORTS"

    echo "Regenerating exports in vyper repo..."
    (
        cd "$VYPER_REPO"
        uv run pytest -s -n 1 --export tests/export -m "not fuzzing" tests/
    )

    echo "Exports regenerated."
}

run_tests() {
    cd "$IVY_REPO"

    # Loud mode: run pytest directly without suppressing output
    if $loud; then
        uv run pytest --tb=short -n auto -x "${args[@]}"
        return
    fi

    # Create temp file for output
    local tmpfile
    tmpfile=$(mktemp)
    trap "rm -f '$tmpfile'" EXIT

    # Run pytest with minimal output, capture everything
    # --tb=short for concise tracebacks, -q for quiet mode
    if uv run pytest -q --tb=short --no-header -n auto -x "${args[@]}" > "$tmpfile" 2>&1; then
        # Success - just print a minimal message
        # Extract the summary line (e.g., "5 passed in 0.12s")
        local summary
        summary=$(grep -E "^[0-9]+ passed" "$tmpfile" | tail -1) || summary=""
        if [[ -n "$summary" ]]; then
            echo "OK: $summary"
        else
            echo "OK"
        fi
    else
        local exit_code=$?
        # Failure - show only the relevant failure info
        echo "FAILED"
        echo ""
        # Filter out noise, keep traceback and assertion errors
        cat "$tmpfile" | \
            grep -v "^=\+" | \
            grep -v "^_\+.*_\+$" | \
            grep -v "^$" | \
            grep -v "^collected " | \
            grep -v "^\.\+$" | \
            grep -v "^[\.FEsx]\+\s*\[" | \
            grep -v "^!!!!!" | \
            grep -v "^FAILED.*- " | \
            grep -v " passed" | \
            grep -v "passed," | \
            grep -v "^[0-9]\+ failed" || true
        exit $exit_code
    fi
}

if $refresh; then
    run_refresh
fi

ensure_exports

if [[ ${#args[@]} -eq 0 ]]; then
    echo "Usage: ivytest <test_module> [::test_name]"
    echo ""
    echo "Examples:"
    echo "  ivytest tests/ivy/test_e2e.py"
    echo "  ivytest tests/ivy/test_e2e.py::test_basic_call"
    echo "  ivytest tests/test_replay.py -k 'test_slice'"
    echo ""
    echo "Options:"
    echo "  --refresh    Regenerate vyper exports before running tests"
    echo "  --loud       Run without suppressing output"
    exit 1
fi

run_tests
