#!/bin/bash

echo "Setting up Ivy development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install dependencies (creates .venv/ automatically)
echo "Installing dependencies with uv..."
uv sync --all-extras

# Run tests to verify installation
echo "Running basic tests to verify installation..."
uv run pytest tests/ivy/test_e2e.py::test_if_control_flow -xvs

echo ""
echo "Development environment setup complete!"
echo ""
echo "To run commands, use 'uv run':"
echo "  uv run pytest"
echo "  uv run python -m src.fuzzer.parallel_fuzzer"
echo ""
echo "To run with Vyper test exports:"
echo "  1. Copy exports to tests/vyper-exports/"
echo "  2. Run: uv run python examples/differential_testing.py"
