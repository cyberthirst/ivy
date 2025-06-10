#!/bin/bash

echo "Setting up Ivy development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [[ $(echo "$python_version < $required_version" | bc) -eq 1 ]]; then
    echo "Error: Python $required_version or higher is required. Current version: $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install package with dev dependencies
echo "Installing Ivy with development dependencies..."
pip install -e ".[dev]"

# Run tests to verify installation
echo "Running basic tests to verify installation..."
PYTHONPATH=src pytest tests/ivy/test_e2e.py::test_if_control_flow -xvs

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To run with Vyper test exports:"
echo "  1. Copy exports to tests/vyper-exports/"
echo "  2. Run: python examples/differential_testing.py"