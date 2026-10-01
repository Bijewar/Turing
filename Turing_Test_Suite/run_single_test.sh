#!/usr/bin/env bash
# Quick test: run a single test file or test pattern.
# Usage:
#   ./run_single_test.sh test_layer1_validator.py
#   ./run_single_test.sh test_layer1_validator.py::TestClassifyGraphJsonExtraction::test_classify_graph_regex_not_greedy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/backend"

if [[ ! -f "conftest.py" ]]; then
    echo "Error: backend/conftest.py not found. Are you running from the right dir?"
    exit 1
fi

# Install test deps if needed
if ! python -c "import pytest" 2>/dev/null; then
    pip install -r requirements-test.txt
fi

python -m pytest -v --tb=long --maxfail=1 "$@"