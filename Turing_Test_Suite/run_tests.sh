#!/usr/bin/env bash
# Run the full Turing-main test suite (backend + frontend).
# Usage: ./run_tests.sh [--backend-only | --frontend-only]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
UPLOAD_DIR="$PROJECT_ROOT"

echo "=========================================="
echo " Turing-main Test Runner"
echo "=========================================="
echo ""

# Parse args
RUN_BACKEND=true
RUN_FRONTEND=true
if [[ "$1" == "--backend-only" ]]; then
    RUN_FRONTEND=false
elif [[ "$1" == "--frontend-only" ]]; then
    RUN_BACKEND=false
fi

# ---- Backend tests ----
if [[ "$RUN_BACKEND" == "true" ]]; then
    echo "[1/2] Running BACKEND tests (pytest)..."
    echo "      Test dir: $SCRIPT_DIR/backend"
    cd "$SCRIPT_DIR/backend"

    # Install test dependencies if not present
    if ! python -c "import pytest" 2>/dev/null; then
        echo "      Installing test dependencies..."
        pip install -r requirements-test.txt
    fi

    # Install project dependencies
    if ! python -c "import fastapi" 2>/dev/null; then
        echo "      Installing project dependencies..."
        pip install -r "$UPLOAD_DIR/python-engine/requirements.txt"
    fi

    # Run pytest with verbose output
    python -m pytest -v --tb=short --maxfail=10 "$@"
    BACKEND_EXIT=$?
    echo ""
    echo "      Backend tests exit code: $BACKEND_EXIT"
else
    echo "[1/2] Skipping backend tests (--frontend-only)"
    BACKEND_EXIT=0
fi

# ---- Frontend tests ----
if [[ "$RUN_FRONTEND" == "true" ]]; then
    echo ""
    echo "[2/2] Running FRONTEND tests (vitest)..."
    echo "      Test dir: $SCRIPT_DIR/frontend"
    cd "$SCRIPT_DIR/frontend"

    # Install node dependencies if not present
    if [[ ! -d "node_modules" ]]; then
        echo "      Installing node dependencies..."
        npm install --no-audit --no-fund
    fi

    # Run vitest
    npm test -- --run
    FRONTEND_EXIT=$?
    echo ""
    echo "      Frontend tests exit code: $FRONTEND_EXIT"
else
    echo "[2/2] Skipping frontend tests (--backend-only)"
    FRONTEND_EXIT=0
fi

# ---- Summary ----
echo ""
echo "=========================================="
echo " Test Summary"
echo "=========================================="
if [[ "$RUN_BACKEND" == "true" ]]; then
    if [[ $BACKEND_EXIT -eq 0 ]]; then
        echo " [PASS] Backend: all tests passed"
    else
        echo " [FAIL] Backend: some tests failed (exit $BACKEND_EXIT)"
    fi
fi
if [[ "$RUN_FRONTEND" == "true" ]]; then
    if [[ $FRONTEND_EXIT -eq 0 ]]; then
        echo " [PASS] Frontend: all tests passed"
    else
        echo " [FAIL] Frontend: some tests failed (exit $FRONTEND_EXIT)"
    fi
fi
echo "=========================================="

# Exit with non-zero if any test failed
[[ $BACKEND_EXIT -eq 0 && $FRONTEND_EXIT -eq 0 ]]