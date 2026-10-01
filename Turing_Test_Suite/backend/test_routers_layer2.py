"""
test_routers_layer2.py

Tests for `python-engine/routers/layer2.py`.

Catches SWARM-4 BUGs #26, #36 and SWARM-6 BUGs #27, #29, #60.
"""

from __future__ import annotations

import inspect
import re
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

import os

ROUTER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "python-engine", "routers", "layer2.py")


def _read_source():
    with open(ROUTER_PATH) as f:
        return f.read()


# ===========================================================================
# simulate endpoint uses typed request model
# ===========================================================================
class TestTypedRequestModel:
    def test_simulate_uses_typed_request_model(self):
        """Verify the `/simulate` endpoint takes a Pydantic model
        (Layer2Request), not `Dict[str, Any]`."""
        source = _read_source()
        if re.search(r"async\s+def\s+simulate\s*\([^)]*:\s*Dict\[str,\s*Any\]", source, re.DOTALL):
            pytest.fail(
                "/simulate takes `Dict[str, Any]` — no schema validation. "
                "Use the existing Layer2Request Pydantic model."
            )


# ===========================================================================
# max_iterations bounded (BUG #60)
# ===========================================================================
class TestMaxIterationsBounded:
    """Locks in SWARM-6 BUG #60: Layer2Request.max_iterations has no upper
    bound. A user passing 1_000_000 spawns 1M LLM calls (self-DoS + huge bill)."""

    def test_simulate_max_iterations_bounded(self):
        """Verify the schema enforces `max_iterations <= 50` (or similar)."""
        from schemas.layer2 import Layer2Request
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Layer2Request(
                graph=MagicMock(), target_node_id="X", max_iterations=1_000_000,
            )

    def test_simulate_max_iterations_ge_1(self):
        from schemas.layer2 import Layer2Request
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            Layer2Request(
                graph=MagicMock(), target_node_id="X", max_iterations=0,
            )


# ===========================================================================
# /simulate and /run-round don't block the event loop (BUG #27 / BUG #26)
# ===========================================================================
class TestNoEventLoopBlocking:
    def test_simulate_uses_to_thread(self):
        """Verify /simulate wraps sync calls in `asyncio.to_thread(...)`."""
        source = _read_source()
        m = re.search(r"async\s+def\s+simulate\s*\([^)]*\).*?(?=\nasync\s+def\s|\Z)",
                      source, re.DOTALL)
        if not m:
            pytest.skip("/simulate endpoint not found")
        body = m.group(0)
        sync_calls = ["Orchestrator(", ".simulate(", "bayesian_optimizer", "do_calculus"]
        for call in sync_calls:
            if call in body and "asyncio.to_thread" not in body:
                pytest.fail(
                    f"/simulate calls sync `{call}` directly — blocks the "
                    "event loop. Wrap in `await asyncio.to_thread(...)`."
                )

    def test_run_round_uses_return_exceptions(self):
        """Locks in SWARM-4 BUG #26: /run-round uses `asyncio.gather(...)`
        without `return_exceptions=True` — one bad agent kills the round."""
        source = _read_source()
        # Find run_round endpoint.
        m = re.search(r"async\s+def\s+run_round\s*\([^)]*\).*?(?=\nasync\s+def\s|\Z)",
                      source, re.DOTALL)
        if not m:
            pytest.skip("/run-round endpoint not found")
        body = m.group(0)
        # Look for asyncio.gather call.
        gather_calls = re.findall(r"asyncio\.gather\([^)]*\)", body, re.DOTALL)
        for call in gather_calls:
            assert "return_exceptions=True" in call, (
                f"/run-round uses `asyncio.gather(...)` without "
                f"`return_exceptions=True`: {call!r}. One bad agent "
                "proposal will crash the round. Use `return_exceptions=True`."
            )


# ===========================================================================
# Internal exceptions not leaked (BUG #29)
# ===========================================================================
class TestNoInternalExceptionLeak:
    def test_no_str_exc_in_http_exception(self):
        source = _read_source()
        bad_patterns = [
            r"HTTPException\([^)]*detail\s*=\s*str\(exc\)",
            r"HTTPException\([^)]*detail\s*=\s*str\(e\)",
            r"HTTPException\([^)]*detail\s*=\s*str\(err\)",
        ]
        for pat in bad_patterns:
            if re.search(pat, source, re.DOTALL):
                pytest.fail(
                    "Router uses `HTTPException(..., detail=str(exc))` — "
                    "internal exception messages leak to the client."
                )


# ===========================================================================
# mode-detect routes correctly (BUG #36)
# ===========================================================================
class TestModeDetectRouting:
    """Locks in SWARM-4 BUG #36: /mode-detect misroutes CSV input to
    LITERATURE mode. Only `"data"` (case-sensitive) triggers SIMULATION.
    A CSV file should trigger SIMULATION, not LITERATURE."""

    def test_csv_routes_to_simulation(self):
        """Verify /mode-detect returns 'simulation' for CSV input, not
        'literature'."""
        source = _read_source()
        # Find the mode-detect function.
        m = re.search(r"async\s+def\s+mode_detect\s*\([^)]*\).*?(?=\nasync\s+def\s|\Z)",
                      source, re.DOTALL)
        if not m:
            pytest.skip("/mode-detect endpoint not found")
        body = m.group(0)
        # Look for the routing logic. The bug: `if input_type == "data":`
        # is too strict — CSV files have input_type "csv" not "data".
        # Verify the routing accepts "csv" or "DATA" or includes a CSV check.
        if '=="data"' in body and 'csv' not in body.lower() and 'CSV' not in body:
            pytest.fail(
                "/mode-detect only triggers SIMULATION for the exact string "
                "'data' — but CSV files have input_type 'csv'. Use a list "
                "check: `if input_type in ('data', 'csv', 'DATA'):`."
            )


# ===========================================================================
# Endpoints have response_model
# ===========================================================================
class TestResponseModelSet:
    @pytest.mark.parametrize("endpoint_name", ["simulate", "run_round", "mode_detect"])
    def test_endpoint_has_response_model(self, endpoint_name):
        source = _read_source()
        idx = source.find(f"async def {endpoint_name}")
        if idx == -1:
            pytest.skip(f"Endpoint {endpoint_name} not found.")
        decorator_start = source.rfind("@router.", 0, idx)
        decorator_end = source.find(")\n", decorator_start)
        decorator = source[decorator_start:decorator_end]
        if "response_model" not in decorator:
            pytest.fail(
                f"Endpoint `{endpoint_name}` has no `response_model=` — "
                "internal fields may leak to the client."
            )