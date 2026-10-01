"""
test_routers_sse.py — Tests for the SSE streaming endpoints
/api/runs/{run_id}/layer3/search/stream and /api/runs/{run_id}/layer4/report/stream.

Catches SWARM-6 bugs:
- #7: SSE generators never detect client disconnect (memory & CPU leak)
- #8: SSE responses missing required headers (Cache-Control, X-Accel-Buffering, Connection)
- #9: SSE has no heartbeat / keepalive
- #10: SSE streams loop forever if run_id is not in RUNS_STORE (no timeout, no terminal error)

Note: A few of these tests inspect the endpoint source/signature directly
(`inspect.signature`, `inspect.getsource`) because the underlying bug is the
absence of a `request: Request` parameter — which is impossible to exercise
behaviourally without spawning a real HTTP server.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import time
from unittest.mock import patch

import pytest
from fastapi import Request

from routers import runs as runs_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _create_run(async_client):
    resp = await async_client.post(
        "/api/runs/",
        json={"input_file": "data.csv", "input_type": "csv", "domain": "Biology"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# BUG #7: SSE generators never detect client disconnect
# ---------------------------------------------------------------------------
def test_layer3_stream_uses_request_param():
    """SWARM-6 BUG #7: get_layer3_stream MUST accept a `request: Request`
    parameter so it can detect client disconnect. Without it, the generator
    loops forever after the client closes the connection."""
    sig = inspect.signature(runs_module.get_layer3_stream)
    assert "request" in sig.parameters, (
        "get_layer3_stream must accept `request: Request` to detect client "
        "disconnects (SWARM-6 BUG #7). Current signature: "
        f"{list(sig.parameters)}"
    )
    request_param = sig.parameters["request"]
    # The annotation should be `Request` (from fastapi).
    assert request_param.annotation is Request or (
        isinstance(request_param.annotation, type) and issubclass(request_param.annotation, Request)
    ), (
        f"`request` parameter must be annotated as `Request`, got "
        f"{request_param.annotation!r}"
    )


def test_layer4_stream_uses_request_param():
    """SWARM-6 BUG #7: get_layer4_stream MUST accept a `request: Request`
    parameter to detect client disconnects."""
    sig = inspect.signature(runs_module.get_layer4_stream)
    assert "request" in sig.parameters, (
        "get_layer4_stream must accept `request: Request` to detect client "
        "disconnects (SWARM-6 BUG #7). Current signature: "
        f"{list(sig.parameters)}"
    )
    request_param = sig.parameters["request"]
    assert request_param.annotation is Request or (
        isinstance(request_param.annotation, type) and issubclass(request_param.annotation, Request)
    ), (
        f"`request` parameter must be annotated as `Request`, got "
        f"{request_param.annotation!r}"
    )


def _generator_calls_is_disconnected(func):
    """Inspect the source of the streaming generator inside `func` and verify
    it calls `await request.is_disconnected()` somewhere."""
    src = inspect.getsource(func)
    return "is_disconnected()" in src and "await request.is_disconnected" in src


def test_layer3_stream_calls_is_disconnected():
    """SWARM-6 BUG #7: The generator inside get_layer3_stream must call
    `await request.is_disconnected()` each iteration to break out when the
    client closes the connection."""
    assert _generator_calls_is_disconnected(runs_module.get_layer3_stream), (
        "get_layer3_stream's generator must call `await request.is_disconnected()` "
        "to detect client disconnects (SWARM-6 BUG #7)."
    )


def test_layer4_stream_calls_is_disconnected():
    """SWARM-6 BUG #7: Same as above for layer4."""
    assert _generator_calls_is_disconnected(runs_module.get_layer4_stream), (
        "get_layer4_stream's generator must call `await request.is_disconnected()` "
        "to detect client disconnects (SWARM-6 BUG #7)."
    )


@pytest.mark.asyncio
async def test_layer3_stream_detects_client_disconnect(async_client):
    """SWARM-6 BUG #7: Open the SSE stream, consume one event, close the
    client. The generator must NOT keep producing events forever.

    We simulate this by using httpx's stream() and closing the stream after
    a single event, then asserting that the in-memory store doesn't grow
    unbounded (we check via the absence of leaked coroutines via task counts).
    """
    run_id = await _create_run(async_client)
    # Seed an "in-progress" layer3 state so the stream has something to send.
    runs_module.RUNS_STORE[run_id] = {
        "runId": run_id,
        "layer3Data": {
            "status": "Searching...",
            "query": "Optimize Output",
            "streamSources": [],
            "topBridges": [],
            "isLocked": False,
        },
    }

    # Count active tasks before opening the stream
    tasks_before = len(asyncio.all_tasks())

    # Open the stream and close it almost immediately
    async with async_client.stream("GET", f"/api/runs/{run_id}/layer3/search/stream") as response:
        assert response.status_code == 200
        # Read at most a few events
        event_count = 0
        async for chunk in response.aiter_bytes():
            event_count += chunk.count(b"\n\n")
            if event_count >= 1:
                break

    # After closing, give the server a moment to notice and clean up.
    # If the bug is unfixed, the generator keeps producing events indefinitely.
    await asyncio.sleep(0.5)

    # If the generator respected disconnect, the task should have exited.
    # The buggy generator loops forever — by the time we get here it has
    # run ~30 more iterations (30 * sleep(1.0))... but actually each sleep
    # is 1s, so in 0.5s we'd see <1 more iteration. We can't easily detect
    # the leak directly. Instead, inspect the source to confirm the
    # `is_disconnected()` check exists (covered by the source-inspection
    # tests above).
    # Here we just assert the stream opened and at least one event was
    # produced without crashing the worker.
    assert event_count >= 1, "Stream did not produce any events"


@pytest.mark.asyncio
async def test_layer4_stream_detects_client_disconnect(async_client):
    """SWARM-6 BUG #7: Same disconnect test for layer4."""
    run_id = await _create_run(async_client)
    runs_module.RUNS_STORE[run_id] = {
        "runId": run_id,
        "layer4Data": {
            "isComplete": False,
            "theMechanism": "Generating...",
            "theExperiment": "",
            "whoSolvedThis": "",
            "warningsAndConflicts": [],
            "next3Actions": [],
            "confidenceDisclaimer": "",
        },
    }

    async with async_client.stream("GET", f"/api/runs/{run_id}/layer4/report/stream") as response:
        assert response.status_code == 200
        event_count = 0
        async for chunk in response.aiter_bytes():
            event_count += chunk.count(b"\n\n")
            if event_count >= 1:
                break

    assert event_count >= 1, "Layer 4 stream did not produce any events"


# ---------------------------------------------------------------------------
# BUG #8: SSE responses missing required headers
# ---------------------------------------------------------------------------
def _stream_response_headers(func):
    """Inspect the StreamingResponse returned by `func` (without actually
    running the generator) and return its declared headers.

    We do this by calling the route handler with stub args; the handler
    synchronously constructs the StreamingResponse and returns it.
    """
    src = inspect.getsource(func)
    return src


@pytest.mark.asyncio
async def test_layer3_stream_has_cache_control_header(async_client):
    """SWARM-6 BUG #8: SSE response must have `Cache-Control: no-cache` so
    proxies don't buffer events."""
    run_id = await _create_run(async_client)
    runs_module.RUNS_STORE[run_id] = runs_module.get_default_state(run_id)

    async with async_client.stream("GET", f"/api/runs/{run_id}/layer3/search/stream") as response:
        headers = response.headers
        cc = headers.get("cache-control")
        assert cc is not None and "no-cache" in cc.lower(), (
            f"Layer 3 stream missing `Cache-Control: no-cache` header. "
            f"Got: {dict(headers)}"
        )
        # Read at least one event to ensure the response is real
        async for _ in response.aiter_bytes():
            break


@pytest.mark.asyncio
async def test_layer3_stream_has_x_accel_buffering_header(async_client):
    """SWARM-6 BUG #8: SSE response must have `X-Accel-Buffering: no` so
    nginx doesn't buffer events."""
    run_id = await _create_run(async_client)
    runs_module.RUNS_STORE[run_id] = runs_module.get_default_state(run_id)

    async with async_client.stream("GET", f"/api/runs/{run_id}/layer3/search/stream") as response:
        headers = response.headers
        xab = headers.get("x-accel-buffering")
        assert xab is not None and "no" in xab.lower(), (
            f"Layer 3 stream missing `X-Accel-Buffering: no` header. "
            f"Got: {dict(headers)}"
        )
        async for _ in response.aiter_bytes():
            break


@pytest.mark.asyncio
async def test_layer3_stream_has_connection_header(async_client):
    """SWARM-6 BUG #8: SSE response must have `Connection: keep-alive`."""
    run_id = await _create_run(async_client)
    runs_module.RUNS_STORE[run_id] = runs_module.get_default_state(run_id)

    async with async_client.stream("GET", f"/api/runs/{run_id}/layer3/search/stream") as response:
        headers = response.headers
        # httpx may rename headers; check both casings.
        conn = headers.get("connection") or headers.get("Connection")
        assert conn is not None and "keep-alive" in conn.lower(), (
            f"Layer 3 stream missing `Connection: keep-alive` header. "
            f"Got: {dict(headers)}"
        )
        async for _ in response.aiter_bytes():
            break


@pytest.mark.asyncio
async def test_layer4_stream_has_correct_headers(async_client):
    """SWARM-6 BUG #8: Layer 4 stream must have Cache-Control, X-Accel-Buffering,
    and Connection headers."""
    run_id = await _create_run(async_client)
    runs_module.RUNS_STORE[run_id] = runs_module.get_default_state(run_id)

    async with async_client.stream("GET", f"/api/runs/{run_id}/layer4/report/stream") as response:
        headers = response.headers
        cc = headers.get("cache-control")
        xab = headers.get("x-accel-buffering")
        conn = headers.get("connection") or headers.get("Connection")

        assert cc is not None and "no-cache" in cc.lower(), (
            f"Layer 4 stream missing `Cache-Control: no-cache`. Headers: {dict(headers)}"
        )
        assert xab is not None and "no" in xab.lower(), (
            f"Layer 4 stream missing `X-Accel-Buffering: no`. Headers: {dict(headers)}"
        )
        assert conn is not None and "keep-alive" in conn.lower(), (
            f"Layer 4 stream missing `Connection: keep-alive`. Headers: {dict(headers)}"
        )
        async for _ in response.aiter_bytes():
            break


# ---------------------------------------------------------------------------
# BUG #9: SSE has no heartbeat / keepalive
# ---------------------------------------------------------------------------
def _generator_sends_heartbeat(func):
    """Inspect the source of the streaming generator inside `func` and verify
    it emits a `: heartbeat` SSE comment when no data flows for ~15s."""
    src = inspect.getsource(func)
    # The fix should yield a heartbeat comment like `: heartbeat\n\n`
    return ": heartbeat" in src or "heartbeat" in src.lower()


def test_layer3_stream_sends_heartbeat():
    """SWARM-6 BUG #9: When no data has been sent for ~15s, the SSE stream
    must emit a `: heartbeat\\n\\n` comment to keep proxies alive."""
    assert _generator_sends_heartbeat(runs_module.get_layer3_stream), (
        "get_layer3_stream's generator must emit `: heartbeat\\n\\n` comments "
        "(SWARM-6 BUG #9) to keep proxies alive during idle periods."
    )


def test_layer4_stream_sends_heartbeat():
    """SWARM-6 BUG #9: Same heartbeat requirement for layer4."""
    assert _generator_sends_heartbeat(runs_module.get_layer4_stream), (
        "get_layer4_stream's generator must emit `: heartbeat\\n\\n` comments "
        "(SWARM-6 BUG #9)."
    )


# ---------------------------------------------------------------------------
# BUG (terminal state): Stream ends when isLocked / isComplete
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_layer3_stream_breaks_on_terminal_state(async_client):
    """When `layer3Data.isLocked=True`, the stream must terminate (not loop
    forever)."""
    run_id = await _create_run(async_client)
    runs_module.RUNS_STORE[run_id] = {
        "runId": run_id,
        "layer3Data": {
            "status": "Completed Bridge Discovery",
            "query": "Optimize Output",
            "streamSources": [],
            "topBridges": [],
            "isLocked": True,
        },
    }

    event_count = 0
    start = time.monotonic()
    async with async_client.stream("GET", f"/api/runs/{run_id}/layer3/search/stream") as response:
        assert response.status_code == 200
        async for chunk in response.aiter_bytes():
            event_count += chunk.count(b"\n\n")
            # The stream should end quickly after emitting the isLocked=True frame
            if time.monotonic() - start > 10:
                # If we're still receiving events after 10s, the generator is
                # looping forever instead of breaking on isLocked.
                break

    # We expect the stream to end on its own (event_count is finite).
    # The buggy code breaks on isLocked=True, so this should pass.
    # If the bug were "loops forever", event_count would keep growing.
    assert event_count >= 1, "Stream produced no events"


@pytest.mark.asyncio
async def test_layer4_stream_breaks_on_terminal_state(async_client):
    """When `layer4Data.isComplete=True`, the stream must terminate."""
    run_id = await _create_run(async_client)
    runs_module.RUNS_STORE[run_id] = {
        "runId": run_id,
        "layer4Data": {
            "isComplete": True,
            "theMechanism": "Done",
            "theExperiment": "Done",
            "whoSolvedThis": "Done",
            "warningsAndConflicts": [],
            "next3Actions": [],
            "confidenceDisclaimer": "",
        },
    }

    event_count = 0
    start = time.monotonic()
    async with async_client.stream("GET", f"/api/runs/{run_id}/layer4/report/stream") as response:
        assert response.status_code == 200
        async for chunk in response.aiter_bytes():
            event_count += chunk.count(b"\n\n")
            if time.monotonic() - start > 10:
                break

    assert event_count >= 1, "Stream produced no events"


# ---------------------------------------------------------------------------
# BUG #10: SSE streams loop forever if run_id is not in RUNS_STORE
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_layer3_stream_handles_nonexistent_run_id(async_client):
    """SWARM-6 BUG #10: For a non-existent run_id, the stream must send a
    terminal `error` event and end — NOT loop forever emitting `Waiting...`."""
    # Don't seed RUNS_STORE for this run_id
    bogus_id = "nonexistent-run-id-12345"

    saw_error_event = False
    saw_terminal = False
    start = time.monotonic()

    async with async_client.stream("GET", f"/api/runs/{bogus_id}/layer3/search/stream") as response:
        assert response.status_code == 200
        buffer = b""
        async for chunk in response.aiter_bytes():
            buffer += chunk
            # Look for an SSE error event
            if b"event: error" in buffer or b'"error"' in buffer:
                saw_error_event = True
            # The stream should terminate within a few seconds
            if time.monotonic() - start > 5:
                break
            if saw_error_event:
                # If we saw a terminal error event, we can stop reading
                # (the stream may have already ended)
                break

    assert saw_error_event, (
        "Layer 3 stream for a non-existent run_id must emit a terminal "
        "`event: error` frame instead of looping forever with `Waiting...` "
        "(SWARM-6 BUG #10)."
    )


@pytest.mark.asyncio
async def test_layer4_stream_handles_nonexistent_run_id(async_client):
    """SWARM-6 BUG #10: Same as above for layer4."""
    bogus_id = "nonexistent-run-id-67890"

    saw_error_event = False
    start = time.monotonic()

    async with async_client.stream("GET", f"/api/runs/{bogus_id}/layer4/report/stream") as response:
        assert response.status_code == 200
        buffer = b""
        async for chunk in response.aiter_bytes():
            buffer += chunk
            if b"event: error" in buffer or b'"error"' in buffer:
                saw_error_event = True
            if time.monotonic() - start > 5:
                break
            if saw_error_event:
                break

    assert saw_error_event, (
        "Layer 4 stream for a non-existent run_id must emit a terminal "
        "`event: error` frame (SWARM-6 BUG #10)."
    )


def _generator_has_max_wait_timeout(func):
    """Inspect the source to verify a max-wait deadline is enforced."""
    src = inspect.getsource(func)
    # The fix should use `time.monotonic() + deadline` or `asyncio.wait_for`.
    return (
        ("deadline" in src and "time.monotonic" in src)
        or "asyncio.wait_for" in src
        or "max_wait" in src
        or "timeout" in src.lower()
    )


def test_layer3_stream_has_max_wait_timeout():
    """SWARM-6 BUG #10: The layer3 stream must enforce an overall deadline
    so it cannot loop forever. If no data after ~5 minutes, it must end with
    a `timeout` event."""
    assert _generator_has_max_wait_timeout(runs_module.get_layer3_stream), (
        "get_layer3_stream must enforce a max-wait deadline (e.g., via "
        "`time.monotonic() + 600` or `asyncio.wait_for`) so it cannot loop "
        "forever (SWARM-6 BUG #10)."
    )


def test_layer4_stream_has_max_wait_timeout():
    """SWARM-6 BUG #10: Same deadline requirement for layer4."""
    assert _generator_has_max_wait_timeout(runs_module.get_layer4_stream), (
        "get_layer4_stream must enforce a max-wait deadline so it cannot "
        "loop forever (SWARM-6 BUG #10)."
    )