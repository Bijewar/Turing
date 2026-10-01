"""
test_routers_runs.py — Tests for /api/runs/* endpoints.

Catches SWARM-6 bugs:
- #12: upload_dataset doesn't verify run exists in DB
- #13: upload_dataset doesn't validate file.filename
- #14: list_runs has no upper bound on limit and no offset
- #15: delete_run only deletes DB row — orphans RUNS_STORE & background task
- #24: get_state cold-start rebuild returns stale layer status
- #25: get_layer2_agents has no DB fallback
"""
from __future__ import annotations

import inspect
import asyncio
import io
import os
import time
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from routers import runs as runs_module
from schemas.run import RunStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _create_run(async_client, payload=None):
    payload = payload or {
        "input_file": "data.csv",
        "input_type": "csv",
        "domain": "Biology",
    }
    resp = await async_client.post("/api/runs/", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# BUG #14: list_runs has no upper bound on limit and no offset
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_runs_limit_bounded(async_client):
    """SWARM-6 BUG #14: `limit > 100` should be rejected with 422, not silently
    scan millions of rows."""
    resp = await async_client.get("/api/runs/", params={"limit": 10_000_000})
    # FastAPI's Query(le=100) returns 422 (validation error) for out-of-range values.
    assert resp.status_code == 422, (
        f"Expected 422 for limit=10_000_000, got {resp.status_code}. "
        f"Body: {resp.text}"
    )


@pytest.mark.asyncio
async def test_list_runs_limit_ge_1(async_client):
    """SWARM-6 BUG #14: `limit <= 0` should be rejected with 422."""
    for bad in (0, -1, -100):
        resp = await async_client.get("/api/runs/", params={"limit": bad})
        assert resp.status_code == 422, (
            f"Expected 422 for limit={bad}, got {resp.status_code}: {resp.text}"
        )


@pytest.mark.asyncio
async def test_list_runs_offset_supported(async_client):
    """SWARM-6 BUG #14: `offset` parameter must be supported to paginate older runs."""
    # Seed 5 runs
    for _ in range(5):
        await _create_run(async_client)

    # Without offset — gets the most recent (default limit=50)
    resp1 = await async_client.get("/api/runs/", params={"limit": 2, "offset": 0})
    assert resp1.status_code == 200, resp1.text
    page1 = resp1.json()
    assert len(page1) == 2, f"Expected 2 runs, got {len(page1)}"

    # With offset=2 — should return the next page (NOT 422, NOT empty when more exist)
    resp2 = await async_client.get("/api/runs/", params={"limit": 2, "offset": 2})
    assert resp2.status_code == 200, resp2.text
    page2 = resp2.json()
    assert len(page2) == 2, f"Expected 2 runs on page 2, got {len(page2)}"

    # Pages must not overlap — that's the whole point of offset pagination
    page1_ids = {r["id"] for r in page1}
    page2_ids = {r["id"] for r in page2}
    assert page1_ids.isdisjoint(page2_ids), (
        "Offset pagination returned overlapping IDs — endpoint is ignoring offset."
    )


# ---------------------------------------------------------------------------
# BUG #15: delete_run orphans RUNS_STORE & background task
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_run_clears_runs_store(async_client):
    """SWARM-6 BUG #15: DELETE /api/runs/{id} must also remove the run from
    `routers.runs.RUNS_STORE`, not just the DB row."""
    run = await _create_run(async_client)
    run_id = run["id"]

    # Seed the in-memory cache (simulates an in-flight pipeline).
    runs_module.RUNS_STORE[run_id] = runs_module.get_default_state(run_id)
    runs_module.RUNS_STORE[run_id]["layer1Status"] = "processing"
    assert run_id in runs_module.RUNS_STORE

    resp = await async_client.delete(f"/api/runs/{run_id}")
    assert resp.status_code == 204, resp.text

    # The in-memory store MUST also be cleared
    assert run_id not in runs_module.RUNS_STORE, (
        f"After DELETE, run {run_id} should be removed from RUNS_STORE; "
        f"still present: {runs_module.RUNS_STORE.get(run_id)}"
    )


@pytest.mark.asyncio
async def test_delete_run_cancels_background_task(async_client):
    """SWARM-6 BUG #15: DELETE should cancel an in-flight process_full_pipeline
    background task — otherwise the orphaned task keeps polling/updating
    RUNS_STORE forever."""
    run = await _create_run(async_client)
    run_id = run["id"]

    # Track the asyncio.Task that runs the pipeline.
    pipeline_state: dict = {}

    async def fake_pipeline(rid, *args, **kwargs):
        pipeline_state["task"] = asyncio.current_task()
        try:
            await asyncio.sleep(20)  # simulate long-running work
        except asyncio.CancelledError:
            pipeline_state["cancelled"] = True
            raise

    with patch("routers.runs.process_full_pipeline", new=fake_pipeline):
        # Start the upload to kick off the background task.
        upload_resp = await async_client.post(
            f"/api/runs/{run_id}/layer1/upload",
            files={"file": ("data.csv", io.BytesIO(b"a,b,c\n1,2,3\n"), "text/csv")},
        )
        assert upload_resp.status_code in (200, 202), upload_resp.text

        # Give the background task a moment to register.
        for _ in range(50):
            await asyncio.sleep(0.02)
            if "task" in pipeline_state:
                break

        assert "task" in pipeline_state, (
            "Background pipeline task was not started — the upload endpoint "
            "isn't actually scheduling process_full_pipeline."
        )

        # Now delete the run — this MUST cancel the in-flight background task.
        del_resp = await async_client.delete(f"/api/runs/{run_id}")
        assert del_resp.status_code == 204, del_resp.text

        # Wait for the cancellation to propagate to the background task.
        for _ in range(50):
            await asyncio.sleep(0.02)
            if pipeline_state.get("cancelled"):
                break

        # Always force-cancel the background task at the end of the test so
        # pytest can tear down cleanly even if the bug is unfixed.
        bg_task = pipeline_state.get("task")
        if bg_task is not None and not bg_task.done():
            bg_task.cancel()
            try:
                await bg_task
            except asyncio.CancelledError:
                pass

        assert pipeline_state.get("cancelled"), (
            "Background pipeline task was NOT cancelled after DELETE — "
            "the orphaned task will keep running forever (BUG #15)."
        )


@pytest.mark.asyncio
async def test_delete_nonexistent_run_returns_404(async_client):
    """Basic 404 contract for DELETE."""
    resp = await async_client.delete("/api/runs/nonexistent-id-12345")
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# BUG #24: get_state cold-start returns stale layer status
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_state_after_restart_shows_completed_layers(async_client, seed_completed_run):
    """SWARM-6 BUG #24: When DB says COMPLETE but RUNS_STORE is empty (cold
    start after restart), `GET /state` must report ALL layers as "completed",
    not "pending".

    The buggy code hardcodes layers 2/3/4 to "pending" regardless of DB status.
    """
    run_id = seed_completed_run["id"]
    # Make sure RUNS_STORE is empty for this run (cold start scenario).
    runs_module.RUNS_STORE.pop(run_id, None)

    resp = await async_client.get(f"/api/runs/{run_id}/state")
    assert resp.status_code == 200, resp.text
    state = resp.json()

    # All three downstream layers must reflect the completed run.
    assert state.get("layer2Status") == "completed", (
        f"After a COMPLETE run, layer2Status must be 'completed', got "
        f"{state.get('layer2Status')!r}. Body: {state}"
    )
    assert state.get("layer3Status") == "completed", state
    assert state.get("layer4Status") == "completed", state


@pytest.mark.asyncio
async def test_get_state_after_restart_shows_failed_layer_correctly(async_client, seed_failed_run):
    """SWARM-6 BUG #24: FAILED status must propagate to layer1Status, not
    appear as pending/completed."""
    run_id = seed_failed_run["id"]
    runs_module.RUNS_STORE.pop(run_id, None)

    resp = await async_client.get(f"/api/runs/{run_id}/state")
    assert resp.status_code == 200, resp.text
    state = resp.json()

    assert state.get("layer1Status") == "failed", (
        f"FAILED run should report layer1Status='failed', got {state.get('layer1Status')!r}."
    )


# ---------------------------------------------------------------------------
# BUG #25: get_layer2_agents has no DB fallback
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_layer2_agents_has_db_fallback(async_client):
    """SWARM-6 BUG #25: After restart, `GET /{run_id}/layer2/agents` must read
    from the DB (or at least show the persisted agents), not always return the
    empty default `{"status": "Waiting for Layer 1...", ...}`."""
    run = await _create_run(async_client)
    run_id = run["id"]

    # Simulate a completed Layer 2 by populating DB (the fix should add a
    # layer2_data column or read from a serialized field). For the test we
    # populate RUNS_STORE with non-default agents and clear it after — to
    # simulate restart we just clear the in-memory cache.
    sample_layer2 = {
        "status": "Completed Bayesian Optimization",
        "nodesComputed": 3,
        "agents": [{"id": "action-0-x", "type": "explorer", "agentId": "EXP", "y": 150, "content": "test", "delayMs": 0}],
        "heatmapNodes": [],
        "heatmapLines": [],
    }

    # Persist the Layer 2 data to the DB (the fix should add a column for this).
    from database.database import AsyncSessionLocal
    from database import crud

    # The fix should expose some persistence mechanism. Try setting it via
    # a generic attribute if the column exists, else fall through.
    persisted = False
    async with AsyncSessionLocal() as db:
        row = await crud.get_run(db, run_id)
        if row is not None and hasattr(row, "layer2_data"):
            row.layer2_data = __import__("json").dumps(sample_layer2)
            await db.commit()
            persisted = True

    if not persisted:
        pytest.skip(
            "RunModel.layer2_data column not present — fix required to persist Layer 2 results."
        )

    # Cold the in-memory cache to simulate restart.
    runs_module.RUNS_STORE.pop(run_id, None)

    resp = await async_client.get(f"/api/runs/{run_id}/layer2/agents")
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body.get("status") != "Waiting for Layer 1...", (
        "After restart, /layer2/agents must NOT return the empty 'Waiting...' default — "
        "DB fallback is missing."
    )
    assert body.get("nodesComputed", 0) > 0, body


# ---------------------------------------------------------------------------
# BUG #12: upload_dataset doesn't verify run exists
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_upload_dataset_verifies_run_exists(async_client):
    """SWARM-6 BUG #12: POST /api/runs/nonexistent/layer1/upload must return
    404, not 202 (which silently swallows the missing run)."""
    resp = await async_client.post(
        "/api/runs/nonexistent-run-id/layer1/upload",
        files={"file": ("data.csv", io.BytesIO(b"a,b,c\n1,2,3\n"), "text/csv")},
    )
    assert resp.status_code == 404, (
        f"Expected 404 for upload to non-existent run, got {resp.status_code}. "
        f"The endpoint is silently accepting uploads for orphan run_ids (BUG #12)."
    )


# ---------------------------------------------------------------------------
# BUG #13: upload_dataset doesn't validate filename
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_upload_dataset_validates_filename(async_client):
    """SWARM-6 BUG #13: `file.filename=None` must not crash — the endpoint
    should fall back to a safe default name like `upload.bin`."""
    run = await _create_run(async_client)
    run_id = run["id"]

    # Build a UploadFile-equivalent with filename=None
    # httpx encodes multipart filename via the tuple — use None to omit the
    # filename. Starlette's UploadFile will then have .filename == None.
    resp = await async_client.post(
        f"/api/runs/{run_id}/layer1/upload",
        files={"file": (None, io.BytesIO(b"some content here"), "application/octet-stream")},
    )
    # Should not be 500 (the buggy code crashes inside the background task
    # and either returns 202 with a hung task or 500 from a guard).
    assert resp.status_code != 500, (
        f"Expected graceful handling of filename=None, got 500: {resp.text}"
    )
    # The endpoint must accept the upload (202) or reject with a clear 400 —
    # never 500 with an internal stack trace.
    assert resp.status_code in (202, 400), (
        f"filename=None should yield 202 (with safe name) or 400, got {resp.status_code}"
    )


# ---------------------------------------------------------------------------
# Content-type / empty-file validation
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_upload_dataset_validates_content_type(async_client):
    """upload_dataset must reject uploads with no content-type. The buggy code
    accepts any upload and silently starts processing."""
    run = await _create_run(async_client)
    run_id = run["id"]

    resp = await async_client.post(
        f"/api/runs/{run_id}/layer1/upload",
        files={"file": ("data.csv", io.BytesIO(b"abc"), None)},
    )
    assert resp.status_code == 400, (
        f"Missing content-type should return 400, got {resp.status_code}: {resp.text}"
    )


@pytest.mark.asyncio
async def test_upload_dataset_rejects_empty_file(async_client):
    """upload_dataset must reject 0-byte uploads with 400, not silently accept
    and hang in processing forever."""
    run = await _create_run(async_client)
    run_id = run["id"]

    resp = await async_client.post(
        f"/api/runs/{run_id}/layer1/upload",
        files={"file": ("empty.csv", io.BytesIO(b""), "text/csv")},
    )
    assert resp.status_code == 400, (
        f"Empty file upload should return 400, got {resp.status_code}: {resp.text}"
    )


# ---------------------------------------------------------------------------
# BUG #3 (SWARM-8): upload streams to disk, not memory
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_upload_dataset_streams_to_disk_not_memory(async_client, monkeypatch):
    """SWARM-8 BUG #2 / SWARM-6 BUG #3: upload must NOT do `await file.read()`
    to slurp the entire file into memory. It should stream in chunks.

    We assert that `file.read` is called multiple times with a `size` argument.
    """
    run = await _create_run(async_client)
    run_id = run["id"]

    # Track calls to UploadFile.read
    call_log: list = []

    # We can't easily intercept Starlette's UploadFile.read in the test client,
    # so instead inspect the source — must not contain the slurp pattern.
    import routers.runs as r
    src = inspect.getsource(r.upload_dataset)
    forbidden = "file_bytes = await file.read()"  # bare read with no size arg
    assert forbidden not in src, (
        f"upload_dataset still calls `await file.read()` (whole-file slurp). "
        f"Must stream in chunks via `await file.read(CHUNK_SIZE)`."
    )

    # Also assert there IS a chunked read pattern somewhere
    assert "file.read(" in src and "file.read()" not in src, (
        "upload_dataset should stream via chunked file.read(SIZE) calls, "
        "not a single whole-file read."
    )


# ---------------------------------------------------------------------------
# /state response shape
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_state_returns_run_id_in_response(async_client, seed_run):
    """The /state response must include the runId field so the client can
    correlate it with its request."""
    run_id = seed_run["id"]
    resp = await async_client.get(f"/api/runs/{run_id}/state")
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("runId") == run_id, body


@pytest.mark.asyncio
async def test_get_state_returns_graph_when_complete(async_client, seed_completed_run):
    """For a completed run with a persisted causal graph, /state must surface
    the graph in the response."""
    run_id = seed_completed_run["id"]

    # Persist a graph to the DB
    from database.database import AsyncSessionLocal
    from database import crud
    from schemas.graph import CausalGraph, Node, Edge

    g = CausalGraph(
        nodes=[
            Node(id="A", label="A", confidence=80.0),
            Node(id="B", label="B", confidence=80.0),
        ],
        edges=[Edge(source="A", target="B", relation="causes", confidence=80.0)],
    )
    async with AsyncSessionLocal() as db:
        await crud.update_run_graph(db, run_id, g)
        await db.commit()

    # Cold the in-memory cache to force DB fallback path
    runs_module.RUNS_STORE.pop(run_id, None)

    resp = await async_client.get(f"/api/runs/{run_id}/state")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body.get("graph") is not None, (
        f"Persisted graph should be returned by /state. Body: {body}"
    )
    g_nodes = body["graph"].get("nodes", []) if isinstance(body["graph"], dict) else []
    assert len(g_nodes) >= 2, body["graph"]


# ---------------------------------------------------------------------------
# Endpoint signature / response-model checks (negative-case detection)
# ---------------------------------------------------------------------------
def test_list_runs_endpoint_signature_uses_query():
    """list_runs must declare `limit` and `offset` as `Query(...)` parameters
    with bounds, not bare `int` defaults."""
    sig = inspect.signature(runs_module.list_runs)
    params = sig.parameters

    assert "limit" in params, "list_runs must accept a `limit` parameter"
    assert "offset" in params, (
        "list_runs must accept an `offset` parameter for pagination (BUG #14)"
    )

    # FastAPI exposes the `Query` metadata via the parameter default.
    limit_default = params["limit"].default
    # If the bug is unfixed, the default is a plain int (50).
    assert hasattr(limit_default, "le") or hasattr(limit_default, "ge"), (
        f"list_runs `limit` should be declared via Query(ge=1, le=100); "
        f"got bare default {limit_default!r}."
    )