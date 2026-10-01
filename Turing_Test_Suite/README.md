# Turing-main Test Suite — Bug-Catching Guardrail Tests

A comprehensive test suite that **fails with the current buggy code** and
**passes only when the bugs are properly fixed** (no fake fixes, no LLM
hallucination green-flag).

## What this is

This test suite is the **verification arm** of the bug audit report at
`Turing_Bug_Audit_Report.md`. Each test
corresponds to a specific bug found in that audit, and is designed so
that:

1. With the **current buggy code**, the test **FAILS** (red).
2. After the bug is **properly fixed**, the test **PASSES** (green).
3. If an LLM **"fakes" the fix** (e.g., always returning a constant value,
   or only handling one case), the test **still fails** because the
   test covers both positive AND negative cases.

This means you can use these tests as guardrails: hand the codebase +
audit report to an LLM (or a developer), tell them to fix the bugs, and
run the tests. The LLM cannot get a "green flag" without actually
fixing the bug correctly.

## Quick start

```bash
# Run the FULL test suite (backend + frontend):
cd Turing_Test_Suite
./run_tests.sh

# Run only backend tests:
./run_tests.sh --backend-only

# Run only frontend tests:
./run_tests.sh --frontend-only

# Run a single test file:
cd Turing_Test_Suite/backend
./../run_single_test.sh test_layer1_validator.py

# Run a single test:
./../run_single_test.sh "test_layer1_validator.py::TestClassifyGraphJsonExtraction::test_classify_graph_regex_not_greedy"

# Run frontend tests:
cd Turing_Test_Suite/frontend
npm test
```

## Directory structure

```
tests/
├── README.md                       ← this file
├── pytest.ini                       ← pytest config (asyncio_mode=auto)
├── run_tests.sh                     ← runs both backend + frontend
├── run_single_test.sh               ← runs a single test
│
├── backend/                         ← pytest tests for Python backend
│   ├── conftest.py                  ← shared fixtures (sys.path, mock LLM, synthetic DataFrames)
│   ├── requirements-test.txt        ← test-only deps
│   ├── test_layer1_*.py             ← Layer 1 (causal discovery)
│   ├── test_layer2_*.py             ← Layer 2 (Bayesian opt + agents)
│   ├── test_layer3_*.py             ← Layer 3 (search + bridge ranking)
│   ├── test_layer4_*.py             ← Layer 4 (report builder)
│   ├── test_database_*.py           ← DB models, CRUD, race, persistence
│   ├── test_routers_*.py            ← FastAPI routers + SSE
│   ├── test_schemas_*.py            ← Pydantic schema validation
│   ├── test_security_*.py           ← CORS, uploads, prompt injection, headers
│   ├── test_anthropic_client.py      ← LLM client wrapper
│   └── test_main.py                  ← FastAPI app config
│
└── frontend/                         ← vitest tests for Next.js frontend
    ├── package.json
    ├── vitest.config.ts
    ├── setup.ts                      ← mocks: EventSource, fetch, matchMedia
    ├── stubs/                        ← Next.js component stubs
    ├── test_D3GraphEngine.tsx
    ├── test_useReportStream.ts
    ├── test_useSearchStream.ts
    ├── test_useRunState.ts
    ├── test_useAgentLoop.ts
    ├── test_useGraphAnimation.ts
    ├── test_useSystemStatus.ts
    ├── test_ui_components.tsx
    ├── test_layer1_components.tsx
    ├── test_layer2_components.tsx
    ├── test_layer3_components.tsx
    ├── test_layer4_components.tsx
    ├── test_shell_components.tsx
    ├── test_run_page.tsx
    ├── test_api.ts
    └── test_types.ts
```

## Test counts

**Backend**: ~500+ tests across 60+ files, covering ~400+ bugs from the audit.

**Frontend**: ~80+ tests across 14 files, covering ~100+ bugs from the audit.

**Total**: ~580+ tests, each one a guardrail against a specific bug.

## How tests catch "fake fixes"

Each test is designed so that simply returning a constant value, or only
handling one case, does NOT pass. Examples:

### Example 1: Contradiction detector (BUG #1)

The buggy code reads `edge["type"]` but every builder writes
`edge["relation"]`. A naive fix would just hardcode `is_contradicted=True`
to make any test pass. **Our tests catch this:**

```python
def test_contradiction_detector_reads_relation_not_type(self):
    """A graph WITH contradictions → is_contradicted=True."""
    # ...build contradicting graph...
    assert edge.is_contradicted is True

def test_contradiction_detector_negative_case(self):
    """A graph WITHOUT contradictions → is_contradicted=False.
    A fake fix that always returns True will FAIL this test."""
    # ...build non-contradicting graph...
    assert edge.is_contradicted is False

def test_contradiction_penalty_applied_correctly(self):
    """The exact penalty value is applied, not just any non-zero value."""
    assert edge.ambiguity_penalty == -25  # or whatever the spec says
```

### Example 2: PC algorithm depth (BUG #3)

The buggy code hardcodes `depth=1`, so PC only tests conditioning sets
of size 1. A naive fix might just always return an empty skeleton (no
edges). **Our tests catch this:**

```python
def test_pc_depth_is_not_hardcoded_to_1(self):
    """With X ⊥ Y | {Z1, Z2} but not marginally, PC must remove X-Y edge
    (requires depth ≥ 2)."""
    df = cond_independent_df  # the fixture
    graph = PCGraphBuilder().build_graph(df)
    edges = [(e["source"], e["target"]) for e in graph["edges"]]
    assert ("X", "Y") not in edges and ("Y", "X") not in edges  # removed
    # AND: must still find the actual causal edges (Z1→X, Z1→Y, Z2→X, Z2→Y,
    # X→Z, Y→Z, etc.). A fake fix that returns an empty graph will FAIL.
    assert len(graph["edges"]) > 0
```

### Example 3: SSE client disconnect (BUG #7)

The buggy code is a `while True:` loop that never checks `request.is_disconnected()`.
A naive fix might just add the check but forget to actually break:

```python
def test_layer3_stream_detects_client_disconnect(self):
    """Open SSE, close connection mid-stream; the generator MUST stop."""
    # Use httpx.AsyncClient.stream(), close it, then check server-side state.
    # A generator that keeps looping will leak memory — detected via a
    # coroutine-count check or by timing out the test.
```

## Bug coverage map

Each test has a docstring that names the **exact bug ID** it catches,
like:

```python
def test_pc_depth_is_not_hardcoded_to_1():
    """Catches SWARM-3 BUG #3: PC algorithm hardcoded depth=1 misses
    conditionally-independent pairs."""
```

To find tests for a specific bug:

```bash
# Find tests catching BUG #1 from the SWARM-3 audit:
grep -l "SWARM-3 BUG #1" tests/

# Find all tests catching Layer 1 validator bugs:
grep -l "SWARM-3 BUG" tests/backend/test_layer1_validator.py
```

## How to use this with an LLM

1. **Give the LLM the audit report**: `Turing_Bug_Audit_Report.md`
2. **Tell it to fix a specific bug** (e.g., SWARM-3 BUG #1).
3. **Run the corresponding test**:
   ```bash
   cd Turing_Test_Suite/backend
   pytest test_layer1_validator.py::TestContradictionDetector::test_contradiction_detector_reads_relation_not_type -v
   ```
4. **The LLM's fix is "good" only when ALL tests for that bug pass** — including
   the negative cases that catch fake fixes.

You can also run the whole suite as a single regression check:

```bash
./run_tests.sh
```

All tests should pass after every bug in the audit is fixed.

## Anti-hallucination features

- **Source inspection tests**: Some tests read the source code via
  `inspect.getsource()` and assert that the buggy pattern is GONE
  (e.g., `assert "errors='ignore'" not in source`). An LLM can't
  "fix" these by changing behavior — the buggy line must be
  physically removed.

- **Two-sided assertions**: Each "positive" test (bug present → expect
  specific outcome) is paired with a "negative" test (bug absent →
  different outcome). Hardcoding one value can't satisfy both.

- **Boundary tests**: Tests for "edge of valid range" (e.g.,
  `confidence=0`, `confidence=100`, `confidence=150`) catch fixes
  that use `< 100` instead of `<= 100`, etc.

- **Reproducibility tests**: Two runs with the same input must produce
  the same output — catches LLMs that introduce randomness to "hide"
  bugs.

- **Async tests**: `pytest.mark.asyncio` tests verify that async
  functions actually use `await` (not sync calls). Catches LLMs that
  "fix" by removing `async` instead of using `await`.

- **Schema tests**: Pydantic schema tests verify field bounds and
  types — catches LLMs that "fix" the schema but leave the validators
  off.

## Skipping tests for missing dependencies

Some tests require optional packages (`causal-learn`, `python-docx`,
`xlrd`, `openpyxl`, PDF parsers) that may not be installed. These tests
use `pytest.importorskip(...)` or `pytest.skip(...)` so the suite still
runs. Install the optional deps to enable them:

```bash
pip install causal-learn python-docx openpyxl pdfplumber
```

## Test isolation

- Each test gets a fresh in-memory SQLite DB (via the `db_session` fixture).
- Each test gets fresh module state (via fixture cleanup).
- LLM calls are mocked — no real API calls.
- External HTTP calls are mocked — no real network calls.
- Frontend tests use jsdom — no browser needed.
- Tests are deterministic — no flakiness, no random failures.

## How this maps to the bug audit

| Audit Section         | Test Files                                                          |
|-----------------------|---------------------------------------------------------------------|
| SWARM-1 (Next pages)  | `test_run_page.tsx`, `test_shell_components.tsx`                    |
| SWARM-2 (Hooks/UI)    | `test_useReportStream.ts`, `test_D3GraphEngine.tsx`, all `test_*_components.tsx` |
| SWARM-3 (Layer 1)     | `test_layer1_*.py` (10 files)                                         |
| SWARM-4 (Layer 2)     | `test_layer2_*.py` (6 files)                                          |
| SWARM-5 (Layer 3)     | `test_layer3_*.py` (10 files)                                         |
| SWARM-6 (Layer 4+Routers+DB+Schemas) | `test_layer4_*.py`, `test_routers_*.py`, `test_database_*.py`, `test_schemas_*.py` |
| SWARM-7 (TS types+lib+config) | `test_types.ts`, `test_api.ts`, `test_useRunState.ts` (type drift) |
| SWARM-8 (Security+Concurrency+Edge) | `test_security_*.py`, `test_routers_sse.py`                       |
| SWARM-9 (DB+Deps+Build+Docs) | `test_database_*.py`, `test_main.py`                          |

## FAQ

### Q: Some tests fail because the source file doesn't exist or has a different structure. Why?

A: The audit found bugs; the tests assume the source structure described
   in the audit. If a test fails with `ImportError` or `AttributeError`,
   it may be that the source file has been refactored since the audit.
   In that case, update the test's import path or use `pytest.skip()`
   with a clear reason.

### Q: Why do some tests inspect source code (`inspect.getsource`) instead of testing behavior?

A: Some bugs are about *patterns that should not exist in the source*
   (e.g., `pd.to_numeric(errors='ignore')` which is removed in pandas 2.2+).
   These are best caught by inspecting the source, not by behavior tests
   that would just crash with `TypeError: unexpected keyword`.

### Q: The test for "no rate limiting" passes (the middleware exists), but I want to verify the rate limit is actually enforced.

A: The source-inspection test catches the "no middleware at all" case.
   For runtime enforcement, add an integration test that fires 1000
   requests and asserts 429s start appearing after the limit.

### Q: A test passes but I know the bug is still there.

A: Two possibilities:
   1. The test may be too weak — check the assertions. If they're too
      loose, tighten them.
   2. The bug may be fixed in a way the test didn't anticipate (rare but
      possible — the test is a guardrail, not a proof).

### Q: I want to add my own tests.

A: Append to the relevant `test_*.py` file. Each test should:
   - Have a docstring naming the bug ID it catches.
   - Test BOTH positive AND negative cases (no fake fixes).
   - Be deterministic (no flakiness).
   - Mock external calls (no real API/network).

## Reporting bugs in the test suite itself

If a test is wrong (false positive, false negative, flaky), open an
issue with:
- The test file and test name.
- What you expected vs. what happened.
- The relevant bug ID from the audit.