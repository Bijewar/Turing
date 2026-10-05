# Turing

Autonomous Causal Discovery & Cross-Domain Abstraction Engine

```mermaid
flowchart TD
    subgraph Client["Frontend Layer (Next.js 14 / React 18)"]
        UI["Web Dashboard :3000"]
        D3["D3.js Force Graph Engine"]
        Hooks["State & SSE Stream Hooks"]
    end

    subgraph Backend["Backend Engine (FastAPI / Python 3.11)"]
        API["FastAPI App :8000"]
        L1["Layer 1: Causal Discovery (PC / NOTEARS / GES)"]
        L2["Layer 2: Bayesian Optimization & Swarm Agents"]
        L3["Layer 3: Cross-Domain Literature & Web Search"]
        L4["Layer 4: Synthesis & Final Report Builder"]
    end

    subgraph Persistence["Storage & Infrastructure"]
        DB[("PostgreSQL 16 / SQLite")]
        Cache[("Redis 7 Broker & Cache")]
    end

    subgraph External["External APIs & Foundation Models"]
        LLM["LiteLLM (Claude 3.5 / GPT-4o / Groq / Ollama)"]
        Search["PubMed / ArXiv / Serper / Wikipedia"]
    end

    UI -->|REST & SSE| API
    API --> L1 --> L2 --> L3 --> L4
    API <--> DB
    API <--> Cache
    L1 & L2 & L4 <--> LLM
    L3 <--> Search
```

## Why

- **Breaks Domain Silos**: Automatically finds structurally isomorphic solutions across entirely different scientific and engineering domains, eliminating manual cross-domain literature review.
- **True Causal Inference, Not Correlation**: Utilizes the PC Algorithm and NOTEARS for directed causal discovery based on conditional independence, avoiding the common pitfalls of confusing correlation with causation.
- **Drastically Reduces Experimental Cost**: An adversarial swarm of Bayesian Agents (Explorer, Exploiter, Contrarian) systematically simulates and tests interventions on the causal graph, rather than requiring hundreds of expensive physical experiments.

## Features

- ✅ **Automated Causal Discovery**: Ingests raw CSV data and maps causal topology using PC Algorithm, GES, NOTEARS, and NetworkX.
- ✅ **Adversarial Agent Swarm**: Three unique agents push boundary exploration, peak refinement, and opposite boundary testing on the generated graph.
- ✅ **Cross-Domain Isomorphism Search**: Uses Graph Edit Distance (GED) and LLM-driven abstractions to find structurally identical architectures across PubMed, ArXiv, Wikipedia, and Serper.
- ✅ **Real-time Pipeline Telemetry**: FastAPI backend streams pipeline progress directly to the Next.js frontend via Server-Sent Events (SSE).
- ✅ **Automated Report Generation**: Synthesizes all findings into a comprehensive, actionable cross-domain research report using LLM reasoning.

## Quick Start (Docker Compose)

The fastest way to launch the entire platform (Frontend, Backend, PostgreSQL, and Redis):

```bash
git clone https://github.com/AutoBots-xyz/Turing.git
cd Turing
cp .env.example .env
docker compose up --build
```
* Access the web dashboard at `http://localhost:3000`
* Access the FastAPI backend API docs at `http://localhost:8000/docs`

---

## Local Development Setup

### Prerequisites
- Node.js 18+ & npm 9+
- Python 3.10+ & pip
- OpenAI or Anthropic API Key

### 1. Setup the Python Engine Backend
```bash
cd python-engine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --port 8000
```

### 2. Setup the Next.js Frontend
In a new terminal window at the project root:
```bash
cp .env.example .env.local
npm install
npm run dev
```

---

## Running Tests

All bug guardrail tests are consolidated in the `Turing_Test_Suite/` package:

```bash
# Run all tests (Frontend + Backend)
make test

# Run only backend pytest suite:
python -m pytest Turing_Test_Suite/backend/

# Run only frontend Vitest suite:
npm test
```

---

## Environment Variables

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `ANTHROPIC_API_KEY` | Optional* | — | Anthropic API key for Claude models. |
| `OPENAI_API_KEY` | Optional* | — | OpenAI API key for GPT-4o models. |
| `GROQ_API_KEY` | Optional | — | Groq API key for fast inference. |
| `DEFAULT_LLM_MODEL` | No | `gpt-4o` | Default LLM model identifier for LiteLLM. |
| `DATABASE_URL` | No | `sqlite+aiosqlite:///./turing.db` | Database connection URL (PostgreSQL / SQLite). |
| `REDIS_URL` | No | `redis://localhost:6379/0` | Redis broker connection URL. |
| `NEXT_PUBLIC_API_URL` | No | `http://localhost:8000` | Base URL of the Python FastAPI backend. |
| `CORS_ORIGINS` | No | `http://localhost:3000` | Comma-separated allowed CORS origins. |

*\*At least one LLM API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) is required for full Layer 1 validation and Layer 4 report synthesis.*

## Architecture Deep Dive


### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                                  │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Next.js Frontend (Port 3000)                     │  │
│  │  Landing Page → File Upload + Prompt → Run Dashboard          │  │
│  │  Tab 1: D3 Causal Graph | Tab 2: Agent Canvas                 │  │
│  │  Tab 3: Cross-Domain Search | Tab 4: Bridge Results           │  │
│  │  Tab 5: Streaming Report                                       │  │
│  └───────────────────────┬───────────────────────────────────────┘  │
└──────────────────────────│──────────────────────────────────────────┘
                           │  REST + SSE (HTTP/1.1)
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│              FastAPI Python Engine (Port 8000)                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌─────────────┐  │
│  │  /api/runs │  │/api/layer1 │  │/api/layer2 │  │ /api/layer3 │  │
│  │  Session   │  │  Ingest    │  │  Simulate  │  │  Search     │  │
│  │  Mgmt +    │  │  Detect    │  │  Bayesian  │  │  Extract    │  │
│  │  Orchestr. │  │  Extract   │  │  Agents    │  │  Match Rank │  │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └──────┬──────┘  │
│        │               │               │                 │          │
│        └───────────────┴───────────────┴─────────────────┘          │
│                              │                                        │
│                    ┌─────────▼─────────┐                             │
│                    │  SQLAlchemy Async  │                             │
│                    │  (SQLite / PG)    │                             │
│                    └───────────────────┘                             │
└─────────────────────────────────────────────────────────────────────┘
         │                    │                        │
         ▼                    ▼                        ▼
  LiteLLM Gateway    Semantic Scholar API       Serper.dev API
  (Claude / GPT-4o   (Academic papers,          (Web search,
   / Ollama)          free, no key needed)       patents search)
```

### Major Components and How They Connect

| Component | Location | Role |
|---|---|---|
| **Next.js App** | `src/` | UI, polling, SSE consumption, D3 graph rendering |
| **FastAPI Engine** | `python-engine/` | REST API, background orchestration, pipeline coordination |
| **Layer 1 Services** | `python-engine/services/layer1/` | File detection, extraction, causal graph building & validation |
| **Layer 2 Services** | `python-engine/services/layer2/` | Bayesian agent swarm, do-calculus simulation |
| **Layer 3 Services** | `python-engine/services/layer3/` | Cross-domain search, relation extraction, isomorphism matching |
| **Layer 4 Services** | `python-engine/services/layer4/` | Context packaging, LLM final report generation |
| **Anthropic Client** | `python-engine/services/anthropic_client.py` | Shared LLM utility used by layers 1, 3, and 4 |
| **Database** | `python-engine/database/` | SQLAlchemy models, CRUD ops, async session management |
| **In-Memory Store** | `RUNS_STORE` dict in `routers/runs.py` | Real-time layer-by-layer state for active runs |

### Technology Stack

#### Frontend
| Tech | Version | Purpose |
|---|---|---|
| Next.js | 14.2.3 | React full-stack framework, App Router |
| React | 18 | Component model |
| TypeScript | 5 | Type safety |
| TailwindCSS | 3.4.1 | Utility CSS styling |
| D3.js | 7.9.0 | Force-directed causal graph visualization |
| d3-force | 3.0.0 | Physics simulation for node layout |
| Framer Motion | 11 | Transition animations |
| Zustand | 4 | Client state management |
| Lucide React | 0.378.0 | Icon set |
| clsx + tailwind-merge | Latest | Conditional class utilities |

#### Backend (Python Engine)
| Tech | Version | Purpose |
|---|---|---|
| FastAPI | ≥0.115.0 | Async REST API framework |
| Uvicorn | ≥0.30.6 | ASGI server |
| Pydantic | ≥2.9.2 | Schema validation |
| SQLAlchemy | ≥2.0.36 | Async ORM |
| aiosqlite | ≥0.20.0 | Async SQLite driver |
| LiteLLM | ≥1.50.0 | Unified LLM gateway (Claude, GPT-4o, Ollama) |
| causal-learn | ≥0.1.4.0 | PC Algorithm for causal discovery |
| scikit-learn | ≥1.5.2 | Gaussian Process regression (RBF, Matérn kernels) |
| numpy | ≥2.1.2 | Numerical computation |
| pandas | ≥2.2.2 | Tabular data manipulation |
| networkx | ≥3.4.2 | Graph operations, GED, cycle detection |
| dowhy | ≥0.12 | Do-calculus causal intervention framework |
| pdfplumber | ≥0.11.0 | PDF text and table extraction |
| httpx | ≥0.27.2 | Async HTTP client for external APIs |
| scipy | Latest | Expected Improvement (EI) computation in Bayesian optimization |
| joblib | Latest | Gaussian Process model serialization to disk |
| python-dotenv | ≥1.0.1 | Environment variable loading |

#### Infrastructure / Database
| Tech | Notes |
|---|---|
| SQLite (default) | `causal_nexus.db`, zero-setup for local dev |
| PostgreSQL (production) | Switch via `DATABASE_URL=postgresql://...` |
| Filesystem Storage | GP models saved as `.joblib` files in `python-engine/storage/models/` |

### Architectural Patterns

- **Layered Pipeline Architecture**: Processing is divided into 4 sequential layers (Ingest → Simulate → Search → Report), each exposed as its own FastAPI router.
- **Dual-Store State Pattern**: Active run state is held in an in-process `RUNS_STORE` dictionary for real-time polling speed, while durable state is simultaneously persisted to the SQL database for crash recovery.
- **ReAct Agent Pattern**: Three adversarial agents (Explorer, Exploiter, Contrarian) each implement a "Think → Decide → Act" cycle within the Bayesian Optimization loop.
- **Background Task Orchestration**: The full 4-layer pipeline runs in a FastAPI `BackgroundTasks` task, freeing the HTTP response immediately while the frontend polls for updates.
- **Server-Sent Events (SSE)**: Layer 3 (search) and Layer 4 (report) stream real-time updates to the frontend via SSE endpoints.
- **Dual-Path Routing**: Based on whether the input file is tabular data (DATA PATH → PC Algorithm) or text/document (TEXT PATH → LLM Extraction), the entire processing pipeline adapts.

---


## Workflow / Project Working

### Step-by-Step End-to-End Operation

```
USER ACTION
    │
    ▼
[1] LANDING PAGE (src/app/page.tsx)
    • User selects a file (CSV, XLSX, PDF, TXT, DOCX, JSON) and types a hypothesis prompt.
    • POST /api/runs/ → creates a Run record in the DB with status=PENDING.
    • POST /api/runs/{runId}/layer1/upload → triggers BackgroundTasks.
    │
    ▼
[2] BACKGROUND PIPELINE STARTED (runs.py → process_full_pipeline)
    • Immediately sets RUNS_STORE[runId]["layer1Status"] = "processing" for polling.
    │
    ▼
[3] LAYER 1: CAUSAL DISCOVERY
    Step 1: UniversalFileDetector.analyze_file() — routes to DATA or TEXT path.
    Step 2: UniversalExtractor.extract_data() or extract_text()
    Step 3: Graph Builder:
        ─ DATA PATH: PCGraphBuilder.build_graph(df) runs PC Algorithm (Fisher Z).
        ─ TEXT PATH: LLMGraphBuilder.build_graph(text, model):
             Stage A → LLM generates domain ontology (entity/relation types)
             Stage B → LLM extracts causal edges chunk-by-chunk
    Step 4: GraphValidator.validate() — breaks cycles, flips impossible edges via LLM, removes isolates.
    Step 5: StructuralFitter.fit_graph() — fits GP for each edge (DATA path only), saves .joblib models.
    Step 6: NodeClassifier.classify_graph() — labels nodes: Source/Sink/Bottleneck/Mediator.
    Step 7: AmbiguityDetector.analyze_graph() — scores edges and ranks unknown nodes.
    Step 8: ConfidenceChecker.evaluate_graph() — decides routing: ROUTE_TO_L2 or SKIP_TO_L3.
    • Graph written to DB via _persist_graph().
    │
    ▼
[4] LAYER 2: BAYESIAN AGENT SIMULATION
    • Identifies source nodes (no incoming edges) as controllable variables.
    • Defines SearchSpace(0.0, 100.0) for each source node.
    • Runs up to 3 iterations of:
        a) BayesianOptimizer.get_base_point() → GP-based Expected Improvement (EI) selects best candidate.
        b) AgentExplorer proposes boundary-extreme values.
        c) AgentExploiter nudges the historically best values by 1% of domain width.
        d) AgentContrarian proposes the opposite boundary to Explorer.
        e) DoCalculusSimulator.simulate() runs each proposal:
            - Cuts incoming edges to intervened nodes (do-operator).
            - Propagates forward through topological order using GPEngine.predict_child().
            - Returns GaussianPrediction(mean, std_dev) for every reachable node.
        f) Winner = agent with highest ambiguity_reduction = mean / (std_dev + 0.001).
        g) Best result appended to historical_data for the next iteration's EI calculation.
    │
    ▼
[5] LAYER 3: CROSS-DOMAIN SEARCH
    • Selects the sink (target) node as the bottleneck.
    • Generates StructuralQuery.structural_description using generate_domain_blind_query() (LLM call).
    • Runs 4 parallel async searches:
        ─ Semantic Scholar API (academic papers, citation-weighted confidence)
        ─ Wikipedia REST API
        ─ Serper.dev web search
        ─ Serper.dev patent search
    • Each source has a 12-second timeout. Results merged via deduplicator.
    • contradiction_detector checks for conflicting evidence across sources.
    • relation_extractor sends each merged result to LLM to extract mini causal graphs.
    • isomorphism.match_graphs() compares each mini graph vs user's target graph using GED.
    • bridge_ranker.rank_bridges() scores each surviving match on 4 factors, returns Top 3.
    │
    ▼
[6] LAYER 4: REPORT GENERATION
    • context_packager.pack_bridges_into_context() serializes Top 3 bridges into a text block.
    • build_report() sends the context to the LLM with a structured prompt.
    • LLM returns JSON with: problem_statement, executive_summary, recommended_experiment.
    • contradiction_warnings collected from bridge metadata.
    • FinalReport assembled and streamed to frontend via SSE.
    │
    ▼
[7] FRONTEND DISPLAY (src/app/(dashboard)/run/[runId]/page.tsx)
    • useRunState() polls /api/runs/{id}/state every 1.5 seconds.
    • Active tab auto-advances as currentLayer changes: 1→Graph, 2→Canvas, 3→Search, 4→Report.
    • D3GraphEngine renders the causal graph with force simulation, drag, hover tooltips.
    • AgentStatusPanel shows the ReAct agent thought/decide/act loop live.
    • SearchStatusPanel consumes the SSE stream for Layer 3.
    • BridgeResultsPanel shows the Top 3 ranked cross-domain bridges.
    • StreamingReport shows the LLM-generated final report via SSE.
```

### Setup Steps

```bash
# 1. Clone the repo
git clone https://github.com/your-org/Turing.git
cd Turing

# 2. Configure environment variables
cp .env.example .env
# Edit .env: set OPENAI_API_KEY or ANTHROPIC_API_KEY, SERPER_API_KEY

# 3. Install frontend dependencies
npm install

# 4. Set up Python engine
cd python-engine
python -m venv venv
venv\Scripts\activate         # Windows
pip install -r requirements.txt
cp .env.example .env          # Set LLM keys again for Python side

# 5. Run backend (from python-engine/)
uvicorn main:app --reload --port 8000

# 6. Run frontend (from project root)
npm run dev                   # Starts on http://localhost:3000
```

### Build Steps (Production)

```bash
# Frontend production build
npm run build
npm start                     # Production server on port 3000

# Backend production (example with Gunicorn)
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Switch to PostgreSQL for production:
# Set DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/turing_db
```

---

##  Frontend

### UI Structure

The frontend is a Next.js 14 App Router application using TailwindCSS for styling. It consists of two main "pages":

1. **Landing Page** (`src/app/page.tsx`): File upload form, hypothesis prompt, recent runs sidebar.
2. **Run Dashboard** (`src/app/(dashboard)/run/[runId]/page.tsx`): 5-tab tabbed interface that auto-advances as the backend pipeline progresses.

### Frameworks / Libraries

| Library | Use |
|---|---|
| Next.js 14 (App Router) | Routing, SSR-compatible client components |
| TailwindCSS 3 | Styling, dark/neobrutalist design tokens |
| D3.js 7 + d3-force | Force-directed SVG graph rendering |
| Framer Motion 11 | Page and panel transition animations |
| Zustand 4 | Global state management (run context) |
| Lucide React | Icon set for UI elements |

### Key Components and Pages

#### Landing Page (`src/app/page.tsx`)
- `FileUploader` — drag-and-drop or click file input.
- System status indicator polling `/health` and `/` for backend version.
- Recent runs list from `GET /api/runs/?limit=10`.
- On submit: `createRun()` → `uploadDataset()` → `router.push(/run/{id})`.

#### Run Dashboard (`src/app/(dashboard)/run/[runId]/page.tsx`)
- `TopNav` — tab bar with 5 tabs: Graph, Canvas, Search, Abstraction, Report.
- `LayerProgress` — overall pipeline progress bar.
- **Tab 1 — Graph** (`GraphPane`): Fetches the causal graph from `/api/runs/{id}/layer1/graph` and renders it with `D3GraphEngine`. Shows node confidence via `ConfidencePanel`.
- **Tab 2 — Canvas** (`AgentStatusPanel`): Polls `/api/runs/{id}/layer2/agents` for ReAct agent activity cards, heatmap, and best-found experiment panel.
- **Tab 3 — Search** (`SearchStatusPanel`): Consumes the SSE stream at `/api/runs/{id}/layer3/search/stream` to display live search source entries.
- **Tab 4 — Abstraction** (`BridgeResultsPanel`): Shows the top 3 cross-domain bridge results with isomorphism scores, mechanism titles, and evidence tiers.
- **Tab 5 — Report** (`StreamingReport`): Consumes the SSE stream at `/api/runs/{id}/layer4/report/stream` to stream the final AI report section by section.

#### D3 Graph Engine (`src/components/graph/D3GraphEngine.tsx`)
- Renders a force-directed graph using D3's force simulation with `forceManyBody`, `forceLink`, `forceCollide`, and `forceCenter`.
- Nodes are color-coded by type: `controllable` (orange), `mediator` (dark blue), `bottleneck` (red), `outcome` (green), `chemistry` (purple).
- Edges are Bézier curves. Cross-domain bridges rendered as dashed purple paths.
- Supports node drag (pinning via `fx`/`fy`), hover tooltips (value, β coefficient, type), and click highlighting.
- Three animation states: `idle`, `simulated` (animated flow arrows), `bottleneck` (pulsing bottleneck nodes).

### State Management

- **Zustand** stores the global `runId` context shared across all components.
- **Custom hooks** handle all asynchronous state:
  - `useRunState` — polls `/api/runs/{id}/state` every 1.5 s to get layer status and auto-advance tabs.
  - `useAgentLoop` — polls `/api/runs/{id}/layer2/agents` for agent activity data.
  - `useSearchStream` — EventSource consumer for Layer 3 SSE.
  - `useReportStream` — EventSource consumer for Layer 4 SSE.
  - `useSystemStatus` — polls `/health` and `/` to show ONLINE/OFFLINE status badge.
  - `useGraphAnimation` — manages which D3 pipeline step is active.

### Frontend ↔ Backend Communication

All API calls go through `src/lib/api.ts`:

```typescript
// Base URL from environment
export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

// REST calls (JSON + FormData)
createRun(inputFile, inputType)           → POST /api/runs/
uploadDataset(runId, file)               → POST /api/runs/{id}/layer1/upload
listRuns(limit)                          → GET  /api/runs/?limit=N
fetchHealth()                            → GET  /health
fetchEngineInfo()                        → GET  /

// Polling endpoints (called by hooks)
GET /api/runs/{id}/state                 → layer-by-layer status dict
GET /api/runs/{id}/layer1/graph          → { nodes, edges }
GET /api/runs/{id}/layer2/agents         → { agents, heatmapNodes, status }

// SSE streams (via EventSource in hooks)
GET /api/runs/{id}/layer3/search/stream  → text/event-stream
GET /api/runs/{id}/layer4/report/stream  → text/event-stream
```

---

##  Backend

### API Structure (Routes / Endpoints)

All routes are mounted under the `/api` prefix in `main.py`.

#### `/api/runs` — Session Management (`routers/runs.py`)

| Method | Path | Description |
|---|---|---|
| POST | `/api/runs/` | Create a new Run (PENDING status) |
| GET | `/api/runs/` | List all runs (newest first, limit=50) |
| GET | `/api/runs/{run_id}` | Get a single run with full state |
| DELETE | `/api/runs/{run_id}` | Delete a run |
| POST | `/api/runs/{run_id}/layer1/upload` | Upload file → trigger background pipeline |
| GET | `/api/runs/{run_id}/state` | Real-time pipeline state (in-memory + DB fallback) |
| GET | `/api/runs/{run_id}/layer1/graph` | Current causal graph (in-memory + DB fallback) |
| GET | `/api/runs/{run_id}/layer2/agents` | Layer 2 agent activity payload |
| GET | `/api/runs/{run_id}/layer3/search/stream` | Layer 3 SSE stream |
| GET | `/api/runs/{run_id}/layer4/report/stream` | Layer 4 SSE stream |

#### `/api/layer1` — Data Ingestion (`routers/layer1.py`)

| Method | Path | Description |
|---|---|---|
| POST | `/api/layer1/upload` | High-level: upload + detect + persist to DB |
| GET | `/api/layer1/status/{run_id}` | Status + causal graph for a run |
| POST | `/api/layer1/detect` | Step 1: Detect file type (DATA/TEXT) |
| POST | `/api/layer1/extract` | Step 2: Extract data/text from file |
| POST | `/api/layer1/build-graph` | Step 3: Build causal graph (PC Alg. or LLM) |
| POST | `/api/layer1/validate-graph` | Step 4: Validate, fix cycles, flip edges |
| POST | `/api/layer1/fit-equations` | Step 5: Fit GP structural equations |
| POST | `/api/layer1/classify-nodes` | Step 6: Label nodes (Source/Sink/Bottleneck) |
| POST | `/api/layer1/detect-ambiguity` | Step 7: Score edges, rank unknown nodes |
| POST | `/api/layer1/confidence-check` | Step 8: Final routing decision |

#### `/api/layer2` — Agent Simulation (`routers/layer2.py`)

| Method | Path | Description |
|---|---|---|
| POST | `/api/layer2/simulate` | Orchestration: full Bayesian agent simulation |
| POST | `/api/layer2/mode-detect` | Detect DATA (Simulation) vs TEXT (Literature) mode |
| POST | `/api/layer2/identify-variables` | Classify nodes as source/sink/intermediate |
| POST | `/api/layer2/simulate-chain` | Run a single do-calculus simulation |
| POST | `/api/layer2/run-round` | Run one agent round (Explorer + Exploiter + Contrarian) |
| POST | `/api/layer2/generate-heatmap` | Generate 2D heatmap of interventions |
| POST | `/api/layer2/find-unexplored-zones` | Find gaps in simulation coverage |

#### `/api/layer3` — Cross-Domain Search (`routers/layer3.py`)

| Method | Path | Description |
|---|---|---|
| POST | `/api/layer3/search` | Step 11: 4-source parallel search |
| POST | `/api/layer3/extract` | Step 12: LLM causal graph extraction from search results |
| POST | `/api/layer3/match` | Step 13: Graph isomorphism matching |
| POST | `/api/layer3/rank` | Step 14: Bridge validity ranking → Top 3 |

#### `/api/layer4` — Report Generation (`routers/layer4.py`)

| Method | Path | Description |
|---|---|---|
| POST | `/api/layer4/report` | Generate final FinalReport from Top 3 bridges |

#### Health Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Engine info: status, service name, version |
| GET | `/health` | Simple health check: `{"status": "healthy"}` |

### Business Logic Organization

Business logic is cleanly separated into the `services/` layer:
- Routers (`routers/`) handle HTTP request/response parsing, dependency injection, and error wrapping.
- Services (`services/`) contain all algorithmic logic, pure Python, no FastAPI imports.
- Schemas (`schemas/`) define Pydantic v2 models shared between routers and services.
- Database (`database/`) holds ORM models and CRUD functions independent of routers.

### Authentication / Authorization

The current implementation has **no authentication layer**. CORS is fully open (`allow_origins=["*"]`). This is appropriate for a local research tool. Production deployments should add API key or OAuth2 authentication.

### Database Communication

- SQLAlchemy async engine (`create_async_engine`) with `aiosqlite` for SQLite.
- Session management via `AsyncSessionLocal` (async session maker).
- All route handlers that need the DB receive an `AsyncSession` via `Depends(get_db)`.
- Each request auto-commits on success or rolls back on exception.
- Dual-write pattern: active runs write to both `RUNS_STORE` (in-memory) and the database simultaneously via `_persist_status()` and `_persist_graph()`.

---

##  Python Engine / Core Processing Layer

### What This Layer Does

The Python engine (`python-engine/`) is a self-contained FastAPI application that implements the entire scientific processing pipeline. It is the brain of Turing: all causal discovery algorithms, agent simulation, cross-domain searching, and report generation happen here.

### Modules / Functions by Layer

#### Layer 1 — Causal Discovery Services

| Module | Key Class / Function | Description |
|---|---|---|
| `file_detector.py` | `UniversalFileDetector.analyze_file()` | In-memory analysis: checks numeric column density for CSV/Excel/JSON, table detection for PDF, token ratio for TXT |
| `file_detector.py` | `detect_input_type()` | Extension-based + content sniffing router |
| `extractor.py` | `UniversalExtractor.extract_data()` | Reads CSV/XLSX/JSON via pandas, returns clean DataFrame + warnings |
| `extractor.py` | `UniversalExtractor.extract_text()` | Reads PDF (pdfplumber), TXT, DOCX; returns clean text string |
| `extractor.py` | `AmbiguityDetector.analyze_graph()` | Scores edges by confidence, ranks nodes with low certainty |
| `pc_algorithm.py` | `PCGraphBuilder.build_graph()` | Runs `causal-learn` PC Algorithm (Fisher Z, α=0.05); falls back to Pearson correlation (|r|>0.3) if no directed edges found |
| `ontology_builder.py` | `LLMGraphBuilder.build_graph()` | Stage A: ontology generation; Stage B: chunk-and-extract causal edges via LLM |
| `validator.py` | `GraphValidator.validate()` | 4-pass: contradiction flagging → cycle breaking → impossible edge flip (LLM) → isolated node removal |
| `validator.py` | `ConfidenceChecker.evaluate_graph()` | Routing logic: `ROUTE_TO_L2_SIMULATION` or `SKIP_TO_L3` |
| `gaussian_process.py` | `StructuralFitter.fit_graph()` | Fits `GaussianProcessRegressor(RBF + WhiteKernel)` for each target node from parent columns |
| `gaussian_process.py` | `GPEngine.predict_child()` | GP-based causal propagation for Layer 2 simulation |
| `gaussian_process.py` | `refine_confidence_with_gp()` | Refines node confidence using graph topology features |
| `classifier.py` | `NodeClassifier.classify_graph()` | Labels nodes: Source (no in-edges), Sink (no out-edges), Bottleneck (high degree), Mediator (both) |

#### Layer 2 — Agent Simulation Services

| Module | Key Class / Function | Description |
|---|---|---|
| `orchestrator.py` | `run_bayesian_optimization()` | Master loop: runs N iterations of EI → propose → simulate → record |
| `bayesian_optimizer.py` | `BayesianOptimizer.get_base_point()` | Fits `GaussianProcessRegressor(Matern + WhiteKernel)` on history, samples 1000 random points, returns max-EI point |
| `do_calculus.py` | `DoCalculusSimulator.simulate()` | Cuts incoming edges to intervened nodes, Kahn's topological sort, forward GP propagation |
| `agent_explorer.py` | `AgentExplorer.propose()` | Pushes each variable to its farthest domain boundary |
| `agent_exploiter.py` | `AgentExploiter.propose()` | Nudges the historical peak by `EXPLOITER_NUDGE_PCT` (default 1%) of domain width |
| `agent_contrarian.py` | `AgentContrarian.propose()` | Picks the opposite boundary to Explorer, guaranteeing unique coverage per round |
| `cliff_detector.py` | `CliffDetector` | Detects performance cliffs at domain boundaries |
| `heatmap.py` | `HeatmapGenerator.generate()` | Generates 2D heatmap of simulation history for UI visualization |

#### Layer 3 — Cross-Domain Search Services

| Module | Key Function | Description |
|---|---|---|
| `search_papers.py` | `search_papers()` | Semantic Scholar Graph API, citation-weighted confidence scoring |
| `search_wikipedia.py` | `search_wikipedia()` | Wikipedia REST search + page content fetch |
| `search_web.py` | `search_web()` | Serper.dev web search, requires `SERPER_API_KEY` |
| `search_patents.py` | `search_patents()` | Serper.dev `/patents` endpoint |
| `deduplicator.py` | `deduplicate_and_merge()` | URL-based dedup, merges titles and summaries |
| `contradiction_detector.py` | `detect_contradictions()` | Cross-result conflict analysis |
| `relation_extractor.py` | `run_relation_extraction()` | LLM extracts mini causal graphs from each merged result; handles contradictions with dual extractions |
| `isomorphism.py` | `match_graphs()` | NetworkX GED (timeout=3s), classifies as PERFECT/STRONG_PARTIAL/WEAK_PARTIAL/DISCARDED |
| `bridge_ranker.py` | `rank_bridges()` | 4-factor geometric product score, LLM rates compatibility and transferability |

#### Layer 4 — Report Generation Services

| Module | Key Function | Description |
|---|---|---|
| `context_packager.py` | `pack_bridges_into_context()` | Serializes Top 3 bridges into a structured LLM prompt context block |
| `report_builder.py` | `build_report()` | Calls LLM for problem_statement, executive_summary, recommended_experiment; graceful fallback if no API key |

### How the Engine Is Invoked

1. **Via REST API**: Frontend calls backend HTTP endpoints. All 4 layer routers are mounted via `app.include_router()` in `main.py`.
2. **Via Background Task**: `POST /api/runs/{id}/layer1/upload` calls `BackgroundTasks.add_task(process_full_pipeline, ...)`. This function in `runs.py` orchestrates all 4 layers sequentially inside a single async task.
3. **Direct Python calls**: The granular `/api/layer1/detect`, `/api/layer1/extract`, etc. endpoints allow testing individual pipeline steps independently.

### Inputs / Outputs

| Layer | Input | Output |
|---|---|---|
| Layer 1 | Raw file bytes (CSV/XLSX/PDF/TXT) | `{nodes: [...], edges: [...]}` JSON causal graph |
| Layer 2 | Causal graph + target node | `Layer2Response(simulation_results, best_intervention, confidence)` |
| Layer 3 | StructuralQuery (domain-blind description) | `Step14Response(top_bridges: [RankedBridge × 3])` |
| Layer 4 | Top 3 RankedBridges | `FinalReport(problem_statement, executive_summary, recommended_experiment, warnings)` |

---

##  Layer-by-Layer Breakdown

### Layer 1 — Causal Discovery

**Responsibilities**: Accept raw user files, extract their data or text content, build a causal graph, validate it, quantify uncertainty, and prepare it for Layer 2.

**Internal Workings**:

```
File Upload
    │
    ▼
UniversalFileDetector           ← Analyzes file in-memory, no temp files
    │
    ├─── DATA PATH (CSV/XLSX/JSON with ≥50% numeric columns)
    │        │
    │        ▼
    │    UniversalExtractor.extract_data()   ← Returns pandas DataFrame
    │        │
    │        ▼
    │    PCGraphBuilder.build_graph()        ← causal-learn PC Algorithm
    │        │                                 Fisher Z independence test (α=0.05)
    │        │                                 Fallback: Pearson |r| > 0.3
    │
    └─── TEXT PATH (PDF/TXT/DOCX/MD)
             │
             ▼
         UniversalExtractor.extract_text()  ← pdfplumber / raw text
             │
             ▼
         LLMGraphBuilder.build_graph()
             ├── Stage A: LLM → entity types + relation types (ontology)
             └── Stage B: LLM → extract edges chunk by chunk (500 char chunks)
    │
    ▼
GraphValidator.validate()
    ├── Flag contradictions (ACTIVATES + INHIBITS on same pair)
    ├── Break cycles (remove lowest-confidence edge in each cycle)
    ├── Agentic edge flip (LLM batch-checks 20 edges: "is this direction physically possible?")
    └── Remove isolated nodes
    │
    ▼
StructuralFitter.fit_graph()     ← DATA PATH ONLY
    ├── For each target node: fit GaussianProcessRegressor(RBF + WhiteKernel)
    │   on parent column(s) → y column mapping
    ├── Stores R² and mean uncertainty per edge
    └── Saves models to storage/models/{session_id}.joblib
    │
    ▼
NodeClassifier.classify_graph()
    └── Source, Sink, Bottleneck, Mediator labels + UI themes
    │
    ▼
AmbiguityDetector.analyze_graph()
    └── urgent_nodes ranked by low confidence
    │
    ▼
ConfidenceChecker.evaluate_graph()
    ├── overall_conf >= 85? → SKIP_TO_L3
    ├── DATA + low conf  → ROUTE_TO_L2_SIMULATION
    └── TEXT + low conf  → SKIP_TO_L3 (no simulation for text)
```

**Adjacent Layer Interaction**: Layer 1 sends its output graph (JSON) to Layer 2 via the `RUNS_STORE` dictionary. It also persists the graph to the database so it survives server restarts.

---

### Layer 2 — Bayesian Agent Simulation

**Responsibilities**: Given a causal graph (DATA path only), mathematically simulate causal interventions on source variables to predict outcomes at the sink node. Use Bayesian Optimization to efficiently search the intervention space.

**Internal Workings**:

```
CausalGraph (from Layer 1)
    │
    ▼
Identify source nodes (in_degree == 0)
Define SearchSpace(0.0, 100.0) per source node
    │
    ▼
ITERATION LOOP (up to max_iterations=3)
    │
    ├── BayesianOptimizer.get_base_point()
    │       └── if history ≥ 2 pts: fit GP(Matern-2.5 + WhiteKernel)
    │               sample 1000 random points, compute EI = μ·Φ(Z) + σ·φ(Z)
    │               return argmax(EI) as base_point
    │
    ├── 3 agents propose values simultaneously:
    │       Explorer:   push each variable to its farthest boundary
    │       Exploiter:  nudge best historical values by 1% of domain width
    │       Contrarian: push each variable to the OPPOSITE of Explorer's choice
    │
    ├── DoCalculusSimulator.simulate() for each proposal:
    │       ├── Cut incoming edges to intervened nodes (do-operator)
    │       ├── Kahn's topological sort (cycle guard)
    │       └── For each node in topo order:
    │               GPEngine.predict_child(parent_predictions, edge_weights)
    │               → fits local GP on 50 synthetic samples from parent distributions
    │               → returns GaussianPrediction(mean, std_dev)
    │
    ├── Score = pred.mean / (pred.std_dev + 0.001)   ← ambiguity reduction metric
    ├── Winner = highest-scoring agent
    └── Append winner to historical_data
    │
    ▼
Layer2Response:
    simulation_results: List[IterationResult]
    best_intervention: str (values dict of the best proposal)
    confidence: float (0–100, heuristic from best score × 10)
```

**Adjacent Layer Interaction**: Layer 2 consumes the Layer 1 causal graph. It feeds agent activity data into `RUNS_STORE[runId]["layer2Data"]` for the frontend to poll.

---

### Layer 3 — Cross-Domain Search

**Responsibilities**: Convert the user's causal bottleneck into a domain-blind structural description, search across 4 external knowledge sources in parallel, deduplicate and analyze contradictions, extract mini causal graphs from each result, match them against the user's target graph using structural isomorphism, and rank surviving matches.

**Internal Workings**:

```
StructuralQuery (domain-blind description of target node)
    │
    ▼
STEP 11: 4-SOURCE PARALLEL SEARCH (12s timeout per source)
    ├── Semantic Scholar Graph API   → up to 5 papers with citation data
    ├── Wikipedia REST API           → up to 5 articles + page content
    ├── Serper.dev Web Search        → up to 10 web results
    └── Serper.dev Patent Search     → up to 10 patent results
    │
    ▼
STEP 11.5: Deduplication + Contradiction Detection
    ├── URL-based dedup, merge summaries
    └── Cross-source: compare snippets for conflicting claims
    │
    ▼
STEP 12: Relation Extraction (LLM per merged result)
    ├── For each MergedResult: extract_causal_graph_from_text(merged_summary)
    │       → LLM returns {nodes: [...], edges: [...]} mini causal graph
    ├── If contradiction detected: additional LLM call for the contradicting mechanism
    └── asyncio.gather(return_exceptions=True) — one failure doesn't kill all
    │
    ▼
STEP 13: Graph Isomorphism Matching
    └── For each ExtractedMechanism:
            calculate_structural_similarity(target_graph, candidate_graph)
            → networkx.graph_edit_distance() (timeout=3s)
            → similarity = 1 - (GED / max_GED) × 100
            → PERFECT (≥90) / STRONG_PARTIAL (≥70) / WEAK_PARTIAL (≥50) / DISCARDED
    │
    ▼
STEP 14: Bridge Validity Ranking
    ├── Factor 1: structural_match = isomorphism_score / 100
    ├── Factor 2: constraint_compatibility (LLM 0.0–1.0)
    ├── Factor 3: solution_transferability (LLM 0.0–1.0)
    ├── Factor 4: evidence_strength (source type × deployment_status × citations)
    ├── final_score = F1 × F2 × F3 × F4  (geometric product)
    └── Return Top 3 RankedBridges sorted by final_score descending
```

**Adjacent Layer Interaction**: Layer 3 receives the causal graph (from `RUNS_STORE`) to use as the target graph for isomorphism matching. Its `step14_res` (Top 3 bridges) is passed directly to Layer 4.

---

### Layer 4 — Report Generation

**Responsibilities**: Package the Top 3 cross-domain bridges into a concise, human-readable final report using LLM synthesis.

**Internal Workings**:

```
Step14Response (Top 3 RankedBridges)
    │
    ▼
context_packager.pack_bridges_into_context()
    └── Serializes: bridge title, source domain, isomorphism score,
                    mechanism description, evidence tier, contradiction flags
    │
    ▼
LLM call (build_report):
    Prompt: "Based on these bridges, generate a JSON with:
             problem_statement | executive_summary | recommended_experiment"
    Model: DEFAULT_LLM_MODEL (env var, default: gpt-4o)
    max_tokens: 600
    │
    ▼
Collect contradiction_warnings from bridge metadata
Compute confidence_disclaimer (avg score, bridge count)
    │
    ▼
FinalReport(
    run_id, problem_statement, top_bridges,
    executive_summary, recommended_experiment,
    contradiction_warnings, confidence_disclaimer
)
```

**Graceful Degradation**: If no LLM API key is configured, the report returns a structured message explaining that keys are needed, rather than a hard 500 error.

**Adjacent Layer Interaction**: Receives data from Layer 3. Outputs to the SSE stream consumed by the frontend's `StreamingReport` component.

---

### Database Layer

**Responsibilities**: Persist run metadata, status, causal graph, and top bridges across server restarts.

**Schema** (`runs` table):

| Column | Type | Notes |
|---|---|---|
| `id` | String(36) | UUID primary key |
| `status` | Enum | PENDING / RUNNING / COMPLETE / FAILED |
| `input_file` | String(512) | Original filename |
| `input_type` | String(32) | `csv` or `text` |
| `created_at` | DateTime | UTC creation timestamp |
| `updated_at` | DateTime | Auto-updates on change |
| `causal_graph` | Text | JSON-serialized `{nodes, edges}` |
| `top_bridges` | Text | JSON-serialized list of RankedBridges |
| `error_message` | Text | Error string if status=FAILED |

---

##  Data Flow

### End-to-End Data Trace

```
INPUT FILE (CSV example)
    │
    │ binary bytes (multipart/form-data)
    ▼
POST /api/runs/{id}/layer1/upload
    │
    │ file_bytes + filename
    ▼
UniversalFileDetector.analyze_file(file_bytes, filename)
    │
    │ {path: "DATA", confidence: 0.95, ...}
    ▼
UniversalExtractor.extract_data(file_bytes, filename)
    │
    │ pandas.DataFrame (rows × numeric columns)
    ▼
PCGraphBuilder.build_graph(df)
    │
    │ causal-learn CG object (adjacency matrix, conditional independence tests)
    │ → Serialized as:
    │   { nodes: [{id, label, type, confidence}], edges: [{source, target, weight, confidence, relation}] }
    ▼
GraphValidator.validate(graph_dict, model)
    │
    │ Cleaned graph dict + validation_logs
    ▼
StructuralFitter.fit_graph(df, graph_dict, "DATA")
    │
    │ Augmented graph dict with fit_metrics per node
    │ Models saved: storage/models/{session_id}.joblib
    ▼
NodeClassifier.classify_graph(graph_dict)
    │
    │ Graph dict with node type labels
    ▼
Persisted to DB (causal_graph column = JSON string)
Stored in RUNS_STORE[runId]["graph"]
    │
    │ Polled by frontend: GET /api/runs/{id}/layer1/graph
    │ Rendered by D3GraphEngine as SVG force simulation
    ▼
Layer2 — Bayesian Optimization
    │
    │ Input: graph dict + target_node_id
    │ Output: Layer2Response {simulation_results, best_intervention, confidence}
    │ Agents write to RUNS_STORE[runId]["layer2Data"]
    │ Consumed by frontend polling GET /api/runs/{id}/layer2/agents
    ▼
Layer3 — Cross-Domain Search
    │
    │ Input: StructuralQuery (LLM-generated domain-blind description of sink node)
    │ Parallel HTTP calls → raw SearchResult objects
    │ Merged → MergedResult objects (URL-deduped, contradiction-tagged)
    │ LLM extraction → ExtractedMechanism objects (mini CausalGraph per result)
    │ GED matching → IsomorphismMatch objects (score 0-100, match type)
    │ LLM + evidence scoring → RankedBridge objects (4-factor final_score)
    │ Output: Step14Response { top_bridges: [RankedBridge × 3] }
    │ RUNS_STORE[runId]["layer3Data"] streamed via SSE
    ▼
Layer4 — Report Generation
    │
    │ Input: Step14Response
    │ LLM call → JSON { problem_statement, executive_summary, recommended_experiment }
    │ Output: FinalReport (Pydantic model)
    │ RUNS_STORE[runId]["layer4Data"] streamed via SSE
    ▼
Frontend StreamingReport component consumes SSE
Renders: problem statement → executive summary → bridges → experiment → warnings
```

### Data Formats at Each Stage

| Stage | Format | Schema |
|---|---|---|
| Raw file upload | `bytes` | Binary file content |
| Tabular extract | `pandas.DataFrame` | N rows × M numeric columns |
| Text extract | `str` | Clean UTF-8 text |
| Causal graph (internal) | `dict` | `{nodes: [{id, label, type, confidence}], edges: [{source, target, weight, confidence, relation}]}` |
| Causal graph (DB) | `TEXT` (JSON string) | Same as above, JSON-serialized |
| GP models | `.joblib` file | scikit-learn `GaussianProcessRegressor` objects |
| Structural query | `StructuralQuery` Pydantic | `{original_node_id, structural_description, original_confidence}` |
| Search results | `SearchResult` Pydantic | `{source, title, summary, url, confidence, citation_count, deployment_status}` |
| Merged results | `MergedResult` Pydantic | Deduplicated + contradiction_analysis attached |
| Extracted mechanism | `ExtractedMechanism` Pydantic | `{source_result: MergedResult, causal_graph: CausalGraph}` |
| Isomorphism match | `IsomorphismMatch` Pydantic | `{mechanism, isomorphism_score, match_type}` |
| Ranked bridge | `RankedBridge` Pydantic | `{match: IsomorphismMatch, scores: ValidityScores}` |
| Final report | `FinalReport` Pydantic | `{run_id, problem_statement, executive_summary, recommended_experiment, top_bridges, contradiction_warnings, confidence_disclaimer}` |
| SSE event | `text/event-stream` | `data: {JSON}\n\n` |

---

##  Agent System

### Agent Overview

Turing's Layer 2 implements a **multi-agent adversarial Bayesian Optimization system** using the ReAct (Reason + Act) pattern. Three agents compete each round; the winner's values are fed back into the Bayesian Optimizer's history.

### The Three Agents

#### 1. `AgentExplorer` (`services/layer2/agent_explorer.py`)

**Role**: Boundary exploration. Finds performance cliffs by pushing each controllable variable to its extremes.

**Logic**:
```
For each variable:
    val = base_point[variable]  (or random uniform if missing)
    dist_to_min = |val - domain.min|
    dist_to_max = |val - domain.max|
    proposed = min if dist_to_min > dist_to_max else max
    tie-break: always go to min
```

**Justification output**: `"Explorer pushed variables to their farthest boundaries to find cliff edges. Decisions: X=50.0->0.0 (pushed to MIN)"`

#### 2. `AgentExploiter` (`services/layer2/agent_exploiter.py`)

**Role**: Peak refinement. Finds the historically best simulation result and nudges values slightly in the direction with the most remaining room.

**Logic**:
```
best_past = max(historical_data, key=lambda x: x[sink_node])
For each variable:
    nudge_size = (domain.max - domain.min) × EXPLOITER_NUDGE_PCT (default 0.01)
    up_val = best_val + nudge_size
    down_val = best_val - nudge_size
    proposed = direction with more room remaining in domain
```

**Config**: `EXPLOITER_NUDGE_PCT` env var (default 1% of domain width).

#### 3. `AgentContrarian` (`services/layer2/agent_contrarian.py`)

**Role**: Anti-duplication. Picks the boundary **opposite** to Explorer's choice, guaranteeing that no two agents test the same corner in the same round.

**Logic**:
```
For each variable:
    if dist_to_min > dist_to_max:  # Explorer picks min
        proposed = domain.max      # Contrarian picks max
    else:                          # Explorer picks max
        proposed = domain.min      # Contrarian picks min
    tie-break: Explorer→min, Contrarian→max
```

### Agent Initialization

Agents are plain Python classes with no internal state (`__init__` takes no parameters except optional `base_noise`). They are instantiated fresh every round in the orchestrator:

```python
# orchestrator.py
optimizer  = BayesianOptimizer()
explorer   = AgentExplorer()
exploiter  = AgentExploiter()
contrarian = AgentContrarian()
simulator  = DoCalculusSimulator()
```

### Agent Communication / Shared State

Agents do **not** communicate with each other directly. They share state only through:
1. `base_point` — the current best point from `BayesianOptimizer.get_base_point()`.
2. `historical_data` — the list of past rounds' best values, accumulates across iterations.
3. `domain_config` — the fixed `SearchSpace(min, max)` per variable.

### Agent Decision Logic and Triggers

Each agent's `propose()` method is called synchronously. All three proposals are then simulated concurrently via `asyncio.gather()`. The winner is determined **after** all simulations complete:

```python
winner_score = max(predictions[sink_node].mean / (predictions[sink_node].std_dev + 0.001))
```

The metric `ambiguity_reduction = mean / (std_dev + 0.001)` rewards proposals that produce high output **and** low uncertainty simultaneously.

### Orchestration / Coordination Logic

```
run_bayesian_optimization(request)  ← orchestrator.py
    │
    ITERATION 1..N (N = max_iterations, default 3):
    │
    ├── optimizer.get_base_point(history, domain_config) → base_point
    │
    ├── proposals = [
    │       explorer.propose(base_point, domain_config),
    │       exploiter.propose(history, domain_config, sink_node),
    │       contrarian.propose(base_point, domain_config)
    │   ]
    │
    ├── asyncio.gather([simulator.simulate(nodes, edges, prop) for prop in proposals])
    │
    ├── score each proposal's sink_node prediction
    ├── winner = highest ambiguity_reduction
    ├── log all AgentAction objects (think, decide, act)
    └── append {sink_node: best_yield, values: best_values} to historical_data
```

The router's `/api/layer2/run-round` endpoint exposes one iteration externally for granular testing.

---

##  LLM Integration

### LLMs Used

| Provider | Model String | Usage |
|---|---|---|
| OpenAI | `gpt-4o` (default) | All LLM calls via LiteLLM |
| Anthropic | `anthropic/claude-3-5-sonnet` | Alternate via LiteLLM |
| Aliyun DashScope | `openai/qwen3.6-plus` | OpenAI-compatible endpoint |
| Local Ollama | `ollama/llama3` | Fully local offline option |

The model is selected via the `DEFAULT_LLM_MODEL` environment variable. LiteLLM's unified API routes the call to the correct provider automatically.

### How LLMs Are Called

All LLM calls use **LiteLLM's `completion()` function**, which provides a uniform interface:

```python
from litellm import completion

response = completion(
    model=os.getenv("DEFAULT_LLM_MODEL", "gpt-4o"),
    messages=[{"role": "user", "content": prompt}],
    max_tokens=600,
    response_format={"type": "json_object"},  # When structured JSON needed
    api_key=os.getenv("NVIDIA_NIM_API_KEY") or os.getenv("NVIDIA_API_KEY") or None
)
raw = response.choices[0].message.content.strip()
```

There is no direct `anthropic` SDK usage in the main pipeline — everything routes through LiteLLM.

The `services/anthropic_client.py` file provides shared utilities used across layers:

| Function | Called By | Tokens | Purpose |
|---|---|---|---|
| `generate_domain_blind_query()` | Layer 3 search setup | 100 | Convert node to domain-blind structural description |
| `extract_causal_graph_from_text()` | Layer 3 Step 12 | 800 | Extract mini causal graph from search result text |
| `evaluate_compatibility_and_transferability()` | Layer 3 Step 14 | 60 | Score constraint compatibility + solution transferability (0.0–1.0 each) |
| `evaluate_deployment_status()` | Layer 3 source ranking | 20 | Classify URL as `deployed / single_study / blog` |

### Prompt Structure / Templates

#### Layer 1: Ontology Builder (Stage A)
```
System: You are an expert causal ontologist.
User:   Read this text snippet and identify:
        1. The top 8 causal entity types
        2. The primary relationship types
        TEXT: {first 2000 chars}
        Respond ONLY with valid JSON:
        {"entity_types": [...], "relation_types": [...]}
```

#### Layer 1: Causal Edge Extraction (Stage B)
```
System: (none)
User:   Extract causal relationships based on this ontology:
        Entities: {entity_types}
        Relations: {relation_types}
        TEXT: {chunk of 500 chars}
        Respond ONLY with JSON array:
        [{"source": "A", "target": "B", "relation": "INHIBITS", "confidence": "DEFINITELY"}]
```

#### Layer 1: Graph Validator (Impossible Edge Check)
```
User:   You are a Physics, Logic, and Causality Validator Agent.
        Look at these causal edges: [{source, target} × 20]
        Are any directions physically/logically impossible?
        Rules: outcome/measurement variables CANNOT cause fundamental inputs.
        Respond with JSON array of edges to flip:
        [{"source": "A", "target": "B", "reason": "Outcome cannot cause input"}]
```

#### Layer 3: Domain-Blind Query Generation
```
User:   You are a cross-domain systems engineer.
        Node '{label}' has {N} incoming and {M} outgoing causal edges.
        Describe the underlying structural mechanism in one sentence using ONLY physics/math terms.
        Remove all domain-specific language.
        Output ONLY the one-sentence description.
```

#### Layer 3: Relation Extraction from Search Text
```
User:   You are a causal graph extractor. Read this scientific text:
        {merged_summary}
        Return JSON: {"nodes": [{id, label, confidence}], "edges": [{source, target, relation, confidence}]}
        Rules: Nodes MUST be nouns. Edges MUST be uppercase verbs (INHIBITS, CAUSES, etc.)
```

#### Layer 3: Compatibility + Transferability Evaluation
```
User:   You are a cross-domain systems engineer.
        TARGET PROBLEM: {domain_context}
        CANDIDATE MECHANISM: {candidate_mechanism}
        Score 0.0–1.0:
        1. constraint_compatibility
        2. solution_transferability
        Return ONLY: {"constraint_compatibility": float, "solution_transferability": float}
```

#### Layer 4: Final Report Generation
```
User:   You are a cross-domain systems analyst.
        Based on the bridges below, generate a concise report:
        {packed bridge context}
        Return JSON with EXACTLY:
        - problem_statement (one sentence)
        - executive_summary (3-5 sentences)
        - recommended_experiment (one specific, measurable experiment)
```

### How Responses Are Parsed

LLM responses are stripped of markdown fences (```` ```json ``` ````) and parsed with `json.loads()`:

```python
raw = response.choices[0].message.content.strip()
if raw.startswith("```"):
    raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]
data = json.loads(raw.strip())
```

Layer 3 extraction also uses regex search as a fallback:
```python
match = re.search(r'\[.*\]', content, re.DOTALL)
if match:
    content = match.group(0)
```

### LLM ↔ Database Connection

- LLM-extracted causal graphs from Layer 1 (ontology_builder) are persisted to the `causal_graph` column in the `runs` table.
- LLM-evaluated bridge scores (compatibility, transferability) are stored in `ValidityScores` Pydantic objects within `RankedBridge`, which are serialized to the `top_bridges` TEXT column.
- The `FinalReport` is not stored in the database — it is streamed via SSE to the frontend.
- There is no vector database or embedding store; LLM calls are stateless and do not use conversation history.

---

##  Impact / Outcomes

### Efficiency Gains

- **Eliminates manual cross-domain literature review**: A process that typically takes weeks is automated in minutes, searching across 4 knowledge sources simultaneously.
- **Automated causal discovery**: What previously required expert-level causal inference knowledge (running PC algorithms, interpreting conditional independence tests) is fully automated and exposed via a simple file upload.
- **Systematic intervention space exploration**: The adversarial agent swarm (Explorer / Exploiter / Contrarian) systematically covers domain boundaries, peaks, and unexplored zones in just 3 iterations rather than requiring hundreds of physical experiments.

### Problems Solved

- **Domain silo problem**: Researchers can now discover solutions from entirely unrelated fields (e.g., applying aerospace fluid dynamics insights to biological signaling pathways) without needing expertise in those fields.
- **Causal vs. correlation confusion**: By using the PC Algorithm (which produces directed causal edges via independence testing, not correlation), Turing avoids the common mistake of treating correlation as causation.
- **Graph validity**: The automated validator catches cycle paradoxes, physically impossible edge directions, and contradictory relationships before they propagate into simulation errors.
- **Uncertainty propagation**: By fitting Gaussian Processes to every causal edge and propagating uncertainty through the do-calculus simulation, every prediction comes with a quantified standard deviation rather than a point estimate.

### Use Cases Enabled

1. **Scientific Research**: Upload experimental CSV data → discover causal structure → find analogous solved problems in other fields.
2. **Engineering Optimization**: Upload process data → identify bottlenecks → simulate interventions → find cross-domain solutions.
3. **Document Analysis**: Upload a research paper or PDF → extract its causal claims → find supporting or contradicting evidence.
4. **Hypothesis Generation**: Use the cross-domain bridge report as a structured hypothesis for new experiments.

---

##  Design Decisions

### Why a 4-Layer Sequential Pipeline?

Each layer produces a well-defined artifact (causal graph → simulation results → ranked bridges → final report) that is independently testable. This makes debugging straightforward: if Layer 3 returns poor bridges, you can inspect the Layer 1 graph output or the Layer 3 search results in isolation using the granular endpoints without rerunning the entire pipeline.

**Trade-off**: Sequential layers mean the total wall-clock time is the sum of all 4 layers. Parallelism was deliberately avoided between layers to maintain clean data dependencies.

### Why Dual-Store (In-Memory + Database)?

**In-memory `RUNS_STORE`**: Real-time polling by the frontend (every 1.5 seconds) would be too slow against SQLite for every request. The in-memory dict allows sub-millisecond state reads during an active run.

**Database**: Server restarts, multi-worker deployments, and long-term run history require durability. Both stores are written in parallel.

**Trade-off**: In-memory store is lost on server restart. The `GET /api/runs/{id}/state` endpoint handles the cold-start case by rebuilding state from the database.

### Why LiteLLM as the LLM Gateway?

LiteLLM provides a single `completion()` interface that routes to Claude, GPT-4o, DashScope/Qwen, or local Ollama models by changing only the `DEFAULT_LLM_MODEL` environment variable. This makes the system provider-agnostic: teams can run fully offline with Ollama or switch between commercial providers without any code changes.

**Trade-off**: LiteLLM adds a thin abstraction layer with minor overhead and occasionally lags behind provider-specific features (like streaming tool calls).

### Why the PC Algorithm for Causal Discovery?

The Peter-Clark (PC) Algorithm is the gold standard for constraint-based causal discovery on observational tabular data. It uses conditional independence tests (Fisher Z for continuous data) to find the Markov equivalence class of the true causal DAG. This is mathematically rigorous compared to correlation-based approaches.

**Trade-off**: PC Algorithm is sensitive to sample size (requires sufficient rows for reliable independence tests) and runs in O(d^k) time where d is the number of variables and k is the maximum degree of the graph. The fallback to Pearson correlation (|r| > 0.3) is used when PC finds no directed edges, acknowledging this limitation.

### Why NetworkX Graph Edit Distance for Isomorphism?

GED is the mathematically correct measure for comparing two graphs that may differ in size. Unlike subgraph isomorphism (which requires an exact structural match and is NP-complete), GED gives a continuous similarity score that gracefully handles partial structural matches across different-sized graphs — which is essential when comparing a user's 10-node graph against mini graphs extracted from academic abstracts.

**Trade-off**: GED is also NP-hard in general, which is why a 3-second timeout is applied. The fallback is node/edge count difference, which is a rough heuristic but prevents pipeline hangs on large graphs.

### Why Three Adversarial Agents?

Each agent fills a distinct role in the exploration-exploitation trade-off:
- **Explorer** maps the unknown territory (boundaries).
- **Exploiter** extracts maximum value from known territory (peaks).
- **Contrarian** prevents both from duplicating each other's proposals.

The trio ensures that every simulation round tests three maximally different regions of the search space, making each of the 3 default iterations maximally informative without wasting compute on redundant proposals.

**Trade-off**: With only 3 iterations, the optimizer may not converge on the true global optimum for large, complex systems. This is a deliberate speed-vs-accuracy trade-off appropriate for the interactive research workflow.

### Why SSE (Server-Sent Events) for Streaming?

Layer 3 (search, up to ~60 seconds) and Layer 4 (report generation) are long-running operations where the user needs to see progress. SSE is simpler than WebSockets for one-directional server-to-client streaming, works natively with `EventSource` in browsers, and integrates cleanly with FastAPI's `StreamingResponse`.

**Trade-off**: SSE is one-directional only. For future bidirectional features (e.g., user steering the agent mid-run), WebSockets would be needed.

### Why SQLite as the Default Database?

Zero-setup for local development. New contributors clone the repo, set environment variables, and run the server — no database installation required. The `DATABASE_URL` environment variable supports a seamless switch to PostgreSQL for production by changing a single configuration line.

**Trade-off**: SQLite has no built-in multi-process write concurrency. For multi-worker production deployments (e.g., Gunicorn with multiple workers), PostgreSQL should be used.
