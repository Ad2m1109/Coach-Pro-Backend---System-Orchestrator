# Football Coach Backend

## Abstract
The backend is a Python/FastAPI service layer that supports the Football Coach platform's transactional workflows, asynchronous video analysis, tactical alerting, and AI-assisted querying. The repository currently contains roughly 85 Python files across `src/`, `tests/`, and `scripts/`. Its architecture is unusual in a productive way: instead of a single API surface, it runs two cooperating FastAPI applications backed by the same MySQL schema and the same authentication model.

Those two applications are:

- the classic backend API on port `8000`, centered on CRUD, RBAC, assistant endpoints, and tactical feedback
- the analysis-management API on port `8001`, centered on uploads, gRPC orchestration, streaming assets, analysis history, retry, and cancellation

Together, they form the application middle tier between the Flutter frontend and the `tracking_engine` gRPC service.

## Runtime Topology

```text
Flutter Frontend
  -> FastAPI app.py           (:8000)  team, player, match, notes, assistant, alerts, auth
  -> FastAPI analysis_app.py  (:8001)  upload, history, status, files, retry, cancel

analysis_app.py / AnalysisJobService
  -> TrackingEngineClient
  -> gRPC AnalysisService     (:50051) in tracking_engine/server.py

MySQL (soccer_analytics)
  <- users, teams, matches, stats, analysis reports/runs, alerts, insights, segments

Remote LLM
  <- assistant RAG answers
  <- tactical advisory generation
```

## Technical Profile

| Area | Implementation |
| --- | --- |
| Web framework | FastAPI |
| App servers | Uvicorn |
| Database access | PyMySQL with manual SQL and `DictCursor` |
| Authentication | OAuth2 password flow + RS256 JWT |
| Authorization | Role and path-based RBAC in dependency layer |
| Analysis transport | gRPC client to tracking engine |
| Assistant architecture | Rule-based intent routing + retrieval + deterministic SQL analytics + remote LLM |
| Realtime transport | WebSocket for tactical alerts, SSE for segment streams |
| Media/file handling | `python-multipart`, static file serving, byte-range streaming |

## Repository Layout

```text
backend/
├── .env.example              # runtime env template
├── data/
│   ├── full_creation.sql     # schema bootstrap
│   └── full_insert.sql       # demo/sample seed data
├── proto/
│   └── analysis.proto        # gRPC contract shared with tracking engine
├── scripts/
│   ├── rotate_jwt_keys.sh
│   ├── add_uuid_ids_to_inserts.py
│   └── smoke_parse_and_persist.py
├── src/
│   ├── app.py                # classic API (:8000)
│   ├── analysis_app.py       # analysis API (:8001)
│   ├── controllers/
│   ├── services/
│   ├── models/
│   ├── security/
│   ├── dependencies.py
│   ├── database.py
│   ├── rbac.py
│   └── config.py
└── tests/
    └── test_analysis_status.py
```

## Dual-Application Architecture

### 1. Classic Application: `src/app.py`
This is the general product API. It:

- mounts the static directory for uploaded assets
- configures CORS from `CORS_ALLOW_ORIGINS`
- exposes `/api/token` and `/api/register`
- includes the full CRUD controller set
- exposes `/api/assistant/*`
- exposes tactical alert feedback and metrics routes
- exposes the `/ws/alerts/{match_id}` WebSocket

This app is the primary source of truth for organization, football entities, and human interactions.

### 2. Analysis Application: `src/analysis_app.py`
This is the operational analysis API used by the Flutter analysis screens. It:

- receives video uploads
- records analysis runs in `analysis_runs`
- submits jobs to the tracking engine over gRPC
- updates status/progress in the database
- exposes analysis history
- serves generated videos and JSON files
- supports retry, cancellation, stale-run cleanup, and segment streams

This app is specialized enough that it should be read as a workflow service rather than a CRUD service.

### Architectural Observation
The repository also still contains a controller-based analysis path inside the classic API (`controllers/analysis_controller.py` + `AnalysisJobService`). That means two analysis orchestration styles currently coexist:

- controller-driven job records stored in `analysis_reports`
- dedicated-analysis-app runs stored in `analysis_runs`

The frontend code points primarily to the dedicated analysis app, so that path appears to be the current production-facing flow.

## Authentication and RBAC

### Authentication
Authentication is based on OAuth2 password login:

- `POST /api/token` validates email + password
- credentials are signed into an RS256 JWT
- the JWT is persisted by the frontend and sent on subsequent requests

JWT key material is loaded by `security/jwt_keys.py` from either:

- PEM contents in environment variables
- PEM file paths in environment variables
- fallback files in `backend/certs/`

### User Model
The backend distinguishes:

- `owner`
- `staff`

For staff users, the token may also include:

- `staff_id`
- `team_id`
- `permission_level`
- `role`
- `app_role`
- `app_permissions`

### Role Derivation
`rbac.py` maps user and staff roles to application roles:

| Input role | App role |
| --- | --- |
| owner | `account_manager` |
| head_coach | `coach` |
| assistant_coach | `assistant_coach` |
| analyst | `analyst` |
| anything else | `player` |

The permission vocabulary is explicit:

- `accounts.read/create/update/delete`
- `football.read/write`
- `analysis.run`
- `notes.read/write`

### Enforcement Model
Authorization is centralized in `dependencies.get_current_active_user()`. The logic is path-aware:

- account managers can view all `/api/*` resources
- only account managers can modify many team/account resources
- assistant routes are intentionally open to any authenticated active user
- tactical decision feedback routes are explicitly allowed for owners

This makes security highly visible in code, but also means route naming conventions matter.

## Data Model and Persistence
The schema is defined in `data/full_creation.sql` and seeded by `data/full_insert.sql`. It uses UUID-like string identifiers for primary entities and a mix of JSON columns for analytics-heavy payloads.

### Core Entity Families

| Family | Tables | Purpose |
| --- | --- | --- |
| Identity | `users`, `staff` | accounts, roles, permission context |
| Club and squad | `teams`, `players`, `formations` | roster and tactical structure |
| Competition | `events`, `matches`, `match_events`, `match_lineups` | fixtures and match context |
| Performance | `player_match_statistics`, `match_team_statistics` | per-player and team analytics |
| Notes and planning | `match_notes`, `reunions`, `training_sessions` | collaboration and planning |
| Analysis state | `analysis_reports`, `analysis_runs`, `video_segments`, `analysis_segments` | async processing records and outputs |
| Tactical intelligence | `tactical_alerts`, `decision_feedback`, `tactical_insights` | alert history, coach feedback, long-term insight memory |

### Important Analytics Tables

- `analysis_reports`: older job/result container, JSON-based report payloads
- `analysis_runs`: newer dedicated analysis-run table used by `analysis_app.py`
- `analysis_segments`: per-window tactical slices persisted by `SegmentService`
- `tactical_alerts`: normalized alert packets emitted by the tracking engine
- `decision_feedback`: human acceptance or dismissal decisions plus later effectiveness evaluation
- `tactical_insights`: durable analytical summaries used as tactical memory by the assistant stack

### Persistence Style
The backend deliberately avoids an ORM. Services execute raw SQL through cursor objects and usually commit immediately after writes. This keeps query logic explicit, but it also means:

- schema drift must be managed carefully
- service methods own most query semantics
- consistency rules live in Python rather than in a data-access layer abstraction

## API Surface by Capability

| Capability | Representative routes | Backing files |
| --- | --- | --- |
| Auth | `/api/token`, `/api/register`, `/api/users/me` | `app.py`, `dependencies.py`, `services/user_service.py` |
| Teams and identity | `/api/teams`, `/api/users`, `/api/staff` | `team_controller.py`, `user_controller.py`, `staff_controller.py` |
| Squad data | `/api/players`, `/api/formations`, `/api/match_lineups` | player/formation/lineup controllers and services |
| Match domain | `/api/matches`, `/api/match_events`, `/api/events` | `match_controller.py`, `match_event_controller.py`, `event_controller.py` |
| Statistics | `/api/player_match_statistics`, `/api/matches/{id}/team_statistics` | statistics controllers and services |
| Notes and planning | `/api/matches/{id}/notes`, `/api/reunions`, `/api/training_sessions` | note, reunion, and training-session services |
| Assistant | `/api/assistant/query`, `/api/assistant/mode` | `assistant_controller.py`, assistant services |
| Tactical alerts | `/api/matches/{match_id}/alerts`, `/api/decision/feedback`, `/api/decision/metrics`, `/ws/alerts/{match_id}` | `tactical_alert_controller.py`, `tactical_alert_service.py` |
| Match-scoped analysis | `/api/matches/{match_id}/analyze`, `/api/analysis/{job_id}/status`, `/api/matches/{match_id}/analysis` | `analysis_controller.py`, `AnalysisJobService` |
| Analysis app endpoints | `/api/analyze_match`, `/api/analysis_status/{id}`, `/api/analysis_history`, `/api/analysis/files`, `/api/analysis/{id}/retry` | `analysis_app.py` |
| Segment APIs | `/api/matches/{id}/segments`, `/api/analysis/{id}/segments`, `/stream` variants | `segment_controller.py`, `segment_service.py` |

## Analysis Orchestration

### Dedicated Analysis Path
The dedicated path in `analysis_app.py` is the most complete workflow in the repository.

High-level lifecycle:

1. A video upload reaches `POST /api/analyze_match`.
2. The upload is stored on disk and an `analysis_runs` record is inserted.
3. The app opens a gRPC stream through `TrackingEngineClient`.
4. Progress packets update run status and message fields.
5. `SEGMENT_DONE` packets are persisted to `analysis_segments` and fanned out to SSE listeners.
6. `ALERT` packets are normalized into `tactical_alerts` and optionally pushed to WebSocket clients.
7. Final artifacts are exposed through file-serving endpoints.
8. The frontend can later:
   - read history
   - retry
   - cancel
   - preview videos
   - fetch JSON payloads
   - replay segment timelines

### File and Stream Handling
`analysis_app.py` contains substantial delivery infrastructure:

- file-path sanitization and output-path resolution
- byte-range video streaming
- JSON preview endpoints
- MP4 "faststart" optimization
- derived preview-video creation

This means the analysis backend is not only a job coordinator but also an asset-serving gateway.

### Legacy Controller Path
`AnalysisJobService` in `services/analysis_job_service.py` handles the classic controller-based analysis flow. It:

- inserts a job shell into `analysis_reports`
- clears prior alert history for the match
- submits the gRPC request
- updates report progress JSON
- persists segments on streamed `SEGMENT_DONE`
- broadcasts alerts on streamed `ALERT`

This path uses `analysis_reports` as its storage substrate, in contrast to the `analysis_runs` table used by `analysis_app.py`.

## Segment Streaming Architecture
`controllers/segment_controller.py` provides both query and SSE interfaces for analysis segments.

Two scopes are supported:

- match-scoped segment retrieval
- analysis-run-scoped segment retrieval

Internally, SSE fan-out is implemented with in-memory `asyncio.Queue` listeners keyed by:

- `match_id`
- `analysis_id`

This makes live segment delivery simple and fast, but it also means fan-out state is process-local rather than shared across multiple app instances.

## Tactical Alert and Decision Feedback Subsystem

`TacticalAlertService` is one of the most operationally interesting parts of the backend. It acts as:

- an in-memory broadcaster
- a persistence layer bootstrapper
- a feedback collector
- a suppression manager
- a decision-effectiveness coordinator

### Responsibilities

1. Ensure `tactical_alerts` and `decision_feedback` tables exist.
2. Broadcast incoming alert packets to live WebSocket connections.
3. Persist alert history for later retrieval.
4. Accept coach feedback (`ACCEPT` or `DISMISS`).
5. Join alert history with user-specific decision feedback.
6. Trigger asynchronous post-feedback evaluation through `DecisionEffectivenessService`.

### Decision-Effectiveness Evaluation
`DecisionEffectivenessService` checks whether the severity profile improves after an intervention:

- accepted recommendations are considered effective when downstream severity drops enough
- dismissed recommendations are considered risky when later severity escalates

This turns the alert pipeline into a measurable feedback system rather than a one-way notification mechanism.

## AI Assistant and RAG Stack

The assistant subsystem is implemented in `services/assistant_service.py` and is more structured than a generic chat endpoint.

Pipeline:

1. `IntentRouter` classifies the question using rule-based fuzzy matching.
2. `RetrievalService` pulls relevant data from the database under a tight timeout budget.
3. `ContextBuilder` formats the data into bounded structured blocks.
4. `AnalyticalService` computes deterministic SQL analytics when the query asks for trends or rankings.
5. `TacticalMemoryService` injects prior stored insights when relevant.
6. `llm_client.call_llm()` sends the final prompt to a remote model endpoint.

Assistant intents include:

- player statistics
- match statistics
- team summary
- tactical knowledge questions
- analytical queries
- general fallback questions

The assistant also has a global mode switch:

- `assistant`
- `analysis`

When the system mode is `analysis`, the assistant returns a structured "unavailable during live analysis" response instead of performing retrieval and generation.

## Deterministic Analytics Layer
`AnalyticalService` is important because it keeps high-value football calculations out of the LLM. Current deterministic outputs include:

- top improving players by sprint-intensity trend
- highest average player ratings over recent windows
- best pressing formations by average pressures

When these insights are computed, they are also stored in `tactical_insights`, which later powers tactical memory in assistant responses.

## Environment Variables

The code reads the following settings directly:

| Variable | Purpose |
| --- | --- |
| `DB_HOST` | MySQL host |
| `DB_USER` | MySQL user |
| `DB_PASSWORD` | MySQL password |
| `DB_NAME` | MySQL database name |
| `CORS_ALLOW_ORIGINS` | comma-separated allowed origins |
| `JWT_PRIVATE_KEY` / `JWT_PUBLIC_KEY` | inline PEM contents |
| `JWT_PRIVATE_KEY_PATH` / `JWT_PUBLIC_KEY_PATH` | filesystem PEM paths |
| `LLM_MODE` | `remote` or `local` |
| `REMOTE_LLM_URL` | assistant and tactical-advisory upstream model endpoint |
| `LLM_TIMEOUT` | remote-LLM timeout seconds |
| `TRACKING_ENGINE_HOST` | gRPC host for the tracking engine |
| `TRACKING_ENGINE_PORT` | gRPC port for the tracking engine |
| `ANALYSIS_OUTPUT_ROOT` | output directory override for analysis assets |
| `ANALYSIS_ENGINE_HOST` / `ANALYSIS_ENGINE_PORT` | legacy analysis gRPC client settings |

`.env.example` already captures the most important backend values for local setup.

## Local Setup

### 1. Python environment

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Database bootstrap

Run:

- `data/full_creation.sql`
- `data/full_insert.sql`

against a MySQL-compatible server. The default code path expects a database named `soccer_analytics`.

### 3. JWT keys
Either:

- provide PEM content/path variables, or
- generate local keys:

```bash
cd backend
./scripts/rotate_jwt_keys.sh
```

### 4. Start the APIs

Classic backend:

```bash
cd backend/src
source ../.venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Analysis backend:

```bash
cd backend/src
source ../.venv/bin/activate
uvicorn analysis_app:app --host 0.0.0.0 --port 8001 --reload
```

Or from the repository root:

```bash
./start_services.sh
```

That script also starts the tracking engine and applies performance-oriented tracking defaults.

## Tests and Utility Scripts

### Tests
Visible test coverage is currently light:

- `tests/test_analysis_status.py`

Run with:

```bash
cd backend
source .venv/bin/activate
pytest
```

### Utilities

| Script | Purpose |
| --- | --- |
| `scripts/rotate_jwt_keys.sh` | generate and rotate RS256 JWT keys |
| `scripts/add_uuid_ids_to_inserts.py` | patch SQL insert scripts to include UUID ids |
| `scripts/smoke_parse_and_persist.py` | smoke-test parsing/persistence workflow |

## Architectural Strengths

- Clear domain-oriented controller and service breakdown
- Explicit SQL makes data flows transparent
- RS256 JWT support is production-friendly compared with simple shared-secret setups
- Realtime tactical feedback is deeply integrated rather than an afterthought
- The assistant stack uses deterministic retrieval and SQL analytics before invoking an LLM
- The analysis app includes practical operational features like stale-run detection, retry, preview generation, and byte-range video streaming

## Architectural Tensions and Limits

- Two analysis orchestration paths coexist, which increases conceptual load.
- Some stateful components, especially segment SSE listeners and active WebSocket maps, are process-local and would need redesign for horizontal scaling.
- Raw SQL offers clarity but comes with higher maintenance cost as the schema evolves.
- Automated test coverage is sparse relative to the complexity of the system.
- A legacy `AnalysisGrpcService` remains in the codebase and appears older than the current `TrackingEngineClient` flow.
- The assistant and tactical advisory pipelines depend on a remote LLM endpoint, which is an operational dependency outside this repository.

## Suggested Reading Order
For a contributor trying to understand the backend quickly, the highest-yield sequence is:

1. `src/app.py`
2. `src/analysis_app.py`
3. `src/dependencies.py`
4. `src/rbac.py`
5. `src/database.py`
6. `src/controllers/analysis_controller.py`
7. `src/controllers/segment_controller.py`
8. `src/services/tracking_engine_client.py`
9. `src/services/tactical_alert_service.py`
10. `src/services/assistant_service.py`

That order follows the real platform path from request entry, through security, into analysis orchestration, then into tactical intelligence and assistant reasoning.
