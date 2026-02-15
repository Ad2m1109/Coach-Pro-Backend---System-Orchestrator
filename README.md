# Football Coach Backend (FastAPI)

Backend is split into two APIs:

- `backend/src/app.py` on `:8000` (core domain APIs)
- `backend/src/analysis_app.py` on `:8001` (analysis upload, status/history, media serving)

Both use the same MySQL database.

## Services Overview

```text
Frontend
  -> :8000 (core APIs)
  -> :8001 (analysis APIs)
       -> gRPC -> Tracking Engine :50051
```

## Database (Current)

Schema is in:

- `backend/data/full_creation.sql`
- `backend/data/full_insert.sql`

Analysis run table is now:

- `analysis_runs`

Important change:

- `analysis_runs` no longer stores output file paths as JSON.
- Outputs are derived dynamically from `analysis_id` + file existence in `tracking_engine/pipeline/outputs`.

## Key Analysis Endpoints (`:8001`)

- `POST /api/analyze_match`
- `GET /api/analysis_status/{analysis_id}`
- `GET /api/analysis_history`
- `DELETE /api/analysis_history/{analysis_id}`
- `GET /api/analysis/stream?path=...&access_token=...`
- `GET /api/analysis/files?path=...&access_token=...`
- `GET /api/analysis/files/json?path=...&access_token=...`

## Output Naming Convention

Tracking engine writes files with analysis ID prefix, for example:

- `outputs/<analysis_id>tracking.mp4`
- `outputs/analytics/<analysis_id>backline.mp4`
- `outputs/heatmaps/<analysis_id>heatmap.png`

`analysis_app.py` computes paths based on this convention.

## Setup

1. Create backend venv and install deps:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Initialize DB:

```bash
mysql -u <user> -p < backend/data/full_creation.sql
mysql -u <user> -p < backend/data/full_insert.sql
```

3. Generate protobuf stubs (or run tracking engine setup):

```bash
cd tracking_engine
source .venv/bin/activate
./setup.sh
```

## Run

Core API:

```bash
cd backend/src
source ../.venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Analysis API:

```bash
cd backend/src
source ../.venv/bin/activate
uvicorn analysis_app:app --reload --host 0.0.0.0 --port 8001
```

## Environment Notes

- `analysis_app.py` reads media from `ANALYSIS_OUTPUT_ROOT`.
- Default is `tracking_engine/pipeline`.
- Legacy fallback paths exist for compatibility.
