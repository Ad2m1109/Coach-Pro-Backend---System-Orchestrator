# Coach Pro Backend - System Orchestrator

A FastAPI-based orchestration layer that serves as the central gateway between mobile clients and high-performance analysis services. This backend manages REST APIs, job scheduling, database persistence, and inter-service communication via gRPC.

## System Role

The **Coach Pro Backend** functions as the **Business Logic Orchestrator** in a distributed microservices architecture. It abstracts the complexity of video processing workflows, manages asynchronous job states, and provides a unified REST interface for frontend clients while coordinating with specialized computational services.

### Architecture Position

```
      Flutter Client (REST)
      /                  \
   (8000)               (8001)
Classic Backend  →  Analysis Management ↔ C++ Inference Engine
(Signs Token)       (Verifies Token)      (gRPC:50051)
      ↓                  ↓
MySQL DB         Buffering (Disk)
```

## Core Responsibilities

### 1. REST API Gateway
Exposes a comprehensive HTTP/JSON API for mobile and web clients with automatic OpenAPI documentation.

### 2. Job Lifecycle Management
Implements a state machine for video analysis workflows:
- **PENDING**: Job queued, awaiting processing
- **PROCESSING**: Active analysis via C++ engine
- **COMPLETED**: Results persisted, ready for retrieval
- **FAILED**: Error state with diagnostic information

### 3. Database Persistence Layer
Manages relational data models including:
- Match records and metadata
- Player statistics (speed, distance, heatmaps)
- Team formations and lineups
- Video segments and annotations
- Analysis reports with computed metrics

### 4. Inter-Service Communication
Maintains a **hybrid communication strategy**:
- **REST (JSON)**: Client-facing APIs for Flutter/Web
- **gRPC (Protocol Buffers)**: Internal communication with C++ engine for low-latency data transfer

## Key Features

### API Documentation
Auto-generated interactive documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Asynchronous Processing
Background task execution using FastAPI's BackgroundTasks or Celery integration for:
- Video upload handling
- Long-running analysis jobs
- Notification dispatching

### Data Validation
Pydantic models ensure type-safe request/response handling with automatic validation and serialization.

## Installation

### Prerequisites
- Python 3.9+
- MySQL 8.0+ (or MariaDB 10.6+)
- Running instance of C++ analysis engine with gRPC enabled

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/coach_pro_backend.git
   cd coach_pro_backend
   ```

2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   Create a `.env` file in the project root:
   ```env
   # Database Configuration
   DB_HOST=localhost
   DB_PORT=3306
   DB_USER=root
   DB_PASSWORD=your_password
   DB_NAME=coach_pro_db
   
   # C++ Engine gRPC Endpoint
   ANALYSIS_ENGINE_HOST=localhost
   ANALYSIS_ENGINE_PORT=50051
   
   # Application Settings
    # SECRET_KEY is no longer used. See "JWT Configuration" below.
    DEBUG=False
    CORS_ORIGINS=["http://localhost:3000"]
   
   # File Storage
    UPLOAD_DIR=/var/coach_pro/uploads
    MAX_UPLOAD_SIZE_MB=500
   ```

5. **Generate RSA Keys (Crucial Step):**
   The system uses RS256 Asymmetric Authentication. You must generate a key pair in the `certs/` directory.

   ```bash
   mkdir -p certs
   # Generate Private Key (for Classic Backend / Auth Authority)
   openssl genrsa -out certs/private.pem 2048
   
   # Extract Public Key (for Analysis Backend / Verifier)
   openssl rsa -in certs/private.pem -outform PEM -pubout -out certs/public.pem
   ```

   **Security Model:**
   - **Classic Backend (Port 8000)**: Uses `private.pem` to SIGN tokens.
   - **Analysis Backend (Port 8001)**: Uses `public.pem` to VERIFY tokens.

5. **Initialize the database:**
   ```bash
   # Import schema and seed data
   mysql -u root -p < data/full_creation.sql
   mysql -u root -p < data/full_insert.sql
   ```

6. **Run database migrations (if using Alembic):**
   ```bash
   alembic upgrade head
   ```

## Running the Service

### Running the Services

The system uses a split-backend architecture:

1. **Classic Backend (CRUD & Management):**
   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Analysis API (Video Processing):**
   ```bash
   uvicorn analysis_app:app --reload --host 0.0.0.0 --port 8001
   ```

### Production Mode
```bash
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120
```

### Docker Deployment
```bash
docker build -t coach-pro-backend .
docker run -d \
  --name coach-pro-backend \
  -p 8000:8000 \
  --env-file .env \
  coach-pro-backend
```

## API Endpoints

### Video Analysis

#### Upload and Analyze Match Video
```http
POST /api/analyze_match
Content-Type: multipart/form-data

file: video.mp4
match_id: uuid (optional)
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "PENDING",
  "match_id": "uuid",
  "estimated_duration_seconds": 120
}
```

#### Check Job Status
```http
GET /api/jobs/{job_id}
```

**Response:**
```json
{
  "job_id": "uuid",
  "status": "COMPLETED",
  "progress_percent": 100,
  "result": {
    "match_id": "uuid",
    "total_frames": 5400,
    "players_tracked": 22,
    "report_id": "uuid"
  }
}
```

### Data Retrieval

#### List Matches
```http
GET /api/matches?limit=20&offset=0
```

#### Get Match Details
```http
GET /api/matches/{match_id}
```

#### Player Statistics
```http
GET /api/player_match_statistics?match_id={match_id}&player_id={player_id}
```

#### Analysis Reports
```http
GET /api/analysis_reports/{report_id}
```

### Team Management

#### Formations
```http
GET /api/formations
POST /api/formations
```

#### Match Lineups
```http
GET /api/match_lineups/{lineup_id}
POST /api/match_lineups
```

## Data Flow Architecture

### Video Analysis & Matching Workflow

The system follows a strict sequential workflow to ensure data integrity and real-time feedback:

1. **Upload & Buffering (Synchronous)**
   - Client uploads video via `POST /api/analyze_match` (Multipart).
   - Backend provides real-time progress for the **sending** phase.
   - Video is buffered to disk in `temp_uploads/` to ensure persistence for background tasks.

2. **Database Registration (Instant)**
   - Once upload is complete, the backend immediately creates:
     - A `Match` record (if not provided).
     - An `AnalysisReport` record with `initializing` status.
   - This ensures the analysis is visible in the **Match History** instantly.

3. **Background Analysis (Asynchronous)**
   - A background task starts `run_streaming_analysis_from_disk`.
   - Video chunks are read from disk and streamed to the C++ Engine via gRPC.
   - Analysis status is updated in real-time (`streaming` → `analyzing` → `completed`).
   - Final metrics are persisted to the database.

4. **Cleanup**
   - Upon completion or failure, the temporary video file is automatically removed from the server.

4. **Retrieval Phase**
   - Frontend queries REST API for structured results
   - Backend returns JSON responses with aggregated statistics
   - Support for filtering, pagination, and sorting

## Database Schema

### Core Tables

- **matches**: Match metadata, timestamps, team info
- **players**: Player profiles, positions, jersey numbers
- **teams**: Team information and configurations
- **analysis_reports**: Aggregated match statistics
- **player_match_statistics**: Per-player, per-match metrics
- **formations**: Tactical formations (4-4-2, 4-3-3, etc.)
- **match_lineups**: Starting XI and substitutions
- **video_segments**: Timestamped video clips for events

### Relationships

```sql
matches 1:N analysis_reports
players N:M matches (through player_match_statistics)
teams 1:N formations
matches 1:1 match_lineups
```

## Testing

### Unit Tests
```bash
pytest tests/unit -v --cov=app
```

### Integration Tests
```bash
pytest tests/integration -v --cov=app --cov-report=html
```

### Smoke Test (CSV Parsing)
```bash
python scripts/smoke_parse_and_persist.py
```

This validates end-to-end data persistence without running the C++ engine.

## Configuration Management

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_HOST` | MySQL host address | `localhost` |
| `DB_PORT` | MySQL port | `3306` |
| `DB_USER` | Database username | `root` |
| `DB_PASSWORD` | Database password | `` |
| `DB_NAME` | Database name | `coach_pro_db` |
| `ANALYSIS_ENGINE_HOST` | C++ engine gRPC host | `localhost` |
| `ANALYSIS_ENGINE_PORT` | C++ engine gRPC port | `50051` |
| `RSA_PRIVATE_KEY` | Path to private key | `certs/private.pem` |
| `RSA_PUBLIC_KEY` | Path to public key | `certs/public.pem` |
| `UPLOAD_DIR` | Video storage directory | `./uploads` |
| `MAX_UPLOAD_SIZE_MB` | Maximum video file size | `500` |

## Monitoring and Logging

### Structured Logging
```python
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Health Check Endpoint
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "engine": "reachable",
  "uptime_seconds": 3600
}
```

## Security Considerations

- **Authentication**: **RS256 Asymmetric JWT**. Tokens are signed by the Authority (Classic Backend) and verified statelessly by microservices (Analysis Backend).
- **Input Validation**: Pydantic models prevent injection attacks
- **File Upload Restrictions**: MIME type validation, size limits, virus scanning
- **CORS Configuration**: Whitelisted origins only
- **Rate Limiting**: Per-IP request throttling (e.g., 100 req/min)

## Project Structure

```
.
├── main.py                     # FastAPI application entry point
├── requirements.txt
├── .env.example
├── alembic/                    # Database migrations
├── app/
│   ├── api/
│   │   ├── endpoints/          # Route handlers
│   │   └── dependencies.py     # Dependency injection
│   ├── models/                 # SQLAlchemy ORM models
│   ├── schemas/                # Pydantic schemas
│   ├── services/
│   │   ├── analysis_service.py # gRPC client for C++ engine
│   │   ├── job_manager.py      # Job state machine
│   │   └── db_service.py       # Database operations
│   └── core/
│       ├── config.py           # Configuration management
│       └── security.py         # Authentication utilities
├── data/
│   ├── full_creation.sql       # Database schema
│   └── full_insert.sql         # Seed data
├── scripts/
│   └── smoke_parse_and_persist.py
└── tests/
    ├── unit/
    └── integration/
```

## Performance Optimization

### Database Indexing
Ensure indexes on frequently queried columns:
```sql
CREATE INDEX idx_match_date ON matches(match_date);
CREATE INDEX idx_player_stats_match ON player_match_statistics(match_id);
```

### Connection Pooling
Configure SQLAlchemy connection pool:
```python
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True
)
```

### Caching Layer
Integrate Redis for frequently accessed data:
```python
import redis
cache = redis.Redis(host='localhost', port=6379, db=0)
```

## Troubleshooting

### Common Issues

**Database Connection Failed**
```bash
# Verify MySQL is running
systemctl status mysql

# Test connection
mysql -u root -p -h localhost
```

**gRPC Connection Refused**
```bash
# Check if C++ engine is running
netstat -tulpn | grep 50051

# Test gRPC endpoint
grpcurl -plaintext localhost:50051 list
```

**Large File Upload Timeout**
Increase Nginx/gunicorn timeout:
```nginx
client_max_body_size 1000M;
proxy_read_timeout 300s;
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-endpoint`
3. Commit changes: `git commit -am 'Add new endpoint'`
4. Push to branch: `git push origin feature/new-endpoint`
5. Submit a pull request

## License

MIT License - See LICENSE file for details

## Support

For integration support or bug reports, open an issue on GitHub or contact the development team.