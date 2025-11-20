# NFL Betting Props - Quick Start Guide

## 1. Virtual Environment Setup

```bash
# Navigate to project
cd /path/to/nfl_backend

# Create virtual environment
python -m venv venv

# Activate it
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Required packages** (if no requirements.txt):
```bash
pip install pandas numpy requests fastapi uvicorn xgboost lightgbm scikit-learn
```

---

## 2. Pull Latest Branch

```bash
# Navigate to your project folder
cd /path/to/nfl_backend

# Fetch the latest from remote
git fetch origin claude/nfl-betting-props-dev-01J82MkeBwHiJEeGdaEhPUn2

# Switch to the branch (if not already on it)
git checkout claude/nfl-betting-props-dev-01J82MkeBwHiJEeGdaEhPUn2

# Pull latest changes
git pull origin claude/nfl-betting-props-dev-01J82MkeBwHiJEeGdaEhPUn2
```

**If you get merge conflicts** (like with `data/nfl_betting.db`):
```bash
# Remove conflicting file (it will be recreated)
rm data/nfl_betting.db

# Pull again
git pull origin claude/nfl-betting-props-dev-01J82MkeBwHiJEeGdaEhPUn2
```

---

## 3. Initialize SQLite Database & Populate Data

### Step 1: Initialize the database
```bash
python -c "from backend.database.local_db import init_database; init_database()"
```

This creates `data/nfl_betting.db` with all tables.

### Step 2: Start the API server (handles data population)
```bash
python start_server.py --auto-update
```

Or manually:
```bash
python api_server.py
```

The server starts on `http://localhost:8000`

### Step 3: Populate data via API endpoints

Open another terminal and run:

```bash
# Populate all data at once
curl -X POST http://localhost:8000/populate/all

# Or individually:
curl -X POST http://localhost:8000/populate/schedules
curl -X POST http://localhost:8000/populate/player-stats
curl -X POST http://localhost:8000/populate/injuries
curl -X POST http://localhost:8000/populate/odds  # Requires ODDS_API_KEY
```

### Step 4: Check database status
```bash
curl http://localhost:8000/database/status
```

---

## 4. Fetch Real Sportsbook Odds

### Set up Odds API key
```bash
# Get free key at https://the-odds-api.com/ (500 requests/month)
export ODDS_API_KEY=your_key_here
```

### Fetch odds into database
```bash
# Via API
curl -X POST http://localhost:8000/populate/odds

# Or directly
python -m backend.ingestion.fetch_odds
```

---

## 5. Start MCP Server (for Claude integration)

```bash
python mcp_server.py
```

The MCP server exposes tools for:
- Fetching game predictions
- Analyzing prop values
- Generating picks
- Accessing player statistics

---

## 6. Generate Picks

### Via command line:
```bash
python -m backend.recommendations.generate_game_picks --team1 HOU --team2 BUF
```

### Via API:
```bash
curl "http://localhost:8000/picks/HOU/BUF"
```

Output saved to `outputs/picks_HOU_BUF.json`

---

## 7. Common Workflows

### Daily workflow (game day):
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Start server with auto-update
python start_server.py --auto-update

# 3. Fetch fresh odds (separate terminal)
curl -X POST http://localhost:8000/populate/odds

# 4. Generate picks for today's games
curl "http://localhost:8000/picks/KC/BUF"
```

### Weekly workflow (update all data):
```bash
# Full data refresh
curl -X POST http://localhost:8000/populate/all
```

### Using the Orchestrator (full pipeline):
```bash
# Run full pipeline: data + features + training + predictions
python -m backend.orchestration.orchestrator --season 2024 --full

# Just fetch data and features (no training)
python -m backend.orchestration.orchestrator --season 2024

# Training + predictions only
python -m backend.orchestration.orchestrator --season 2024 --train --predict

# Generate picks for specific game
python -m backend.orchestration.orchestrator --season 2024 --picks --team1 HOU --team2 BUF

# Backtest mode
python -m backend.orchestration.orchestrator --season 2023 --backtest

# List all pipeline stages
python -m backend.orchestration.orchestrator --list-stages
```

The orchestrator runs stages in order:
1. Initialize SQLite database
2. Fetch nflverse data (PBP, player stats, rosters)
3. Fetch NFL schedules
4. Fetch sportsbook odds (requires ODDS_API_KEY)
5. Fetch injury reports
6. Extract player PBP features
7. Apply smoothing and rolling windows
8. Build roster & injury indexes
9. (Optional) Train models
10. (Optional) Generate projections
11. (Optional) Run backtests
12. (Optional) Generate game picks

---

## 8. Troubleshooting

### "No module named X"
```bash
pip install X
```

### Database locked
```bash
# Kill any running servers, then restart
pkill -f "python.*server"
python start_server.py
```

### SQLAlchemy import error
The system uses SQLite (local_db.py), not PostgreSQL. If you see SQLAlchemy errors, they're from unused PostgreSQL code - ignore them.

### No odds data
1. Check ODDS_API_KEY is set
2. Run: `curl -X POST http://localhost:8000/populate/odds`
3. Check: `curl http://localhost:8000/database/status`

---

## 9. File Locations

| What | Where |
|------|-------|
| SQLite Database | `data/nfl_betting.db` |
| Input CSVs | `inputs/` |
| Trained Models | `outputs/models/` |
| Generated Picks | `outputs/picks_*.json` |
| API Server | `api_server.py` (port 8000) |
| MCP Server | `mcp_server.py` |

---

## 10. Key API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Server status & database overview |
| `/database/status` | GET | Data inventory |
| `/populate/all` | POST | Refresh all data |
| `/populate/odds` | POST | Fetch sportsbook lines |
| `/picks/{team1}/{team2}` | GET | Generate picks for matchup |
| `/predictions/{game_id}` | GET | Get projections for game |
| `/injuries/{team}` | GET | Team injury report |
