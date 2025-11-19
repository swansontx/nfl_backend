# PostgreSQL Database Setup

This document outlines the PostgreSQL database layer for the NFL Props prediction system.

## Overview

The database layer stores:
- **User accounts** - Authentication, profiles, subscription tiers
- **Bet history** - User bets, outcomes, performance tracking
- **Model predictions** - Historical predictions for backtesting
- **Cache/optimization** - Frequently accessed data (schedules, rosters, lines)

## Database Schema

### Users Table
```sql
CREATE TABLE users (
    user_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(50) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    subscription_tier VARCHAR(20) DEFAULT 'free', -- free, basic, premium
    created_at TIMESTAMP DEFAULT NOW(),
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    preferences JSONB DEFAULT '{}'::jsonb -- User settings, notifications, favorite teams
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
```

### Bet History Table
```sql
CREATE TABLE bet_history (
    bet_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    game_id VARCHAR(50) NOT NULL,
    player_id VARCHAR(50),
    bet_type VARCHAR(50) NOT NULL, -- 'over', 'under', 'spread', 'moneyline'
    prop_type VARCHAR(50), -- 'passing_yards', 'rushing_yards', etc.
    line_value DECIMAL(10, 2),
    odds DECIMAL(10, 2), -- American odds
    stake DECIMAL(10, 2) DEFAULT 1.00, -- Units bet
    result VARCHAR(20), -- 'win', 'loss', 'push', 'pending'
    actual_value DECIMAL(10, 2), -- Actual stat value
    payout DECIMAL(10, 2), -- Profit/loss
    placed_at TIMESTAMP DEFAULT NOW(),
    settled_at TIMESTAMP,
    notes TEXT,
    metadata JSONB DEFAULT '{}'::jsonb -- Additional bet details
);

CREATE INDEX idx_bet_history_user ON bet_history(user_id);
CREATE INDEX idx_bet_history_game ON bet_history(game_id);
CREATE INDEX idx_bet_history_player ON bet_history(player_id);
CREATE INDEX idx_bet_history_result ON bet_history(result);
CREATE INDEX idx_bet_history_placed ON bet_history(placed_at DESC);
```

### Model Predictions Table
```sql
CREATE TABLE model_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id VARCHAR(50) NOT NULL,
    player_id VARCHAR(50) NOT NULL,
    prediction_type VARCHAR(50) NOT NULL, -- 'passing_yards', 'rushing_yards', etc.
    predicted_value DECIMAL(10, 2) NOT NULL,
    confidence DECIMAL(5, 4), -- 0.0 to 1.0
    vegas_line DECIMAL(10, 2),
    model_edge DECIMAL(10, 2), -- Predicted value - Vegas line
    actual_value DECIMAL(10, 2), -- Filled in after game
    prediction_error DECIMAL(10, 2), -- abs(predicted - actual)
    model_version VARCHAR(50), -- Model identifier
    created_at TIMESTAMP DEFAULT NOW(),
    game_date DATE,
    season INTEGER,
    week INTEGER,
    features JSONB DEFAULT '{}'::jsonb -- Feature values used in prediction
);

CREATE INDEX idx_predictions_game ON model_predictions(game_id);
CREATE INDEX idx_predictions_player ON model_predictions(player_id);
CREATE INDEX idx_predictions_type ON model_predictions(prediction_type);
CREATE INDEX idx_predictions_created ON model_predictions(created_at DESC);
CREATE INDEX idx_predictions_season_week ON model_predictions(season, week);
```

### Cached Lines Table
```sql
CREATE TABLE cached_lines (
    line_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    game_id VARCHAR(50) NOT NULL,
    player_id VARCHAR(50),
    sportsbook VARCHAR(50) NOT NULL, -- 'draftkings', 'fanduel', etc.
    market_type VARCHAR(50) NOT NULL, -- 'player_prop', 'game_total', 'spread'
    prop_type VARCHAR(50), -- 'passing_yards', 'rushing_yards', etc.
    line_value DECIMAL(10, 2),
    over_odds DECIMAL(10, 2),
    under_odds DECIMAL(10, 2),
    fetched_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP, -- TTL for cache invalidation
    metadata JSONB DEFAULT '{}'::jsonb
);

CREATE INDEX idx_cached_lines_game ON cached_lines(game_id);
CREATE INDEX idx_cached_lines_player ON cached_lines(player_id);
CREATE INDEX idx_cached_lines_expires ON cached_lines(expires_at);
CREATE INDEX idx_cached_lines_fetched ON cached_lines(fetched_at DESC);
```

### User Bankroll Table
```sql
CREATE TABLE user_bankroll (
    bankroll_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    total_units DECIMAL(10, 2) DEFAULT 100.00,
    starting_units DECIMAL(10, 2) DEFAULT 100.00,
    total_bets INTEGER DEFAULT 0,
    winning_bets INTEGER DEFAULT 0,
    losing_bets INTEGER DEFAULT 0,
    push_bets INTEGER DEFAULT 0,
    total_profit DECIMAL(10, 2) DEFAULT 0.00,
    roi DECIMAL(10, 4), -- Return on investment %
    win_rate DECIMAL(5, 4), -- Win percentage
    last_bet_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_bankroll_user ON user_bankroll(user_id);
```

### User Watchlist Table
```sql
CREATE TABLE user_watchlist (
    watchlist_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(user_id) ON DELETE CASCADE,
    game_id VARCHAR(50),
    player_id VARCHAR(50),
    prop_type VARCHAR(50),
    target_line DECIMAL(10, 2), -- Alert when line reaches this value
    target_odds DECIMAL(10, 2),
    alert_triggered BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW(),
    notes TEXT
);

CREATE INDEX idx_watchlist_user ON user_watchlist(user_id);
CREATE INDEX idx_watchlist_game ON user_watchlist(game_id);
```

## Setup Instructions

### 1. Install PostgreSQL

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

**macOS (Homebrew):**
```bash
brew install postgresql
brew services start postgresql
```

**Windows:**
Download installer from https://www.postgresql.org/download/windows/

### 2. Create Database

```bash
# Login as postgres user
sudo -u postgres psql

# Create database
CREATE DATABASE nfl_props;

# Create application user
CREATE USER nfl_props_user WITH ENCRYPTED PASSWORD 'your_secure_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE nfl_props TO nfl_props_user;

# Exit psql
\q
```

### 3. Run Schema Migrations

```bash
# Connect to database
psql -U nfl_props_user -d nfl_props -h localhost

# Run schema from this file (copy SQL above)
# Or use migration tool (Alembic)
```

### 4. Configure Environment Variables

Add to `.env` file:
```env
# Database
DATABASE_URL=postgresql://nfl_props_user:your_secure_password@localhost:5432/nfl_props
DB_HOST=localhost
DB_PORT=5432
DB_NAME=nfl_props
DB_USER=nfl_props_user
DB_PASSWORD=your_secure_password

# Connection Pool
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```

### 5. Initialize with Alembic (Optional)

```bash
# Install Alembic
pip install alembic psycopg2-binary

# Initialize Alembic
alembic init alembic

# Create migration
alembic revision -m "Initial schema"

# Run migrations
alembic upgrade head
```

## Database Module Structure

```
backend/
  database/
    __init__.py          # Database connection, session management
    models.py            # SQLAlchemy ORM models
    crud.py              # CRUD operations
    schemas.py           # Pydantic schemas for validation
    migrations/          # Alembic migration files
      versions/
```

## Usage Examples

### Python Database Module (SQLAlchemy)

```python
from backend.database import get_db_session
from backend.database.models import User, BetHistory
from backend.database.crud import create_bet, get_user_bets

# Create bet
session = get_db_session()
bet = create_bet(
    session=session,
    user_id=user_id,
    game_id="2024_17_BUF_KC",
    player_id="00-0038120",
    bet_type="over",
    prop_type="passing_yards",
    line_value=275.5,
    odds=-110,
    stake=1.0
)

# Query user's bet history
user_bets = get_user_bets(session, user_id, limit=50)
```

### API Integration

```python
from fastapi import Depends
from backend.database import get_db

@app.post("/api/v1/bets")
def place_bet(bet_data: BetCreate, db: Session = Depends(get_db)):
    """Place a new bet."""
    bet = create_bet(db, bet_data)
    return {"bet_id": str(bet.bet_id), "status": "placed"}

@app.get("/api/v1/users/{user_id}/bets")
def get_user_bet_history(user_id: str, db: Session = Depends(get_db)):
    """Get user's bet history."""
    bets = get_user_bets(db, user_id)
    return {"bets": bets}
```

## Performance Optimization

### Indexes
All critical columns have indexes for fast queries:
- User lookups: email, username
- Bet queries: user_id, game_id, player_id, result, placed_at
- Prediction queries: game_id, player_id, season/week
- Line caching: game_id, player_id, expires_at

### Partitioning (Optional)
For large datasets, consider partitioning by date:
```sql
-- Partition bet_history by month
CREATE TABLE bet_history_2024_01 PARTITION OF bet_history
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
```

### Connection Pooling
Use SQLAlchemy connection pooling:
```python
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True  # Verify connections
)
```

## Backup Strategy

### Automated Backups
```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/backups/nfl_props"
DATE=$(date +%Y%m%d_%H%M%S)

pg_dump -U nfl_props_user nfl_props > "$BACKUP_DIR/nfl_props_$DATE.sql"

# Compress
gzip "$BACKUP_DIR/nfl_props_$DATE.sql"

# Delete backups older than 30 days
find $BACKUP_DIR -name "*.sql.gz" -mtime +30 -delete
```

### Restore from Backup
```bash
gunzip nfl_props_20240101_120000.sql.gz
psql -U nfl_props_user -d nfl_props < nfl_props_20240101_120000.sql
```

## Security Best Practices

1. **Never commit database credentials** - Use environment variables
2. **Use strong passwords** - Generate with password manager
3. **Enable SSL** - For production deployments
4. **Limit privileges** - Application user should not have SUPERUSER
5. **Regular backups** - Automated daily backups with off-site storage
6. **Monitor queries** - Use pg_stat_statements for slow query detection
7. **Connection limits** - Prevent resource exhaustion

## Next Steps

1. Create `backend/database/` module with SQLAlchemy models
2. Set up Alembic migrations
3. Implement CRUD operations
4. Add authentication endpoints to FastAPI
5. Create user registration/login flows
6. Build bet tracking dashboard
7. Implement analytics queries for user performance

## Resources

- PostgreSQL Docs: https://www.postgresql.org/docs/
- SQLAlchemy ORM: https://docs.sqlalchemy.org/
- Alembic Migrations: https://alembic.sqlalchemy.org/
- FastAPI + SQLAlchemy: https://fastapi.tiangolo.com/tutorial/sql-databases/
