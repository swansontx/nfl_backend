# Data Freshness Guide

## How Fresh is Your Data?

### ✅ Real-Time (Every Request)
These are calculated **live** when the frontend makes a request:
- Recommendation scores
- Matchup analysis (opponent rankings)
- Signal combinations
- Parlay correlations
- Confidence calculations

**No caching** - every API call gets fresh calculations.

### 📊 Database Data (Updated Manually)
These come from your PostgreSQL database:
- Game schedules
- Player rosters
- Historical stats (play-by-play)
- Game results/scores

**Update frequency:** Weekly (or daily during season)

---

## Update Schedule

### During NFL Season (Sep - Feb)

**Weekly Updates (Recommended):**
```bash
# Run every Tuesday after Monday Night Football
python scripts/update_data.py
```

This updates:
- Latest game results from previous week
- New player stats
- Updated injury reports (if scraped)
- Retrained models with new data

**Daily Updates (Optional for Production):**
```bash
# Run at 2 AM daily via cron
0 2 * * * cd /path/to/nfl_backend && python scripts/update_data.py
```

### Off-Season (Mar - Aug)

**Monthly Updates:**
```bash
# Run once a month to get roster changes, trades, draft picks
python scripts/update_data.py
```

---

## Manual Update

To update right now:

```bash
# Option 1: Use the update script (runs all 3 steps)
python scripts/update_data.py

# Option 2: Run steps individually
python scripts/load_nfl_data.py      # 1. Load NFL data
python scripts/generate_features.py  # 2. Generate features
python scripts/train_models.py       # 3. Retrain models
```

**Time required:** 10-15 minutes total

---

## Frontend Polling Strategy

### Option 1: Manual Refresh
Frontend users click "Refresh" button to get latest data:
```javascript
const refreshData = async () => {
  const response = await fetch('http://localhost:8000/api/v1/games/?season=2024');
  const data = await response.json();
  // Update UI
};
```

### Option 2: Auto-Refresh
Refresh data every N minutes:
```javascript
// Refresh every 5 minutes
setInterval(async () => {
  const response = await fetch('http://localhost:8000/api/v1/games/today');
  const data = await response.json();
  // Update UI
}, 5 * 60 * 1000);
```

### Option 3: WebSocket (Future Enhancement)
For real-time updates, could add WebSocket support:
- Push updates when games start/end
- Live score updates
- Instant recommendation changes

---

## Data Staleness Indicators

### Check Data Freshness

**1. Health Endpoint:**
```bash
curl http://localhost:8000/health

# Returns:
{
  "status": "healthy",
  "timestamp": "2024-11-17T12:00:00Z",
  "database": "connected",
  "games": 284,
  "players": 3215
}
```

**2. Latest Game Date:**
```bash
curl 'http://localhost:8000/api/v1/games/?season=2024' | jq '.games[-1].game_date'
```

If the latest game is >1 week old, run an update.

**3. Database Query:**
```sql
SELECT MAX(game_date) as latest_game
FROM games
WHERE season = 2024;
```

---

## External Data Sources (Future)

### Real-Time Data (when API keys added)

**1. Odds API (The Odds API)**
```python
# backend/odds/odds_fetcher.py
# Fetches latest betting lines every 15 minutes
# Requires: ODDS_API_KEY in .env
```

**2. News API**
```python
# backend/news/news_fetcher.py
# Fetches injury reports every hour
# Requires: NEWS_API_KEY in .env
```

**3. Weather API**
```python
# backend/weather/weather_fetcher.py
# Fetches game day weather 6 hours before kickoff
# Requires: WEATHER_API_KEY in .env
```

**To enable:**
1. Get API keys (most have free tiers)
2. Add to `backend/config/.env`
3. APIs auto-fetch on request (with caching)

---

## Caching Strategy (Optional)

For production with high traffic, add Redis caching:

**Cache recommendations for 5 minutes:**
```python
@cache_recommendations(ttl=300)  # 5 minutes
def get_recommendations(game_id):
    # Expensive calculation
    return recommendations
```

**Benefits:**
- Faster response times
- Reduced database load
- Lower compute costs

**Trade-offs:**
- Data can be up to 5 minutes stale
- Need Redis server

---

## Automation Scripts

### Mac (using cron)

Edit crontab:
```bash
crontab -e
```

Add weekly update (Tuesdays at 2 AM):
```
0 2 * * 2 cd /path/to/nfl_backend && /path/to/python scripts/update_data.py >> logs/update.log 2>&1
```

### Linux (systemd timer)

Create `/etc/systemd/system/nfl-update.service`:
```ini
[Unit]
Description=Update NFL data

[Service]
Type=oneshot
WorkingDirectory=/path/to/nfl_backend
ExecStart=/usr/bin/python3 scripts/update_data.py
```

Create `/etc/systemd/system/nfl-update.timer`:
```ini
[Unit]
Description=Run NFL data update weekly

[Timer]
OnCalendar=Tue *-*-* 02:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable:
```bash
sudo systemctl enable nfl-update.timer
sudo systemctl start nfl-update.timer
```

---

## Monitoring Data Freshness

### Add to Frontend

Show data age in UI:
```javascript
const DataFreshnessIndicator = () => {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetch('http://localhost:8000/health')
      .then(r => r.json())
      .then(setHealth);
  }, []);

  const lastUpdate = new Date(health?.timestamp);
  const hoursAgo = (Date.now() - lastUpdate) / 1000 / 60 / 60;

  return (
    <div className={hoursAgo > 24 ? 'warning' : 'ok'}>
      Data last updated: {hoursAgo.toFixed(0)}h ago
    </div>
  );
};
```

### Add to Backend

Create endpoint to show data age:
```python
@router.get("/data-status")
def get_data_status():
    with get_db() as session:
        latest_game = session.query(Game).order_by(Game.game_date.desc()).first()
        latest_feature = session.query(PlayerGameFeature).order_by(
            PlayerGameFeature.created_at.desc()
        ).first()

        return {
            "latest_game_date": latest_game.game_date,
            "latest_feature_generated": latest_feature.created_at,
            "games_count": session.query(Game).count(),
            "features_count": session.query(PlayerGameFeature).count(),
            "is_stale": (datetime.utcnow() - latest_game.game_date).days > 7
        }
```

---

## Summary

| Data Type | Freshness | Update Method | Frequency |
|-----------|-----------|---------------|-----------|
| Recommendations | Real-time | Auto-calculated | Every request |
| Matchup analysis | Real-time | Auto-calculated | Every request |
| Game results | Manual | `update_data.py` | Weekly |
| Player stats | Manual | `update_data.py` | Weekly |
| Odds (future) | Auto | External API | Every 15 min |
| News (future) | Auto | External API | Every hour |

**Current state:** ✅ Real-time calculations, weekly data updates needed

**To stay fresh:**
1. Run `python scripts/update_data.py` weekly during season
2. API handles rest automatically
3. Frontend gets fresh recommendations on every request
