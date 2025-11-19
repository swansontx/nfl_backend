# NFL Props System - Quick Start

## For You (Running Backend Locally)

### 1. Install Dependencies

```bash
cd nfl_backend
pip install -r requirements.txt
```

### 2. Start the API

```bash
# Simple way
uvicorn backend.api.app:app --reload --port 8000

# Or with startup script
./start.sh
```

API runs at: **http://localhost:8000**

### 3. Test It's Working

```bash
# Health check
curl http://localhost:8000/health

# Interactive docs (open in browser)
open http://localhost:8000/docs
```

### 4. Analyze a Game

```bash
# Run demo analysis
python scripts/demo_analysis.py

# Or validate system
python scripts/validate_system.py --season 2024
```

---

## For Frontend Developer

### API Base URL
```
http://localhost:8000
```

### Key Endpoints

**Games:**
```javascript
// Get today's games
GET /api/v1/games/today

// Get games for week 17
GET /api/v1/games/?season=2024&week=17
```

**Recommendations:**
```javascript
// Get top props for a game
GET /api/v1/recommendations/{game_id}?limit=10&min_confidence=0.7

// Get parlays
GET /api/v1/recommendations/{game_id}/parlays?parlay_size=3
```

**Backtesting:**
```javascript
// Historical performance
GET /api/v1/backtest/?season=2024
```

### Example Frontend Code

```typescript
// types.ts
interface PropRecommendation {
  player_name: string;
  position: string;
  team: string;
  market: string;
  line: number;
  calibrated_prob: number;
  overall_score: number;
  recommendation_strength: 'elite' | 'strong' | 'moderate' | 'weak' | 'avoid';
  confidence: number;
  reasoning: string[];
  flags: string[];
  edge?: number;
}

interface Game {
  game_id: string;
  season: number;
  week: number;
  home_team: string;
  away_team: string;
  game_date: string;
  completed: boolean;
}

// api.ts
const API_URL = 'http://localhost:8000';

export const getGames = async () => {
  const res = await fetch(`${API_URL}/api/v1/games/today`);
  return res.json();
};

export const getRecommendations = async (gameId: string) => {
  const res = await fetch(
    `${API_URL}/api/v1/recommendations/${gameId}?limit=20&min_confidence=0.7`
  );
  return res.json();
};

export const getParlays = async (gameId: string) => {
  const res = await fetch(
    `${API_URL}/api/v1/recommendations/${gameId}/parlays?parlay_size=3`
  );
  return res.json();
};

// component.tsx
import { useEffect, useState } from 'react';

export const GameAnalysis = ({ gameId }) => {
  const [recs, setRecs] = useState([]);

  useEffect(() => {
    getRecommendations(gameId).then(data => setRecs(data.recommendations));
  }, [gameId]);

  return (
    <div>
      {recs.map(rec => (
        <div key={`${rec.player_name}-${rec.market}`}>
          <h3>{rec.player_name} - {rec.market}</h3>
          <p>Line: {rec.line}</p>
          <p>Probability: {(rec.calibrated_prob * 100).toFixed(1)}%</p>
          <p>Score: {rec.overall_score.toFixed(3)}</p>
          <p className={`strength-${rec.recommendation_strength}`}>
            {rec.recommendation_strength.toUpperCase()}
          </p>

          <ul>
            {rec.reasoning.map((r, i) => <li key={i}>{r}</li>)}
          </ul>

          {rec.edge && <p>Edge: +{(rec.edge * 100).toFixed(1)}%</p>}
        </div>
      ))}
    </div>
  );
};
```

---

## Interactive API Docs

Once backend is running, visit:

**http://localhost:8000/docs**

This gives you a full interactive API explorer where you can:
- See all endpoints
- Try them out with real data
- See request/response schemas
- Copy curl commands

---

## Troubleshooting

**Backend won't start:**
```bash
# Check database connection
psql -U postgres -d nfl_props

# Check if port 8000 is free
lsof -i :8000

# Try different port
uvicorn backend.api.app:app --reload --port 8001
```

**No recommendations:**
- Make sure game data exists in database
- Models need to be trained
- Player features need to be generated

**CORS errors:**
- Already configured for localhost
- If frontend on different machine, update CORS in `backend/api/app.py`

---

## File Locations

- **API Code**: `backend/api/`
- **Models**: `backend/models/`, `backend/recommendations/`
- **Scripts**: `scripts/`
- **Docs**: `API_DOCUMENTATION.md`, `VALIDATION.md`

---

## Next Steps

1. **Test API**: `curl http://localhost:8000/health`
2. **Explore Docs**: Open http://localhost:8000/docs
3. **Run Analysis**: `python scripts/demo_analysis.py`
4. **Frontend Integration**: Use endpoints in React/Next.js

---

## Full Documentation

- **API Docs**: `API_DOCUMENTATION.md`
- **Validation**: `VALIDATION.md`
- **Validation Checklist**: `VALIDATION_CHECKLIST.md`
