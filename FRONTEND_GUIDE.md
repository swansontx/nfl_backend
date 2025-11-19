# Frontend Integration Guide

Quick guide for frontend developers to integrate with the NFL Props API.

## 🚀 Quick Start

### API Base URL
```
http://localhost:8000
```

### Interactive Documentation
```
http://localhost:8000/docs
```

---

## 📋 Setup Test Data

Before testing the frontend, run this script to create sample data:

```bash
# On your Mac (with venv activated)
python scripts/quick_setup_for_testing.py
```

This creates:
- A Thursday night game (PIT @ CLE)
- Sample players with realistic stats
- Mock player features for recommendations

---

## 🔌 Key API Endpoints

### 1. Get Games

**Get all games:**
```javascript
GET /api/v1/games/?season=2024

// Response
{
  "games": [
    {
      "game_id": "2024_12_PIT_CLE",
      "season": 2024,
      "week": 12,
      "home_team": "CLE",
      "away_team": "PIT",
      "game_date": "2024-11-21T20:15:00",
      "completed": false
    }
  ],
  "total_count": 1
}
```

**Get today's games:**
```javascript
GET /api/v1/games/today
```

**Get specific game:**
```javascript
GET /api/v1/games/2024_12_PIT_CLE
```

### 2. Get Recommendations

**Get prop recommendations for a game:**
```javascript
GET /api/v1/recommendations/2024_12_PIT_CLE?limit=10&min_confidence=0.6

// Response
{
  "game_id": "2024_12_PIT_CLE",
  "recommendations": [
    {
      "player_name": "George Pickens",
      "position": "WR",
      "team": "PIT",
      "market": "player_rec_yds",
      "line": 68.5,
      "calibrated_prob": 0.64,
      "overall_score": 0.76,
      "recommendation_strength": "strong",
      "confidence": 0.82,
      "reasoning": [
        "Favorable matchup vs CLE secondary",
        "Hot streak: 85+ yards in last 3 games"
      ],
      "flags": [],
      "edge": 0.12
    }
  ]
}
```

**Query Parameters:**
- `limit` (int, 1-100): Max recommendations (default: 10)
- `min_confidence` (float, 0-1): Minimum confidence (default: 0.6)
- `markets` (list): Filter by specific markets

### 3. Get Parlays

**Get parlay recommendations:**
```javascript
GET /api/v1/recommendations/2024_12_PIT_CLE/parlays?parlay_size=3

// Response
{
  "game_id": "2024_12_PIT_CLE",
  "parlays": [
    {
      "props": [
        {
          "player_name": "George Pickens",
          "market": "player_rec_yds",
          "line": 68.5,
          "probability": 0.64
        },
        {
          "player_name": "David Njoku",
          "market": "player_receptions",
          "line": 4.5,
          "probability": 0.61
        },
        {
          "player_name": "Jaylen Warren",
          "market": "player_rush_yds",
          "line": 45.5,
          "probability": 0.58
        }
      ],
      "raw_probability": 0.231,
      "adjusted_probability": 0.278,
      "overall_score": 0.72,
      "correlation_impact": "positive",
      "confidence": 0.735
    }
  ]
}
```

**Query Parameters:**
- `parlay_size` (int, 2-6): Number of props (default: 3)
- `limit` (int, 1-20): Max parlays (default: 5)

### 4. Health Check

**Check API status:**
```javascript
GET /health

// Response
{
  "status": "healthy",
  "timestamp": "2024-11-19T12:00:00Z",
  "database": "connected",
  "games": 1,
  "players": 8
}
```

---

## 💻 Frontend Code Examples

### React/Next.js

```typescript
// lib/api.ts
const API_URL = 'http://localhost:8000';

export async function getGames(season: number = 2024) {
  const res = await fetch(`${API_URL}/api/v1/games/?season=${season}`);
  return res.json();
}

export async function getRecommendations(gameId: string) {
  const res = await fetch(
    `${API_URL}/api/v1/recommendations/${gameId}?limit=20&min_confidence=0.6`
  );
  return res.json();
}

export async function getParlays(gameId: string, size: number = 3) {
  const res = await fetch(
    `${API_URL}/api/v1/recommendations/${gameId}/parlays?parlay_size=${size}`
  );
  return res.json();
}
```

### React Component Example

```typescript
// components/GameAnalysis.tsx
'use client';

import { useEffect, useState } from 'react';
import { getRecommendations } from '@/lib/api';

interface PropRecommendation {
  player_name: string;
  position: string;
  team: string;
  market: string;
  line: number;
  calibrated_prob: number;
  overall_score: number;
  recommendation_strength: string;
  confidence: number;
  reasoning: string[];
  edge?: number;
}

export default function GameAnalysis({ gameId }: { gameId: string }) {
  const [recs, setRecs] = useState<PropRecommendation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getRecommendations(gameId)
      .then(data => {
        setRecs(data.recommendations);
        setLoading(false);
      })
      .catch(err => {
        console.error('Failed to load recommendations:', err);
        setLoading(false);
      });
  }, [gameId]);

  if (loading) return <div>Loading recommendations...</div>;

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold">Top Props</h2>

      {recs.map((rec, i) => (
        <div key={i} className="border rounded-lg p-4">
          <div className="flex justify-between items-start">
            <div>
              <h3 className="font-semibold text-lg">
                {rec.player_name} ({rec.position})
              </h3>
              <p className="text-sm text-gray-600">
                {rec.team} • {rec.market}
              </p>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold">
                {(rec.calibrated_prob * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-gray-600">
                Line: {rec.line}
              </div>
            </div>
          </div>

          <div className="mt-3">
            <div className="flex gap-2 mb-2">
              <span className={`px-2 py-1 rounded text-sm font-medium ${
                rec.recommendation_strength === 'strong'
                  ? 'bg-green-100 text-green-800'
                  : rec.recommendation_strength === 'moderate'
                  ? 'bg-yellow-100 text-yellow-800'
                  : 'bg-gray-100 text-gray-800'
              }`}>
                {rec.recommendation_strength.toUpperCase()}
              </span>
              <span className="px-2 py-1 rounded text-sm bg-blue-100 text-blue-800">
                {(rec.confidence * 100).toFixed(0)}% Confidence
              </span>
              {rec.edge && (
                <span className="px-2 py-1 rounded text-sm bg-purple-100 text-purple-800">
                  +{(rec.edge * 100).toFixed(1)}% Edge
                </span>
              )}
            </div>

            <ul className="space-y-1 text-sm">
              {rec.reasoning.map((reason, j) => (
                <li key={j} className="flex items-start">
                  <span className="mr-2">•</span>
                  <span>{reason}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      ))}
    </div>
  );
}
```

---

## 📊 TypeScript Types

```typescript
// types/api.ts

export interface Game {
  game_id: string;
  season: number;
  week: number;
  home_team: string;
  away_team: string;
  game_date: string;
  completed: boolean;
  stadium?: string;
  game_type?: string;
}

export interface PropRecommendation {
  player_id?: string;
  player_name: string;
  position: string;
  team: string;
  market: string;
  line: number;
  model_prob?: number;
  calibrated_prob: number;
  base_signal?: number;
  matchup_signal?: number;
  trend_signal?: number;
  news_signal?: number;
  roster_signal?: number;
  overall_score: number;
  recommendation_strength: 'elite' | 'strong' | 'moderate' | 'weak' | 'avoid';
  confidence: number;
  reasoning: string[];
  flags: string[];
  edge?: number;
}

export interface ParlayRecommendation {
  props: Array<{
    player_name: string;
    market: string;
    line: number;
    probability: number;
    score?: number;
  }>;
  raw_probability: number;
  adjusted_probability: number;
  adjustment_factor?: number;
  overall_score: number;
  correlation_impact: string;
  confidence: number;
  reasoning: string[];
}

export interface GamesResponse {
  games: Game[];
  total_count: number;
  season?: number;
  week?: number;
}

export interface RecommendationsResponse {
  game_id: string;
  game_time?: string;
  home_team: string;
  away_team: string;
  recommendations: PropRecommendation[];
  total_count: number;
  markets_analyzed?: string[];
  min_confidence?: number;
}

export interface ParlaysResponse {
  game_id: string;
  parlays: ParlayRecommendation[];
  total_count: number;
  parlay_size: number;
}
```

---

## 🧪 Testing the API

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Get games
curl http://localhost:8000/api/v1/games/?season=2024

# Get recommendations
curl "http://localhost:8000/api/v1/recommendations/2024_12_PIT_CLE?limit=10"

# Get parlays
curl "http://localhost:8000/api/v1/recommendations/2024_12_PIT_CLE/parlays?parlay_size=3"
```

### Using the Interactive Docs

1. Open http://localhost:8000/docs
2. Click on any endpoint
3. Click "Try it out"
4. Fill in parameters
5. Click "Execute"
6. See the response

---

## 🎨 Recommendation Strength Colors

Suggested color scheme for the frontend:

```typescript
const strengthColors = {
  elite: 'bg-purple-100 text-purple-800 border-purple-300',
  strong: 'bg-green-100 text-green-800 border-green-300',
  moderate: 'bg-yellow-100 text-yellow-800 border-yellow-300',
  weak: 'bg-orange-100 text-orange-800 border-orange-300',
  avoid: 'bg-red-100 text-red-800 border-red-300',
};
```

---

## ⚡ Performance Tips

1. **Cache game lists** - Games don't change frequently
2. **Debounce recommendation fetches** - Don't refetch on every filter change
3. **Use loading states** - API calls can take 500ms-2s
4. **Handle errors gracefully** - Show user-friendly error messages

---

## 🐛 Common Issues

**CORS errors:**
- Already configured for localhost
- Backend allows all origins in development

**No recommendations returned:**
- Make sure you ran `python scripts/quick_setup_for_testing.py`
- Check the game_id format: `SEASON_WEEK_AWAY_HOME`
- Verify the game exists: `curl http://localhost:8000/api/v1/games/GAME_ID`

**500 errors:**
- Check backend logs in the terminal where uvicorn is running
- Database might not be connected
- Missing player features

---

## 📞 Need Help?

- **API Docs**: http://localhost:8000/docs
- **Full API Documentation**: See `API_DOCUMENTATION.md`
- **Backend Logs**: Check the terminal running uvicorn
