# Service Status & Next Steps

**Last Updated:** 2025-11-24
**After:** Removal of all placeholder/fallback data

---

## ✅ COMPLETED

### All Fallback Data Removed
- ✅ No placeholder weather data
- ✅ No sample odds/events
- ✅ No fake news items
- ✅ No placeholder insights/narratives
- ✅ No synthetic player data
- ✅ All APIs fail gracefully with clear error messages

### Dynamic Date Resolution
- ✅ All critical files use `get_current_nfl_season()`
- ✅ No hardcoded 2025/2024 in production code
- ✅ Analysis scripts intentionally hardcoded (point-in-time) are documented

### Real Data Integration
- ✅ Player stats from NFLverse CSVs
- ✅ Team defensive stats from CSVs
- ✅ Schedule data from CSVs
- ✅ Injury data from Sleeper API (no key required)

---

## 🔴 SERVICES WITH CONSISTENT ISSUES

### 1. Weather API (OpenWeather)
**Status:** ❌ BLOCKS FEATURES
**Impact:** HIGH

**Issue:**
- Requires `OPENWEATHER_API_KEY` environment variable
- Used by:
  - `GET /api/v1/games/{game_id}/weather`
  - `backend/features/extract_weather_features.py`
  - Weather insights in game analysis

**Current Behavior:**
- Raises `ValueError: OPENWEATHER_API_KEY not set`
- No fallback - completely blocks weather features

**Resolution Options:**
1. **Get API key** - Free tier: 1000 calls/day (~$0/month)
   - Sign up: https://openweathermap.org/api
   - Set: `export OPENWEATHER_API_KEY=your_key_here`

2. **Use NFLverse fallback** - Historical weather from play-by-play data
   - Implement: `load_weather_from_nflverse()` in extract_weather_features.py
   - Limitation: Only historical, not forecasts

3. **Dome-only** - Only return data for dome stadiums
   - Already implemented for domes (controlled environment)
   - Skip outdoor games

**Recommendation:** Get API key (free tier sufficient)

---

### 2. Odds API (The Odds API)
**Status:** ⚠️ DEGRADES FEATURES
**Impact:** HIGH

**Issue:**
- Requires `ODDS_API_KEY` environment variable
- Used by:
  - `GET /api/v1/odds`
  - `GET /api/v1/props/value` (prop value analysis)
  - `/api/v1/props/trending` (line movement)

**Current Behavior:**
- Returns empty list `[]` with error message
- Downstream features return "No prop lines available"

**Resolution:**
1. **Get API key** - $25/month for 50 requests/day
   - Sign up: https://the-odds-api.com/
   - Set: `export ODDS_API_KEY=your_key_here`

2. **Manual CSV** - Create odds files manually
   - Format: `outputs/odds/odds_{game_id}.json`
   - Scrape from public sources (legal gray area)

**Recommendation:** Get API key or operate without prop lines (predictions only)

---

### 3. Model Projections Loading
**Status:** ⚠️ NOT IMPLEMENTED
**Impact:** HIGH

**Issue:**
- Model files exist in `outputs/models/`
- Projection files exist in `outputs/predictions/`
- But loading not wired into API endpoints

**Affected Endpoints:**
- `GET /api/v1/props/value` - "No projections available"
- `GET /api/v1/tools/player-comparison` - "Feature not implemented"
- `GET /api/v1/games/{game_id}/insights` - "Feature not implemented"

**Current Behavior:**
- Returns JSON with `{"error": "Feature not implemented"}`

**Resolution:**
See P4_EXTERNAL_SERVICE_REQUIREMENTS.md items #4, #8, #9

**Recommendation:** Wire up model loading (HIGH PRIORITY)

---

### 4. News/Content APIs
**Status:** ℹ️ LOW PRIORITY
**Impact:** LOW

**Issues:**
- `GET /api/v1/news` - Only returns Sleeper injury news
- `GET /api/v1/games/{game_id}/narrative` - Requires LLM API
- `GET /api/v1/games/{game_id}/content` - Requires YouTube/RSS APIs

**Current Behavior:**
- News: Returns only injury items (works, but limited)
- Narratives: Returns `{"error": "Feature not implemented"}`
- Content: Returns `{"error": "Feature not implemented"}`

**Resolution:**
See P4_EXTERNAL_SERVICE_REQUIREMENTS.md items #3, #6, #7

**Recommendation:** Defer until core features complete

---

## 🟢 WORKING SERVICES

### NFLverse Data (All FREE)
- ✅ Player stats CSV loading
- ✅ Team defensive stats
- ✅ Schedule/standings data
- ✅ Depth charts
- ✅ NextGen Stats
- ✅ Snap counts
- ✅ Red zone stats

### Sleeper API (FREE, No Key Required)
- ✅ Player injury data
- ✅ Real-time status updates
- ✅ Automatic caching (15 min TTL)

### Stadium Database (LOCAL)
- ✅ All 32 NFL stadiums
- ✅ Coordinates for weather API
- ✅ Dome/outdoor flags
- ✅ Automatic game-to-stadium mapping

---

## 📋 NEXT STEPS (Priority Order)

### Immediate (Must Do)
1. **Decision: Get Weather API Key?**
   - Cost: $0 (free tier)
   - Impact: Enables weather features
   - Time: 5 minutes to sign up
   - **Action:** Set `OPENWEATHER_API_KEY` or disable weather endpoints

2. **Decision: Get Odds API Key?**
   - Cost: $25/month
   - Impact: Enables prop betting features
   - Alternative: Run predictions-only mode
   - **Action:** Set `ODDS_API_KEY` or document "predictions-only" mode

3. **Test All API Endpoints**
   - Run: `pytest tests/test_api.py`
   - Verify error messages are clear
   - Check that real data endpoints work
   - **Action:** Fix any failing tests

### Short Term (This Week)
4. **Wire Up Model Loading** (HIGH PRIORITY)
   - Load projections from `outputs/predictions/*.csv`
   - Integrate into `/api/v1/props/value`
   - Enable player comparison endpoint
   - See: P4_EXTERNAL_SERVICE_REQUIREMENTS.md item #8
   - **Estimated:** 3-4 hours

5. **Implement Game Insights** (HIGH PRIORITY)
   - Pull from `backend/features/`
   - Use actual model predictions
   - Add weather integration (if API key available)
   - See: P4_EXTERNAL_SERVICE_REQUIREMENTS.md item #4
   - **Estimated:** 4-6 hours

6. **Fix CORS for Production**
   - Set allowed origins in environment
   - See: P4_EXTERNAL_SERVICE_REQUIREMENTS.md item #1
   - **Estimated:** 5 minutes

### Medium Term (Next 2 Weeks)
7. **Orchestration Integration**
   - Wire `/admin/recompute` into orchestrator
   - Add job status tracking
   - See: P4_EXTERNAL_SERVICE_REQUIREMENTS.md item #2
   - **Estimated:** 2-4 hours

8. **Generate Prop Sheet** (Core Feature)
   - Aggregate props from Odds API
   - Run model projections
   - Calculate EV and value grades
   - See: P4_EXTERNAL_SERVICE_REQUIREMENTS.md item #9
   - **Estimated:** 4-6 hours

### Long Term (Optional)
9. **News API Integration**
   - ESPN/NFL.com RSS feeds
   - See: P4_EXTERNAL_SERVICE_REQUIREMENTS.md item #3
   - **Estimated:** 2-3 hours

10. **LLM Narratives**
    - OpenAI or Anthropic API
    - See: P4_EXTERNAL_SERVICE_REQUIREMENTS.md item #6
    - **Estimated:** 1-2 hours

11. **Content Aggregation**
    - YouTube, podcasts, articles
    - See: P4_EXTERNAL_SERVICE_REQUIREMENTS.md item #7
    - **Estimated:** 3-4 hours

---

## 🔍 HOW TO IDENTIFY ISSUES

### Check API Endpoint Health
```bash
# Test weather endpoint (will fail if no API key)
curl http://localhost:8000/api/v1/games/2025_12_KC_BUF/weather

# Test odds endpoint (will return empty if no API key)
curl http://localhost:8000/api/v1/odds

# Test player stats (should work - uses CSV)
curl http://localhost:8000/api/v1/players/00-0033873/stats

# Test standings (should work - uses CSV)
curl http://localhost:8000/api/v1/standings
```

### Check Error Messages
All error responses now follow this format:
```json
{
  "error": "Error type",
  "message": "Detailed explanation with fix instructions",
  "data": []
}
```

Or raise exceptions with clear messages:
```python
ValueError: "OPENWEATHER_API_KEY not set - cannot fetch weather data. Set environment variable: export OPENWEATHER_API_KEY=your_key_here"
```

### Check Logs
```bash
# Start API server and watch logs
python backend/api/app.py

# Look for:
# ✅ "Loaded X players from inputs/players.csv"
# ✅ "Loaded X stats from inputs/player_stats_2024_2025.csv"
# ❌ "ERROR: ODDS_API_KEY not set"
# ❌ "ValueError: OPENWEATHER_API_KEY not set"
```

---

## 📊 FEATURE COMPLETENESS MATRIX

| Feature | Real Data | API Integration | Status |
|---------|-----------|-----------------|--------|
| Player Stats | ✅ | N/A | ✅ WORKING |
| Team Stats | ✅ | N/A | ✅ WORKING |
| Schedule/Standings | ✅ | N/A | ✅ WORKING |
| Injuries | N/A | ✅ Sleeper (free) | ✅ WORKING |
| Weather | ✅ NFLverse (hist) | ⚠️ OpenWeather (need key) | ⚠️ BLOCKED |
| Odds/Props | N/A | ⚠️ OddsAPI (paid) | ⚠️ BLOCKED |
| Projections | ✅ Model files exist | ❌ Loading not wired | ❌ NOT IMPL |
| Insights | ✅ Feature extractors | ❌ Not wired to API | ❌ NOT IMPL |
| Narratives | N/A | ❌ LLM API (paid) | ❌ NOT IMPL |
| News | N/A | ✅ Sleeper (limited) | ⚠️ LIMITED |
| Content | N/A | ❌ YouTube/RSS APIs | ❌ NOT IMPL |

**Legend:**
- ✅ WORKING - Fully functional with real data
- ⚠️ BLOCKED - Needs API key to function
- ⚠️ LIMITED - Works but incomplete
- ❌ NOT IMPL - Feature exists but not wired up

---

## 🚨 CRITICAL DECISIONS NEEDED

### 1. Weather Strategy
**Options:**
- A) Get OpenWeather API key (free) - **RECOMMENDED**
- B) Use NFLverse historical only (no forecasts)
- C) Disable weather endpoints entirely

**Your choice:** ___________

### 2. Odds/Props Strategy
**Options:**
- A) Get OddsAPI key ($25/month) - For full prop betting features
- B) Predictions-only mode (no odds integration) - Free
- C) Manual odds entry (CSV files)

**Your choice:** ___________

### 3. LLM Integration Priority
**Options:**
- A) High priority - Get OpenAI/Anthropic key now
- B) Medium priority - After core features work
- C) Low priority - Defer indefinitely

**Your choice:** ___________

---

## 💡 QUICK WINS (< 1 hour each)

1. ✅ **Set CORS origins** - 5 minutes
   ```bash
   export CORS_ORIGINS="https://yourdomain.com"
   ```

2. ⏳ **Add betting lines to standings** - 1 hour
   - Enrich standings with current week's lines
   - Uses existing odds API integration

3. ⏳ **Stadium location for insights** - 30 minutes
   - Wire up stadium database to insights endpoint
   - Already have all infrastructure

4. ⏳ **Get weather API key** - 5 minutes
   - Sign up at openweathermap.org
   - Set environment variable
   - Immediately enables all weather features

---

## 📝 ENVIRONMENT VARIABLES CHECKLIST

```bash
# Core Data (All FREE, no keys needed)
✅ NFLverse CSVs in inputs/ directory

# Optional APIs
⚠️ OPENWEATHER_API_KEY=________________  # FREE tier available
⚠️ ODDS_API_KEY=_______________________  # $25/month
❌ OPENAI_API_KEY=_____________________  # For LLM narratives
❌ YOUTUBE_API_KEY=____________________  # For video content

# Production Config
❌ CORS_ORIGINS=_______________________  # Comma-separated domains
```

**Status:**
- ✅ = Have and configured
- ⚠️ = Need for key features
- ❌ = Optional/low priority

---

## 🎯 SUCCESS CRITERIA

**Minimal Viable Product:**
- [x] All player/team stats work (real NFLverse data)
- [x] Injuries work (Sleeper API)
- [ ] Weather works (need API key) OR disabled with clear messaging
- [ ] Model projections load correctly
- [ ] At least 1 value prop analysis endpoint works

**Full Feature Set:**
- [ ] All above + Odds API integration
- [ ] Game insights with ML models
- [ ] Prop sheet generation
- [ ] Orchestration pipeline working

**Enhanced Features:**
- [ ] LLM narratives
- [ ] Content aggregation
- [ ] News feed (beyond injuries)

---

## 📞 SUPPORT

**Questions about:**
- NFLverse data: Check API_DATA_INTEGRATION_STATUS.md
- External services: Check P4_EXTERNAL_SERVICE_REQUIREMENTS.md
- Implementation status: Check HARDCODED_AND_TODO_FIXES.md

**Common Issues:**
- "OPENWEATHER_API_KEY not set" → Get free API key
- "No prop lines available" → Get ODDS_API_KEY or accept predictions-only
- "Feature not implemented" → See P4_EXTERNAL_SERVICE_REQUIREMENTS.md

**All services now fail with clear, actionable error messages.**
