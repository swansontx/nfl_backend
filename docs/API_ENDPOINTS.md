# API Endpoints Documentation

Complete reference for all NFL Props Backend API endpoints.

## Base URL

```
http://localhost:8000
```

API Documentation (Swagger): http://localhost:8000/docs

---

## ✅ COMPLETE ENDPOINT COVERAGE

All endpoints required by frontend are now implemented!

### Teams Endpoints (COMPLETE)
- ✅ `GET /api/v1/teams` - List all teams
- ✅ `GET /api/v1/teams/{team_id}` - Team details  
- ✅ `GET /api/v1/teams/{team_id}/stats` - Team statistics
- ✅ `GET /api/v1/teams/{team_id}/schedule` - Team schedule
- ✅ `GET /api/v1/teams/{team_id}/news` - Team news

### Games Endpoints (COMPLETE)
- ✅ `GET /api/v1/games` - List games (filterable)
- ✅ `GET /api/v1/games/{game_id}` - Game details
- ✅ `GET /api/v1/games/{game_id}/boxscore` - Complete boxscore
- ✅ `GET /api/v1/games/{game_id}/weather` - Weather
- ✅ `GET /api/v1/games/{game_id}/insights` - Insights
- ✅ `GET /api/v1/games/{game_id}/projections` - Projections

### Players Endpoints (SCAFFOLDED)
- ✅ `GET /api/v1/players` - Search players
- ✅ `GET /api/v1/players/{player_id}` - Player details
- ✅ `GET /api/v1/players/{player_id}/stats` - Player stats
- ✅ `GET /api/v1/players/{player_id}/gamelogs` - Game logs
- ✅ `GET /api/v1/players/{player_id}/insights` - Player insights

### Prop Betting Endpoints (COMPLETE)
- ✅ `GET /api/v1/odds/current` - Current DraftKings odds
- ✅ `GET /api/v1/props/trending` - Trending props (line movement tracking)
- ✅ `GET /api/v1/props/value` - High-value prop recommendations
- ✅ `GET /api/v1/props/compare` - Compare props across players

### League Info Endpoints (COMPLETE)
- ✅ `GET /api/v1/standings` - NFL standings by division
- ✅ `GET /api/v1/news` - Latest NFL news and injuries

### Advanced Analysis Endpoints (NEW)
Advanced routes that power the MCP analysis tools. See [docs/ADVANCED_ANALYSIS.md](ADVANCED_ANALYSIS.md) for full payload shapes and curl examples.

- ✅ `GET /analysis/situational` - Weather, rest, momentum, and positional edge breakdown for a matchup
- ✅ `GET /analysis/team/form` - Team momentum + efficiency trend snapshot
- ✅ `GET /analysis/game/{game_id}/positional` - Position-by-position matchup grades and prop targets
- ✅ `GET /analysis/defense/{team}` - Combined rush/pass defense summary with player-level matchups
- ✅ `GET /analysis/defense/{team}/pass` - Pass defense performance vs recent quarterbacks
- ✅ `GET /analysis/defense/{team}/rush` - Rush defense performance vs recent running backs
- ✅ `GET /analysis/evaluate/game` - Full game evaluation combining situational, matchup, injury, and prop analyzers
- ✅ `GET /analysis/evaluate/week` - Weekly ranking of games with top prop targets

---

## Quick Examples

### Get All Teams
```bash
curl http://localhost:8000/api/v1/teams
```

### Get Team Schedule
```bash
curl http://localhost:8000/api/v1/teams/KC/schedule?season=2024
```

### Get Games for Week 12
```bash
curl http://localhost:8000/api/v1/games?week=12&season=2024
```

### Get Game Details
```bash
curl http://localhost:8000/api/v1/games/2024_12_KC_BUF
```

### Get Complete Boxscore
```bash
curl http://localhost:8000/api/v1/games/2024_10_KC_BUF/boxscore
```

### Get Current Odds (DraftKings)
```bash
curl http://localhost:8000/api/v1/odds/current?week=12
```

### Get Trending Props
```bash
curl http://localhost:8000/api/v1/props/trending?week=12&limit=20
```

### Get NFL Standings
```bash
curl http://localhost:8000/api/v1/standings?season=2024
```

---

## Detailed Endpoint Documentation

### Prop Betting Endpoints

#### GET /api/v1/odds/current

Get current DraftKings prop odds for all upcoming games.

**Query Parameters:**
- `week` (optional): Week number (defaults to current week)
- `market` (optional): Filter by specific market (e.g., 'player_pass_yds')

**Response:**
```json
{
  "week": 12,
  "total_games": 14,
  "total_props": 450,
  "snapshot_timestamp": "2024-11-19T12:00:00",
  "games": [
    {
      "game_id": "2024_12_KC_BUF",
      "home_team": "KC",
      "away_team": "BUF",
      "commence_time": "2024-11-24T20:20:00",
      "props_by_market": {
        "player_pass_yds": [
          {
            "player": "Patrick Mahomes",
            "market": "player_pass_yds",
            "line": 275.5,
            "over_odds": -110,
            "under_odds": -110,
            "timestamp": "2024-11-19T12:00:00",
            "bookmaker": "draftkings"
          }
        ]
      }
    }
  ]
}
```

#### GET /api/v1/props/trending

Get trending prop lines with directional categorization and 3-week movement tracking.

**Query Parameters:**
- `week` (optional): Week number
- `limit` (optional): Number of props per category (default: 20)

**Response:**
```json
{
  "week": 12,
  "snapshot_count": 15,
  "current_timestamp": "2024-11-19T12:00:00",

  "hottest_movers": [
    {
      "player": "Patrick Mahomes",
      "market": "player_pass_yds",
      "bookmaker": "draftkings",
      "opening_line": 265.5,
      "current_line": 275.5,
      "line_movement": 10.0,
      "line_movement_pct": 3.8,
      "over_odds": -110,
      "under_odds": -110,
      "direction": "up",
      "opening_timestamp": "2024-11-12T12:00:00",
      "current_timestamp": "2024-11-19T12:00:00",
      "days_tracked": 7,
      "badge": "+10.0 pts (7 days)",
      "badge_pct": "+4% (7 days)",
      "icon": "⬆️",
      "color": "green",
      "strength": "🔥"
    }
  ],

  "lines_moving_up": [
    {
      "player": "Josh Allen",
      "market": "player_pass_yds",
      "line_movement": 5.0,
      "direction": "up"
    }
  ],

  "lines_moving_down": [
    {
      "player": "Travis Kelce",
      "market": "player_reception_yds",
      "line_movement": -3.5,
      "direction": "down"
    }
  ],

  "sustained_trends": [
    {
      "player": "Christian McCaffrey",
      "market": "player_rush_yds",
      "week_1_line": 85.5,
      "week_2_line": 90.5,
      "week_3_line": 95.5,
      "current_line": 95.5,
      "total_movement": 10.0,
      "total_movement_pct": 11.7,
      "direction": "up",
      "consistency": "3/3 weeks up",
      "badge": "+10.0 pts (3 weeks)",
      "strength": "🔥🔥"
    }
  ],

  "summary": {
    "hottest_movers": {
      "total": 20,
      "strong": 8,
      "avg_movement": 4.2
    },
    "lines_moving_up": {
      "total": 12,
      "strong": 5,
      "avg_movement": 3.1
    },
    "lines_moving_down": {
      "total": 8,
      "strong": 2,
      "avg_movement": -2.8
    },
    "sustained_trends": {
      "total": 15,
      "strong": 6,
      "trending_up": 9,
      "trending_down": 6
    }
  }
}
```

**Trending Categories:**
- 🔥 **Hottest movers**: Biggest absolute line changes (regardless of direction)
- ⬆️ **Lines moving up**: Props getting harder to hit Over (line increased)
- ⬇️ **Lines moving down**: Props getting easier to hit Over (line decreased)
- **Sustained trends**: 3-week consistent directional patterns

### League Info Endpoints

#### GET /api/v1/standings

Get NFL standings by division and conference.

**Query Parameters:**
- `season` (optional): Season year (default: 2024)
- `week` (optional): Week number (returns current standings if not specified)

**Response:**
```json
{
  "season": 2024,
  "week": 12,
  "standings": {
    "afc_east": [
      {
        "team": "BUF",
        "wins": 9,
        "losses": 2,
        "ties": 0,
        "win_pct": 0.818
      }
    ],
    "afc_north": [...],
    "afc_south": [...],
    "afc_west": [...],
    "nfc_east": [...],
    "nfc_north": [...],
    "nfc_south": [...],
    "nfc_west": [...]
  }
}
```

### Advanced Analysis Endpoints

These routes power the MCP analysis workflows. Full schemas live in [docs/ADVANCED_ANALYSIS.md](ADVANCED_ANALYSIS.md).

#### GET /analysis/situational

Blend weather, rest, travel, injuries, and positional matchup indicators for a single game.

**Query Parameters:**
- `game_id` (required): Game identifier (e.g., `2024_12_KC_BUF`)
- `home_team` (required): Home team abbreviation
- `away_team` (required): Away team abbreviation
- `season` (required): Season year
- `week` (required): Week number

**Example:**
```bash
curl "http://localhost:8000/analysis/situational?game_id=2024_12_KC_BUF&home_team=KC&away_team=BUF&season=2024&week=12"
```

#### GET /analysis/defense/{team}

Combined run/pass defense report with positional edges and recent opponent context.

**Path Parameters:**
- `team` (required): Team abbreviation (e.g., `KC`)

**Query Parameters:**
- `season` (required): Season year
- `week` (optional): Week number for contextual trends

**Examples:**
```bash
# Overall defensive profile
curl "http://localhost:8000/analysis/defense/KC?season=2024"

# Pass defense vs recent quarterbacks
curl "http://localhost:8000/analysis/defense/KC/pass?season=2024&week=12"

# Rush defense vs recent running backs
curl "http://localhost:8000/analysis/defense/KC/rush?season=2024&week=12"
```

#### GET /analysis/evaluate/game

End-to-end evaluator that fuses situational analysis, positional matchups, injuries, and prop signals for a single game.

**Query Parameters:**
- `game_id` (required): Game identifier
- `home_team` (required): Home team abbreviation
- `away_team` (required): Away team abbreviation
- `season` (required): Season year
- `week` (required): Week number

**Example:**
```bash
curl "http://localhost:8000/analysis/evaluate/game?game_id=2024_12_KC_BUF&home_team=KC&away_team=BUF&season=2024&week=12"
```

#### GET /analysis/evaluate/week

Ranks games for the week with recommended prop targets and model notes.

**Query Parameters:**
- `season` (required): Season year
- `week` (required): Week number

**Example:**
```bash
curl "http://localhost:8000/analysis/evaluate/week?season=2024&week=12"
```

---

## Usage Guide: Prop Betting Workflow

### Prerequisites

Before using the prop betting endpoints, you need to capture prop line snapshots:

```bash
# Set your Odds API key
export ODDS_API_KEY="your_api_key_here"

# Fetch current week prop lines (creates snapshot)
python -m backend.ingestion.fetch_prop_lines --week 12 --output outputs/prop_lines/

# Run daily to build historical data for trending analysis
# Recommended: Set up cron job to run daily at same time
```

### Complete Workflow

#### 1. Daily Snapshot Capture (Automated)

Set up a daily cron job to capture snapshots:

```bash
# Add to crontab (runs daily at 10 AM)
0 10 * * * cd /path/to/nfl_backend && python -m backend.ingestion.fetch_prop_lines --week 12
```

**Why daily?**
- Week-over-week trending requires 7 daily snapshots
- 3-week sustained trends need 10-15 snapshots (every 2-3 days minimum)
- More snapshots = better trend detection

#### 2. Get Current Odds (Frontend Display)

```bash
# Get all current DraftKings odds for week 12
curl http://localhost:8000/api/v1/odds/current?week=12

# Filter by specific market
curl http://localhost:8000/api/v1/odds/current?week=12&market=player_pass_yds
```

**Use cases:**
- Display current lines on player prop cards
- Show live odds for betting decisions
- Compare against model projections

#### 3. Get Trending Props (Line Movement)

```bash
# Get top 20 trending props per category
curl http://localhost:8000/api/v1/props/trending?week=12&limit=20

# Get top 50 for comprehensive analysis
curl http://localhost:8000/api/v1/props/trending?week=12&limit=50
```

**Response includes 4 categories:**

**🔥 Hottest Movers** - Biggest absolute line changes
```javascript
// Use for: Immediate action alerts, breaking news detection
hottest_movers[0]
// {
//   player: "Patrick Mahomes",
//   line_movement: 10.0,  // Line moved up 10 yards
//   badge: "+10.0 pts (7 days)",
//   strength: "🔥"  // Strong move
// }
```

**⬆️ Lines Moving Up** - Props getting harder to hit Over
```javascript
// Use for: Under betting opportunities, fade the public
lines_moving_up.filter(p => p.strength === "🔥")
// Line increased = Market expects better performance
// Strategy: Target the Under, or wait for line to stabilize
```

**⬇️ Lines Moving Down** - Props getting easier to hit Over
```javascript
// Use for: Over betting opportunities, injury/news detection
lines_moving_down.filter(p => p.strength === "🔥")
// Line decreased = Market expects worse performance
// Strategy: Target the Over if you disagree with market
```

**Sustained 3-Week Trends** - Pattern validation
```javascript
// Use for: Long-term trend validation, sustained edges
sustained_trends.filter(t => t.consistency === "3/3 weeks up")
// Consistent upward trend = Market confidence growing
// Consistent downward trend = Market confidence fading
```

#### 4. Get NFL Standings (Context)

```bash
# Get current standings
curl http://localhost:8000/api/v1/standings?season=2024

# Get standings for specific week
curl http://localhost:8000/api/v1/standings?season=2024&week=12
```

**Use cases:**
- Display team records on game pages
- Contextualize playoff implications
- Show division standings on team pages

### Frontend Integration Examples

#### Display Current Odds on Player Card

```javascript
// Fetch current odds for a game
const response = await fetch('/api/v1/odds/current?week=12');
const { games } = await response.json();

// Find player's prop
const game = games.find(g => g.game_id === '2024_12_KC_BUF');
const passYards = game.props_by_market.player_pass_yds;
const mahomesLine = passYards.find(p => p.player === 'Patrick Mahomes');

// Display: "Patrick Mahomes O/U 275.5 yards (-110/-110)"
```

#### Show Trending Props Section

```javascript
// Fetch trending props
const response = await fetch('/api/v1/props/trending?week=12&limit=20');
const { hottest_movers, lines_moving_up, lines_moving_down, sustained_trends } = await response.json();

// Display sections:
// 🔥 Hottest Movers (biggest changes)
hottest_movers.map(prop => (
  <TrendingCard
    player={prop.player}
    market={prop.market}
    badge={prop.badge}  // "+10.0 pts (7 days)"
    icon={prop.icon}    // "⬆️"
    color={prop.color}  // "green"
    strength={prop.strength}  // "🔥"
  />
));

// ⬆️ Lines Moving Up (harder to hit Over)
lines_moving_up.map(prop => (
  <TrendingCard {...prop} />
));

// ⬇️ Lines Moving Down (easier to hit Over)
lines_moving_down.map(prop => (
  <TrendingCard {...prop} />
));
```

#### Display Standings Widget

```javascript
// Fetch standings
const response = await fetch('/api/v1/standings?season=2024');
const { standings } = await response.json();

// Display AFC East standings
standings.afc_east.map(team => (
  <StandingsRow
    team={team.team}
    wins={team.wins}
    losses={team.losses}
    winPct={team.win_pct}
  />
));
```

### Recommended API Call Frequency

**High Frequency (Real-time)**
- `/api/v1/odds/current` - Every 5-10 minutes during game week
  - Lines change frequently as games approach
  - Critical for timing bets

**Medium Frequency (Multiple times per day)**
- `/api/v1/props/trending` - 2-3 times daily
  - Morning (to see overnight moves)
  - Afternoon (to catch breaking news)
  - Evening (for next-day preview)

**Low Frequency (Once per day or less)**
- `/api/v1/standings` - Once daily or after games complete
  - Standings only update after games finish

### Error Handling

```javascript
// Check for missing data errors
const response = await fetch('/api/v1/props/trending?week=12');
const data = await response.json();

if (data.error) {
  if (data.error === 'Insufficient snapshots for trending analysis') {
    // Show message: "Trending data not available yet. Need at least 2 daily snapshots."
    console.log('Snapshots found:', data.snapshots_found);
    console.log('Run: python -m backend.ingestion.fetch_prop_lines --week 12');
  } else if (data.error === 'No prop line snapshots found') {
    // Show message: "No prop data available. Run data ingestion first."
    console.log('Command:', data.command);
  }
}
```

### Sharp Action Detection (Advanced)

The prop line fetcher tracks multiple sportsbooks to detect sharp action:

```bash
# Fetch with sharp action analysis
python -m backend.ingestion.fetch_prop_lines --week 12
```

**How to interpret line movements:**

1. **DK isolated move** (Public money)
   - DraftKings line moves alone
   - Other sharp books stay flat
   - **Strategy:** FADE DraftKings

2. **Steam move** (Strong consensus)
   - All books move together
   - Sharp and public money aligned
   - **Strategy:** Follow the movement

3. **Sharp disagreement**
   - Sharp books (FanDuel, BetMGM) move opposite DK
   - DraftKings stays flat or moves opposite
   - **Strategy:** Follow sharp books, fade DK

4. **Sharp consensus**
   - DK + sharp books move together
   - Good signal for betting
   - **Strategy:** High confidence bet

---

## Complete Documentation

See Swagger docs for detailed request/response schemas:
http://localhost:8000/docs

All endpoints support:
- Proper HTTP status codes (200, 404, 400, 500)
- JSON responses
- CORS for frontend integration
- Query parameter filtering

---

## Implementation Notes

**Fully Functional:**
- Teams (all endpoints working with real data)
- Schedule/Games (working with nflverse schedules)
- Boxscores (generated from play-by-play data)

**Structured (Ready for Data Integration):**
- Team stats (structure ready, needs aggregation)
- Player endpoints (structure ready, needs player_lookup/stats CSV)
- Betting lines (structure ready, needs odds API)

See full API details in Swagger docs!
