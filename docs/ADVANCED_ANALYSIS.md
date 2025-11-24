# Advanced Analysis API

Detailed documentation for advanced analysis routes used by the MCP tools.

## Situational Analysis
- `GET /analysis/situational` — Weather, rest, momentum, and positional edges for a matchup.

**Example**
```bash
curl "http://localhost:8000/analysis/situational?game_id=2024_12_KC_BUF&home_team=KC&away_team=BUF&season=2024&week=12"
```

## Team Form Trends
- `GET /analysis/team/form` — Team momentum and efficiency trend snapshot.

**Example**
```bash
curl "http://localhost:8000/analysis/team/form?team=KC&season=2024&week=12"
```

## Positional Matchups
- `GET /analysis/game/{game_id}/positional` — Position-by-position matchup grades and prop targets.

**Example**
```bash
curl "http://localhost:8000/analysis/game/2024_12_KC_BUF/positional?home_team=KC&away_team=BUF&season=2024&week=12"
```

## Defense Profiles
- `GET /analysis/defense/{team}` — Combined rush/pass defense summary with player matchups.
- `GET /analysis/defense/{team}/pass` — Pass defense performance vs recent quarterbacks.
- `GET /analysis/defense/{team}/rush` — Rush defense performance vs recent running backs.

**Example**
```bash
curl "http://localhost:8000/analysis/defense/KC?season=2024"
```

## Evaluation Pipelines
- `GET /analysis/evaluate/game` — Full game evaluation combining situational, matchup, injury, and prop analyzers.
- `GET /analysis/evaluate/week` — Weekly ranking of games with top prop targets.

**Examples**
```bash
curl "http://localhost:8000/analysis/evaluate/game?game_id=2024_12_KC_BUF&home_team=KC&away_team=BUF&season=2024&week=12"
```
```bash
curl "http://localhost:8000/analysis/evaluate/week?season=2024&week=12"
```
