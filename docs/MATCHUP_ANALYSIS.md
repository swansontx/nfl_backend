# Advanced Matchup Analysis Guide

Complete guide to using EPA, CPOE, and success rate metrics for matchup analysis and prop betting edges.

## Overview

Matchup analysis compares **offensive advanced metrics** vs **defensive advanced metrics** to identify exploitable edges for prop betting.

### Key Question
*"Does Patrick Mahomes' elite EPA production (+0.15 per play) outweigh the Broncos' strong pass defense (-0.02 EPA allowed)?"*

**Answer**: Yes - Mahomes has a **+0.17 EPA edge** (95th percentile) → **STRONG VALUE on passing props**

---

## Matchup Types

### 1. QB vs Pass Defense

**Metrics Compared:**
- QB EPA vs Passing EPA Allowed
- CPOE vs CPOE Allowed (accuracy vs secondary quality)
- Success Rate vs Success Rate Allowed
- OL Pressure Rate Allowed vs DL Pressure Rate Generated

**Example:**
```python
from backend.api.matchup_analyzer import matchup_analyzer, OffensiveProfile, DefensiveProfile

# Josh Allen's offensive profile
josh_allen = OffensiveProfile(
    player_id='00-0036389',
    player_name='Josh Allen',
    position='QB',
    team='BUF',
    avg_epa=0.12,
    qb_epa=0.14,
    cpoe=2.8,
    success_rate=51.0,
    attempts=35,
    pressure_rate_faced=26.0,
    games_played=10
)

# Miami's defensive profile (weak pass defense example)
miami_defense = DefensiveProfile(
    team='MIA',
    passing_epa_allowed=0.06,  # Poor (allows high EPA)
    rushing_epa_allowed=-0.02,
    receiving_epa_allowed=0.05,
    total_epa_allowed=0.04,
    cpoe_allowed=1.8,  # Weak secondary
    success_rate_allowed=50.0,
    pressure_rate_generated=23.0,  # Weak pass rush
    games_played=10
)

# Generate matchup report
report = matchup_analyzer.generate_matchup_report(josh_allen, miami_defense)

print(report)
```

**Output:**
```json
{
  "player": {
    "name": "Josh Allen",
    "position": "QB",
    "team": "BUF"
  },
  "opponent_defense": "MIA",
  "overall_matchup_grade": "A",
  "overall_percentile": 82.5,
  "overall_confidence": 0.78,
  "overall_recommendation": "STRONG PLAY",
  "edges": [
    {
      "type": "passing_epa",
      "offensive_value": 0.14,
      "defensive_value": 0.06,
      "edge": 0.08,
      "percentile": 75,
      "confidence": 0.76,
      "recommendation": "VALUE: passing OVER props",
      "explanation": "Offensive EPA: +0.140 | Defensive EPA allowed: +0.060 | Edge: +0.080 EPA per play (75th percentile)"
    },
    {
      "type": "cpoe_accuracy",
      "offensive_value": 2.8,
      "defensive_value": 1.8,
      "edge": 1.0,
      "percentile": 55,
      "confidence": 0.68,
      "recommendation": "VALUE: QB accuracy advantage",
      "explanation": "QB CPOE: +2.8% | Defense CPOE allowed: +1.8% | Edge: +1.0% CPOE"
    },
    {
      "type": "success_rate",
      "offensive_value": 51.0,
      "defensive_value": 50.0,
      "edge": 1.0,
      "percentile": 52,
      "confidence": 0.62,
      "recommendation": "NEUTRAL: Even efficiency matchup",
      "explanation": "Offensive Success Rate: 51.0% | Defensive Success Rate Allowed: 50.0% | Edge: +1.0%"
    },
    {
      "type": "pressure_rate",
      "offensive_value": 26.0,
      "defensive_value": 23.0,
      "edge": -3.0,
      "percentile": 56,
      "confidence": 0.71,
      "recommendation": "VALUE: OL has advantage, clean pocket expected",
      "explanation": "OL Pressure Rate Allowed: 26.0% | DL Pressure Rate Generated: 23.0% | Edge: -3.0% (negative = OL advantage)"
    }
  ]
}
```

**Interpretation:**
- **Overall Grade A** (82.5th percentile)
- **EPA edge** (+0.08) favors Josh Allen
- **CPOE edge** (+1.0%) favors accuracy
- **Pressure edge** (-3.0%) favors Buffalo's OL
- **Recommendation**: STRONG PLAY on Josh Allen passing props

---

### 2. RB vs Run Defense

**Metrics Compared:**
- Rushing EPA vs Rushing EPA Allowed
- Success Rate vs Success Rate Allowed

**Example:**
```python
# Saquon Barkley's profile
saquon = OffensiveProfile(
    player_id='00-0035704',
    player_name='Saquon Barkley',
    position='RB',
    team='PHI',
    avg_epa=0.08,
    rushing_epa=0.09,
    success_rate=48.0,
    attempts=20,
    games_played=10
)

# Arizona's run defense (weak example)
arizona_defense = DefensiveProfile(
    team='ARI',
    passing_epa_allowed=0.03,
    rushing_epa_allowed=0.05,  # Poor run defense
    receiving_epa_allowed=0.02,
    total_epa_allowed=0.04,
    success_rate_allowed=49.0,  # Allows high success rate
    games_played=10
)

report = matchup_analyzer.generate_matchup_report(saquon, arizona_defense)
```

**Expected Result:**
- **Rushing EPA edge**: +0.09 - 0.05 = **+0.04** → VALUE
- **Success rate edge**: 48% - 49% = **-1%** → NEUTRAL
- **Overall**: **GOOD PLAY** for Saquon rushing props

---

### 3. WR vs Pass Defense

**Metrics Compared:**
- Receiving EPA vs Receiving EPA Allowed
- Air Yards per Attempt vs Air Yards Allowed

**Example:**
```python
# Tyreek Hill's profile
tyreek = OffensiveProfile(
    player_id='00-0033357',
    player_name='Tyreek Hill',
    position='WR',
    team='MIA',
    avg_epa=0.18,
    receiving_epa=0.20,
    air_yards_per_attempt=12.5,  # Deep threat
    success_rate=45.0,
    games_played=10
)

# New England's pass defense (strong vs deep)
patriots_defense = DefensiveProfile(
    team='NE',
    passing_epa_allowed=-0.02,  # Good pass defense
    rushing_epa_allowed=0.01,
    receiving_epa_allowed=-0.01,  # Limits receiving EPA
    total_epa_allowed=-0.01,
    air_yards_allowed=6.8,  # Limits deep passing
    games_played=10
)

report = matchup_analyzer.generate_matchup_report(tyreek, patriots_defense)
```

**Expected Result:**
- **Receiving EPA edge**: +0.20 - (-0.01) = **+0.21** → STRONG VALUE
- **Air yards edge**: 12.5 - 6.8 = **+5.7** → STRONG VALUE (deep threat advantage)
- **Overall**: **STRONG PLAY** for Tyreek Hill receiving props

---

## Edge Interpretation Guide

### EPA Edges

| Edge | Percentile | Interpretation | Recommendation |
|------|-----------|----------------|----------------|
| +0.20+ | 95+ | Elite mismatch | STRONG BUY |
| +0.10 to +0.20 | 75-95 | Significant advantage | BUY |
| +0.05 to +0.10 | 60-75 | Moderate advantage | VALUE |
| -0.05 to +0.05 | 45-60 | Even matchup | NEUTRAL |
| -0.10 to -0.05 | 35-45 | Moderate disadvantage | FADE |
| -0.20 to -0.10 | 15-35 | Significant disadvantage | STRONG FADE |
| -0.20- | <15 | Elite defense dominates | AVOID |

### CPOE Edges (Percentage Points)

| Edge | Interpretation | Recommendation |
|------|----------------|----------------|
| +3%+ | Elite accuracy vs weak secondary | STRONG BUY |
| +1% to +3% | Accuracy advantage | VALUE |
| -1% to +1% | Even matchup | NEUTRAL |
| -3% to -1% | Secondary has edge | FADE |
| -3%- | Elite secondary vs poor accuracy | STRONG FADE |

### Success Rate Edges

| Edge | Interpretation | Recommendation |
|------|----------------|----------------|
| +8%+ | Dominant efficiency edge | STRONG BUY |
| +3% to +8% | Efficiency advantage | VALUE |
| -3% to +3% | Even matchup | NEUTRAL |
| -8% to -3% | Defense has edge | FADE |
| -8%- | Defense dominates efficiency | STRONG FADE |

### Pressure Rate Edges

| Edge | Interpretation | Recommendation |
|------|----------------|----------------|
| -8%- | Elite OL vs weak pass rush | STRONG BUY (clean pocket) |
| -3% to -8% | OL advantage | VALUE |
| -3% to +3% | Even matchup | NEUTRAL |
| +3% to +8% | DL has edge | FADE (pressure expected) |
| +8%+ | Elite pass rush vs weak OL | STRONG FADE |

**Note**: Negative pressure edge = GOOD for offense (OL allows less pressure than DL generates)

---

## Integration with API

The matchup analyzer integrates with the insights API endpoint:

```bash
# Get matchup analysis for a specific game
GET /api/v1/games/2024_12_BUF_MIA/insights

# Response includes matchup edges
{
  "insights": [
    {
      "type": "matchup_epa",
      "player": "Josh Allen",
      "title": "Elite EPA Matchup vs MIA",
      "edge": 0.08,
      "percentile": 75,
      "recommendation": "VALUE: passing OVER props"
    }
  ]
}
```

---

## Building Defensive Profiles

Defensive profiles are calculated by aggregating opponents' performance:

```python
def build_defensive_profile(team: str, pbp_data: pd.DataFrame) -> DefensiveProfile:
    """Build defensive profile from play-by-play data.

    Args:
        team: Team abbreviation
        pbp_data: Play-by-play DataFrame with EPA metrics

    Returns:
        DefensiveProfile with aggregated defensive metrics
    """
    # Filter to plays against this defense
    defense_plays = pbp_data[
        (pbp_data['defteam'] == team) &
        (pbp_data['play_type'].isin(['pass', 'run']))
    ]

    # Calculate EPA allowed
    passing_epa_allowed = defense_plays[
        defense_plays['play_type'] == 'pass'
    ]['epa'].mean()

    rushing_epa_allowed = defense_plays[
        defense_plays['play_type'] == 'run'
    ]['epa'].mean()

    # CPOE allowed (how accurate QBs are vs this defense)
    cpoe_allowed = defense_plays[
        defense_plays['play_type'] == 'pass'
    ]['cpoe'].mean()

    # Success rate allowed
    success_rate_allowed = (
        defense_plays['success'].sum() / len(defense_plays) * 100
    )

    # Pressure rate generated
    pressure_plays = defense_plays[defense_plays['play_type'] == 'pass']
    pressure_rate = (
        (pressure_plays['qb_hit'].sum() +
         pressure_plays['qb_hurry'].sum() +
         pressure_plays['sack'].sum()) /
        len(pressure_plays) * 100
    )

    return DefensiveProfile(
        team=team,
        passing_epa_allowed=passing_epa_allowed,
        rushing_epa_allowed=rushing_epa_allowed,
        receiving_epa_allowed=passing_epa_allowed,  # Same as passing
        total_epa_allowed=defense_plays['epa'].mean(),
        cpoe_allowed=cpoe_allowed,
        success_rate_allowed=success_rate_allowed,
        pressure_rate_generated=pressure_rate,
        games_played=defense_plays['game_id'].nunique()
    )
```

---

## Practical Betting Workflow

### Step 1: Identify Games
```bash
# Get week 12 games
GET /api/v1/games?week=12&season=2024
```

### Step 2: Analyze Key Matchups
```python
# For each game, analyze QB vs defense
game = "2024_12_BUF_MIA"

# Load player and defensive profiles
josh_allen_profile = load_player_profile('00-0036389')
miami_defense_profile = load_defensive_profile('MIA')

# Generate matchup report
report = matchup_analyzer.generate_matchup_report(
    josh_allen_profile,
    miami_defense_profile
)
```

### Step 3: Filter for Strong Edges
```python
# Find all grade A matchups
strong_plays = [
    report for report in all_matchups
    if report['overall_matchup_grade'] in ['A', 'B']
    and report['overall_percentile'] >= 70
]
```

### Step 4: Combine with Other Factors
- **Weather**: High wind? Fade passing, boost rushing
- **Injuries**: Key WR out? Target RB receiving props
- **Vegas Lines**: Model predicts 280 yards, line is 245? OVER value
- **Trends**: Player on EPA uptrend? Extra confidence

### Step 5: Place Bets
- **Grade A matchups** (75+ percentile): 2-3 unit plays
- **Grade B matchups** (65-74 percentile): 1-2 unit plays
- **Grade C matchups** (55-64 percentile): 0.5-1 unit plays
- **Grade D/F**: Fade or avoid

---

## Advanced Use Cases

### 1. Stack Building (DFS)
Find correlated matchups:
```python
# QB + WR1 + WR2 stack with favorable matchups
qb_edges = analyze_qb_matchup(mahomes, broncos_defense)
wr1_edges = analyze_wr_matchup(kelce, broncos_defense)
wr2_edges = analyze_wr_matchup(rice, broncos_defense)

# All have EPA edges > 0.05? Strong stack
```

### 2. Contrarian Plays
Find undervalued players with elite matchups:
```python
# Low ownership but A-grade matchup
if player_ownership < 10% and matchup_grade == 'A':
    print(f"CONTRARIAN PLAY: {player_name}")
```

### 3. Live Betting Adjustments
```python
# If QB is getting pressured more than expected
if live_pressure_rate > expected_pressure_rate + 5%:
    print("FADE: QB props look worse than pre-game analysis")
```

---

## Resources

- **nflverse Play-by-Play**: https://github.com/nflverse/nflverse-data
- **EPA Documentation**: https://www.espn.com/nfl/story/_/id/8379024/explaining-expected-points-added
- **CPOE Explained**: https://www.nfl.com/news/next-gen-stats-intro-to-completion-probability
- **Success Rate**: Baldwin's success rate definition

---

## Quick Reference: Matchup Grades

| Grade | Percentile | Edge Magnitude | Action |
|-------|-----------|----------------|--------|
| A | 75-100 | Large advantage | STRONG PLAY (2-3 units) |
| B | 65-74 | Moderate advantage | GOOD PLAY (1-2 units) |
| C | 55-64 | Slight advantage | NEUTRAL (0.5-1 unit) |
| D | 45-54 | Slight disadvantage | FADE |
| F | 0-44 | Large disadvantage | STRONG FADE |

Remember: **Multiple edges in same direction = higher confidence**. If QB has EPA edge, CPOE edge, AND pressure edge, that's a much stronger play than just one metric.
