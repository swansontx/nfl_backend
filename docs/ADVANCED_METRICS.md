# Advanced Metrics Available from nflverse

This document outlines the advanced analytics and metrics available in the nflverse data.

## Play-by-Play Advanced Metrics

The play-by-play CSV includes **300+ columns** with sophisticated analytics. Here are the key advanced metrics:

### Expected Points Added (EPA)

EPA measures the value of each play in terms of expected points scored.

**Key Columns:**
- `epa` - Overall EPA for the play
- `qb_epa` - EPA attributed to QB (passing plays)
- `total_home_epa` - Cumulative home team EPA
- `total_away_epa` - Cumulative away team EPA
- `rushing_epa` - EPA from rushing plays
- `passing_epa` - EPA from passing plays
- `air_epa` - EPA from air yards (pre-catch)
- `yac_epa` - EPA from yards after catch
- `comp_air_epa` - Completed pass air EPA
- `comp_yac_epa` - Completed pass YAC EPA

**Usage:** EPA > 0 indicates a successful play that increased expected points.

### Win Probability Added (WPA)

WPA measures how much each play changes the team's probability of winning.

**Key Columns:**
- `wpa` - Win Probability Added for the play
- `wp` - Current win probability
- `home_wp` - Home team win probability
- `away_wp` - Away team win probability
- `vegas_wpa` - WPA adjusted for Vegas line
- `vegas_home_wpa` - Home WPA with Vegas adjustment

**Usage:** High WPA plays are "clutch" plays that significantly shift game outcome.

### Completion Percentage Over Expected (CPOE)

CPOE measures QB accuracy relative to expected completion percentage based on:
- Pass depth
- Receiver separation
- Down and distance
- Defensive coverage

**Key Columns:**
- `cpoe` - Completion Percentage Over Expected

**Usage:** Positive CPOE indicates QB is outperforming expected accuracy.

### Air Yards & YAC (Yards After Catch)

Separates passing yards into pre-catch (air) and post-catch (YAC) components.

**Key Columns:**
- `air_yards` - Yards ball traveled in air
- `yards_after_catch` - Yards gained after reception
- `complete_pass` - Whether pass was completed
- `incomplete_pass` - Whether pass was incomplete

**Expected YAC (xYAC):**
- `xyac_epa` - Expected YAC EPA
- `xyac_success` - Whether YAC exceeded expected
- `xyac_fd` - Expected first down probability
- `xyac_mean_yardage` - Mean expected YAC
- `xyac_median_yardage` - Median expected YAC

**Usage:** Compare actual YAC to expected YAC to evaluate receiver/RB performance.

### Success Rate

Binary metric: did the play succeed based on down and distance?

**Key Columns:**
- `success` - 1 if EPA > 0, 0 otherwise
- `series_success` - Success rate for the drive

**Typical Success Criteria:**
- 1st down: Gain 40%+ of yards to go
- 2nd down: Gain 60%+ of yards to go
- 3rd/4th down: Convert first down

**Usage:** Success rate is more stable than EPA for evaluating consistency.

### QB Pressure & Pass Rush

Tracks defensive pressure on the quarterback.

**Key Columns:**
- `qb_hit` - QB was hit on the play
- `qb_hurry` - QB was hurried
- `qb_knockdown` - QB was knocked down
- `sack` - Play resulted in sack

**Usage:** Pressure metrics help evaluate offensive line and pass rush effectiveness.

### Personnel & Formation

Tracks offensive and defensive personnel groupings.

**Key Columns:**
- `offense_personnel` - Offensive personnel (e.g., "1 RB, 1 TE, 3 WR")
- `defense_personnel` - Defensive personnel
- `defenders_in_box` - Number of defenders in the box (run defense)
- `number_of_pass_rushers` - Pass rushers on the play

**Usage:** Analyze performance by formation/personnel grouping.

### Target & Route Data

Identifies pass targets and routes run.

**Key Columns:**
- `receiver_player_id` - Player who was targeted
- `receiver_player_name` - Receiver name
- `pass_length` - Short/deep pass categorization
- `pass_location` - Left/middle/right
- `route_run` - Type of route run (if available)

**Usage:** Evaluate receiver targets, QB distribution, and route effectiveness.

### Game Context

Situational factors that impact play outcomes.

**Key Columns:**
- `down` - Down (1-4)
- `ydstogo` - Yards to go for first down
- `yardline_100` - Yards from opponent's end zone
- `score_differential` - Point difference
- `game_seconds_remaining` - Time left in game
- `half_seconds_remaining` - Time left in half
- `drive` - Drive number
- `qtr` - Quarter (1-5, 5=OT)

## Next Gen Stats (Optional Dataset)

**Player Tracking Metrics:**
- Average speed (mph)
- Max speed (mph)
- Distance traveled (yards)
- Time to throw (QB)
- Cushion at snap (WR/CB)
- Separation at catch (WR)
- Expected completion percentage
- Aggressiveness (throwing into tight windows)

**Available for:**
- Passing (QB aggressiveness, time to throw)
- Rushing (yards before contact, rush probability)
- Receiving (separation, cushion, catch probability)

**Usage:**
```bash
# Download Next Gen Stats
python -m backend.ingestion.fetch_nflverse --year 2024 --nextgen
```

## Snap Counts (Optional Dataset)

**Participation Metrics:**
- Total snaps played
- Offensive snaps
- Defensive snaps
- Special teams snaps
- Snap percentage (vs. team total)

**Usage:**
```bash
# Download snap counts
python -m backend.ingestion.fetch_nflverse --year 2024 --snap-counts
```

## Player Stats Dataset

**Weekly Aggregated Stats:**
- Passing: attempts, completions, yards, TDs, INTs, sacks
- Rushing: attempts, yards, TDs, fumbles
- Receiving: targets, receptions, yards, TDs
- Fantasy: fantasy points (various formats)

## Feature Engineering with Advanced Metrics

### Recommended EPA-Based Features

1. **Player EPA Trends**
   - Rolling 3-game EPA average
   - Season EPA percentile
   - EPA vs. defensive strength

2. **Success Rate Features**
   - Success rate by down
   - Success rate in red zone
   - Success rate vs. expected (context-adjusted)

3. **Efficiency Metrics**
   - EPA per play
   - EPA per dropback
   - Success rate on early downs

4. **Matchup Features**
   - QB CPOE vs. opponent secondary
   - WR separation vs. opponent CBs
   - RB YAC vs. opponent front 7

### Example: Building EPA-Based Player Features

```python
import pandas as pd

# Load play-by-play
pbp = pd.read_csv('inputs/play_by_play_2024.csv')

# Filter to passing plays for a specific QB
qb_plays = pbp[
    (pbp['passer_player_id'] == '00-0036212') &  # Patrick Mahomes
    (pbp['play_type'] == 'pass')
]

# Calculate rolling EPA
qb_plays['rolling_epa_3g'] = qb_plays.groupby('week')['epa'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)

# Calculate CPOE average
avg_cpoe = qb_plays['cpoe'].mean()

# Calculate success rate
success_rate = qb_plays['success'].mean()

print(f"QB EPA/play: {qb_plays['epa'].mean():.3f}")
print(f"QB CPOE: {avg_cpoe:.3f}")
print(f"Success rate: {success_rate:.1%}")
```

## Model Training with Advanced Metrics

### Recommended Features for Prop Predictions

**QB Passing Props:**
- Rolling EPA/play (3, 5, 8 games)
- CPOE trend
- Air yards per attempt
- Time to throw (Next Gen)
- Success rate vs. defensive EPA allowed
- Pressure rate allowed by OL

**RB Rushing Props:**
- Yards before contact
- YAC over expected
- Success rate
- Rush EPA
- Defenders in box faced
- Snap count percentage

**WR Receiving Props:**
- Separation at catch (Next Gen)
- Air yards share
- Target share
- YAC over expected
- Route diversity
- QB CPOE when targeting player

## Resources

- **nflverse Documentation**: https://nflreadr.nflverse.com/
- **EPA Explained**: https://www.espn.com/nfl/story/_/id/8379024/nfl-explaining-expected-points-added-nfl-statistical-analysis
- **Next Gen Stats**: https://nextgenstats.nfl.com/
- **Play-by-Play Column Reference**: https://nflreadr.nflverse.com/articles/dictionary_pbp.html

## Quick Reference: Key Metrics

| Metric | Column | Range | Interpretation |
|--------|--------|-------|----------------|
| EPA | `epa` | -7 to +7 typically | Value added in expected points |
| CPOE | `cpoe` | -50% to +50% | QB accuracy vs. expected |
| Success | `success` | 0 or 1 | Binary success indicator |
| WPA | `wpa` | -50% to +50% | Win probability shift |
| Air Yards | `air_yards` | -10 to +70 | Yards ball traveled |
| YAC | `yards_after_catch` | 0 to +90 | Post-catch yards |

## Next Steps

1. **Extract EPA features** in `extract_player_pbp_features.py`
2. **Add CPOE, success rate** to feature engineering
3. **Integrate Next Gen Stats** for tracking metrics
4. **Build EPA-based models** for more accurate predictions
5. **Backtest EPA predictions** vs. basic yardage models
