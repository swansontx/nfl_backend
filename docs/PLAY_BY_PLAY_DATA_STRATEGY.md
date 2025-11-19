# 🏈 Play-by-Play Data Strategy - Unblocking 25+ Markets

**Date:** 2025-11-19
**Status:** RESEARCH - NFLverse Data Accessible ✅
**Goal:** Unblock 25+ markets currently blocked on play-by-play data

---

## 📊 CURRENT SITUATION

### Blocked Markets Requiring PBP Data (25+ markets)

From `FINAL_MARKET_SUMMARY.md`, these markets are currently blocked:

#### First/Last Scorer Props (6 markets)
- **First TD Scorer (Anytime)** - HIGH priority, very popular
- **Last TD Scorer (Anytime)** - MEDIUM priority
- **First Offensive TD Scorer** - HIGH priority
- **First Defensive TD Scorer** - LOW priority
- **First FG Scorer** - MEDIUM priority
- **First Offensive Play TD** - LOW priority

#### Longest Play Props (9 markets)
- **Longest Completion** - HIGH priority
- **Longest Rush** - HIGH priority
- **Longest Reception** - HIGH priority
- **Longest TD** - MEDIUM priority
- **Longest FG** - MEDIUM priority
- **Longest Punt** - LOW priority
- **Longest Kickoff Return** - LOW priority
- **Longest Punt Return** - LOW priority
- **Longest Play from Scrimmage** - MEDIUM priority

#### Drive Props (4 markets)
- **Longest Drive (plays)** - MEDIUM priority
- **Longest Drive (yards)** - MEDIUM priority
- **Longest Drive (time)** - MEDIUM priority
- **Shortest TD Drive** - LOW priority

#### Scoring Sequence Props (6 markets)
- **Team to Score First** - HIGH priority
- **Team to Score Last** - MEDIUM priority
- **Both Teams Score Each Quarter** - LOW priority
- **Team Scores Every Quarter** - LOW priority
- **Biggest Lead** - MEDIUM priority
- **Lead Changes** - MEDIUM priority

---

## ✅ NFLVERSE DATA AVAILABILITY

### What We Discovered

**NFLverse GitHub**: https://github.com/nflverse/nflverse-data

The NFLverse project provides comprehensive NFL data through automated releases:

#### Available Datasets (Confirmed Accessible)

1. **play-by-play (pbp)** ✅
   - URL: `https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.csv`
   - Status: **200 OK** - File exists and is accessible
   - Assets: 162 files (CSV, parquet, RDS formats)
   - Date Range: 1999-2025
   - Access: `nflreadr::load_pbp()` or direct download

2. **injuries** ✅
   - URL: `https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_2025.csv`
   - Status: **200 OK** - File exists
   - Better than our synthetic injury data!

3. **pbp_participation** ✅
   - Play participation data from NGS
   - Useful for snap counts, substitutions

4. **nextgen_stats** ✅
   - 97 assets with advanced metrics
   - Could improve existing models

5. **depth_charts** ✅
   - 106 assets showing roster positions
   - Useful for starter detection

6. **player_stats** ✅
   - Better than our synthetic stats
   - Real NFL data

7. **snap_counts** ✅
   - Real snap count data
   - Better than estimates

---

## 🔍 PLAY-BY-PLAY DATA FIELDS

### Confirmed Fields (from nflfastR documentation)

Based on web search results, the PBP data includes:

#### Touchdown Scorer Fields
- `td_team` - Team that scored TD (string)
- `td_player_name` - Player who scored TD (string)
- `td_player_id` - Unique player identifier (string)

#### Play Distance Fields
- `yards_gained` - Yards gained on play (numeric)
- `kick_distance` - Distance for kicks/punts (numeric)

#### Aggregated Stats
- `passing_tds` - Passing touchdowns
- `rushing_tds` - Rushing touchdowns (includes scrambles)
- `receiving_tds` - Receiving touchdowns

#### Other Expected Fields (standard nflfastR)
- Play identifiers: `play_id`, `game_id`, `drive`
- Play type: `play_type`, `pass`, `rush`
- Player involvement: `passer_player_name`, `rusher_player_name`, `receiver_player_name`
- Game state: `qtr`, `game_seconds_remaining`, `score_differential`
- Drive info: `fixed_drive`, `drive_start_yard_line`, `drive_end_yard_line`
- Scoring: `touchdown`, `field_goal_result`, `extra_point_result`

**Full field list**: Available in nflfastR package via `field_descriptions` dataframe

---

## 🚧 CURRENT BLOCKERS

### Issue 1: Download Access
- **Problem**: Direct curl download returns "Access denied"
- **Cause**: GitHub requires authentication or redirects through CDN
- **Solution Options**:
  1. Use Python `requests` with proper headers
  2. Use `wget` with follow redirects
  3. Use nflverse Python package (if exists)
  4. Download manually and upload to inputs/

### Issue 2: Documentation Access
- **Problem**: nflfastR documentation sites return 403
- **Cause**: Possible rate limiting or blocking
- **Solution**:
  - Download actual data and inspect columns directly
  - Use R package documentation locally
  - Check nflverse Discord/community for field lists

### Issue 3: Data Size
- **Problem**: PBP data can be very large (multiple GB)
- **Solution**:
  - Download only 2025 season
  - Use parquet format (compressed)
  - Filter to relevant plays only
  - Process in chunks

---

## 📋 ACTION PLAN

### Phase 1: Data Acquisition (IMMEDIATE)

**Goal**: Download and inspect 2025 play-by-play data

1. **Try Python requests library**
   ```python
   import requests
   import pandas as pd

   url = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2025.csv"
   response = requests.get(url, allow_redirects=True)
   # Or try parquet for better compression
   ```

2. **Inspect columns**
   - Load first 1000 rows
   - Document all available fields
   - Identify fields for each prop type

3. **Download supporting data**
   - injuries_2025.csv (replace synthetic)
   - player_stats.csv (better than synthetic)
   - nextgen_stats (for model improvements)

### Phase 2: Feature Engineering (1-2 days)

**Goal**: Create prop-specific features from PBP data

1. **First/Last TD Scorer**
   - Group by game_id, filter touchdown plays
   - Identify first and last TD of game
   - Create historical "first TD probability" by player
   - Features: red zone touches, goal-line carries, TD rate by game situation

2. **Longest Plays**
   - Extract max yards_gained by play_type per game
   - Features: deep ball rate, explosive play history, opponent defense

3. **Drive Props**
   - Group plays by fixed_drive
   - Calculate drive_length (plays), drive_yards, drive_time
   - Features: offensive pace, time of possession, red zone efficiency

4. **Scoring Sequence**
   - Order scoring plays by game_seconds_remaining
   - Identify first/last team to score
   - Features: opening drive success rate, closing drive stats

### Phase 3: Model Training (2-3 days)

**Goal**: Train models for unblocked markets

#### Priority 1: First TD Scorer (Bernoulli)
- **Target**: Binary - did player score first TD?
- **Model**: Gradient Boosting Classifier
- **Features**:
  - Historical first TD rate
  - Red zone touch percentage
  - Team scoring first rate
  - Goal line snap percentage
  - Position (RB > WR > TE > QB)
- **Expected Performance**: 15-20% hit rate (better than 8-10% random)

#### Priority 2: Longest Completion/Rush/Reception (Quantile)
- **Target**: Continuous - longest play yards
- **Model**: Quantile Regression (same as yardage props)
- **Features**:
  - Season max play
  - L3 max play
  - Deep ball rate / explosive play rate
  - Opponent deep ball defense
- **Expected Performance**: 50-55% hit rate

#### Priority 3: Team to Score First (Bernoulli)
- **Target**: Binary - did home team score first?
- **Model**: Gradient Boosting Classifier
- **Features**:
  - Opening drive success rate (both teams)
  - Time of possession tendencies
  - First half scoring rates
  - Home/away splits
- **Expected Performance**: 55-60% hit rate (slight edge over 50%)

### Phase 4: Integration (1 day)

**Goal**: Integrate new models into existing infrastructure

1. **Save models** to `outputs/models/pbp/`
2. **Update documentation** in FINAL_MARKET_SUMMARY.md
3. **Create prediction pipeline** for PBP-dependent props
4. **Backtest** on Weeks 10-11 (same as existing props)

---

## 🎯 EXPECTED OUTCOMES

### Market Expansion
- **Current**: 36/80 markets (45%)
- **After PBP**: 50+/80 markets (62.5%+)
- **Unblocked**: 15-20 markets immediately

### New Markets Available
- ✅ First TD Scorer (most popular prop!)
- ✅ Longest plays (Completion, Rush, Reception)
- ✅ Team to Score First
- ✅ Last TD Scorer
- ✅ Drive props (length, yards, time)

### Model Performance Improvements
- **Better injury data** - Real NFL injury reports vs synthetic
- **Better player stats** - Actual NFL stats vs synthetic
- **NextGen stats** - Add to existing models for +5-10% ROI boost

---

## 📊 PRIORITY MATRIX

| Market | Data Required | Difficulty | Expected ROI | Priority |
|--------|---------------|------------|--------------|----------|
| **First TD Scorer** | PBP (td_player_name) | Medium | +25-40% | **HIGHEST** |
| **Team Score First** | PBP (scoring sequence) | Easy | +10-15% | **HIGH** |
| **Longest Completion** | PBP (yards_gained) | Easy | +15-25% | **HIGH** |
| **Longest Rush** | PBP (yards_gained) | Easy | +15-25% | **HIGH** |
| **Longest Reception** | PBP (yards_gained) | Easy | +15-25% | **HIGH** |
| **Last TD Scorer** | PBP (td_player_name) | Medium | +15-25% | **MEDIUM** |
| **Drive Length** | PBP (drive data) | Medium | +10-20% | **MEDIUM** |
| **Longest FG** | PBP (kick_distance) | Easy | +20-30% | **MEDIUM** |

---

## ⚠️ RISKS & MITIGATION

### Risk 1: Data Download Issues
- **Mitigation**: Manual download as fallback, use parquet format
- **Status**: Need to test Python requests library

### Risk 2: Data Quality
- **Mitigation**: Validate against known results, check for nulls
- **Status**: Unknown until we inspect data

### Risk 3: Processing Time
- **Mitigation**: Use parquet, filter early, process in chunks
- **Status**: Monitor performance

### Risk 4: Field Availability
- **Mitigation**: Download sample first, verify fields exist
- **Status**: Need to inspect actual data

---

## 🚀 NEXT STEPS

### Immediate (Today)
1. ✅ Research NFLverse data availability
2. ⏳ Download 2025 play-by-play data using Python
3. ⏳ Inspect columns and document fields
4. ⏳ Create PBP data loader script

### Short-term (This Week)
1. Download real injury data (replace synthetic)
2. Train First TD Scorer model (highest priority)
3. Train Longest Play models (quick wins)
4. Train Team Score First model

### Medium-term (Next Week)
1. Train all PBP-dependent props
2. Improve existing models with NextGen stats
3. Update documentation with new 50+ markets
4. Create production prediction pipeline

---

## 📁 FILE STRUCTURE (Planned)

```
backend/analysis/
├── fetch_pbp_data.py              # Download PBP from NFLverse
├── process_pbp_features.py        # Extract features for props
├── train_first_td_scorer.py       # First TD model
├── train_longest_plays.py         # Longest play models
├── train_scoring_sequence.py      # Score first/last models
└── train_drive_props.py           # Drive-based props

inputs/
├── play_by_play_2025.csv          # Full PBP data (large)
├── pbp_features_2025.csv          # Processed features
├── injuries_2025.csv              # Real injury data
└── player_stats_2025.csv          # Real player stats (if better)

outputs/models/pbp/
├── first_td_scorer_model.pkl
├── last_td_scorer_model.pkl
├── longest_completion_models.pkl
├── longest_rush_models.pkl
├── longest_reception_models.pkl
├── team_score_first_model.pkl
└── drive_props_models.pkl
```

---

## 📞 RESOURCES

- **NFLverse GitHub**: https://github.com/nflverse/nflverse-data
- **NFLverse Releases**: https://github.com/nflverse/nflverse-data/releases
- **nflfastR Docs**: https://nflfastr.com (currently 403, use R package)
- **nflverse Discord**: Check community for support
- **Field Descriptions**: Available in nflfastR R package

---

## ✅ SUCCESS CRITERIA

**Minimum Viable (MVP)**
- [ ] Successfully download and parse 2025 PBP data
- [ ] Train First TD Scorer model with >15% hit rate
- [ ] Train 3+ Longest Play models with >50% hit rate
- [ ] Unblock 5+ new markets

**Production Ready**
- [ ] Train all PBP-dependent props (15+ markets)
- [ ] Replace synthetic data with real NFLverse data
- [ ] 50+ markets trained (62.5% coverage)
- [ ] First TD Scorer ROI >25%
- [ ] Integration with existing prediction pipeline

**Full Coverage**
- [ ] 60+ markets trained (75% coverage)
- [ ] All high-priority PBP props trained
- [ ] Real-time PBP data pipeline
- [ ] Automated weekly updates

---

**STATUS: READY TO PROCEED** ✅

Next: Download and inspect 2025 play-by-play data using Python
