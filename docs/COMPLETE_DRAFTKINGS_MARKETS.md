# Complete DraftKings NFL Markets - Full Training Requirements

## Market Categories

This document maps ALL DraftKings NFL betting markets to our training data and modeling requirements.

---

## 1. GAME PROPS (Full Game)

### 1.1 Core Game Markets ✅ PARTIALLY SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Game Total** | home_score + away_score | Quantile Regression | HIGH | ✅ Trained |
| **Spread** | home_score - away_score | Quantile Regression | HIGH | ✅ Trained |
| **Moneyline** | home_won (binary) | Logistic Regression | HIGH | ✅ Trained |

### 1.2 Team Total Markets ❌ NOT SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Home Team Total** | home_score | Quantile Regression | HIGH | ❌ Need to train |
| **Away Team Total** | away_score | Quantile Regression | HIGH | ❌ Need to train |

### 1.3 Scoring Props ❌ NOT SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **First Score Type** | TD/FG/Safety | Multinomial | MEDIUM | ❌ Need play-by-play |
| **Last Score Type** | TD/FG/Safety | Multinomial | LOW | ❌ Need play-by-play |
| **First Team to Score** | Home/Away | Bernoulli | MEDIUM | ❌ Need play-by-play |
| **First Team to 10** | Home/Away/Neither | Multinomial | LOW | ❌ Need play-by-play |
| **Will there be OT?** | Binary (0/1) | Bernoulli | LOW | ❌ Need to extract |
| **Will there be a safety?** | Binary (0/1) | Bernoulli | LOW | ❌ Rare event |
| **Will there be a 2-pt conversion?** | Binary (0/1) | Bernoulli | LOW | ❌ Need play-by-play |
| **Winning Margin** | abs(home_score - away_score) | Categorical | LOW | ❌ Need to train |

### 1.4 Quarter/Period Props ❌ NOT SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Highest Scoring Quarter** | Q1/Q2/Q3/Q4 | Multinomial | LOW | ❌ Need quarter scores |
| **Highest Scoring Half** | 1H/2H | Bernoulli | LOW | ❌ Need half scores |
| **Will there be scoreless quarter?** | Binary | Bernoulli | LOW | ❌ Need quarter scores |
| **Race to X Points** | Which team first | Multinomial | LOW | ❌ Need play-by-play |

---

## 2. QUARTER & HALF DERIVATIVE MARKETS

### 2.1 First Quarter (1Q) Markets ❌ NOT SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **1Q Total** | Q1 total points | Quantile Regression | MEDIUM | ❌ Need Q1 scores |
| **1Q Spread** | Q1 point diff | Quantile Regression | MEDIUM | ❌ Need Q1 scores |
| **1Q Moneyline** | Q1 winner | Logistic Regression | MEDIUM | ❌ Need Q1 scores |
| **1Q Team Totals** | Each team Q1 points | Quantile Regression | LOW | ❌ Need Q1 scores |

### 2.2 First Half (1H) Markets ❌ NOT SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **1H Total** | H1 total points | Quantile Regression | HIGH | ❌ Need H1 scores |
| **1H Spread** | H1 point diff | Quantile Regression | HIGH | ❌ Need H1 scores |
| **1H Moneyline** | H1 winner | Logistic Regression | HIGH | ❌ Need H1 scores |
| **1H Team Totals** | Each team H1 points | Quantile Regression | MEDIUM | ❌ Need H1 scores |

### 2.3 Second Half (2H) Markets ❌ NOT SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **2H Total** | H2 total points | Quantile Regression | LOW | ❌ Need H2 scores |
| **2H Spread** | H2 point diff | Quantile Regression | LOW | ❌ Need H2 scores |

---

## 3. PLAYER PROPS (Full Game)

### 3.1 Passing Props
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Pass Yards** | passing_yards | Quantile Regression | HIGH | ✅ Trained (40-50%) |
| **Pass TDs** | passing_tds | Poisson | HIGH | ⏳ In progress |
| **Completions** | completions | Quantile Regression | HIGH | ⏳ In progress |
| **Pass Attempts** | attempts | Quantile Regression | MEDIUM | ⏳ In progress |
| **Interceptions** | interceptions | Bernoulli | MEDIUM | ⏳ In progress |
| **Pass + Rush Yards** | passing_yards + rushing_yards | Quantile | MEDIUM | ❌ Need combo model |
| **Longest Completion** | max(pass_length) | Extreme Value | LOW | ❌ Need play-by-play |
| **1Q Pass Yards** | Q1 passing_yards | Quantile | LOW | ❌ Need Q1 splits |
| **1H Pass Yards** | H1 passing_yards | Quantile | MEDIUM | ❌ Need H1 splits |

### 3.2 Rushing Props
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Rush Yards** | rushing_yards | Quantile Regression | HIGH | ✅ Trained (75% W11) |
| **Rush Attempts** | carries | Quantile Regression | MEDIUM | ⏳ In progress |
| **Rush TDs** | rushing_tds | Poisson | HIGH | ⏳ In progress |
| **Longest Rush** | max(rush_length) | Extreme Value | LOW | ❌ Need play-by-play |
| **1Q Rush Yards** | Q1 rushing_yards | Quantile | LOW | ❌ Need Q1 splits |
| **1H Rush Yards** | H1 rushing_yards | Quantile | MEDIUM | ❌ Need H1 splits |

### 3.3 Receiving Props
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Receiving Yards** | receiving_yards | Quantile Regression | HIGH | ✅ Trained (80% W10) |
| **Receptions** | receptions | Quantile Regression | HIGH | ⏳ In progress |
| **Targets** | targets | Quantile Regression | MEDIUM | ⏳ In progress |
| **Rec TDs** | receiving_tds | Poisson | HIGH | ⏳ In progress |
| **Rec + Rush Yards** | receiving_yards + rushing_yards | Quantile | MEDIUM | ❌ Need combo model |
| **Longest Reception** | max(reception_length) | Extreme Value | LOW | ❌ Need play-by-play |
| **1Q Rec Yards** | Q1 receiving_yards | Quantile | LOW | ❌ Need Q1 splits |
| **1H Rec Yards** | H1 receiving_yards | Quantile | MEDIUM | ❌ Need H1 splits |

### 3.4 Touchdown Scorer Props
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Anytime TD** | rush_tds + rec_tds > 0 | Bernoulli | HIGH | ⏳ In progress |
| **First TD Scorer** | first_td indicator | Multinomial | MEDIUM | ❌ Need play-by-play |
| **Last TD Scorer** | last_td indicator | Multinomial | LOW | ❌ Need play-by-play |
| **2+ TDs** | total_tds >= 2 | Bernoulli | MEDIUM | ❌ Need to train |
| **3+ TDs** | total_tds >= 3 | Bernoulli | LOW | ❌ Need to train |
| **1Q Anytime TD** | Q1 TD scorer | Bernoulli | LOW | ❌ Need Q1 play-by-play |
| **1H Anytime TD** | H1 TD scorer | Bernoulli | MEDIUM | ❌ Need H1 play-by-play |

### 3.5 Kicker Props
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **FG Made** | fg_made | Poisson | MEDIUM | ❌ Need kicker data |
| **XP Made** | xp_made | Poisson | LOW | ❌ Need kicker data |
| **Total Kicker Points** | (3 * fg_made) + xp_made | Poisson | MEDIUM | ❌ Need kicker data |
| **Longest FG** | max(fg_distance) | Extreme Value | LOW | ❌ Need play-by-play |
| **1H FG Made** | H1 fg_made | Poisson | LOW | ❌ Need H1 kicker splits |

### 3.6 Defense Props
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Sacks** | sacks (defensive player) | Poisson | MEDIUM | ❌ Need defensive stats |
| **Tackles** | total_tackles | Poisson | LOW | ❌ Need defensive stats |
| **Interceptions** | ints_caught (defense) | Bernoulli | MEDIUM | ❌ Need defensive stats |
| **Defensive TDs** | def_tds | Bernoulli | LOW | ❌ Need defensive stats |

---

## 4. TEAM PROPS

### 4.1 Team Scoring Props ❌ NOT SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Team Total Points** | team_score | Quantile Regression | HIGH | ❌ Same as team total |
| **Team 1Q Points** | Q1_team_score | Quantile | MEDIUM | ❌ Need Q1 scores |
| **Team 1H Points** | H1_team_score | Quantile | HIGH | ❌ Need H1 scores |
| **Team Total TDs** | total_tds (team) | Poisson | MEDIUM | ❌ Need aggregation |
| **Team Total FGs** | total_fgs (team) | Poisson | LOW | ❌ Need aggregation |

### 4.2 Team Performance Props ❌ NOT SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Team Longest TD** | max(td_distance) | Extreme Value | LOW | ❌ Need play-by-play |
| **Team First TD Type** | rush/pass/return | Multinomial | LOW | ❌ Need play-by-play |
| **Team to Score in All Quarters** | Binary | Bernoulli | LOW | ❌ Need Q scores |
| **Team Largest Lead** | max(score_diff) | Quantile | LOW | ❌ Need in-game data |

---

## 5. SPECIAL/EXOTIC PROPS

### 5.1 Exact Score Props ❌ NOT SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Exact Final Score** | home_score, away_score | Multinomial | LOW | ❌ 2500+ outcomes |
| **Score Bands** | Ranges (0-13, 14-20, etc.) | Categorical | LOW | ❌ Need to train |

### 5.2 Drive Props (Live Only) ❌ NOT SUPPORTED
| Market | Data Needed | Model Type | Priority | Status |
|--------|-------------|------------|----------|--------|
| **Next Drive Result** | TD/FG/Punt/Turnover | Multinomial | N/A | ❌ Live betting only |
| **Will next drive be a 3-and-out?** | Binary | Bernoulli | N/A | ❌ Live betting only |

---

## DATA REQUIREMENTS

### Critical Missing Data Sources

#### 1. Quarter & Half Scores ⚠️ CRITICAL
**Why:** Required for all 1Q, 1H, 2H derivative markets
**Source:** NFLverse `pbp` data has quarter-by-quarter scoring
**Implementation:**
```python
# Extract from play-by-play
q1_scores = pbp[(pbp['qtr'] == 1) & (pbp['play_type'].isin(['field_goal', 'extra_point', 'touchdown']))].groupby('posteam')['points'].sum()
h1_scores = pbp[(pbp['qtr'].isin([1,2]))...].groupby('posteam')['points'].sum()
```

**Markets Unlocked:** 1Q Total, 1Q Spread, 1H Total, 1H Spread (~10 high-value markets)

#### 2. Kicker Stats ⚠️ CRITICAL
**Why:** Required for all kicker props (popular market)
**Source:** NFLverse `pbp` data - filter `play_type == 'field_goal'` and `play_type == 'extra_point'`
**Implementation:**
```python
kicker_stats = pbp[pbp['play_type'] == 'field_goal'].groupby(['kicker_player_id', 'game_id']).agg({
    'field_goal_result': lambda x: (x == 'made').sum(),  # FGs made
    'kick_distance': 'max'  # Longest FG
})
```

**Markets Unlocked:** FG Made, XP Made, Total Kicker Points, Longest FG (~4 markets)

#### 3. Play-by-Play Data ⚠️ HIGH PRIORITY
**Why:** Required for first/last score, drive props, longest plays
**Source:** NFLverse `pbp` CSV
**Implementation:** Full play-level data ingestion
**Markets Unlocked:** ~15-20 additional markets

#### 4. Defensive Player Stats 🔵 MEDIUM PRIORITY
**Why:** Required for defense props (sacks, tackles, INTs)
**Source:** NFLverse defensive stats or `pbp` aggregation
**Markets Unlocked:** ~5 defensive markets

---

## IMPLEMENTATION PRIORITY

### Phase 1: HIGH ROI, LOW EFFORT ✅ IN PROGRESS
**Goal:** Get the most popular player props working with injury filtering

**Tasks:**
- [x] Yards props (pass, rush, rec) - DONE
- [⏳] TD props (pass_tds, rush_tds, rec_tds) - IN PROGRESS
- [⏳] Volume props (completions, receptions) - IN PROGRESS
- [⏳] Anytime TD - IN PROGRESS
- [⏳] Injury filtering integration - IN PROGRESS

**Expected Impact:** 60-70% of betting volume, 15-25% ROI

---

### Phase 2: TEAM TOTALS & 1H MARKETS ⚠️ NEXT PRIORITY
**Goal:** Add high-volume derivative markets

**Tasks:**
- [ ] Extract quarter scores from games data
- [ ] Train 1H Total model
- [ ] Train 1H Spread model
- [ ] Train Team Total models (home/away)
- [ ] Backtest on Weeks 10-11

**Data Needed:**
```python
# From games.csv or pbp data
- home_score_q1, away_score_q1
- home_score_q2, away_score_q2
- home_score_h1, away_score_h1 (Q1+Q2)
```

**Expected Impact:** 20-30% additional betting volume

---

### Phase 3: KICKER PROPS 🔵 MEDIUM PRIORITY
**Goal:** Add kicker market coverage

**Tasks:**
- [ ] Extract kicker stats from play-by-play
- [ ] Train FG Made model (Poisson)
- [ ] Train XP Made model (Poisson)
- [ ] Train Total Kicker Points model
- [ ] Backtest

**Expected Impact:** 5-10% additional volume

---

### Phase 4: EXOTIC PROPS 🔵 LOW PRIORITY
**Goal:** Round out coverage for completeness

**Tasks:**
- [ ] First/Last TD Scorer (need pbp)
- [ ] First Score Type (need pbp)
- [ ] Defensive props (need defensive stats)
- [ ] Longest plays (need pbp)

**Expected Impact:** <5% volume, specialist bets

---

## CURRENT STATUS SUMMARY

### ✅ FULLY SUPPORTED (3 markets)
- Pass Yards (40-50% hit rate)
- Rush Yards (75% hit rate Week 11)
- Rec Yards (80% hit rate Week 10)

### ⏳ IN PROGRESS (9 markets)
- Pass TDs, Rush TDs, Rec TDs
- Completions, Receptions, Targets, Attempts, Carries
- Anytime TD

### ❌ NOT STARTED - HIGH PRIORITY (6 markets)
- 1H Total, 1H Spread
- Team Totals (home/away)
- FG Made, XP Made

### ❌ NOT STARTED - MEDIUM PRIORITY (10-15 markets)
- 1Q markets
- 2+ TDs, 3+ TDs
- First/Last TD
- Defensive props

### ❌ NOT STARTED - LOW PRIORITY (20+ markets)
- 2H markets
- Exotic scoring props
- Exact score/bands
- Drive props (live only)

---

## RECOMMENDED NEXT STEPS

1. **FINISH Phase 1** - Complete training all in-progress player props
2. **ADD injury filtering** - Integrate before any predictions
3. **EXTRACT quarter scores** - Critical for Phase 2
4. **TRAIN 1H markets** - Highest incremental ROI
5. **ADD kicker stats** - Popular market, easy wins
6. **BACKTEST comprehensively** - Validate all models on Weeks 10-12

---

## SUCCESS METRICS BY PHASE

**Phase 1 Complete:**
- 12 prop types trained
- >55% average hit rate
- >10% average ROI
- 0 bets on inactive players (injury filtering working)

**Phase 2 Complete:**
- 18 prop types trained (add 1H, team totals)
- >58% average hit rate
- >15% average ROI
- 70%+ market coverage

**Phase 3 Complete:**
- 22 prop types trained (add kicker props)
- >60% average hit rate
- >18% average ROI
- 85%+ market coverage

**Production Ready:**
- 30+ prop types
- >60% hit rate across all types
- >20% ROI
- Complete DraftKings coverage
- Calibrated probabilities (Brier < 0.20)
