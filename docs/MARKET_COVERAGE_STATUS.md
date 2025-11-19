# NFL Betting Market Coverage Status

**Last Updated:** 2025-11-19

**Overall Progress:** 27/80+ markets (33.75% complete)

---

## ✅ FULLY TRAINED MARKETS (27 total)

### Player Props - Yards (3 markets)
| Market | Model Type | Hit Rate | ROI | Status |
|--------|------------|----------|-----|--------|
| **Pass Yards** | Quantile Regression | 52.5% | -0.2% | ✅ TRAINED |
| **Rush Yards** | Quantile Regression | 50.0% | -5.0% | ✅ TRAINED |
| **Rec Yards** | Quantile Regression | 57.0% | +8.4% | ✅ TRAINED |

### Player Props - TDs (3 markets)
| Market | Model Type | Hit Rate | ROI | Status |
|--------|------------|----------|-----|--------|
| **Pass TDs** | Poisson | 62.6% | +19.0% | ⭐ TRAINED |
| **Rush TDs** | Poisson | 75.8% | +44.1% | ⭐⭐ TRAINED |
| **Rec TDs** | Poisson | 0% (no bets) | 0% | ✅ TRAINED |

### Player Props - Volume (6 markets)
| Market | Model Type | Hit Rate | ROI | Status |
|--------|------------|----------|-----|--------|
| **Completions** | Quantile Regression | 36.7% | -30.2% | ⚠️ NEEDS IMPROVEMENT |
| **Pass Attempts** | Quantile Regression | 39.7% | -24.5% | ⚠️ NEEDS IMPROVEMENT |
| **Receptions** | Quantile Regression | 45.1% | -14.4% | ⚠️ NEEDS IMPROVEMENT |
| **Targets** | Quantile Regression | 44.3% | -15.9% | ⚠️ NEEDS IMPROVEMENT |
| **Carries** | Quantile Regression | 0% (no bets) | -100% | ⚠️ NEEDS IMPROVEMENT |
| **Interceptions** | Bernoulli | 0% (no bets) | 0% | ✅ TRAINED |

### Game Derivative Markets (10 markets)
| Market | Model Type | MAE | Priority | Status |
|--------|------------|-----|----------|--------|
| **1H Total** | Quantile Regression | 6.61 | HIGH | ✅ TRAINED |
| **1H Spread** | Quantile Regression | 5.50 | HIGH | ✅ TRAINED |
| **1H Moneyline** | Classification | 0.45 | HIGH | ✅ TRAINED |
| **Home Team Total** | Quantile Regression | 9.16 | HIGH | ✅ TRAINED |
| **Away Team Total** | Quantile Regression | 8.53 | HIGH | ✅ TRAINED |
| **1Q Total** | Quantile Regression | 2.80 | MEDIUM | ✅ TRAINED |
| **1Q Spread** | Quantile Regression | 2.52 | MEDIUM | ✅ TRAINED |
| **1Q Moneyline** | Classification | 0.48 | LOW | ✅ TRAINED |
| **2H Total** | Quantile Regression | 7.42 | LOW | ✅ TRAINED |
| **2H Spread** | Quantile Regression | 6.21 | LOW | ✅ TRAINED |

### Kicker Props (3 markets)
| Market | Model Type | Hit Rate | ROI | Status |
|--------|------------|----------|-----|--------|
| **FG Made** | Poisson | 50% (1 bet) | +45% | ✅ TRAINED |
| **XP Made** | Poisson | 65.0% | +23.5% | ⭐ TRAINED |
| **Total Kicker Points** | Poisson | 75.0% | +42.5% | ⭐⭐ TRAINED |

### Combo Props (2 markets)
| Market | Model Type | Hit Rate | ROI | Status |
|--------|------------|----------|-----|--------|
| **Pass + Rush Yards** | Quantile Regression | 27.3% | -56.1% | ⚠️ NEEDS IMPROVEMENT |
| **Rec + Rush Yards** | Quantile Regression | 41.7% | -20.8% | ⚠️ NEEDS IMPROVEMENT |

---

## 🚧 DATA AVAILABLE - READY TO TRAIN (15+ markets)

### Trainable from Existing Data

#### Multi-TD Props (4 markets)
- **2+ TDs** (any position) - Bernoulli on (total_tds >= 2)
- **3+ TDs** (any position) - Bernoulli on (total_tds >= 3)
- **2+ Pass TDs** - Bernoulli on (passing_tds >= 2)
- **2+ Rush+Rec TDs** - Bernoulli on (rushing_tds + receiving_tds >= 2)

Data: ✅ Available in player_stats_2025_synthetic.csv
Model Type: Bernoulli classification
Estimated Time: 30 mins

#### Anytime TD Scorer (1 market)
- **Anytime TD** - Bernoulli on (rushing_tds + receiving_tds > 0)

Data: ✅ Available in player_stats_2025_synthetic.csv
Model Type: Bernoulli classification
Estimated Time: 15 mins

#### Scoring Props - Trainable (6 markets)
- **Will there be OT?** - Bernoulli (extract from games where overtime=1)
- **Will there be a 2PT conversion?** - Bernoulli (estimate from scores)
- **Winning Margin** - Categorical on abs(home_score - away_score)
- **Highest Scoring Quarter** - Multinomial on quarter scores
- **Highest Scoring Half** - Bernoulli on (H1 vs H2)
- **Will there be scoreless quarter?** - Bernoulli on quarter scores

Data: ✅ Can derive from games_2025_with_quarters.csv
Model Type: Classification
Estimated Time: 1 hour

#### 1Q/1H Derivative Player Props (4 markets)
- **1Q Pass Yards** - Need quarter-level player splits
- **1H Pass Yards** - Need half-level player splits
- **1Q/1H Rush Yards** - Need splits
- **1Q/1H Rec Yards** - Need splits

Data: ⚠️ Need to generate quarter/half player stat splits
Model Type: Quantile Regression
Estimated Time: 2 hours (data gen + training)

**TOTAL TRAINABLE NOW:** 15 markets

---

## ❌ DATA NOT AVAILABLE - BLOCKED (38+ markets)

### Requires Play-by-Play Data

#### First/Last Scorer Props (8 markets)
- First TD Scorer
- Last TD Scorer
- First Score Type (TD/FG/Safety)
- Last Score Type
- First Team to Score
- First Team to 10 Points
- Race to X Points
- Team First TD Type (rush/pass/return)

**Blocker:** Needs play-by-play data with scoring sequence
**Source:** NFLverse pbp.csv (currently blocked by API 403 errors)

#### Drive Props (4 markets - Live Only)
- Next Drive Result (TD/FG/Punt/TO)
- Will next drive be 3-and-out?
- Points scored on this drive
- Longest play on drive

**Blocker:** Live betting only (not applicable for pre-game)

#### Longest Play Props (6 markets)
- Longest Completion
- Longest Rush
- Longest Reception
- Longest FG
- Team Longest TD
- Longest Play from Scrimmage

**Blocker:** Needs play-level data (max play length)
**Source:** NFLverse pbp.csv

#### Defense Props (5 markets)
- Sacks (individual player)
- Tackles (individual player)
- Interceptions (defensive)
- Defensive TDs
- QB Hits

**Blocker:** Needs defensive player stats
**Source:** NFLverse player_stats.csv or defensive aggregation

#### Exotic Scoring Props (10+ markets)
- Exact Final Score
- Score Bands (0-13, 14-20, 21-27, etc.)
- Will there be a safety?
- Team to Score in All Quarters
- Team Largest Lead
- Most consecutive scoring drives
- Shutdown quarter (team doesn't score)
- Both teams score in Q1
- Game goes under in regulation, over in OT
- Pick-6 scored

**Blocker:** Various - needs play-by-play, drive data, in-game tracking

#### Position-Specific Props (5+ markets)
- TE Receiving Yards
- TE Receptions
- TE Rec TDs
- Specific player milestones
- Rookie of the Year props

**Blocker:** Needs position splits and season-long data

**TOTAL BLOCKED:** 38+ markets

---

## 📊 PERFORMANCE SUMMARY

### Top Performing Models (ROI > 20%)
1. **Rush TDs**: 75.8% hit rate, +44.1% ROI ⭐⭐⭐
2. **Total Kicker Points**: 75.0% hit rate, +42.5% ROI ⭐⭐⭐
3. **XP Made**: 65.0% hit rate, +23.5% ROI ⭐⭐

### Solid Performers (ROI 5-20%)
1. **Pass TDs**: 62.6% hit rate, +19.0% ROI ⭐
2. **Rec Yards**: 57.0% hit rate, +8.4% ROI ✅

### Break-Even (ROI ±5%)
1. **Pass Yards**: 52.5% hit rate, -0.2% ROI
2. **Rush Yards**: 50.0% hit rate, -5.0% ROI
3. **1H Total**: MAE 6.61 pts
4. **1H Spread**: MAE 5.50 pts

### Needs Improvement (ROI < -10%)
1. **Completions**: 36.7% hit rate, -30.2% ROI ⚠️
2. **Attempts**: 39.7% hit rate, -24.5% ROI ⚠️
3. **Receptions**: 45.1% hit rate, -14.4% ROI ⚠️
4. **Targets**: 44.3% hit rate, -15.9% ROI ⚠️
5. **Pass + Rush Yards**: 27.3% hit rate, -56.1% ROI ⚠️
6. **Rec + Rush Yards**: 41.7% hit rate, -20.8% ROI ⚠️

**Improvement Strategy for Underperforming Props:**
- Add opponent defensive features (EPA allowed, etc.)
- Add weather data (wind, temp, dome)
- Add snap count data (usage rates)
- Add target share / carry share features
- Add game script indicators

---

## 🎯 NEXT STEPS TO HIT 80 MARKETS

### Immediate Actions (Can do now - 15 markets)
1. ✅ Train 2+ TD, 3+ TD props (4 markets) - 30 mins
2. ✅ Train Anytime TD scorer (1 market) - 15 mins
3. ✅ Train scoring props (OT, 2PT, margins, quarters) (6 markets) - 1 hour
4. ⚠️ Generate 1Q/1H player stat splits (4 markets) - 2 hours

**Expected Result:** 27 → 42 markets (52.5% complete)

### Medium Priority (Need data generation - 10+ markets)
1. Generate defensive stats from game data
2. Create TE-specific props from WR data
3. Add weather data integration
4. Build opponent defense features

**Expected Result:** 42 → 52+ markets (65% complete)

### Long-Term (Blocked on NFLverse API - 28 markets)
1. Wait for NFLverse API access OR
2. Manually parse play-by-play CSVs from GitHub
3. Train first/last scorer models
4. Train longest play models
5. Train exotic scoring props

**Expected Result:** 52 → 80 markets (100% complete)

---

## 📁 MODEL FILES

All trained models are saved in `outputs/models/`:

```
outputs/models/
├── comprehensive/          # Player props (12 models)
│   ├── pass_yards_models.pkl
│   ├── rush_yards_models.pkl
│   ├── rec_yards_models.pkl
│   ├── pass_tds_models.pkl
│   ├── rush_tds_models.pkl
│   ├── rec_tds_models.pkl
│   ├── completions_models.pkl
│   ├── attempts_models.pkl
│   ├── receptions_models.pkl
│   ├── targets_models.pkl
│   ├── carries_models.pkl
│   └── interceptions_models.pkl
├── derivative/             # Game derivative markets (10 models)
│   ├── h1_total_model.pkl
│   ├── h1_spread_model.pkl
│   ├── h1_home_won_model.pkl
│   ├── q1_total_model.pkl
│   ├── q1_spread_model.pkl
│   ├── q1_home_won_model.pkl
│   ├── h2_total_model.pkl
│   ├── h2_spread_model.pkl
│   ├── home_total_model.pkl
│   └── away_total_model.pkl
├── kicker/                 # Kicker props (3 models)
│   ├── fg_made_model.pkl
│   ├── xp_made_model.pkl
│   └── total_points_model.pkl
└── combo/                  # Combo props (2 models)
    ├── pass_rush_models.pkl
    └── rec_rush_models.pkl
```

---

## 🔄 CONTINUOUS IMPROVEMENT

### Weekly Tasks
- [ ] Backtest all models on new weeks
- [ ] Retrain with updated data
- [ ] Monitor hit rates and ROI
- [ ] Adjust edge thresholds

### Monthly Tasks
- [ ] Add new features (defense, weather, etc.)
- [ ] Hyperparameter tuning
- [ ] Model calibration checks (Brier score)
- [ ] Compare to Vegas closing lines

### Quarterly Tasks
- [ ] Full model rebuild with season data
- [ ] Feature importance analysis
- [ ] Remove underperforming models
- [ ] Add new market types

---

## 📈 ROADMAP TO 100% COVERAGE

**Phase 1: Foundation** ✅ COMPLETE (27/80 markets - 33.75%)
- Core player props
- Game derivative markets
- Kicker props
- Basic infrastructure

**Phase 2: Data-Driven Expansion** ⏳ IN PROGRESS (Target: 42/80 markets - 52.5%)
- Multi-TD props
- Anytime TD scorer
- Scoring props
- 1Q/1H player derivatives

**Phase 3: Feature Enhancement** 🔜 NEXT (Target: 52/80 markets - 65%)
- Add opponent defense stats
- Add weather integration
- Add snap count data
- Improve underperforming models

**Phase 4: Play-by-Play Integration** 🔮 FUTURE (Target: 80/80 markets - 100%)
- First/Last TD scorer
- Longest play props
- Drive-based props
- Exotic scoring props

**Estimated Timeline:**
- Phase 2: 4-6 hours
- Phase 3: 1-2 weeks
- Phase 4: Pending NFLverse API access

---

## ✅ SUCCESS CRITERIA

### Minimum Viable Product (MVP)
- ✅ 25+ markets trained (ACHIEVED - 27 markets)
- ✅ 3+ props with >60% hit rate (ACHIEVED - Rush TDs, XP, Total Kicker Pts)
- ✅ Injury filtering integrated (COMPLETE)
- ✅ Comprehensive backtesting (COMPLETE)

### Production Ready
- ⏳ 50+ markets trained (Target: 52 markets)
- ⏳ Average hit rate >55% across all props
- ⏳ Average ROI >10%
- ⏳ Zero bets on inactive players
- ⏳ Model calibration (Brier < 0.20)

### Full DraftKings Parity
- ❌ 80+ markets trained
- ❌ All high-volume markets covered
- ❌ Real-time line integration with Odds API
- ❌ Automated daily predictions
- ❌ Live CLV tracking

---

**Current Status:** 27/80 markets complete (33.75%)

**Next Milestone:** 42 markets (52.5%) - Achievable in next 4-6 hours

**Ultimate Goal:** 80+ markets (100%) - Full DraftKings coverage
