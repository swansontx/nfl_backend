# Additional Signals & Data Sources for Backtesting

**Date:** 2025-11-28
**Purpose:** Identify high-value signals not currently in use
**Current Status:** Using only play-by-play, schedules, basic team stats

---

## HIGH PRIORITY SIGNALS (Expect Large Impact)

### 1. 🏥 INJURY IMPACT ★★★★★
**Data:** `injuries_2024_2025.csv` (6,264 records)
**Fields:** position, injury type, status (Out/Questionable/Doubtful), practice status

**Potential Signals:**
- **Star Player Out:** QB/RB1/WR1 missing = massive impact
- **Injury Severity Score:** Weight by position importance
  - QB Out: -6 to -10 points
  - RB1 Out: -3 to -5 points
  - WR1 Out: -2 to -4 points
  - OL starters: -1 to -2 points (cumulative)
- **Questionable/Limited Practice:** Reduced performance (-30% to -50%)
- **Backup Quality Metric:** Starter vs backup dropoff
- **Multiple Injuries:** Cumulative effect on team

**Backtesting Approach:**
- Parse injuries for each game week
- Calculate team injury impact score
- Adjust spread/total predictions
- Compare vs games with full health

**Expected Impact:** ✅ **HUGE** - Injuries are often 3-7 point swing

---

### 2. 🏈 REST & TRAVEL ★★★★★
**Data:** `schedules_2024_2025.csv` (has rest days, home/away)

**Potential Signals:**
- **Short Rest (Thursday/Friday games):** Team fatigue
  - Typical: -1 to -3 points for short rest team
  - Worse if traveling
- **Extra Rest (Bye week, Monday after bye):** Team advantage
  - Typical: +2 to +3 points
- **Cross-Country Travel (3+ hour time zones):**
  - West → East early game: -2 to -3 points
  - East → West late game: +1 to +2 points
- **Back-to-back road games:** Fatigue effect

**Backtesting Approach:**
- Calculate rest differential (home rest - away rest)
- Adjust spread by rest advantage
- Test Thursday night games separately (notoriously low-scoring)

**Expected Impact:** ✅ **HIGH** - Rest/travel affects 20%+ of games

---

### 3. 📊 NEXT GEN STATS (Advanced Metrics) ★★★★☆
**Data:**
- `ngs_passing_2024_2025.csv` - Passing efficiency
- `ngs_receiving_2024_2025.csv` - Route running, separation
- `ngs_rushing_2024_2025.csv` - Rush efficiency

**Potential Signals:**

**Passing NGS:**
- `avg_time_to_throw`: Pressure indicator (faster = more pressure)
- `avg_completed_air_yards`: Downfield passing ability
- `aggressiveness`: Risk-taking tendency (deep ball %)
- `avg_air_yards_to_sticks`: Conservative vs aggressive

**Receiving NGS:**
- `avg_separation`: WR vs DB quality
- `avg_cushion`: Coverage style faced
- `percent_share_of_targets`: Target concentration

**Rushing NGS:**
- `efficiency`: Yards over expected
- `avg_time_to_los`: OL run blocking quality

**Backtesting Approach:**
- Aggregate NGS metrics by team
- Create "downfield threat" score (air yards + aggressiveness)
- Create "pressure vulnerability" score (time to throw)
- Test correlation with totals and spreads

**Expected Impact:** ✅ **MEDIUM-HIGH** - More granular than basic stats

---

### 4. 👔 REFEREE TENDENCIES ★★★☆☆
**Data:** `officials_2024_2025.csv` (3,204 records)

**Potential Signals:**
- **Penalty-Heavy Refs:** More flags = longer games, more variance
  - Average penalties per game by referee
  - Affects total (more penalties = more possessions = higher scores)
- **Pass Interference Rate:** Some refs call more DPI (affects spreads)
- **Home Team Bias:** Do certain refs favor home teams?

**Backtesting Approach:**
- Calculate historical penalty rates per referee
- Test if high-penalty refs → higher totals
- Test if certain refs favor home teams statistically

**Expected Impact:** ✅ **MEDIUM** - Subtle but measurable

---

### 5. 📈 QUARTER-BY-QUARTER TRENDS ★★★★☆
**Data:** `games_2025_with_quarters.csv`

**Potential Signals:**
- **Q1 Scoring Tendency:**
  - Some teams slow starters (SF, BAL historically)
  - Some teams fast starters (KC, BUF)
- **Second Half Adjustments:**
  - Coaching quality indicator
  - H2 spread vs H1 spread
- **Comeback Ability:**
  - Teams that perform better when trailing
- **Garbage Time Scoring:**
  - Affects totals in blowouts

**Backtesting Approach:**
- Calculate team Q1 vs Q2-Q4 scoring rates
- Create "slow start" and "strong finish" indicators
- Test if these predict 1H vs 2H performance

**Expected Impact:** ✅ **MEDIUM** - Useful for live betting, derivatives

---

## MEDIUM PRIORITY SIGNALS

### 6. 🎯 RED ZONE EFFICIENCY (Enhanced) ★★★☆☆
**Data:** `red_zone_stats_2025.csv`

**Potential Signals:**
- **Red zone attempts per game:** Offensive quality
- **Red zone TD vs FG rate:** Scoring efficiency
- **Goal-to-go success rate:** Short-yardage ability

**Note:** We already use some red zone data from PBP, but this could be more detailed.

**Expected Impact:** ✅ **SMALL-MEDIUM** - Already using RZ%, this adds detail

---

### 7. 📦 SNAP COUNT TRENDS ★★★☆☆
**Data:** `snap_counts_2024_2025.csv` (4.3M records)

**Potential Signals:**
- **Offensive snap % trend:** Is player usage increasing/decreasing?
- **Fresh legs:** RB snap count drop = fresh backup coming
- **Defensive rotation quality:** Deep rotations = less fatigue late game

**Backtesting Approach:**
- Track player snap % trends week-over-week
- Detect workload increases (injury risk)
- Detect workload decreases (reduced role)

**Expected Impact:** ✅ **SMALL-MEDIUM** - More for player props

---

### 8. 🏟️ WEATHER CONDITIONS ★★★★☆
**Data:** `schedules_2024_2025.csv` (has temp, wind, roof)

**Potential Signals:**
- **Wind Speed:**
  - >15 mph: Passing game severely impacted
  - Typical: -3 to -7 points on total
- **Temperature:**
  - Cold (<32°F): Ball harder to throw/catch
  - Typical: -2 to -4 points on total
- **Precipitation (if available):**
  - Rain/Snow: Running game favored, lower totals

**Current Usage:** We have temp and wind in schedules!

**Backtesting Approach:**
- Test wind threshold effects on totals
- Test cold weather effects on passing efficiency
- Compare dome vs outdoor games

**Expected Impact:** ✅ **MEDIUM-HIGH** - Weather games are predictable

---

### 9. 🆚 HISTORICAL MATCHUP DATA ★★☆☆☆
**Data:** Can derive from `schedules_2024_2025.csv` historical games

**Potential Signals:**
- **Recent H2H results:** Last 3 meetings
- **Divisional game variance:** Division games more volatile
- **Rivalry game intensity:** Higher intensity = lower totals?

**Backtesting Approach:**
- Calculate H2H history (spread, total)
- Test if divisional games have lower totals
- Test if recent H2H predicts current game

**Expected Impact:** ✅ **SMALL** - NFL parity limits H2H predictiveness

---

## ADVANCED / EXPERIMENTAL SIGNALS

### 10. 🧠 COACHING ADJUSTMENTS ★★★☆☆
**Derive from:** Play-by-play + game results

**Potential Signals:**
- **Second Half Adjustments:** H2 spread vs H1 spread by coach
- **Timeout Usage:** Clock management quality
- **4th Down Aggressiveness:** Modern analytics adoption
- **Challenge Success Rate:** Film study quality

**Expected Impact:** ✅ **SMALL-MEDIUM** - Hard to quantify

---

### 11. 💰 BETTING MARKET MOVEMENT ★★★★☆
**Data:** Would need to acquire (line movement data)

**Potential Signals:**
- **Sharp Money Indicators:** Line moves against public
- **Steam Moves:** Sudden large line movements
- **Reverse Line Movement:** Line moves opposite of public %

**Expected Impact:** ✅ **POTENTIALLY HIGH** - Market is efficient

**Note:** Would need to acquire this data separately

---

### 12. 📱 PLAYER SOCIAL MEDIA / NEWS SENTIMENT ★☆☆☆☆
**Data:** Would need to scrape

**Potential Signals:**
- Locker room issues (team morale)
- Contract disputes (player motivation)
- Personal issues (distraction)

**Expected Impact:** ⚠️ **VERY UNCERTAIN** - Hard to quantify

---

## RECOMMENDED IMPLEMENTATION PRIORITY

### Phase 1: Quick Wins (Implement Now) 🚀
1. **Weather Adjustments** - Data already in schedules
   - Wind >15mph: -5 points to total
   - Temp <32°F: -3 points to total
   - **Effort:** LOW, **Impact:** MEDIUM-HIGH

2. **Rest/Travel Differentials** - Data already in schedules
   - Short rest: -2 points
   - Extra rest: +2 points
   - **Effort:** LOW, **Impact:** HIGH

3. **Divisional Game Flags** - Data in schedules
   - Test if divisional games have different characteristics
   - **Effort:** LOW, **Impact:** SMALL-MEDIUM

---

### Phase 2: High-Value Additions (Next Sprint) 🎯
4. **Injury Impact Scoring System**
   - Parse injuries, weight by position
   - Create injury severity score
   - **Effort:** MEDIUM, **Impact:** VERY HIGH

5. **Referee Penalty Tendencies**
   - Aggregate official stats
   - Test penalty rate → total correlation
   - **Effort:** MEDIUM, **Impact:** MEDIUM

6. **Next Gen Stats Integration**
   - Add NGS passing/receiving/rushing metrics
   - Create composite "threat scores"
   - **Effort:** MEDIUM, **Impact:** MEDIUM-HIGH

---

### Phase 3: Advanced Features (Future) 🔮
7. **Quarter-by-Quarter Modeling**
   - Build Q1, H1, H2 specific models
   - **Effort:** HIGH, **Impact:** MEDIUM

8. **Snap Count Trend Analysis**
   - Track player usage trends
   - **Effort:** HIGH, **Impact:** SMALL-MEDIUM

9. **Coaching Adjustment Metrics**
   - Quantify second-half coaching
   - **Effort:** HIGH, **Impact:** SMALL-MEDIUM

---

## DATA COMPILATION CHECKLIST

### ✅ Already Have (Not Using)
- [x] Injuries (6,264 records)
- [x] Weather (temp, wind in schedules)
- [x] Rest days (in schedules)
- [x] Officials (3,204 records)
- [x] NGS Stats (passing, receiving, rushing)
- [x] Quarter-by-quarter scores
- [x] Snap counts (4.3M records)
- [x] Red zone stats (detailed)
- [x] Depth charts (45MB!)

### ❌ Need to Acquire
- [ ] Historical line movement data
- [ ] Precipitation data (rain/snow)
- [ ] Advanced coaching metrics
- [ ] Betting percentages (public vs sharp money)

---

## BACKTESTING FRAMEWORK UPDATES NEEDED

To test these new signals, we'll need to:

1. **Extend `game_metrics_features.py`:**
   - Add weather adjustment method
   - Add rest adjustment method
   - Add injury impact calculation

2. **Create New Modules:**
   - `backend/features/injury_impact.py`
   - `backend/features/weather_adjustments.py`
   - `backend/features/referee_metrics.py`
   - `backend/features/ngs_metrics.py`

3. **Update Backtesting:**
   - Add injury data to game context
   - Add weather data to game context
   - Track which adjustments help most

---

## EXPECTED IMPACT SUMMARY

| Signal | Expected Impact | Effort | Priority |
|--------|----------------|--------|----------|
| **Injuries** | ★★★★★ HUGE | Medium | 🔥 Phase 2 |
| **Rest/Travel** | ★★★★★ HIGH | Low | 🚀 Phase 1 |
| **Weather** | ★★★★☆ MEDIUM-HIGH | Low | 🚀 Phase 1 |
| **NGS Metrics** | ★★★★☆ MEDIUM-HIGH | Medium | 🎯 Phase 2 |
| **Quarter Trends** | ★★★★☆ MEDIUM | High | 🔮 Phase 3 |
| **Referees** | ★★★☆☆ MEDIUM | Medium | 🎯 Phase 2 |
| **Divisional Flags** | ★★★☆☆ SMALL-MED | Low | 🚀 Phase 1 |
| **Red Zone Detail** | ★★★☆☆ SMALL-MED | Low | 🔮 Phase 3 |
| **Snap Counts** | ★★☆☆☆ SMALL-MED | High | 🔮 Phase 3 |
| **H2H History** | ★★☆☆☆ SMALL | Medium | 🔮 Phase 3 |

---

## QUICK WIN IMPLEMENTATION

The **easiest immediate improvements** are already in our data:

```python
# Weather adjustments (already have temp, wind)
if wind > 15:
    total_adj -= 5.0
if temp < 32:
    total_adj -= 3.0

# Rest adjustments (already have rest days)
rest_diff = home_rest - away_rest
if rest_diff >= 7:  # Home team off bye
    spread_adj += 2.0
elif rest_diff <= -3:  # Home team on short rest
    spread_adj -= 2.0

# Divisional game flag
if div_game:
    # Test if these should be treated differently
    pass
```

These could be added **today** with minimal effort and tested immediately!

---

## CONCLUSION

We have **massive untapped potential** in our existing data:
- 6K+ injury records (not using)
- Weather data (not using)
- Rest/travel data (not using)
- NGS metrics (not using)
- Officials data (not using)

**Recommendation:** Start with Phase 1 (weather, rest, divisional) since the data is already in `schedules_2024_2025.csv` and can be implemented in ~1 hour. Then move to Phase 2 (injuries, refs, NGS) for the biggest impact.

The injury system alone could be worth **3-7 points of edge** in games with key player absences!

---

**Author:** Claude Code
**Date:** 2025-11-28
**Status:** Ready for implementation
