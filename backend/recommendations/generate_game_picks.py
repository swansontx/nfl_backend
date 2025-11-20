"""
Generate value-optimized parlay picks for a specific game.

Uses our validated strategy:
- Focus on UNDERs with buffers
- Target 70-85% hit rate
- Reasonable odds (-120 to -250)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import json


class GamePicksGenerator:
    """Generate parlay picks for a specific matchup."""

    def __init__(self, inputs_dir: str = "inputs"):
        self.inputs_dir = Path(inputs_dir)

        # Prefer enhanced file which has snap count data
        # Try files in order of preference
        files_to_try = [
            "player_stats_enhanced_2024_2025.csv",
            "player_stats_enhanced_2025.csv",
            "player_stats_2024_2025.csv",
        ]

        for filename in files_to_try:
            filepath = self.inputs_dir / filename
            if filepath.exists():
                self.stats = pd.read_csv(filepath, low_memory=False)
                break
        else:
            raise FileNotFoundError("No player stats file found")

        # Load injury data
        self.injuries = self._load_injuries()

        # Load depth chart data
        self.depth_charts = self._load_depth_charts()

        # Calculate team defensive rankings
        self.def_rankings = self._calculate_defensive_rankings()

        # Stat to defensive category mapping
        self.stat_to_def_category = {
            'rushing_yards': 'rush_yds_allowed',
            'rushing_tds': 'rush_tds_allowed',
            'carries': 'rush_yds_allowed',
            'receiving_yards': 'rec_yds_allowed',
            'receiving_tds': 'rec_tds_allowed',
            'receptions': 'rec_yds_allowed',
            'targets': 'rec_yds_allowed',
            'passing_yards': 'pass_yds_allowed',
            'passing_tds': 'pass_tds_allowed',
            'completions': 'pass_yds_allowed',
            'attempts': 'pass_yds_allowed',
            'interceptions': 'ints_forced',
        }

        self.prop_map = {
            'passing_yards': 'passing_yards',
            'passing_tds': 'passing_tds',
            'rushing_yards': 'rushing_yards',
            'rushing_tds': 'rushing_tds',
            'receiving_yards': 'receiving_yards',
            'receptions': 'receptions',
            'receiving_tds': 'receiving_tds',
            'completions': 'completions',
            'attempts': 'attempts',
            'interceptions': 'passing_interceptions',
            'carries': 'carries',
            'targets': 'targets',
        }

        # Our validated value strategies
        self.value_strategies = [
            {'prop': 'receiving_tds', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 82.2, 'odds': -120, 'ev': 50.7, 'category': 'scoring'},
            {'prop': 'rushing_tds', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 79.4, 'odds': -140, 'ev': 36.1, 'category': 'scoring'},
            {'prop': 'passing_tds', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 75.0, 'odds': -150, 'ev': 25.0, 'category': 'qb_scoring'},
            {'prop': 'rushing_yards', 'dir': 'UNDER', 'buffer': 15, 'hit_rate': 81.0, 'odds': -200, 'ev': 21.5, 'category': 'ground'},
            {'prop': 'rushing_yards', 'dir': 'UNDER', 'buffer': 10, 'hit_rate': 77.8, 'odds': -200, 'ev': 16.7, 'category': 'ground'},
            {'prop': 'rushing_yards', 'dir': 'UNDER', 'buffer': 5, 'hit_rate': 70.9, 'odds': -140, 'ev': 21.5, 'category': 'ground'},
            {'prop': 'carries', 'dir': 'UNDER', 'buffer': 3, 'hit_rate': 80.1, 'odds': -200, 'ev': 20.1, 'category': 'volume'},
            {'prop': 'carries', 'dir': 'UNDER', 'buffer': 2, 'hit_rate': 74.6, 'odds': -165, 'ev': 19.9, 'category': 'volume'},
            {'prop': 'receiving_yards', 'dir': 'UNDER', 'buffer': 12.5, 'hit_rate': 77.0, 'odds': -200, 'ev': 15.6, 'category': 'air'},
            {'prop': 'receiving_yards', 'dir': 'UNDER', 'buffer': 7.5, 'hit_rate': 70.9, 'odds': -165, 'ev': 13.9, 'category': 'air'},
            {'prop': 'receptions', 'dir': 'UNDER', 'buffer': 1.5, 'hit_rate': 81.6, 'odds': -300, 'ev': 5.7, 'category': 'volume'},
            {'prop': 'receptions', 'dir': 'UNDER', 'buffer': 1, 'hit_rate': 76.8, 'odds': -250, 'ev': 11.9, 'category': 'volume'},
            {'prop': 'targets', 'dir': 'UNDER', 'buffer': 2, 'hit_rate': 75.0, 'odds': -200, 'ev': 12.5, 'category': 'volume'},
            {'prop': 'passing_yards', 'dir': 'UNDER', 'buffer': 25, 'hit_rate': 72.0, 'odds': -165, 'ev': 14.0, 'category': 'qb'},
            {'prop': 'completions', 'dir': 'UNDER', 'buffer': 3, 'hit_rate': 70.0, 'odds': -140, 'ev': 12.0, 'category': 'qb'},
            {'prop': 'attempts', 'dir': 'UNDER', 'buffer': 4, 'hit_rate': 68.0, 'odds': -130, 'ev': 10.0, 'category': 'qb'},
            {'prop': 'interceptions', 'dir': 'UNDER', 'buffer': 0.5, 'hit_rate': 73.3, 'odds': -165, 'ev': 17.8, 'category': 'qb_turnover'},
            {'prop': 'interceptions', 'dir': 'OVER', 'buffer': -0.5, 'hit_rate': 45.0, 'odds': +110, 'ev': -2.0, 'category': 'qb_turnover'},
        ]

        # Narrative themes
        self.narratives = {
            'ground_game_stuffed': {
                'name': 'Ground Game Stuffed',
                'description': 'Strong run defense limits RB production',
                'positions': ['RB'],
                'props': ['rushing_yards', 'carries', 'rushing_tds'],
            },
            'secondary_lockdown': {
                'name': 'Secondary Lockdown',
                'description': 'Receivers struggle against tight coverage',
                'positions': ['WR', 'TE'],
                'props': ['receiving_yards', 'receptions', 'targets', 'receiving_tds'],
            },
            'qb_under_pressure': {
                'name': 'QB Under Pressure',
                'description': 'Pass rush disrupts timing',
                'positions': ['QB'],
                'props': ['passing_yards', 'completions', 'attempts', 'passing_tds', 'interceptions'],
            },
            'red_zone_woes': {
                'name': 'Red Zone Struggles',
                'description': 'Team struggles to punch it in',
                'positions': ['QB', 'RB', 'WR', 'TE'],
                'props': ['rushing_tds', 'receiving_tds', 'passing_tds'],
            },
        }

    def _load_injuries(self) -> Dict:
        """Load current injury statuses."""
        injury_file = self.inputs_dir / "injuries_2024_2025.csv"
        if not injury_file.exists():
            return {}

        injuries_df = pd.read_csv(injury_file)
        max_week = int(self.stats['week'].max())

        # Get latest injury report
        latest = injuries_df[injuries_df['week'] == max_week]

        # Build lookup by player name
        injury_status = {}
        for _, row in latest.iterrows():
            name = row['full_name']
            status = row.get('report_status', '')
            if pd.notna(status):
                injury_status[name] = status

        return injury_status

    def is_player_out(self, player_name: str) -> bool:
        """Check if player is ruled OUT."""
        status = self.injuries.get(player_name, '')
        return status.lower() == 'out' if status else False

    def get_injury_status(self, player_name: str) -> str:
        """Get player's injury status."""
        return self.injuries.get(player_name, '')

    def _load_depth_charts(self) -> pd.DataFrame:
        """Load depth chart data."""
        depth_file = self.inputs_dir / "depth_charts_2024_2025.csv"
        if depth_file.exists():
            df = pd.read_csv(depth_file, low_memory=False)
            # Filter to skill positions for offensive players
            skill_positions = ['QB', 'RB', 'WR', 'TE']
            df = df[df['depth_position'].isin(skill_positions)].copy()
            return df
        return pd.DataFrame()

    def get_depth_chart_multiplier(self, player_name: str, position: str) -> float:
        """Get depth chart multiplier based on player's position on depth chart.

        Starters (depth_team=1) get boosted, backups get reduced.

        Returns:
            Multiplier between 0.7 and 1.05
        """
        if len(self.depth_charts) == 0:
            return 1.0

        # Get most recent week
        max_week = int(self.stats['week'].max())

        # Find player's most recent depth chart entry
        player_dc = self.depth_charts[
            (self.depth_charts['full_name'] == player_name) &
            (self.depth_charts['week'] <= max_week)
        ].sort_values('week', ascending=False)

        if len(player_dc) == 0:
            # Try matching on last name + first name
            name_parts = player_name.split()
            if len(name_parts) >= 2:
                player_dc = self.depth_charts[
                    (self.depth_charts['first_name'] == name_parts[0]) &
                    (self.depth_charts['last_name'] == name_parts[-1]) &
                    (self.depth_charts['week'] <= max_week)
                ].sort_values('week', ascending=False)

        if len(player_dc) == 0:
            return 1.0

        depth_team = player_dc.iloc[0].get('depth_team', 1)

        # Starter (1st string) gets boost, backups get reduction
        if depth_team == 1:
            return 1.05  # 5% boost for starters
        elif depth_team == 2:
            return 0.85  # 15% reduction for backups
        else:
            return 0.70  # 30% reduction for 3rd string

    def _calculate_defensive_rankings(self) -> Dict:
        """Calculate team defensive rankings based on what they ALLOW to opponents."""
        rankings = {}

        # Load schedule to get matchup info
        schedule_file = self.inputs_dir / "schedules_2024_2025.csv"
        if not schedule_file.exists():
            return self._default_rankings()

        schedule = pd.read_csv(schedule_file)
        max_week = int(self.stats['week'].max())
        min_week = max(1, max_week - 5)

        # Track what each team ALLOWS
        team_allowed = {}

        for _, game in schedule[
            (schedule['week'] >= min_week) &
            (schedule['week'] <= max_week) &
            (schedule['game_type'] == 'REG')
        ].iterrows():
            home = game['home_team']
            away = game['away_team']
            week = game['week']

            # Away team's production = what home team allowed
            away_stats = self.stats[
                (self.stats['team'] == away) & (self.stats['week'] == week)
            ]
            # Home team's production = what away team allowed
            home_stats = self.stats[
                (self.stats['team'] == home) & (self.stats['week'] == week)
            ]

            # Home team allowed away team's stats
            if home not in team_allowed:
                team_allowed[home] = {'games': 0, 'rush': 0, 'rec': 0, 'pass': 0}
            team_allowed[home]['games'] += 1
            team_allowed[home]['rush'] += away_stats['rushing_yards'].sum()
            team_allowed[home]['rec'] += away_stats['receiving_yards'].sum()
            team_allowed[home]['pass'] += away_stats['passing_yards'].sum()

            # Away team allowed home team's stats
            if away not in team_allowed:
                team_allowed[away] = {'games': 0, 'rush': 0, 'rec': 0, 'pass': 0}
            team_allowed[away]['games'] += 1
            team_allowed[away]['rush'] += home_stats['rushing_yards'].sum()
            team_allowed[away]['rec'] += home_stats['receiving_yards'].sum()
            team_allowed[away]['pass'] += home_stats['passing_yards'].sum()

        # Calculate per-game averages and rank
        team_avgs = []
        for team, allowed in team_allowed.items():
            if allowed['games'] > 0:
                team_avgs.append({
                    'team': team,
                    'rush_avg': allowed['rush'] / allowed['games'],
                    'rec_avg': allowed['rec'] / allowed['games'],
                    'pass_avg': allowed['pass'] / allowed['games'],
                })

        if not team_avgs:
            return self._default_rankings()

        df = pd.DataFrame(team_avgs)

        # Calculate league averages for multiplier calculation
        league_rush_avg = df['rush_avg'].mean()
        league_rec_avg = df['rec_avg'].mean()
        league_pass_avg = df['pass_avg'].mean()

        # Rank: 1 = best defense (allows least)
        df['rush_rank'] = df['rush_avg'].rank(ascending=True)
        df['rec_rank'] = df['rec_avg'].rank(ascending=True)
        df['pass_rank'] = df['pass_avg'].rank(ascending=True)

        for _, row in df.iterrows():
            rankings[row['team']] = {
                'rank': {
                    'rush': int(row['rush_rank']),
                    'rec': int(row['rec_rank']),
                    'pass': int(row['pass_rank']),
                },
                'avg': {
                    'rush': row['rush_avg'],
                    'rec': row['rec_avg'],
                    'pass': row['pass_avg'],
                },
                'league_avg': {
                    'rush': league_rush_avg,
                    'rec': league_rec_avg,
                    'pass': league_pass_avg,
                }
            }

        return rankings

    def _default_rankings(self) -> Dict:
        """Return average rankings when data unavailable."""
        teams = self.stats['team'].unique()
        return {team: {'rank': {'rush': 16, 'rec': 16, 'pass': 16}} for team in teams}

    def get_def_context(self, opponent: str, prop: str) -> str:
        """Get defensive context string for opponent matchup."""
        if opponent not in self.def_rankings:
            return ""

        def_cat = self.stat_to_def_category.get(prop, '')
        if not def_cat:
            return ""

        ranks = self.def_rankings[opponent].get('rank', {})

        # Map defensive category to rank key
        if 'rush' in def_cat:
            rank = ranks.get('rush', 16)
            cat_name = 'vs run'
        elif 'rec' in def_cat:
            rank = ranks.get('rec', 16)
            cat_name = 'vs catch'
        elif 'pass' in def_cat:
            rank = ranks.get('pass', 16)
            cat_name = 'vs pass'
        elif 'int' in def_cat:
            rank = 32 - ranks.get('pass', 16)  # Inverse for INTs forced
            cat_name = 'forcing INTs'
        else:
            return ""

        # Describe the matchup
        if rank <= 8:
            quality = "tough"
        elif rank <= 16:
            quality = "avg"
        elif rank <= 24:
            quality = "soft"
        else:
            quality = "very soft"

        return f"vs {opponent} ({quality} {cat_name}, #{rank})"

    def get_opponent_multiplier(self, opponent: str, stat_col: str) -> float:
        """Get opponent adjustment multiplier based on defensive quality.

        Returns a multiplier > 1 for bad defenses, < 1 for good defenses.
        """
        if opponent not in self.def_rankings:
            return 1.0

        opp_data = self.def_rankings[opponent]
        if 'avg' not in opp_data or 'league_avg' not in opp_data:
            return 1.0

        # Map stat to defensive category
        if stat_col in ['rushing_yards', 'carries', 'rushing_tds']:
            cat = 'rush'
        elif stat_col in ['receiving_yards', 'receptions', 'targets', 'receiving_tds']:
            cat = 'rec'
        elif stat_col in ['passing_yards', 'completions', 'attempts', 'passing_tds']:
            cat = 'pass'
        else:
            return 1.0

        opp_avg = opp_data['avg'].get(cat, 0)
        league_avg = opp_data['league_avg'].get(cat, 1)

        if league_avg == 0:
            return 1.0

        # Calculate multiplier
        # More aggressive for rushing (0.75-1.35), less for passing (0.85-1.20)
        multiplier = opp_avg / league_avg
        if cat == 'rush':
            return max(0.75, min(1.35, multiplier))
        elif cat == 'pass':
            return max(0.85, min(1.20, multiplier))
        else:
            return max(0.80, min(1.25, multiplier))

    def get_snap_usage_multiplier(self, player_id: str, stat_col: str) -> float:
        """Get snap usage multiplier based on recent vs season average snap share.

        A player with increasing snap share is getting more opportunities,
        so we boost their projection. Decreasing snap share = fewer opportunities.

        Returns:
            Multiplier between 0.85 and 1.15
        """
        player_df = self.stats[
            self.stats['player_id'] == player_id
        ].sort_values('week', ascending=False)

        if len(player_df) < 2:
            return 1.0

        # Check if snap data is available
        if 'offense_pct' not in player_df.columns:
            return 1.0

        snap_values = player_df['offense_pct'].dropna()
        if len(snap_values) < 2:
            return 1.0

        season_avg = snap_values.mean()
        l3_avg = snap_values.head(3).mean()

        if season_avg == 0:
            return 1.0

        # Calculate snap trend (recent vs season)
        snap_trend = l3_avg / season_avg

        # Apply different caps based on stat type
        # Volume stats (yards, receptions) are more affected by snap count
        if stat_col in ['rushing_yards', 'receiving_yards', 'receptions', 'targets', 'carries']:
            # More aggressive adjustment for volume stats (0.85-1.15)
            return max(0.85, min(1.15, snap_trend))
        elif stat_col in ['passing_yards', 'completions', 'attempts']:
            # Moderate adjustment for passing (QBs usually play full game)
            multiplier = 0.5 + (snap_trend * 0.5)  # Dampened effect
            return max(0.90, min(1.10, multiplier))
        else:
            # TD stats are less correlated with pure volume
            multiplier = 0.7 + (snap_trend * 0.3)  # Very dampened
            return max(0.95, min(1.05, multiplier))

    def get_projection(self, player_id: str, stat_col: str, n_weeks: int = 4, opponent: str = None) -> float:
        """Get weighted average projection for a player, adjusted for opponent.

        Args:
            player_id: Player's ID
            stat_col: Stat column name
            n_weeks: Number of recent weeks to consider
            opponent: Optional opponent team for defensive adjustment
        """
        player_df = self.stats[
            self.stats['player_id'] == player_id
        ].sort_values('week', ascending=False)

        if len(player_df) == 0 or stat_col not in player_df.columns:
            return 0

        values = player_df[stat_col].head(n_weeks).dropna().tolist()
        if not values:
            return 0

        # Recency weighted
        weights = [0.4, 0.3, 0.2, 0.1][:len(values)]
        base_projection = sum(v * w for v, w in zip(values, weights)) / sum(weights)

        # Apply opponent adjustment for rushing/receiving yards
        if opponent and stat_col in ['rushing_yards', 'receiving_yards', 'passing_yards']:
            opp_multiplier = self.get_opponent_multiplier(opponent, stat_col)
            base_projection = base_projection * opp_multiplier

        # Apply snap usage adjustment (opportunity indicator)
        snap_multiplier = self.get_snap_usage_multiplier(player_id, stat_col)
        base_projection = base_projection * snap_multiplier

        # Apply depth chart adjustment (starter/backup status)
        player_name = player_df.iloc[0].get('player_display_name', '')
        position = player_df.iloc[0].get('position', '')
        dc_multiplier = self.get_depth_chart_multiplier(player_name, position)
        base_projection = base_projection * dc_multiplier

        return base_projection

    def get_player_trend(self, player_id: str, stat_col: str) -> Dict:
        """Get player's recent trend for a stat."""
        player_df = self.stats[
            self.stats['player_id'] == player_id
        ].sort_values('week', ascending=False)

        if len(player_df) < 2 or stat_col not in player_df.columns:
            return {'trend': 'stable', 'recent': [], 'consistency': 'unknown'}

        values = player_df[stat_col].head(4).dropna().tolist()
        if len(values) < 2:
            return {'trend': 'stable', 'recent': values, 'consistency': 'unknown'}

        # Trend: compare recent to older
        recent_avg = np.mean(values[:2]) if len(values) >= 2 else values[0]
        older_avg = np.mean(values[2:]) if len(values) > 2 else recent_avg

        if recent_avg > older_avg * 1.15:
            trend = 'up'
        elif recent_avg < older_avg * 0.85:
            trend = 'down'
        else:
            trend = 'stable'

        # Consistency: low variance = consistent
        if len(values) >= 3:
            cv = np.std(values) / np.mean(values) if np.mean(values) > 0 else 1
            if cv < 0.25:
                consistency = 'very_consistent'
            elif cv < 0.4:
                consistency = 'consistent'
            else:
                consistency = 'volatile'
        else:
            consistency = 'unknown'

        return {'trend': trend, 'recent': values, 'consistency': consistency}

    def generate_justification(self, player: str, team: str, prop: str, direction: str,
                                projection: float, line: float, buffer: float,
                                trend_info: Dict, opponent: str) -> str:
        """Generate a short justification for why this pick makes sense."""
        reasons = []

        # Matchup context FIRST
        def_context = self.get_def_context(opponent, prop)
        if def_context:
            reasons.append(def_context)

        # Trend reasoning
        if trend_info['trend'] == 'down' and direction == 'UNDER':
            reasons.append("trending ↓")
        elif trend_info['trend'] == 'up' and direction == 'OVER':
            reasons.append("trending ↑")
        elif trend_info['consistency'] == 'very_consistent':
            reasons.append("very consistent")

        # Recent values context
        if trend_info['recent']:
            recent_str = '/'.join([str(int(v)) if v == int(v) else str(round(v, 1)) for v in trend_info['recent'][:3]])
            reasons.append(f"L3: {recent_str}")

        # Buffer as cushion
        buffer_pct = (buffer / projection * 100) if projection > 0 else 0
        if buffer_pct >= 20:
            reasons.append(f"{buffer_pct:.0f}% cushion")

        return "; ".join(reasons[:3])

    def calculate_alt_line_suggestion(self, projection: float, line: float,
                                       direction: str, prop_type: str,
                                       trend_info: Dict, consistency: str) -> Dict:
        """Calculate alternate line suggestion for better hit rate.

        Based on backtest data, optimal buffers by prop type:
        - passing_yards: 10% (~25 yds on 250 line)
        - rushing_yards: 5% (~4 yds on 80 line)
        - receiving_yards: 5% (~3 yds on 60 line)

        Always provides alt line option - user decides whether to use it.
        """
        edge = abs(projection - line)
        edge_pct = (edge / line * 100) if line > 0 else 0

        # Optimal buffer sizes from backtest analysis
        optimal_buffers = {
            'passing_yards': 0.10,    # 10% = ~25 yards
            'rushing_yards': 0.05,    # 5% = ~4 yards
            'receiving_yards': 0.05,  # 5% = ~3 yards
            'completions': 0.10,
            'attempts': 0.15,
            'receptions': 0.10,
            'targets': 0.10,
            'carries': 0.10,
            'passing_tds': 0.0,       # TDs don't benefit from alt lines
            'rushing_tds': 0.0,
            'receiving_tds': 0.0,
        }

        buffer_pct = optimal_buffers.get(prop_type, 0.10)

        # Skip alt lines for TD props (they already have large edges)
        if 'tds' in prop_type or buffer_pct == 0:
            base_hit_rate = 55 + min(edge_pct * 1.5, 25)
            return {
                'use_alt': False,
                'alt_line': line,
                'alt_odds': -110,
                'reason': None,
                'confidence': 'high' if edge_pct > 20 else 'medium',
                'estimated_hit_rate': round(min(base_hit_rate, 85), 0),
                'standard_hit_rate': round(min(base_hit_rate, 85), 0),
            }

        # Calculate alt line
        if direction == 'OVER':
            alt_line = line * (1 - buffer_pct)
        else:
            alt_line = line * (1 + buffer_pct)

        # Round to standard increments
        if prop_type == 'passing_yards':
            alt_line = round(alt_line / 5) * 5
        elif prop_type in ['rushing_yards', 'receiving_yards']:
            alt_line = round(alt_line / 2.5) * 2.5
        else:
            alt_line = round(alt_line * 2) / 2

        # Calculate line difference in actual yards
        line_diff = abs(alt_line - line)

        # Estimate hit rates (UNDERs hit ~7% more than OVERs based on backtest)
        if direction == 'UNDER':
            base_hit_rate = 52 + min(edge_pct * 1.2, 18)
            alt_hit_rate = base_hit_rate + (buffer_pct * 60)  # 5% buffer = +3%, 10% = +6%
        else:
            base_hit_rate = 45 + min(edge_pct * 1.2, 18)
            alt_hit_rate = base_hit_rate + (buffer_pct * 50)

        # Calculate odds at alt line
        odds_map = {0.05: -130, 0.10: -150, 0.15: -180}
        alt_odds = odds_map.get(buffer_pct, -150)

        # Determine recommendation strength
        reasons = []

        # Always recommend for small/medium edges
        if edge_pct < 8:
            reasons.append(f"+{line_diff:.0f} yds buffer")

        # Extra reasons
        if consistency == 'volatile':
            reasons.append("volatile")
        if direction == 'OVER' and trend_info.get('trend') == 'down':
            reasons.append("↓ trend")
        elif direction == 'UNDER' and trend_info.get('trend') == 'up':
            reasons.append("↑ trend")

        # Always recommend alt for yards props (user decides)
        use_alt = len(reasons) > 0 or edge_pct < 12

        return {
            'use_alt': use_alt,
            'alt_line': alt_line,
            'alt_odds': alt_odds,
            'reason': ", ".join(reasons) if reasons else f"+{line_diff:.0f} yds safety",
            'confidence': 'high' if alt_hit_rate >= 65 else 'medium',
            'estimated_hit_rate': round(min(alt_hit_rate, 85), 0),
            'standard_hit_rate': round(min(base_hit_rate, 80), 0),
            'buffer_pct': buffer_pct * 100,
            'line_diff': line_diff
        }

    def generate_picks(self, team1: str, team2: str) -> List[Dict]:
        """Generate value picks for a matchup."""
        picks = []

        # Get players from both teams
        teams = [team1, team2]
        max_week = int(self.stats['week'].max())

        players = self.stats[
            (self.stats['week'] >= max_week - 3) &
            (self.stats['team'].isin(teams)) &
            (self.stats['position'].isin(['QB', 'RB', 'WR', 'TE']))
        ][['player_id', 'player_display_name', 'team', 'position']].drop_duplicates()

        for _, player in players.iterrows():
            player_id = player['player_id']
            player_name = player['player_display_name']
            team = player['team']
            position = player['position']

            # Skip players who are OUT
            injury_status = self.injuries.get(player_name, '')
            if injury_status.lower() == 'out':
                continue

            # Determine which props to check
            if position == 'QB':
                props = ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards']
            elif position == 'RB':
                props = ['rushing_yards', 'rushing_tds', 'carries', 'receptions', 'receiving_yards']
            elif position in ['WR', 'TE']:
                props = ['receptions', 'receiving_yards', 'receiving_tds', 'targets']
            else:
                continue

            # Determine opponent for this player
            opponent = team2 if team == team1 else team1

            for strategy in self.value_strategies:
                if strategy['prop'] not in props:
                    continue

                stat_col = self.prop_map.get(strategy['prop'], strategy['prop'])
                # Pass opponent for defensive adjustment
                projection = self.get_projection(player_id, stat_col, opponent=opponent)

                if projection <= 0:
                    continue

                # Skip very low projections
                if strategy['prop'] in ['receiving_tds', 'rushing_tds'] and projection < 0.2:
                    continue
                if strategy['prop'] == 'interceptions' and projection < 0.3:
                    continue

                # Get trend info for justification
                trend_info = self.get_player_trend(player_id, stat_col)

                # Calculate the line
                if strategy['dir'] == 'UNDER':
                    line = projection + strategy['buffer']
                else:
                    line = projection - strategy['buffer']

                # Round line to typical sportsbook increments
                if strategy['prop'] in ['passing_yards']:
                    line = round(line / 5) * 5
                elif strategy['prop'] in ['rushing_yards', 'receiving_yards']:
                    line = round(line * 2) / 2
                else:
                    line = round(line * 2) / 2

                # Determine if player is a backup based on projection thresholds
                is_backup = False
                if strategy['prop'] == 'rushing_yards' and projection < 40:
                    is_backup = True
                elif strategy['prop'] == 'receiving_yards' and projection < 35:
                    is_backup = True
                elif strategy['prop'] == 'carries' and projection < 8:
                    is_backup = True
                elif strategy['prop'] == 'receptions' and projection < 3:
                    is_backup = True

                # Generate justification
                justification = self.generate_justification(
                    player_name, team, strategy['prop'], strategy['dir'],
                    projection, line, strategy['buffer'], trend_info, opponent
                )

                # Calculate alternate line suggestion
                alt_suggestion = self.calculate_alt_line_suggestion(
                    projection, line, strategy['dir'], strategy['prop'],
                    trend_info, trend_info.get('consistency', 'unknown')
                )

                pick_data = {
                    'player': player_name,
                    'team': team,
                    'position': position,
                    'prop': strategy['prop'],
                    'direction': strategy['dir'],
                    'projection': round(projection, 1),
                    'line': line,
                    'buffer': strategy['buffer'],
                    'est_hit_rate': strategy['hit_rate'],
                    'est_odds': strategy['odds'],
                    'est_ev': strategy['ev'],
                    'category': strategy.get('category', 'other'),
                    'is_backup': is_backup,
                    'justification': justification,
                    'trend': trend_info['trend'],
                    'recent_values': trend_info['recent'][:3],
                }

                # Add alt line info
                pick_data['alt_line_recommended'] = alt_suggestion['use_alt']
                pick_data['alt_line'] = alt_suggestion['alt_line']
                pick_data['alt_odds'] = alt_suggestion['alt_odds']
                pick_data['alt_reason'] = alt_suggestion.get('reason', '')
                pick_data['alt_hit_rate'] = alt_suggestion['estimated_hit_rate']
                pick_data['standard_hit_rate'] = alt_suggestion.get('standard_hit_rate', strategy['hit_rate'])
                pick_data['confidence'] = alt_suggestion['confidence']
                pick_data['line_diff'] = alt_suggestion.get('line_diff', 0)

                picks.append(pick_data)

        # Sort by EV
        picks.sort(key=lambda x: -x['est_ev'])

        return picks

    def build_narrative_parlays(self, picks: List[Dict], team1: str, team2: str) -> Dict[str, List[Dict]]:
        """Build parlays organized by narrative themes."""
        parlays = {}

        for narrative_key, narrative in self.narratives.items():
            # Filter picks matching this narrative
            matching = []
            for p in picks:
                # Check if prop type matches narrative's props
                if 'props' in narrative and p['prop'] not in narrative['props']:
                    continue

                # Check position match
                if p['position'] not in narrative['positions']:
                    continue

                # Check backup role if specified
                if narrative.get('role') == 'backup' and not p.get('is_backup', False):
                    continue
                if narrative.get('role') == 'starter' and p.get('is_backup', False):
                    continue

                # Handle OVER props - only include if narrative allows
                if p['direction'] == 'OVER' and not narrative.get('include_overs', False):
                    continue

                matching.append(p)

            # Sort by hit rate (safer bets first), then by EV
            matching.sort(key=lambda x: (-x['est_hit_rate'], -x['est_ev']))

            # Remove duplicates (same player+prop, keep highest hit rate)
            seen = set()
            unique = []
            for p in matching:
                key = f"{p['player']}_{p['prop']}"
                if key not in seen:
                    seen.add(key)
                    unique.append(p)

            parlays[narrative_key] = unique

        return parlays

    def print_picks(self, picks: List[Dict], game: str, team1: str, team2: str):
        """Print formatted narrative-based parlays."""
        print(f"\n{'='*70}")
        print(f"NARRATIVE PARLAY PICKS: {game}")
        print(f"{'='*70}")

        # Build narrative parlays
        narrative_parlays = self.build_narrative_parlays(picks, team1, team2)

        # Print each narrative
        for narrative_key, narrative in self.narratives.items():
            parlay_picks = narrative_parlays.get(narrative_key, [])

            if len(parlay_picks) < 2:
                continue

            print(f"\n\n{'='*70}")
            print(f"NARRATIVE: {narrative['name']}")
            print(f"{narrative['description']}")
            print(f"{'='*70}")

            # Show individual picks
            print("\nAvailable Picks:")
            print("-" * 70)
            for i, p in enumerate(parlay_picks[:6], 1):
                backup_tag = " [BACKUP]" if p.get('is_backup') else ""
                trend_arrow = "↓" if p.get('trend') == 'down' else "↑" if p.get('trend') == 'up' else "→"
                print(f"{i}. {p['player']}{backup_tag} ({p['team']}) {trend_arrow}")
                print(f"   {p['prop']} {p['direction']} {p['line']}")
                print(f"   Proj: {p['projection']} | Hit: {p['est_hit_rate']}% | Odds: {p['est_odds']}")
                if p.get('justification'):
                    print(f"   WHY: {p['justification']}")

            # Build suggested parlays for this narrative
            if len(parlay_picks) >= 2:
                print("\nSuggested 2-Leg:")
                legs = parlay_picks[:2]
                combined = 1
                teams_involved = set()
                for p in legs:
                    print(f"  • {p['player']} {p['prop']} {p['direction']} {p['line']} ({p['est_hit_rate']}%)")
                    combined *= (p['est_hit_rate']/100)
                    teams_involved.add(p['team'])
                print(f"  Combined hit rate: {combined*100:.1f}%")
                # Parlay justification
                if len(teams_involved) == 2:
                    print(f"  WHY: Props from both teams - uncorrelated outcomes")
                elif len(teams_involved) == 1:
                    print(f"  WHY: Same team narrative - if defense shows up, both hit")

            if len(parlay_picks) >= 3:
                print("\nSuggested 3-Leg:")
                legs = parlay_picks[:3]
                combined = 1
                teams_involved = set()
                props_involved = set()
                for p in legs:
                    print(f"  • {p['player']} {p['prop']} {p['direction']} {p['line']} ({p['est_hit_rate']}%)")
                    combined *= (p['est_hit_rate']/100)
                    teams_involved.add(p['team'])
                    props_involved.add(p['prop'])
                print(f"  Combined hit rate: {combined*100:.1f}%")
                # Parlay justification
                diversified = len(teams_involved) > 1 or len(props_involved) > 1
                if diversified:
                    print(f"  WHY: Diversified across {'teams' if len(teams_involved) > 1 else 'prop types'}")

        # Cross-narrative "Best of" parlay
        print(f"\n\n{'='*70}")
        print("RECOMMENDED: CROSS-NARRATIVE PARLAY")
        print("Mix of ground, air, and volume for maximum diversification")
        print(f"{'='*70}")

        # Get best from each category (excluding TDs)
        best_ground = [p for p in picks if p['category'] == 'ground']
        best_air = [p for p in picks if p['category'] == 'air']
        best_volume = [p for p in picks if p['category'] == 'volume']

        # Sort each by hit rate
        best_ground.sort(key=lambda x: -x['est_hit_rate'])
        best_air.sort(key=lambda x: -x['est_hit_rate'])
        best_volume.sort(key=lambda x: -x['est_hit_rate'])

        cross_parlay = []
        if best_ground:
            cross_parlay.append(best_ground[0])
        if best_air:
            cross_parlay.append(best_air[0])
        if best_volume and len(cross_parlay) < 3:
            # Avoid duplicate players
            for p in best_volume:
                if p['player'] not in [x['player'] for x in cross_parlay]:
                    cross_parlay.append(p)
                    break

        if len(cross_parlay) >= 2:
            print("\nDiversified 3-Leg Parlay:")
            combined = 1
            for p in cross_parlay[:3]:
                print(f"  • {p['player']} {p['prop']} {p['direction']} {p['line']} ({p['est_hit_rate']}%)")
                combined *= (p['est_hit_rate']/100)
            print(f"  Combined hit rate: {combined*100:.1f}%")

        # Optional: Add one TD for "spice"
        td_picks = [p for p in picks if p['category'] == 'td']
        if td_picks and len(cross_parlay) >= 2:
            td_picks.sort(key=lambda x: -x['est_hit_rate'])
            spice_pick = td_picks[0]
            print("\n  Optional 'Spice' Add:")
            print(f"  + {spice_pick['player']} {spice_pick['prop']} {spice_pick['direction']} {spice_pick['line']} ({spice_pick['est_hit_rate']}%)")
            combined_with_spice = combined * (spice_pick['est_hit_rate']/100)
            print(f"  4-Leg with spice: {combined_with_spice*100:.1f}% hit rate")


def main():
    import argparse
    p = argparse.ArgumentParser(description='Generate value-optimized parlay picks')
    p.add_argument('--team1', type=str, default='HOU',
                   help='Away team abbreviation (default: HOU)')
    p.add_argument('--team2', type=str, default='BUF',
                   help='Home team abbreviation (default: BUF)')
    p.add_argument('--inputs', type=str, default='inputs',
                   help='Input directory for player stats')
    args = p.parse_args()

    team1, team2 = args.team1.upper(), args.team2.upper()

    generator = GamePicksGenerator(inputs_dir=args.inputs)

    # Generate picks
    picks = generator.generate_picks(team1, team2)
    game_name = f"{team1} @ {team2}"
    generator.print_picks(picks, game_name, team1, team2)

    # Save to file
    output_file = Path(f"outputs/picks_{team1}_{team2}.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(picks, f, indent=2)

    print(f"\n📁 Full picks saved to: {output_file}")


if __name__ == "__main__":
    main()
