"""Event to game mapping with confidence scoring"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz
from backend.config.logging_config import get_logger
from backend.database.models import Game

logger = get_logger(__name__)


@dataclass
class GameMatchResult:
    """Result of game matching"""
    game_id: Optional[str]
    confidence_score: float
    method: str
    metadata: Dict


class GameMapper:
    """
    Map odds API events to canonical nflverse game IDs

    Uses event metadata (start_time, home/away team strings) and matches
    against nflverse schedule with confidence scoring.
    """

    def __init__(self, games: List[Game]):
        """
        Initialize game mapper

        Args:
            games: List of canonical Game objects
        """
        self.games = {g.game_id: g for g in games}

        # Build indices
        self._build_indices()

        # Team name normalization map
        self.team_abbrev_map = {
            # Standard abbreviations
            "arizona cardinals": "ARI",
            "atlanta falcons": "ATL",
            "baltimore ravens": "BAL",
            "buffalo bills": "BUF",
            "carolina panthers": "CAR",
            "chicago bears": "CHI",
            "cincinnati bengals": "CIN",
            "cleveland browns": "CLE",
            "dallas cowboys": "DAL",
            "denver broncos": "DEN",
            "detroit lions": "DET",
            "green bay packers": "GB",
            "houston texans": "HOU",
            "indianapolis colts": "IND",
            "jacksonville jaguars": "JAX",
            "kansas city chiefs": "KC",
            "las vegas raiders": "LV",
            "los angeles chargers": "LAC",
            "los angeles rams": "LAR",
            "miami dolphins": "MIA",
            "minnesota vikings": "MIN",
            "new england patriots": "NE",
            "new orleans saints": "NO",
            "new york giants": "NYG",
            "new york jets": "NYJ",
            "philadelphia eagles": "PHI",
            "pittsburgh steelers": "PIT",
            "san francisco 49ers": "SF",
            "seattle seahawks": "SEA",
            "tampa bay buccaneers": "TB",
            "tennessee titans": "TEN",
            "washington commanders": "WAS",
            # Alternate names
            "raiders": "LV",
            "chargers": "LAC",
            "rams": "LAR",
            "49ers": "SF",
            "commanders": "WAS",
        }

    def _normalize_team_name(self, name: str) -> str:
        """Normalize team name to standard abbreviation"""
        name_lower = name.lower().strip()

        # Check direct mapping
        if name_lower in self.team_abbrev_map:
            return self.team_abbrev_map[name_lower]

        # Check if already an abbreviation (2-3 letters uppercase)
        if len(name) <= 3 and name.upper() == name:
            return name.upper()

        # Try fuzzy match on team names
        best_match = None
        best_score = 0

        for full_name, abbrev in self.team_abbrev_map.items():
            score = fuzz.ratio(name_lower, full_name)
            if score > best_score:
                best_score = score
                best_match = abbrev

        if best_score >= 80:
            return best_match

        # Return normalized original if no match
        return name.upper()

    def _build_indices(self):
        """Build lookup indices for games"""
        # Index by date
        self.date_index: Dict[str, List[str]] = {}  # date_str -> [game_ids]

        # Index by teams
        self.team_index: Dict[Tuple[str, str], str] = {}  # (home, away) -> game_id

        for game_id, game in self.games.items():
            # Date index (by date only, ignoring time)
            date_str = game.game_date.strftime("%Y-%m-%d")
            if date_str not in self.date_index:
                self.date_index[date_str] = []
            self.date_index[date_str].append(game_id)

            # Team index
            team_key = (game.home_team, game.away_team)
            self.team_index[team_key] = game_id

    def map_event_to_game(
        self,
        event_data: Dict,
        home_team: Optional[str] = None,
        away_team: Optional[str] = None,
        start_time: Optional[datetime] = None,
    ) -> GameMatchResult:
        """
        Map an odds API event to a game ID

        Args:
            event_data: Raw event data from odds API
            home_team: Home team name/abbreviation
            away_team: Away team name/abbreviation
            start_time: Game start time

        Returns:
            GameMatchResult with game_id and confidence
        """
        # Extract data from event if not provided
        if not home_team:
            home_team = event_data.get("home_team")
        if not away_team:
            away_team = event_data.get("away_team")
        if not start_time:
            start_time_str = event_data.get("commence_time") or event_data.get("start_time")
            if start_time_str:
                start_time = self._parse_datetime(start_time_str)

        if not all([home_team, away_team, start_time]):
            logger.warning(
                "insufficient_event_data",
                home=home_team,
                away=away_team,
                time=start_time
            )
            return GameMatchResult(
                game_id=None,
                confidence_score=0.0,
                method="insufficient_data",
                metadata={"event_data": event_data}
            )

        # Normalize team names
        home_abbrev = self._normalize_team_name(home_team)
        away_abbrev = self._normalize_team_name(away_team)

        # Step 1: Try exact team match
        team_key = (home_abbrev, away_abbrev)
        if team_key in self.team_index:
            game_id = self.team_index[team_key]
            game = self.games[game_id]

            # Verify time proximity (within 24 hours)
            time_diff = abs((game.game_date - start_time).total_seconds())
            if time_diff <= 24 * 3600:  # Within 24 hours
                logger.info(
                    "exact_team_match",
                    game_id=game_id,
                    home=home_abbrev,
                    away=away_abbrev
                )
                return GameMatchResult(
                    game_id=game_id,
                    confidence_score=100.0,
                    method="exact_team_match",
                    metadata={
                        "home_team": home_abbrev,
                        "away_team": away_abbrev,
                        "time_diff_hours": time_diff / 3600
                    }
                )

        # Step 2: Find games on same date
        date_str = start_time.strftime("%Y-%m-%d")
        candidate_game_ids = self.date_index.get(date_str, [])

        if not candidate_game_ids:
            # Try adjacent dates (±1 day)
            for offset in [-1, 1]:
                adj_date = start_time + timedelta(days=offset)
                adj_date_str = adj_date.strftime("%Y-%m-%d")
                candidate_game_ids.extend(self.date_index.get(adj_date_str, []))

        if not candidate_game_ids:
            logger.warning(
                "no_games_found_for_date",
                date=date_str,
                home=home_abbrev,
                away=away_abbrev
            )
            return GameMatchResult(
                game_id=None,
                confidence_score=0.0,
                method="no_games_on_date",
                metadata={"date": date_str, "home": home_abbrev, "away": away_abbrev}
            )

        # Step 3: Score candidates by team match and time proximity
        scored_candidates = []
        for game_id in candidate_game_ids:
            game = self.games[game_id]

            # Score team match
            home_match_score = fuzz.ratio(home_abbrev.lower(), game.home_team.lower())
            away_match_score = fuzz.ratio(away_abbrev.lower(), game.away_team.lower())
            team_score = (home_match_score + away_match_score) / 2

            # Score time proximity
            time_diff = abs((game.game_date - start_time).total_seconds())
            time_score = max(0, 100 - (time_diff / 3600))  # Decay by hour

            # Combined score
            combined_score = 0.7 * team_score + 0.3 * time_score

            scored_candidates.append((game_id, combined_score, {
                "team_score": team_score,
                "time_score": time_score,
                "time_diff_hours": time_diff / 3600
            }))

        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        if not scored_candidates:
            return GameMatchResult(
                game_id=None,
                confidence_score=0.0,
                method="no_candidates",
                metadata={}
            )

        # Return best match if confidence is sufficient
        best_game_id, best_score, best_metadata = scored_candidates[0]

        # Determine confidence threshold
        if best_score >= 90:
            method = "high_confidence_match"
        elif best_score >= 70:
            method = "medium_confidence_match"
        else:
            method = "low_confidence_match"

        logger.info(
            "game_mapped",
            game_id=best_game_id,
            score=best_score,
            method=method,
            home=home_abbrev,
            away=away_abbrev
        )

        return GameMatchResult(
            game_id=best_game_id if best_score >= 70 else None,
            confidence_score=best_score,
            method=method,
            metadata={
                "home_team": home_abbrev,
                "away_team": away_abbrev,
                "start_time": start_time.isoformat(),
                **best_metadata
            }
        )

    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string in various formats"""
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"Unable to parse datetime: {dt_str}")

    def batch_map(self, events: List[Dict]) -> List[GameMatchResult]:
        """Batch map multiple events to games"""
        results = []
        for event in events:
            result = self.map_event_to_game(event)
            results.append(result)
        return results
