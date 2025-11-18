"""Player name matching and canonicalization with fuzzy logic"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from rapidfuzz import fuzz, process
from backend.config.logging_config import get_logger
from backend.database.models import Player

logger = get_logger(__name__)


@dataclass
class MatchResult:
    """Result of a player matching attempt"""
    player_id: Optional[str]
    display_name: Optional[str]
    match_score: float
    method: str
    confidence: str  # high, medium, low
    candidates: List[Tuple[str, str, float]]  # [(player_id, name, score)]
    metadata: Dict


class PlayerMatcher:
    """
    Multi-step player name matching pipeline

    Pipeline steps:
    1. Exact normalized match
    2. Team-scoped exact match
    3. Token-based normalization
    4. Nickname mapping
    5. Fuzzy matching with thresholds
    """

    def __init__(self, players: List[Player], manual_overrides: Optional[Dict[str, str]] = None):
        """
        Initialize player matcher

        Args:
            players: List of canonical Player objects
            manual_overrides: Dict mapping book names to player_id
        """
        self.players = {p.player_id: p for p in players}
        self.manual_overrides = manual_overrides or {}

        # Build normalized lookup indices
        self._build_indices()

        # Nickname mapping (common variations)
        self.nickname_map = {
            "aj": ["a.j.", "aj", "a. j."],
            "jj": ["j.j.", "jj", "j. j."],
            "tj": ["t.j.", "tj", "t. j."],
            "cj": ["c.j.", "cj", "c. j."],
            "dj": ["d.j.", "dj", "d. j."],
            "chris": ["christopher", "chris"],
            "mike": ["michael", "mike"],
            "matt": ["matthew", "matt"],
            "rob": ["robert", "rob", "bobby"],
            "bill": ["william", "bill"],
            "jim": ["james", "jim"],
            "joe": ["joseph", "joe"],
            "dan": ["daniel", "dan"],
            "dave": ["david", "dave"],
            "steve": ["stephen", "steve", "steven"],
            "josh": ["joshua", "josh"],
            "pat": ["patrick", "pat"],
        }

    def _normalize_name(self, name: str) -> str:
        """Normalize name: lowercase, remove punctuation, diacritics, extra whitespace"""
        # Remove diacritics
        name = unicodedata.normalize('NFKD', name)
        name = name.encode('ASCII', 'ignore').decode('ASCII')

        # Lowercase and remove punctuation
        name = name.lower()
        name = re.sub(r'[^\w\s]', '', name)

        # Normalize whitespace
        name = ' '.join(name.split())

        return name

    def _tokenize_name(self, name: str) -> List[str]:
        """Split name into tokens"""
        normalized = self._normalize_name(name)
        return normalized.split()

    def _build_indices(self):
        """Build lookup indices for fast matching"""
        self.normalized_index: Dict[str, str] = {}  # normalized_name -> player_id
        self.token_index: Dict[str, List[str]] = {}  # last_name -> [player_ids]
        self.team_name_index: Dict[Tuple[str, str], str] = {}  # (team, normalized_name) -> player_id

        for player_id, player in self.players.items():
            normalized = self._normalize_name(player.display_name)

            # Normalized index
            if normalized not in self.normalized_index:
                self.normalized_index[normalized] = player_id

            # Team-scoped index
            if player.team:
                key = (player.team, normalized)
                if key not in self.team_name_index:
                    self.team_name_index[key] = player_id

            # Token index (by last name)
            tokens = self._tokenize_name(player.display_name)
            if tokens:
                last_name = tokens[-1]
                if last_name not in self.token_index:
                    self.token_index[last_name] = []
                self.token_index[last_name].append(player_id)

    def _expand_nickname(self, name: str) -> List[str]:
        """Expand name with possible nickname variants"""
        tokens = self._tokenize_name(name)
        variations = [name]

        for i, token in enumerate(tokens):
            if token in self.nickname_map:
                for variant in self.nickname_map[token]:
                    new_tokens = tokens.copy()
                    new_tokens[i] = variant
                    variations.append(' '.join(new_tokens))

        return variations

    def match(
        self,
        name: str,
        team: Optional[str] = None,
        position: Optional[str] = None,
    ) -> MatchResult:
        """
        Match a player name to canonical player ID

        Args:
            name: Player name from sportsbook
            team: Team abbreviation (helps with disambiguation)
            position: Position (helps with disambiguation)

        Returns:
            MatchResult with player_id and confidence metrics
        """
        # Check manual overrides first
        if name in self.manual_overrides:
            player_id = self.manual_overrides[name]
            player = self.players.get(player_id)
            return MatchResult(
                player_id=player_id,
                display_name=player.display_name if player else None,
                match_score=100.0,
                method="manual_override",
                confidence="high",
                candidates=[(player_id, player.display_name, 100.0)] if player else [],
                metadata={"override": True}
            )

        # Step 1: Exact normalized match
        result = self._exact_match(name)
        if result:
            return result

        # Step 2: Team-scoped exact match
        if team:
            result = self._team_scoped_match(name, team)
            if result:
                return result

        # Step 3: Token-based match (last name + first initial)
        result = self._token_match(name, team, position)
        if result:
            return result

        # Step 4: Nickname expansion
        result = self._nickname_match(name, team)
        if result:
            return result

        # Step 5: Fuzzy matching
        result = self._fuzzy_match(name, team, position)
        if result:
            return result

        # No match found
        logger.warning("no_match", name=name, team=team, position=position)
        return MatchResult(
            player_id=None,
            display_name=None,
            match_score=0.0,
            method="no_match",
            confidence="none",
            candidates=[],
            metadata={"input_name": name, "team": team, "position": position}
        )

    def _exact_match(self, name: str) -> Optional[MatchResult]:
        """Step 1: Exact normalized match"""
        normalized = self._normalize_name(name)
        player_id = self.normalized_index.get(normalized)

        if player_id:
            player = self.players[player_id]
            logger.info("exact_match", name=name, player_id=player_id)
            return MatchResult(
                player_id=player_id,
                display_name=player.display_name,
                match_score=100.0,
                method="exact_normalized",
                confidence="high",
                candidates=[(player_id, player.display_name, 100.0)],
                metadata={}
            )
        return None

    def _team_scoped_match(self, name: str, team: str) -> Optional[MatchResult]:
        """Step 2: Team-scoped exact match"""
        normalized = self._normalize_name(name)
        key = (team, normalized)
        player_id = self.team_name_index.get(key)

        if player_id:
            player = self.players[player_id]
            logger.info("team_scoped_match", name=name, team=team, player_id=player_id)
            return MatchResult(
                player_id=player_id,
                display_name=player.display_name,
                match_score=100.0,
                method="team_scoped_exact",
                confidence="high",
                candidates=[(player_id, player.display_name, 100.0)],
                metadata={"team": team}
            )
        return None

    def _token_match(self, name: str, team: Optional[str], position: Optional[str]) -> Optional[MatchResult]:
        """Step 3: Token-based matching (e.g., 'G. Davis' matches 'Gabe Davis')"""
        tokens = self._tokenize_name(name)
        if not tokens:
            return None

        last_name = tokens[-1]
        candidates_ids = self.token_index.get(last_name, [])

        if not candidates_ids:
            return None

        # Filter by team and position if provided
        filtered_candidates = []
        for pid in candidates_ids:
            player = self.players[pid]

            # Check team match
            if team and player.team != team:
                continue

            # Check position match
            if position and player.position != position:
                continue

            # Check if first initial matches
            player_tokens = self._tokenize_name(player.display_name)
            if len(tokens) > 1 and len(player_tokens) > 1:
                # Check first initial
                if tokens[0][0] == player_tokens[0][0]:
                    filtered_candidates.append(pid)
            elif len(tokens) == 1:  # Just last name
                filtered_candidates.append(pid)

        if len(filtered_candidates) == 1:
            player_id = filtered_candidates[0]
            player = self.players[player_id]
            logger.info("token_match", name=name, player_id=player_id)
            return MatchResult(
                player_id=player_id,
                display_name=player.display_name,
                match_score=95.0,
                method="token_based",
                confidence="high",
                candidates=[(player_id, player.display_name, 95.0)],
                metadata={"team": team, "position": position}
            )

        return None

    def _nickname_match(self, name: str, team: Optional[str]) -> Optional[MatchResult]:
        """Step 4: Try matching with nickname expansions"""
        variants = self._expand_nickname(name)

        for variant in variants:
            # Try exact match with variant
            result = self._exact_match(variant)
            if result:
                result.method = "nickname_expansion"
                result.match_score = 92.0
                logger.info("nickname_match", name=name, variant=variant, player_id=result.player_id)
                return result

            # Try team-scoped with variant
            if team:
                result = self._team_scoped_match(variant, team)
                if result:
                    result.method = "nickname_team_scoped"
                    result.match_score = 92.0
                    logger.info("nickname_match", name=name, variant=variant, player_id=result.player_id)
                    return result

        return None

    def _fuzzy_match(self, name: str, team: Optional[str], position: Optional[str]) -> Optional[MatchResult]:
        """Step 5: Fuzzy matching with thresholds"""
        # Build candidate list
        candidates = []
        for player_id, player in self.players.items():
            # Filter by team if provided
            if team and player.team != team:
                continue
            # Filter by position if provided
            if position and player.position != position:
                continue

            candidates.append((player.display_name, player_id))

        if not candidates:
            return None

        # Get fuzzy matches
        matches = process.extract(
            name,
            [c[0] for c in candidates],
            scorer=fuzz.token_sort_ratio,
            limit=5
        )

        if not matches:
            return None

        # Get best match
        best_name, best_score, _ = matches[0]

        # Find player_id for best match
        player_id = next(pid for n, pid in candidates if n == best_name)
        player = self.players[player_id]

        # Determine confidence based on score and context
        if best_score >= 95:
            confidence = "high"
        elif best_score >= 80:
            # Medium confidence requires team match
            if team and player.team == team:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "low"

        # Only accept if confidence is medium or high
        if confidence in ["high", "medium"]:
            candidate_list = [
                (next(pid for n, pid in candidates if n == match[0]), match[0], match[1])
                for match in matches
            ]

            logger.info(
                "fuzzy_match",
                name=name,
                player_id=player_id,
                score=best_score,
                confidence=confidence
            )

            return MatchResult(
                player_id=player_id,
                display_name=player.display_name,
                match_score=best_score,
                method="fuzzy",
                confidence=confidence,
                candidates=candidate_list,
                metadata={"team": team, "position": position, "threshold_used": best_score}
            )

        # Score too low or context doesn't match
        logger.warning(
            "fuzzy_match_rejected",
            name=name,
            best_match=best_name,
            score=best_score,
            confidence=confidence
        )
        return None

    def batch_match(
        self,
        names: List[Tuple[str, Optional[str], Optional[str]]]
    ) -> List[MatchResult]:
        """
        Batch match multiple players

        Args:
            names: List of (name, team, position) tuples

        Returns:
            List of MatchResults
        """
        results = []
        for name, team, position in names:
            result = self.match(name, team, position)
            results.append(result)

        return results

    def get_match_stats(self, results: List[MatchResult]) -> Dict:
        """Get statistics on batch matching results"""
        total = len(results)
        matched = sum(1 for r in results if r.player_id is not None)

        methods = {}
        confidences = {}

        for result in results:
            methods[result.method] = methods.get(result.method, 0) + 1
            confidences[result.confidence] = confidences.get(result.confidence, 0) + 1

        return {
            "total": total,
            "matched": matched,
            "match_rate": matched / total if total > 0 else 0,
            "methods": methods,
            "confidences": confidences,
        }
