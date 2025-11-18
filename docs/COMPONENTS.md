# Components and Interfaces

- ingestion/
  - fetch_nflverse.py -> outputs: inputs/stats_player_week_YYYY.csv, inputs/player_lookup.json
  - fetch_odds.py -> outputs: cache/web_event_<id>.json
  - fetch_injuries.py -> outputs: outputs/injuries_YYYYMMDD_parsed.json

- canonical/
  - map_event_to_game.py -> inputs: odds event JSON, nflverse schedule -> outputs: canonical game_id
  - player_map.py -> utilities to map book player names -> nflverse player_id

- features/
  - extract_player_pbp_features.py -> outputs: player_pbp_features_by_id.json
  - smoothing_and_rolling.py -> outputs: smoothed JSONs

- modeling/
  - model_runner.py -> loads features, team profiles -> writes props CSVs

- calib_backtest/
  - calibrate.py -> fits Platt/Isotonic on historical data
  - backtest.py -> evaluates model vs outcomes

- roster_injury/
  - build_game_roster_index.py -> outputs: game_rosters_YYYY.json
  - build_injury_game_index.py -> outputs: injury_game_index_YYYY.json
  - service: roster_lookup.py -> function: get_player_status(game_id, player_id)

- api/
  - app.py (FastAPI) with endpoints:
    - GET /health
    - GET /game/{game_id}/projections
    - POST /admin/recompute
