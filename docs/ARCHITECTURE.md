# Architecture Overview

Components:
- ingestion: fetch nflverse CSVs, OddsAPI, injury/news sources
- canonical: player/game id mapping and canonicalization
- features: PBP feature extraction, smoothing, aggregates
- modeling: prop models (counts, TDs, yards)
- calib_backtest: calibration mapping and backtest tooling
- roster_injury: roster builder and injury index service
- api: FastAPI app exposing projections and admin recompute endpoints
- orchestration: docker-compose and runner scripts

Services interact via file artifacts in a canonical `data/` or `outputs/` layout and small JSON APIs for status.
