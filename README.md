# trading-models

Hermes trading research starter repository for Ahmed Abaza.

This repo currently includes:
- A curated 25-account X research watchlist for macro, investing, energy, commodities, AI, and semiconductors
- A script to fetch recent tweets from the X watchlist using x-cli
- A script to fetch Alpha Vantage quotes for a core market watchlist
- A script to build a simple text brief combining both data sources

## Structure

- `config/accounts.json` — curated X accounts with categories and rationale
- `scripts/fetch_x_watchlist.py` — fetch 5 recent tweets per handle
- `scripts/fetch_alpha_watchlist.py` — fetch quotes for the default watchlist
- `scripts/build_daily_brief.py` — combine X + Alpha Vantage into a plain-text brief
- `data/x/latest.json` — latest fetched X snapshot
- `data/alpha_vantage/latest.json` — latest Alpha Vantage snapshot
- `reports/daily_brief.txt` — generated combined brief

## Requirements

Credentials live in `~/.hermes/.env`:
- `X_API_KEY`
- `X_API_SECRET`
- `X_BEARER_TOKEN`
- `X_ACCESS_TOKEN`
- `X_ACCESS_TOKEN_SECRET`
- `ALPHA_VANTAGE_API_KEY`

Python 3 and `x-cli` are required.

## Usage

```bash
python3 scripts/fetch_x_watchlist.py
python3 scripts/fetch_alpha_watchlist.py
python3 scripts/build_daily_brief.py
cat reports/daily_brief.txt
```

## Notes

- Current `x-cli` supports timeline/search/bookmark/like/retweet/post flows, but not follow/unfollow or list management.
- For persistent monitoring, this repo stores a curated account list and polls their timelines.
- This is a starter pipeline and should be extended with scoring, deduplication, sentiment, and portfolio relevance ranking.
