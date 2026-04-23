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

For reproducible runs, use the project-managed `uv` environment rather than the ambient shell Python.

```bash
uv run python scripts/fetch_x_watchlist.py
uv run python scripts/fetch_alpha_watchlist.py
uv run python scripts/build_daily_brief.py
cat reports/daily_brief.txt

uv run python scripts/run_baseline.py
uv run --with pytest python -m pytest tests -q
```

## Notes

- Current `x-cli` supports timeline/search/bookmark/like/retweet/post flows, but not follow/unfollow or list management.
- For persistent monitoring, this repo stores a curated account list and polls their timelines.
- The baseline model now excludes the final `FORWARD_DAYS` rows from supervised evaluation so backtest labels only use observations with a known forward close; live inference still scores the freshest feature row.
- Baseline outputs now rank names by a skill-aware conviction score (`balanced_accuracy_edge * signal_edge`), where `balanced_accuracy_edge = balanced_accuracy - 0.5` and `signal_edge` measures confidence in the predicted side (`prob_up` for BUY signals, `1 - prob_up` for SELL signals); this penalizes sub-chance models instead of letting strong live probabilities from historically weak names float to the top of the watchlist.
- `reports/latest_predictions.csv` and `reports/RESULTS.md` now include `signal_confidence` plus `balanced_accuracy_edge`, so the ranking formula is auditable and it is obvious when a signal is coming from a model with real edge above chance versus one that is merely confident.
- Baseline outputs now also include `train_rows` and `test_rows` so each ticker's headline metrics can be read alongside the amount of labeled history used for fitting and out-of-sample evaluation.
- The ranked baseline outputs now include `train_positive_rate` and `test_positive_rate`, which show the share of forward-up labels in each split; this makes it easier to spot when a ticker's recent evaluation window was unusually bullish or bearish and to interpret balanced accuracy in that context.
- The baseline report/CSV also derive `up_rate_delta = test_positive_rate - train_positive_rate`, surfacing which tickers are seeing the biggest bullish or bearish label-regime shift versus their training history.
- The ranked baseline outputs now also include `regime_edge`, which measures how far the live BUY/SELL confidence sits above the recent test-window base rate of the predicted side; this helps distinguish genuinely differentiated signals from probabilities that mostly echo a bullish or bearish regime.
- The baseline watchlist now surfaces each signal's `latest_date` and `latest_close`, making the markdown/CSV output easier to audit against the exact market snapshot that generated the live BUY/SELL call.
- Baseline outputs now also include out-of-sample `brier_score`, so the markdown/CSV can show which tickers had the most calibrated test-set probabilities in addition to directional accuracy.
- The baseline feature set now also includes short/medium and medium/long moving-average spread features (`ma_10_vs_ma20`, `ma_20_vs_ma50`) so the model can distinguish trend alignment from simple price-vs-average distance without reducing the usable sample history.
- The feature set now also includes `ma_20_slope_5d`, defined as the 5-day change in the 20-day moving average; this gives the classifier a smoother medium-trend slope signal while keeping warmup below the existing 50-day moving-average cap, and zero-denominator windows are treated as a neutral `0.0`.
- The feature set now also includes `range_position_20`, which places the latest close within its rolling 20-day high/low range (with flat ranges treated as neutral at `0.5`) so the model can distinguish breakouts from pullbacks without adding more warmup than the existing 20-day windows.
- The feature set now also includes `drawdown_from_high_20`, defined as the percentage gap from the rolling 20-day high; this gives the classifier an explicit pullback-depth signal that reuses the same 20-day range window and naturally reads as `0.0` when price is at the high.
- The baseline feature set now also includes `vol_ratio_5d_20d`, a short-vs-long realized volatility ratio that highlights volatility expansion/contraction regimes; flat windows are treated as a neutral ratio of `1.0`, and the feature does not increase warmup beyond the existing 20-day volatility lookback.
- The feature set now also includes `ret_5d_minus_ret_20d`, a short-vs-medium return spread that gives the model an explicit momentum-acceleration signal while reusing the existing 20-day lookback window.
- The feature set now also includes `ret_5d_per_vol_5d`, a 5-day return scaled by 5-day realized volatility; this gives the classifier a short-horizon risk-adjusted momentum signal while reusing the existing 5-day return/volatility window, and flat volatility windows are treated as a neutral `0.0`.
- The feature set now also includes `ret_20d_per_vol_20d`, a 20-day return scaled by 20-day realized volatility; this gives the classifier a simple risk-adjusted momentum signal while reusing the existing 20-day return/volatility windows, and flat volatility windows are treated as a neutral `0.0`.
- This is a starter pipeline and should be extended with scoring, deduplication, sentiment, and portfolio relevance ranking.
