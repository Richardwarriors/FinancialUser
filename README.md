# FinancialUser

Predict an investor's next trading-day activity, buy/sell direction and ranked tickers using their historical profile, recent behavior and market context.

## Setup

Python 3.11+ on Linux or macOS.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
cp .env.example .env
```

Edit `.env` with your API key, chat-completions endpoint and model ID. `run.py` loads it automatically. The endpoint must support JSON responses and structured profile output.

## Your data

Provide these locally (datasets are not included):

```text
data/
  raw_records/   # One numeric-named JSON file per anonymous user
  users.txt      # One anonymous user ID per line
  symbols.txt    # One market symbol per line
```

Each JSON has `user_id` plus string fields `behavior_events`, `trade_executions` and `account_snapshots`. The exact formats are defined in [the parser](vendor/parse_financial_payloads.py). For a different data source, adapt it to [the SQLite schema](load_data.py) or [the data interface](app/datasource.py).

## Run

```bash
# Import data, generate seed profiles and fetch market bars.
python run.py prepare

# Build rolling samples and check them.
python run.py build --coverage-through 2026-10-07

# Validate prompts without API calls, then predict.
python run.py predict --dry-run
python run.py predict --resume
```

`prepare` recreates the local input databases and uses model/network calls. Only set `--coverage-through` when execution data is complete through that date; otherwise omit it. For shorter data coverage, pass the same `--end` to `prepare` and `build`.

Defaults: T0 `2026-08-01`, a seven-session lookback and 40 prediction sessions ending by `2026-10-07`. Change T0 in `app/rolling_config.py`; use `--end` and `--sessions` for the horizon. Model calls may incur charges.

Predictions and metrics are saved under `data/rolling_prediction/predictions/<model>/`. `summary.json` reports classification and buy/sell ticker-ranking scores; unknown outcomes are excluded. Resume only with the same model and configuration. Rebuild profiles/samples when their schema changes.

```bash
# Optional: small trial in a separate directory.
python run.py predict --limit 1 --out outputs/smoke

# Optional: information ablation or model comparison.
python ablate.py --arm market --all --model YOUR_MODEL
python compare.py --dir data/rolling_prediction/predictions
```

Use `python run.py --help` or `python <script>.py --help` for options. Direct entry points: `load_data.py`, `build_samples.py`, `predict.py`, `ablate.py`, `check_samples.py`, `audit.py`, `compare.py`, `summarize.py` and `export_profiles.py`.

## Tests

```bash
for test in tests/test_*.py; do python "$test" || exit 1; done
```

Credentials, private inputs and generated outputs are excluded from Git. Use anonymized data and keep `.env` local.
