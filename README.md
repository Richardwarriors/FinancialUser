# FinancialUser: Rolling Prediction

Code for the rolling next-trading-day financial behavior prediction experiment. This repository includes the point-in-time profile engine, rolling sample construction, hierarchical prediction, information ablations and evaluation. Datasets, user profiles, experiment outputs, Explanation experiments, credentials and participant information are excluded.

## Requirements

- Python 3.11 or newer; Linux or macOS (`fcntl` is used for build locks).
- Your own anonymized behavioral events, executions and daily holdings snapshots.
- A chat-completions-compatible model endpoint supporting JSON output. Profile generation uses LangChain structured output; prediction uses HTTP chat completions.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
cp .env.example .env
# Edit .env locally: set the API key, endpoint and model IDs.
set -a
source .env
set +a
```

The historical model defaults in the source are experiment identifiers. Set all four `LUNA_MODEL*` fields to model IDs available at your endpoint. `.env.example` contains no credentials. The environment must be loaded before running any script, since the profile engine reads model settings at import time. The predictor also accepts `--model`.

The lock file records the dependency versions used for offline validation (Python 3.12). `requirements.txt` lists the three direct dependencies.

## Prediction protocol

The fixed profile boundary (`T0`) is `2026-08-01` in `app/rolling_config.py`. Each sample uses an expanding-history profile, the last seven US trading sessions and market context available at the previous session's 16:00 America/New_York cutoff. The first target is `2026-08-12`; samples advance one trading session at a time. The default horizon is 40 prediction sessions ending by `2026-10-07`. Holidays and weekends are excluded.

The primary task first predicts `trade` / `no_trade`, then `buy_only` / `sell_only` / `buy_and_sell` on trade days, and separately ranks up to five buy and sell tickers. Unobserved outcomes remain unknown and are excluded from scoring. No executions prove `no_trade` only when known boundary holdings books are unchanged. Forecast fields and target-side evidence are excluded from model prompts. Legacy cash-direction and position-action labels remain for audit compatibility.

## 1. Prepare local inputs

No research dataset is distributed, so paper results cannot be reproduced from this repository alone. Use your own authorized, anonymized data. Do not put names, email addresses, account identifiers, free-text personal disclosures or credential values into shared artifacts. `app/pii.py` and the evidence projection apply additional filtering, but input anonymization remains necessary.

For the included loader, place one numeric-named JSON file per anonymous user under `data/raw_records/`. The document has `user_id` (an anonymous identifier) and string fields `behavior_events`, `trade_executions`, `account_snapshots`:

- `behavior_events`: timestamped behavior records (source timezone Asia/Shanghai).
- `trade_executions`: `|`-separated orders with `#`-separated fields; true execution time comes from field 18 as epoch milliseconds, not the extraction date.
- `account_snapshots`: daily account/holdings snapshots.

The exact payload decoders are `vendor/parse_financial_payloads.py`. For another source format, implement an adapter producing the SQLite tables in `load_observations_db.py:SCHEMA`, or adapt `app/datasource.py`. Keep execution timestamps in the documented naive UTC representation and snapshot dates in the source session convention.

The payload field names, SQLite table names and paths are semantic names throughout the code. Existing databases and input files must be converted to the schema shown above or rebuilt with this loader; old identifiers are not accepted as aliases. Profile settings use the `PROFILE_` environment-variable prefix (for example, `PROFILE_EVIDENCE_CACHE` and `PROFILE_SOURCE_KIND`).

Create `data/users.txt` containing one anonymous user ID per line. Create `data/symbols.txt` containing the portfolio, trade and viewed symbols needed for market context, one symbol per line. These local files are intentionally not shipped.

```bash
# Physically truncated seed database. This command recreates the specified output.
python load_observations_db.py --raw data/raw_records --out data/observations_t0.db \
  --cutoff 2026-08-01 --users-file data/users.txt

# Full observation database, through the last date in your source extraction.
# Choose a different output path so that T0 remains truncated.
python load_observations_db.py --raw data/raw_records --out data/observations_full.db \
  --cutoff 2026-10-07 --users-file data/users.txt
```

Do not use the full database as the seed database. If changing T0, update `PROFILE_CUTOFF`, the loader cutoff and the experiment dates together; the calendar implementation is an experiment calendar, not a live exchange-calendar service.

## 2. Build seed profiles and cache market bars

These profile calls may consume paid model tokens. Without credentials the engine can use deterministic fallbacks; such a run is not equivalent to the LLM experiment. Verify the recorded agent/model provenance before comparing results.

```bash
python scripts/build_t0_profiles.py --db data/observations_t0.db --users-file data/users.txt

# Replace N with the number of users in data/users.txt (no header).
python export_point_in_time_profiles.py --db data/observations_t0.db \
  --out data/rolling_prediction/base_t0 --users-file data/users.txt --expected-users N

# Explicit network access; the builder itself only reads cached bars.
python scripts/cache_market.py --symbols-file data/symbols.txt \
  --cache-dir data/market_raw --start 2026-07-01 --end 2026-10-07
```

Market fetching includes SPY and QQQ and maps options to their underlying symbol. Supply every needed symbol; missing cached bars reduce market coverage. The cache uses Yahoo's public chart endpoint and may fail under network or service limits.

## 3. Build and validate rolling samples

```bash
# Calendar/cohort inspection: no model calls.
python build_rolling_dataset.py --t0-db data/observations_t0.db \
  --full-db data/observations_full.db --users-file data/users.txt \
  --out data/rolling_prediction --base-dir data/rolling_prediction/base_t0 \
  --market-cache data/market_raw --end 2026-10-07 --prediction-sessions 40 --dry-run

# Omit --dry-run to build. Declare coverage ONLY if execution extraction is complete.
python build_rolling_dataset.py --t0-db data/observations_t0.db \
  --full-db data/observations_full.db --users-file data/users.txt \
  --out data/rolling_prediction --base-dir data/rolling_prediction/base_t0 \
  --market-cache data/market_raw --end 2026-10-07 --prediction-sessions 40 \
  --execution-coverage-through 2026-10-07

python check_rolling_dataset.py --out data/rolling_prediction \
  --users-file data/users.txt --end 2026-10-07 --prediction-sessions 40
python audit_rolling_leakage.py --samples-dir data/rolling_prediction/samples
```

If your source is incomplete or ends earlier, shorten `--end` and declare the actual known-complete date. Omitting `--execution-coverage-through` preserves unknown labels for unverified inactivity. The audit's independent label recheck reads `data/observations_full.db` by default.

The builder refreshes changed profile dimensions and writes immutable daily profiles plus `samples/<user_id>/<prediction_date>.json`. It can resume existing profiles and samples at the current schema. Schema identifiers use the `financial_` namespace. Seed profiles and rolling samples with older identifiers must be regenerated with the current exporter and builder; do not bypass schema checks to mix experiments.

## 4. Run prediction and evaluation

```bash
# Validate prompt projections and report token sizes without model calls.
python predict_rolling.py --samples-dir data/rolling_prediction/samples \
  --all --dry-run

# Start with a small per-user run; use a fresh output directory for this smoke run.
python predict_rolling.py --samples-dir data/rolling_prediction/samples \
  --users-file data/users.txt --limit 1 --model "$LUNA_MODEL" --out outputs/smoke

# Full experiment; interrupted predictions can resume from per-user outputs.
python predict_rolling.py --samples-dir data/rolling_prediction/samples \
  --all --model "$LUNA_MODEL" --out outputs/models/model_a --resume

# Rebuild the complete summary if a run was sharded across processes.
python aggregate_prediction_shards.py --out outputs/models/model_a --model "$LUNA_MODEL"

# Compare multiple model directories under outputs/models/.
python compare_rolling_hierarchical.py --dir outputs/models
```

Per-user `pred_*.json` files store predictions, truth, cutoff metadata and latency. `summary.json` includes classification accuracy and class metrics, buy/sell Hit@5, MRR, Recall@5, candidate-pool oracle recall, observation coverage and leakage checks. Use separate output directories for different models/configurations. Only resume with the same model, schema and prompt settings. Dry-run validates inputs but does not establish endpoint compatibility or reproduction of paper scores.

## Optional rolling information ablations

```bash
for arm in full market holdings behavior7d trades profile; do
  python predict_rolling_ablation.py --arm "$arm" --all \
    --samples-dir data/rolling_prediction/samples --model "$LUNA_MODEL" \
    --out "data/rolling_prediction/predictions_ablation/$arm/$LUNA_MODEL" --resume
done
python score_ablation.py "$LUNA_MODEL"
```

Use a model identifier safe as a directory name for this command, or adapt the scoring path. Ablations operate on the projected prompt view and replace blocks with explicit withheld markers. They remove the recent information block; the long-term profile may still summarize historical information from the same stream. The scorer compares paired samples with McNemar's exact test and a bootstrap over users. Run all arms on the same cohort before scoring.

## Offline tests

All included tests use synthetic in-memory or temporary inputs; no private dataset or API key is needed.

```bash
for test in tests/test_rolling_*.py tests/test_point_in_time_datasource.py; do
  python "$test" || exit 1
done
```

## Repository contents and privacy

`app/` contains the dependency modules used by rolling construction and prediction, including the profile engine. `scripts/` supplies portable T0 and market-cache entry points. The top-level Python files build, validate, predict and score experiments; `tests/` contains synthetic contract tests. No Explanation evaluation/export scripts, data archives, pretrained user profiles, outputs, manuscript files or local environment files are included. Credentials are read from environment variables; private server addresses and source-machine paths have been replaced. Runtime inputs/outputs and credential files are ignored by Git.
