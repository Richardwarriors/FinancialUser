#!/usr/bin/env python3
"""Rebuild a full-cohort summary from per-user rolling prediction files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from predict_rolling import (
    PROMPT_FORMAT, TRADE_ACTIVITY, TRADE_SIDES, by_confidence,
    contract_layers, scores, ticker_scores,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", type=Path,
                        help="full-cohort summary whose sample IDs must match")
    args = parser.parse_args()

    files = sorted(args.out.glob("pred_*.json"))
    if not files:
        raise SystemExit("no per-user prediction files")
    rows = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model") != args.model:
            raise ValueError(f"model mismatch in {path}")
        user_rows = payload.get("results") or []
        if any(str(row["user_id"]) != str(payload["user_id"]) for row in user_rows):
            raise ValueError(f"user mismatch in {path}")
        rows.extend(user_rows)

    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate sample IDs across per-user files")
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        reference_ids = {row["sample_id"] for row in reference["results"]}
        if set(sample_ids) != reference_ids:
            raise ValueError(f"reference mismatch: {len(reference_ids - set(sample_ids))} "
                             f"missing, {len(set(sample_ids) - reference_ids)} extra")
    rows.sort(key=lambda row: row["sample_id"])
    observed = [row for row in rows if row["gold"]["trade_activity"] is not None]
    trade = [row for row in observed if row["gold"]["trade_activity"] == "trade"]
    summary = {
        "model": args.model,
        "prompt_format": PROMPT_FORMAT,
        "n_samples": len(rows),
        "n_users": len({row["user_id"] for row in rows}),
        "n_errors": 0 if args.reference else None,
        "prompt_tokens_approx": {
            "max": max(row["prompt_tokens_approx"] for row in rows),
            "mean": round(sum(row["prompt_tokens_approx"] for row in rows) / len(rows)),
        },
        "leakage_violations": sum(row["leakage_clean"] is False for row in rows),
        "trade_activity": scores(rows, "trade_activity", TRADE_ACTIVITY),
        "trade_sides_on_true_trade_days": scores(trade, "trade_sides", TRADE_SIDES[:3]),
        "ticker_ranking": {side: ticker_scores(rows, side) for side in ("buy", "sell")},
        "contract_layers": contract_layers(rows),
        "accuracy_by_confidence": {
            "trade_activity": by_confidence(rows, "trade_activity"),
            "trade_sides": by_confidence(trade, "trade_sides"),
        },
        "coverage": {"observed": len(observed), "unknown": len(rows) - len(observed),
                     "true_trade_days": len(trade)},
        "corpus_schema_version": "financial_rolling_prediction_en_v7",
        "aggregation_source": "per-user pred_*.json files",
    }
    output = args.out / "summary.json"
    output.write_text(json.dumps({"summary": summary, "results": rows},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("model", "n_samples", "n_users", "coverage", "trade_activity")},
                     indent=2))


if __name__ == "__main__":
    main()
