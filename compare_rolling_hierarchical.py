#!/usr/bin/env python3
"""Compare the hierarchical (v7) rolling-prediction groups.

The v7 task is three questions, not one, and they fail for different reasons:

    activity    trade vs no_trade. Under v7 labels a zero-execution day is only
                `no_trade` when both boundary books are known and unchanged, so
                the 509 v6 `hold` days that the position book contradicts or
                cannot confirm are simply absent here rather than scored as
                inactivity. That moves the majority-class baseline from .824 to
                about .758 -- a weaker baseline, and the honest one
    sides       buy_only / sell_only / buy_and_sell on the days a trade really
                happened. `buy_and_sell` is the largest trading-day class and the
                reason the flat `net_trade_action` label was abandoned: it
                collapsed every rebalancing day onto whichever side was larger
                in dollars
    tickers     which names, scored as answered AND restricted to the candidate
                pool, with the pool's own ceiling reported beside both

Everything is conditioned on the GOLD side, never on a group's own correctness.
Scoring stage 2 only where a group got stage 1 right gives every group a
different, self-selected denominator and rewards a timid stage 1: a group that
says `trade` only when it is surest gets an easier stage 2 than one whose recall
reaches the marginal days. Coverage is reported per group instead.

Usage
-----
    ./.venv/bin/python compare_rolling_hierarchical.py
    ./.venv/bin/python compare_rolling_hierarchical.py --out report.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINAL = HERE / "data" / "rolling_prediction" / "predictions" / "final_v7"

ACTIVITY = ("no_trade", "trade")
SIDES = ("buy_only", "sell_only", "buy_and_sell")


def load_group(d: Path) -> dict[str, dict]:
    """-> {sample_id: row}, from the per-user files only.

    Never `summary.json`: it is written by whichever shard exits last, and a
    crashed final pass leaves it stale while the per-user files stay correct.
    """
    rows = {}
    for f in sorted(d.glob("pred_*.json")):
        for row in json.loads(f.read_text(encoding="utf-8")).get("results") or []:
            rows[row["sample_id"]] = row
    return rows


def pairs(rows, ids, field, *, gold_in=None):
    """(sample_id, gold, predicted) where the label exists and was answered."""
    out = []
    for sid in ids:
        row = rows.get(sid)
        if not row:
            continue
        g = (row.get("gold") or {}).get(field)
        p = (row.get("predicted") or {}).get(field)
        if g is None or g == "unknown":
            continue
        if gold_in is not None and g not in gold_in:
            continue
        out.append((sid, g, p))
    return out


def metrics(triples, allowed):
    """Accuracy and per-class P/R/F1. An unanswered field counts as wrong.

    Nothing is dropped for being malformed: a group that answers nothing on the
    hard days would otherwise score like one that answers them correctly.
    """
    if not triples:
        return {"n": 0}
    gold = Counter(g for _, g, _ in triples)
    pred = Counter(p for _, _, p in triples if p is not None)
    hits = sum(1 for _, g, p in triples if g == p)
    f1s, per_class = [], {}
    for label in allowed:
        tp = sum(1 for _, g, p in triples if g == label and p == label)
        prec = tp / pred[label] if pred[label] else 0.0
        rec = tp / gold[label] if gold[label] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if gold[label]:
            f1s.append(f1)
        per_class[label] = {"support": gold[label], "predicted": pred[label],
                            "precision": round(prec, 4), "recall": round(rec, 4),
                            "f1": round(f1, 4)}
    return {"n": len(triples), "accuracy": round(hits / len(triples), 4),
            "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
            "majority_class_baseline": round(max(gold.values()) / len(triples), 4),
            "n_unanswered": sum(1 for _, _, p in triples if p is None),
            "true_distribution": dict(gold), "predicted_distribution": dict(pred),
            "per_class": per_class}


def bootstrap_f1(triples, label, *, draws=2000, seed=0):
    """Percentile CI for one class's F1.

    `buy_and_sell` has a support in the low hundreds, so a bare F1 invites a
    comparison the sample size does not support.
    """
    if not triples:
        return None
    rng = random.Random(seed)
    n, out = len(triples), []
    for _ in range(draws):
        sample = [triples[rng.randrange(n)] for _ in range(n)]
        tp = sum(1 for _, g, p in sample if g == label and p == label)
        fp = sum(1 for _, g, p in sample if g != label and p == label)
        fn = sum(1 for _, g, p in sample if g == label and p != label)
        out.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    out.sort()
    return {"f1_median": round(out[draws // 2], 4),
            "ci95": [round(out[int(.025 * draws)], 4),
                     round(out[int(.975 * draws)], 4)]}


def mcnemar(a_triples, b_triples):
    """Exact two-sided McNemar over the samples both groups scored."""
    a = {sid: (g, p) for sid, g, p in a_triples}
    b = {sid: (g, p) for sid, g, p in b_triples}
    shared = a.keys() & b.keys()
    a_only = sum(1 for s in shared if a[s][1] == a[s][0] and b[s][1] != b[s][0])
    b_only = sum(1 for s in shared if b[s][1] == b[s][0] and a[s][1] != a[s][0])
    n = a_only + b_only
    if not n:
        return {"n_shared": len(shared), "a_only_correct": 0, "b_only_correct": 0,
                "p_value": 1.0, "note": "right on the same samples"}
    k = min(a_only, b_only)
    return {"n_shared": len(shared), "a_only_correct": a_only,
            "b_only_correct": b_only,
            "p_value": round(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1))
                                 / (2 ** n)), 6)}


def ticker_layers(rows, ids, side, k=5):
    """The ticker question at three levels, plus the pool's own ceiling."""
    key, named_key = f"{side}_tickers", f"{side}_tickers_named"
    scored = [rows[s] for s in ids if s in rows
              and (rows[s].get("gold") or {}).get("trade_activity") == "trade"
              and (rows[s].get("gold") or {}).get(key)]
    n = len(scored)
    if not n:
        return {"n_scored": 0}

    def rank(pred_key):
        hits, rr, rec = 0, 0.0, []
        for row in scored:
            truth = set(row["gold"][key])
            # Fall back to the filtered list for rows written before the
            # predictor kept the unfiltered one.
            pred = ((row["predicted"].get(pred_key)
                     or row["predicted"].get(key)) or [])[:k]
            ranks = [i + 1 for i, t in enumerate(pred) if t in truth]
            hits += bool(ranks)
            rr += 1 / min(ranks) if ranks else 0
            rec.append(len(truth & set(pred)) / len(truth))
        return {f"hit@{k}": round(hits / n, 4), "mrr": round(rr / n, 4),
                f"recall@{k}": round(sum(rec) / n, 4)}

    reachable = sum(1 for r in scored
                    if set(r["gold"][key]) & set(r.get("candidate_tickers") or []))
    unnameable = sum(1 for r in scored
                     if not set(r["gold"][key]) <= set(r.get("candidate_tickers") or []))
    return {"n_scored": n, "as_answered": rank(named_key),
            "within_candidates": rank(key),
            "candidate_oracle_recall": round(reachable / n, 4),
            "gold_days_with_an_unnameable_ticker": unnameable,
            "share_with_an_unnameable_ticker": round(unnameable / n, 4),
            "days_the_model_used_an_out_of_pool_name": sum(
                1 for r in scored if r["predicted"].get("unnameable_tickers"))}


def contract_layers(rows, ids):
    """Well-formedness per layer. A ticker the pool never offered is not a fault."""
    got = [rows[s]["predicted"] for s in ids if s in rows]
    n = len(got)
    if not n:
        return {"n": 0}
    return {"n": n,
            "activity_parsed": round(sum(1 for p in got if p.get("trade_activity")) / n, 4),
            "sides_parsed": round(sum(1 for p in got if p.get("trade_sides")) / n, 4),
            "full_ticker_contract_valid": round(
                sum(1 for p in got if p.get("contract_valid")) / n, 4),
            "answers_naming_an_out_of_pool_ticker": sum(
                1 for p in got if p.get("unnameable_tickers"))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default=str(FINAL))
    ap.add_argument("--out", help="also write a markdown table here")
    ap.add_argument("--expected", type=int, default=0,
                    help="samples per group; 0 means the largest group")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.is_dir():
        print(f"no hierarchical predictions yet under {root}")
        return 1
    groups = {d.name: load_group(d) for d in sorted(root.iterdir()) if d.is_dir()}
    groups = {k: v for k, v in groups.items() if v}
    if not groups:
        print(f"no group has predictions yet under {root}")
        return 1

    common = set.intersection(*(set(v) for v in groups.values()))
    expected = args.expected or max(len(v) for v in groups.values())
    report = {
        "groups": sorted(groups),
        "task": "hierarchical: activity -> sides -> tickers, v7 labels",
        "scored_on": "samples answered by every group, conditioned on gold",
        "coverage": {k: {"answered": len(v), "expected": expected,
                         "share": round(len(v) / expected, 4)}
                     for k, v in groups.items()},
        "n_common_samples": len(common),
        "activity": {}, "sides_on_gold_trade_days": {}, "tickers": {},
        "contract_layers": {}, "pairwise_mcnemar": {},
    }

    for name, rows in groups.items():
        act = pairs(rows, common, "trade_activity")
        sd = pairs(rows, common, "trade_sides", gold_in=SIDES)
        report["activity"][name] = metrics(act, ACTIVITY)
        report["sides_on_gold_trade_days"][name] = dict(
            metrics(sd, SIDES),
            buy_and_sell_bootstrap=bootstrap_f1(sd, "buy_and_sell"))
        report["tickers"][name] = {s: ticker_layers(rows, common, s)
                                   for s in ("buy", "sell")}
        report["contract_layers"][name] = contract_layers(rows, common)

    for field, allowed, gold_in in (("trade_activity", ACTIVITY, None),
                                    ("trade_sides", SIDES, SIDES)):
        report["pairwise_mcnemar"][field] = {
            f"{a} vs {b}": mcnemar(pairs(groups[a], common, field, gold_in=gold_in),
                                   pairs(groups[b], common, field, gold_in=gold_in))
            for a, b in combinations(sorted(groups), 2)}

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.out:
        L = ["# Hierarchical rolling prediction (v7 labels)", "",
             f"Scored on the {len(common)} samples every group answered. Stage 2 is",
             "conditioned on the gold side, so the denominator is identical for",
             "every group.", "",
             "## Stage 1 -- trade vs no_trade", "",
             "| group | n | accuracy | trade recall | trade precision | macro-F1 | unanswered | coverage |",
             "|---|---|---|---|---|---|---|---|"]
        for name in sorted(groups):
            m = report["activity"][name]
            c = (m.get("per_class") or {}).get("trade") or {}
            L.append(f"| {name} | {m.get('n')} | {m.get('accuracy')} | "
                     f"{c.get('recall')} | {c.get('precision')} | "
                     f"{m.get('macro_f1')} | {m.get('n_unanswered')} | "
                     f"{report['coverage'][name]['share']} |")
        first = report["activity"][sorted(groups)[0]]
        L += [f"| *majority class* | {first.get('n')} | "
              f"{first.get('majority_class_baseline')} | 0.0 | | | | |", "",
              "## Stage 2 -- sides on gold trading days", "",
              "| group | n | accuracy | macro-F1 | buy_and_sell F1 [95% CI] | "
              "buy_only F1 | sell_only F1 |", "|---|---|---|---|---|---|---|"]
        for name in sorted(groups):
            m = report["sides_on_gold_trade_days"][name]
            pc = m.get("per_class") or {}
            bs = m.get("buy_and_sell_bootstrap") or {}
            ci = bs.get("ci95") or ["", ""]
            L.append(f"| {name} | {m.get('n')} | {m.get('accuracy')} | "
                     f"{m.get('macro_f1')} | "
                     f"{(pc.get('buy_and_sell') or {}).get('f1')} "
                     f"[{ci[0]}, {ci[1]}] | "
                     f"{(pc.get('buy_only') or {}).get('f1')} | "
                     f"{(pc.get('sell_only') or {}).get('f1')} |")
        L += ["", "## Stage 3 -- tickers", "",
              "`as answered` scores the names the model gave. `in pool` keeps only",
              "names the candidate pool contained. `oracle` is the share of gold",
              "days whose true ticker is in the pool at all: the ceiling for the",
              "`in pool` column, and the reason the two differ.", "",
              "| group | side | n | hit@5 as answered | hit@5 in pool | oracle | "
              "gold days unnameable |", "|---|---|---|---|---|---|---|"]
        for name in sorted(groups):
            for side in ("buy", "sell"):
                t = report["tickers"][name][side]
                if not t.get("n_scored"):
                    continue
                L.append(f"| {name} | {side} | {t['n_scored']} | "
                         f"{t['as_answered']['hit@5']} | "
                         f"{t['within_candidates']['hit@5']} | "
                         f"{t['candidate_oracle_recall']} | "
                         f"{t['share_with_an_unnameable_ticker']} |")
        L += ["", "## Answer well-formedness", "",
              "| group | activity parsed | sides parsed | full ticker contract | "
              "answers naming an out-of-pool ticker |", "|---|---|---|---|---|"]
        for name in sorted(groups):
            c = report["contract_layers"][name]
            L.append(f"| {name} | {c.get('activity_parsed')} | "
                     f"{c.get('sides_parsed')} | "
                     f"{c.get('full_ticker_contract_valid')} | "
                     f"{c.get('answers_naming_an_out_of_pool_ticker')} |")
        for field in ("trade_activity", "trade_sides"):
            L += ["", f"McNemar on {field} (exact, two-sided):", ""]
            for pair, st in report["pairwise_mcnemar"][field].items():
                L.append(f"- {pair}: p={st.get('p_value')} "
                         f"({st.get('a_only_correct')}/{st.get('b_only_correct')} "
                         f"discordant)")
        Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
