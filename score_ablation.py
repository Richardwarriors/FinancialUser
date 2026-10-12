#!/usr/bin/env python3
"""Cross-arm causal attribution for the rolling prediction task.

Paired, not independent. Every arm answers the SAME 1763 samples, so an
unpaired comparison of two accuracies throws away the pairing and is far less
sensitive than it needs to be. Each arm is compared against `full` with
McNemar's exact test on the discordant pairs, and the accuracy delta gets a
paired bootstrap interval resampled over USERS (not samples): a user's ~22 days
are correlated, so resampling samples would understate the interval.
"""
from __future__ import annotations
import json, random, sys
from collections import defaultdict
from math import comb
from pathlib import Path

B = Path("data/rolling_prediction/predictions_ablation")
MODEL = sys.argv[1] if len(sys.argv) > 1 else "gpt-5.6-luna"
ARMS = ("full", "market", "holdings", "behavior7d", "trades", "profile")
STAGES = {
    "S1_trade_activity":  ("trade_activity",  lambda g: g["trade_activity"] is not None),
    "S2_trade_sides":     ("trade_sides",     lambda g: g["trade_activity"] == "trade"
                                                        and g["trade_sides"] is not None),
    "S4_position_action": ("position_action", lambda g: g["trade_activity"] == "trade"
                                                        and g["position_action"] is not None),
}

def load(arm):
    out = {}
    for f in sorted((B / arm / MODEL).glob("pred_*.json")):
        for r in json.load(open(f)).get("results") or []:
            out[r["sample_id"]] = r
    return out

def mcnemar_exact(b, c):
    """Two-sided exact binomial on the discordant pairs. Exact rather than the
    chi-square approximation because b+c is small for S2/S4 (n<=275)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)

data = {a: load(a) for a in ARMS}
report = {"model": MODEL, "stages": {}}

for sname, (field, keep) in STAGES.items():
    full = data["full"]
    ids = [s for s, r in full.items() if keep(r["gold"])]
    base_hits = {s: full[s]["predicted"][field] == full[s]["gold"][field] for s in ids}
    base_acc = sum(base_hits.values()) / len(ids)
    st = {"n_scored": len(ids), "full_accuracy": round(base_acc, 4), "arms": {}}
    for arm in ARMS:
        if arm == "full":
            continue
        d = data[arm]
        ids_a = [s for s in ids if s in d]
        hits = {s: d[s]["predicted"][field] == d[s]["gold"][field] for s in ids_a}
        acc = sum(hits.values()) / len(ids_a)
        b = sum(1 for s in ids_a if base_hits[s] and not hits[s])   # full right, arm wrong
        c = sum(1 for s in ids_a if not base_hits[s] and hits[s])   # arm right, full wrong
        # paired bootstrap over users
        by_user = defaultdict(list)
        for s in ids_a:
            by_user[full[s]["user_id"]].append(s)
        users = list(by_user)
        rng = random.Random(20260920)
        deltas = []
        for _ in range(2000):
            samp = [u for _ in users for u in [rng.choice(users)]]
            sel = [s for u in samp for s in by_user[u]]
            if not sel:
                continue
            deltas.append(sum(hits[s] for s in sel) / len(sel)
                          - sum(base_hits[s] for s in sel) / len(sel))
        deltas.sort()
        lo, hi = deltas[int(.025 * len(deltas))], deltas[int(.975 * len(deltas))]
        st["arms"][arm] = {
            "accuracy": round(acc, 4),
            "delta_vs_full": round(acc - base_acc, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
            "mcnemar_b_full_right_arm_wrong": b,
            "mcnemar_c_arm_right_full_wrong": c,
            "mcnemar_p": round(mcnemar_exact(b, c), 6),
        }
    report["stages"][sname] = st

print(json.dumps(report, indent=2))
Path("data/rolling_prediction/predictions_ablation/ablation_report.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8")


# --- Conditional stage 2/4, added after the unconditional numbers proved to be
# an artifact -------------------------------------------------------------
# Removing the trade log made the model answer "trade" on 58.8% of days instead
# of 29.2% (true rate 24.8%). Stage 2 and 4 are scored only where the gold is a
# trade, so an arm that over-predicts trading gets MORE chances to be graded on
# them -- and `full`'s "no_trade" answers score `not_applicable`, which is always
# wrong there. That inflated the trades arm to +0.013 on S2 and +0.066 on S4,
# reversing the true sign. Conditioning on days where BOTH arms said "trade"
# compares like with like: S2 then reads -0.1105, not +0.0130.
def conditional(model=MODEL):
    D = {a: load(a) for a in ARMS}
    full = D["full"]
    gold_trade = [s for s, r in full.items()
                  if r["gold"]["trade_activity"] == "trade" and r["gold"]["trade_sides"]]
    out = {}
    for a in ARMS:
        if a == "full":
            continue
        d = D[a]
        both = [s for s in gold_trade
                if full[s]["predicted"]["trade_activity"] == "trade"
                and d[s]["predicted"]["trade_activity"] == "trade"]
        def acc(src, f, ss):
            v = [s for s in ss if src[s]["gold"][f] is not None]
            return sum(src[s]["predicted"][f] == src[s]["gold"][f] for s in v) / len(v) if v else None
        b4 = [s for s in both if full[s]["gold"]["position_action"] is not None]
        out[a] = {"n_both_said_trade": len(both),
                  "S2_full": round(acc(full, "trade_sides", both), 4),
                  "S2_arm": round(acc(d, "trade_sides", both), 4),
                  "S2_delta": round(acc(d, "trade_sides", both) - acc(full, "trade_sides", both), 4),
                  "S4_n": len(b4),
                  "S4_full": round(acc(full, "position_action", b4), 4),
                  "S4_arm": round(acc(d, "position_action", b4), 4),
                  "S4_delta": round(acc(d, "position_action", b4) - acc(full, "position_action", b4), 4)}
    return out


if __name__ == "__main__":
    rp = Path("data/rolling_prediction/predictions_ablation/ablation_report.json")
    rep = json.loads(rp.read_text())
    rep["conditional_on_both_predicting_trade"] = conditional()
    rep["predicted_trade_rate"] = {}
    D = {a: load(a) for a in ARMS}
    ids = [s for s, r in D["full"].items() if r["gold"]["trade_activity"] is not None]
    for a in ARMS:
        rep["predicted_trade_rate"][a] = round(
            sum(D[a][s]["predicted"]["trade_activity"] == "trade" for s in ids) / len(ids), 4)
    rep["predicted_trade_rate"]["GOLD"] = round(
        sum(D["full"][s]["gold"]["trade_activity"] == "trade" for s in ids) / len(ids), 4)
    rp.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print("wrote", rp)
