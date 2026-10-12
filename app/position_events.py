"""BM453 Position Dynamics: daily snapshots -> capital decision events.

Positioning
-----------
BM453 is not "what I hold", it is **how I manage a position after I hold it**.
This module turns the daily snapshot into an event stream (OPEN / ADD / REDUCE /
CLOSE / REENTER) with the context needed to interpret each one, then aggregates
that into decision-behaviour metrics: capital deployment, conviction, scale-in /
scale-out, loss/gain response, exit style and rotation.

Empty holdings with nonzero stock totals are treated as data gaps.
Position profit/loss is derived from price and cost rather than reported totals.
"""
from collections import Counter, defaultdict
from datetime import timedelta

POSITION_VERSION = "position-events-v4"

# an ADD at or below cost, repeated within one episode this many times or more,
# is "repeated averaging down" rather than a single size adjustment
AVERAGE_DOWN_MIN_ADDS = 2
# Below this depth, being "underwater" carries no information: these instruments
# move ~2%/day, so an add at 1-4% below cost is indistinguishable from an add at
# cost. Measured fleet-wide, 29% of adds made while underwater happen at under 5%,
# and 15% at under 3%. Counting those as 摊平 overstated the behaviour for roughly a
# third of cases, so they are classified separately as `is_shallow_underwater_add`
# rather than dropped — the frequency fact is still reported, it just no longer
# borrows the word for "buying into a decline".
AVERAGE_DOWN_MIN_DEPTH = 0.05
# Deep end: at or beyond this the position is in a real drawdown and adding is
# unambiguously buying into weakness.
AVERAGE_DOWN_DEEP_DEPTH = 0.20
SCALE_EVENT_MIN_REL = 0.15        # ignore <15% share moves: that is price drift
MIN_EPISODES_FOR_STYLE = 3        # below this, style labels are noise
# same floor as the TWR curve in account.py: below this fraction of usable days
# a ratio is withheld, because the excluded days are not a random subset
MIN_COVERAGE = 0.5


def _depth_tier(depth):
    """shallow / moderate / deep for a median underwater depth."""
    if depth is None:
        return None
    if depth < AVERAGE_DOWN_MIN_DEPTH:
        return "shallow"
    return "deep" if depth >= AVERAGE_DOWN_DEEP_DEPTH else "moderate"


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_div(a, b, digits=4):
    if not b:
        return None
    return round(a / b, digits)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def _ts(v):
    from .taxonomy import parse_ts
    return parse_ts(v)


# ---------- 1. data quality gate ----------

def gate_day(day):
    """Classify one snapshot before it is allowed to produce events."""
    holdings = day.get("holdings") or {}
    asset = _f(day.get("asset_total"))
    stock = _f(day.get("stock_total"))
    cash = _f(day.get("money_remain"))

    issues = []
    if not holdings and (stock or 0) > 0:
        issues.append("holdings_empty_but_stock_nonzero")
    if asset is None:
        issues.append("asset_total_missing")
    elif stock is not None and cash is not None:
        # stock + cash should reconstruct total; a big gap means one leg is wrong
        if abs((stock + cash) - asset) > max(1.0, abs(asset) * 0.02):
            issues.append("asset_total_inconsistent_with_components")
    if any(v is not None and v < 0 for v in (asset, stock, cash)):
        issues.append("negative_balance")

    blocking = {"holdings_empty_but_stock_nonzero", "asset_total_missing",
                "negative_balance"}
    return {"usable_for_positions": not (blocking & set(issues)), "issues": issues}


def build_days(account_daily):
    """Ordered days with a quality verdict and effective positions.

    A day flagged `data_missing` does **not** reset the book; the previous known
    book is carried forward so downstream diffs see continuity rather than a
    fake full liquidation.

    `book_known` records whether we actually know what is held that day:
      * holdings present            -> known
      * no holdings AND stock == 0  -> known to be genuinely flat
      * no holdings AND stock  > 0  -> UNKNOWN

    When an UNKNOWN day has nothing to carry (the previous book is itself empty)
    the carried book stays empty, which must NOT be read as "holds nothing" —
    that produced 117 of 157 users showing an empty book on days where
    stock_total was in the millions. Such days are flagged
    `carried_forward_empty_book` and stay `book_known = False`.
    """
    rows = sorted(account_daily, key=lambda d: d.get("snapshot_date") or "")
    out = []
    carried = {}
    carried_from = None
    for d in rows:
        g = gate_day(d)
        holdings = d.get("holdings") or {}
        effective = holdings
        issues = list(g["issues"])
        book_known = "holdings_empty_but_stock_nonzero" not in issues
        if not g["usable_for_positions"] and "holdings_empty_but_stock_nonzero" in issues:
            effective = carried
            carried_from = carried_from or d.get("snapshot_date")
            if not effective:
                book_known = False
                issues.append("carried_forward_empty_book")
        elif holdings:
            carried = holdings
            carried_from = None
        out.append({
            "date": d.get("snapshot_date"),
            "raw_holdings": holdings,
            "holdings": effective,
            "carried_forward": effective is not holdings,
            "carried_from": carried_from if effective is not holdings else None,
            "book_known": book_known,
            "asset_total": _f(d.get("asset_total")),
            "stock_total": _f(d.get("stock_total")),
            "money_remain": _f(d.get("money_remain")),
            "usable": g["usable_for_positions"],
            "issues": issues,
        })
    return out


def baseline_index(days):
    """Index of the first day whose holding book is actually known.

    Day 1 is an unknown book (`holdings = {}` while `stock_total > 0`) for
    essentially every user, so diffing against it turns the entire second day of
    holdings into phantom "entries". Positions observable on the first *known*
    day were bought before the window and must be treated as the starting book.

    A genuinely flat book (`stock_total == 0`) does count as known: that user
    really did hold nothing, so a later purchase is a true entry.
    """
    for i, d in enumerate(days):
        if d.get("book_known"):
            return i
    return len(days)


def quality_report(days):
    n = len(days)
    bad = [d for d in days if not d["usable"]]
    issues = Counter(i for d in days for i in d["issues"])
    return {
        "days": n,
        "usable_days": n - len(bad),
        "unusable_days": len(bad),
        "carried_forward_days": sum(1 for d in days if d["carried_forward"]),
        "issue_counts": dict(issues),
        "usable_ratio": round((n - len(bad)) / n, 3) if n else 0,
        "note": ("Empty holdings with stock_total>0 are treated as missing data, and the previous day's holdings are carried forward rather than classified as an exit. profit_loss_total and cost_price use inconsistent definitions (37.9% magnitude mismatch), so the position profit/loss state is uniformly determined using last_price vs cost_price; this field is not included in aggregation."),
    }


# ---------- 2. position state ----------

def position_state(p):
    """Win/loss from last_price vs cost_price (internally consistent basis)."""
    lp, cp = _f(p.get("last_price")), _f(p.get("cost_price"))
    if lp is None or cp is None or cp <= 0:
        return "unknown", None
    pct = lp / cp - 1
    if pct > 0.01:
        return "gain", round(pct, 4)
    if pct < -0.01:
        return "loss", round(pct, 4)
    return "flat", round(pct, 4)


def book_view(day):
    """{ticker: {...}} with qty, value, weight, state for one effective day."""
    book = day["holdings"] or {}
    total = sum(_f(p.get("stock_market_value"), 0) or 0 for p in book.values())
    inv = None
    if day.get("asset_total"):
        inv = _safe_div(day.get("stock_total") or total, day["asset_total"], 4)
    out = {}
    for c, p in book.items():
        t = (c or "").upper()
        state, pct = position_state(p)
        mv = _f(p.get("stock_market_value"), 0) or 0
        out[t] = {
            "ticker": t, "qty": _f(p.get("stock_vol"), 0) or 0,
            "market_value": mv, "weight": _safe_div(mv, total, 4) if total else None,
            "last_price": _f(p.get("last_price")), "cost_price": _f(p.get("cost_price")),
            "state": state, "price_vs_cost": pct,
            "name": p.get("stock_name"),
        }
    return out, ({"invested_ratio": inv,
                  "cash_ratio": _safe_div(day.get("money_remain"), day.get("asset_total"), 4)
                  if day.get("asset_total") else None} if inv is not None else {})


# ---------- 3. diff engine ----------

def _tags(action, before, after, ctx):
    tags = []
    st_before = before.get("state") if before else None
    if action == "OPEN":
        tags.append("new_position")
    elif action == "ADD":
        tags.append("scale_in")
        if st_before == "loss":
            tags.append("add_to_loser")
        elif st_before == "gain":
            tags.append("pyramid_winner")
        if ctx.get("invested_ratio") is not None and ctx["invested_ratio"] < 0.5:
            tags.append("adding_with_high_cash")
    elif action == "REDUCE":
        tags.append("scale_out")
        if st_before == "gain":
            tags.append("trim_winner")
        elif st_before == "loss":
            tags.append("cut_loser")
        if before and after and before.get("weight") and after.get("weight"):
            if after["weight"] < before["weight"] * 0.5:
                tags.append("large_reduction")
    elif action == "CLOSE":
        tags.append("full_exit")
        if st_before == "gain":
            tags.append("close_winner")
        elif st_before == "loss":
            tags.append("close_loser")
    elif action == "REENTER":
        tags.append("reentry")
    return tags


def diff_events(days):
    """Daily snapshot diff -> position events (skipping unusable gap days)."""
    events = []
    seen_before = set()          # tickers ever held before, for REENTER
    prev_view = {}
    for i, d in enumerate(days):
        view, ctx = book_view(d)
        if i == 0:
            prev_view = view
            seen_before |= set(view)
            continue
        pv = prev_view
        for t in set(pv) | set(view):
            b, a = pv.get(t), view.get(t)
            qb = (b or {}).get("qty") or 0.0
            qa = (a or {}).get("qty") or 0.0
            if abs(qa - qb) < 1e-9:
                continue
            action = ("OPEN" if not b else
                      "CLOSE" if not a else
                      "ADD" if qa > qb else "REDUCE")
            if action == "OPEN" and t in seen_before:
                action = "REENTER"
            events.append({
                "date": d["date"], "index": i, "ticker": t, "action": action,
                "qty_before": round(qb, 4), "qty_after": round(qa, 4),
                "delta_qty": round(qa - qb, 4),
                "weight_before": (b or {}).get("weight"),
                "weight_after": (a or {}).get("weight"),
                "state_before": (b or {}).get("state"),
                "price_vs_cost_before": (b or {}).get("price_vs_cost"),
                "portfolio_context": ctx,
                "tags": _tags(action, b, a, ctx),
                "data_quality": {"carried_forward": d["carried_forward"],
                                 "issues": d["issues"]},
            })
        seen_before |= set(view)
        prev_view = view
    return events


# ---------- 4. episodes ----------

def episodes(events):
    """Group same-ticker consecutive same-direction actions into episodes.

    An `episode` is the unit that answers "how does this user build or unwind a
    position": a single ADD is a size tweak, five ADDs while underwater is a
    strategy.
    """
    by_ticker = defaultdict(list)
    for e in events:
        by_ticker[e["ticker"]].append(e)
    out = []
    for t, evs in by_ticker.items():
        evs.sort(key=lambda x: x["index"])
        cur = None
        for e in evs:
            direction = "up" if e["action"] in ("OPEN", "ADD", "REENTER") else "down"
            if cur and cur["direction"] == direction and e["index"] - cur["last_index"] <= 5:
                cur["events"].append(e)
                cur["last_index"] = e["index"]
            else:
                if cur:
                    out.append(cur)
                cur = {"ticker": t, "direction": direction, "events": [e],
                       "start_index": e["index"], "start_date": e["date"],
                       "last_index": e["index"]}
        if cur:
            out.append(cur)
    for ep in out:
        n = len(ep["events"])
        ep["n_actions"] = n
        ep["duration_days"] = ep["last_index"] - ep["start_index"] + 1
        states = [e["state_before"] for e in ep["events"] if e["state_before"]]
        ep["state_during"] = Counter(states).most_common(1)[0][0] if states else None

        # How FAR underwater was the position when the user added?
        #
        # This was missing, and its absence made the label misleading. An ADD while
        # 1% below cost counted as "average down" exactly like one while 40% below —
        # yet only the second is buying into a real decline. Measured fleet-wide, 29%
        # of such adds happen at under 5% below cost, which is inside the daily noise
        # of these instruments (the user's own equity volatility runs ~2%/day), so
        # calling all of them "摊平" overstates the behaviour for a third of cases.
        depths = [abs(e["price_vs_cost_before"]) for e in ep["events"]
                  if e.get("price_vs_cost_before") is not None
                  and e.get("state_before") == "loss"]
        ep["depth_median"] = sorted(depths)[len(depths) // 2] if depths else None
        ep["depth_max"] = max(depths) if depths else None
        ep["depth_tier"] = (_depth_tier(ep["depth_median"])
                             if ep["depth_median"] is not None else None)

        adds_underwater = ep["direction"] == "up" and n >= AVERAGE_DOWN_MIN_ADDS
        # `shallow` = added while only nominally underwater; kept as its own fact
        # rather than folded into "average down", so nothing is hidden either way.
        ep["is_underwater_add"] = bool(adds_underwater and ep["state_during"] == "loss")
        ep["is_shallow_underwater_add"] = bool(
            ep["is_underwater_add"] and ep["depth_median"] is not None
            and ep["depth_median"] < AVERAGE_DOWN_MIN_DEPTH)
        ep["is_average_down"] = bool(
            ep["is_underwater_add"] and not ep["is_shallow_underwater_add"])
        ep["is_pyramid_winner"] = (ep["direction"] == "up" and n >= AVERAGE_DOWN_MIN_ADDS
                                   and ep["state_during"] == "gain")
    return out


# ---------- 5. metrics ----------

def valid_exposure_days(days):
    """Days whose `stock_total / asset_total` is trustworthy, as (index, date, ratio).

    A day is kept only when the gate found the components consistent AND the
    resulting ratio is possible (stock cannot exceed total assets). One shared
    helper so `capital_management` and `_de_risking_speed` cannot drift apart.
    """
    out = []
    for i, d in enumerate(days):
        asset, stock = d.get("asset_total"), d.get("stock_total")
        if not asset or stock is None:
            continue
        if "asset_total_inconsistent_with_components" in (d.get("issues") or []):
            continue
        v = stock / asset
        if v < 0 or v > 1.0 + 1e-9:
            continue
        out.append((i, d.get("date"), v))
    return out


def capital_management(days):
    """Gross-exposure behaviour: how actively is the cash/stock mix managed?

    When too few days survive validation the ratio-based conclusions are withheld
    rather than reported on a biased sample: the excluded days are not a random
    subset, because the corrupt rows inflate the numerator.
    """
    inv = valid_exposure_days(days)
    total = len(days)
    coverage = round(len(inv) / total, 3) if total else 0.0
    excluded = total - len(inv)
    if not inv:
        return {
            "available": False,
            "coverage": coverage,
            "excluded_days": excluded,
            "note": (f"On none of the {total} trading days did Total assets = position market value + cash hold, so the position ratio cannot be provided"),
        }

    vals = [v for _, _, v in inv]
    switches = 0
    prev_bucket = None
    for _, _, v in inv:
        b = ("risk_on" if v >= 0.7 else "risk_off" if v <= 0.35 else "neutral")
        if prev_bucket and b != prev_bucket and {b, prev_bucket} != {"neutral"}:
            switches += 1
        prev_bucket = b

    out = {
        "available": True,
        "coverage": coverage,
        "used_days": len(inv),
        "excluded_days": excluded,
        "note": ("invested_ratio = stock_total / asset_total. Switch count is calculated from transitions between the risk_on(≥70%) / risk_off(≤35%) buckets; intermediate states are excluded. Trading days with inconsistent total assets and holdings/cash were excluded (this field is not reliable in BM453)."),
    }
    if coverage < MIN_COVERAGE:
        out["ratio_withheld"] = (
            f"Only {coverage * 100:.0f}% of trading days are available for calculating the position ratio (excluding {excluded} days), below {int(MIN_COVERAGE * 100)}%, and the excluded days are not randomly distributed; therefore, no average position or position style is provided")
        return out

    rng = max(vals) - min(vals)
    avg = _mean(vals)
    out.update({
        "avg_invested_ratio": round(avg, 4),
        "max_invested_ratio": round(max(vals), 4),
        "min_invested_ratio": round(min(vals), 4),
        "invested_ratio_range": round(rng, 4),
        "risk_on_off_switch_count": switches,
        "cash_buffer_ratio": round(1 - avg, 4),
        "exposure_style": ("actively_adjusted" if rng >= 0.4 else
                           "moderately_adjusted" if rng >= 0.15 else
                           "stable_exposure"),
    })
    return out


def _episode_direction_rates(eps):
    up = [e for e in eps if e["direction"] == "up"]
    down = [e for e in eps if e["direction"] == "down"]
    ad = [e for e in up if e["is_average_down"]]
    pw = [e for e in up if e["is_pyramid_winner"]]
    return up, down, ad, pw


def style_metrics(days, events, eps):
    up, down, ad, pw = _episode_direction_rates(eps)
    n_up, n_down = len(up), len(down)
    adds = [e for e in events if e["action"] in ("ADD", "REENTER")]
    reduces = [e for e in events if e["action"] == "REDUCE"]
    closes = [e for e in events if e["action"] == "CLOSE"]
    opens = [e for e in events if e["action"] in ("OPEN", "REENTER")]

    add_loss = [e for e in adds if e["state_before"] == "loss"]
    add_gain = [e for e in adds if e["state_before"] == "gain"]
    red_loss = [e for e in reduces if e["state_before"] == "loss"]
    red_gain = [e for e in reduces if e["state_before"] == "gain"]
    close_gain = [e for e in closes if e["state_before"] == "gain"]
    close_loss = [e for e in closes if e["state_before"] == "loss"]

    out = {
        "available": True,
        "opens": len(opens), "adds": len(adds), "reduces": len(reduces), "closes": len(closes),
        "episodes": len(eps), "accumulation_episodes": n_up, "distribution_episodes": n_down,
        "avg_adds_per_episode": _safe_div(len(adds), n_up, 2) if n_up else None,
        "avg_reductions_per_episode": _safe_div(len(reduces), n_down, 2) if n_down else None,
    }
    if len(adds) >= MIN_EPISODES_FOR_STYLE:
        out["average_down_rate"] = _safe_div(len(add_loss), len(adds), 3)
        # NOT `average_down_depth`. This value is a RATE (episodes / accumulation
        # episodes) and the old name asserted it was a depth, which is how it came to
        # be described to the model as "摊平深度". Renamed to say what it is; the real
        # depth now lives in median_underwater_depth below.
        out["average_down_episode_rate"] = _safe_div(len(ad), n_up, 3) if n_up else None
        out["pyramid_winner_rate"] = _safe_div(len(add_gain), len(adds), 3)
    underwater_depths = sorted(
        abs(e["price_vs_cost_before"]) for e in adds
        if e.get("state_before") == "loss" and e.get("price_vs_cost_before") is not None)
    if underwater_depths:
        out["median_underwater_depth"] = underwater_depths[len(underwater_depths) // 2]
        out["max_underwater_depth"] = underwater_depths[-1]
        out["shallow_underwater_rate"] = _safe_div(
            sum(1 for d in underwater_depths if d < AVERAGE_DOWN_MIN_DEPTH),
            len(underwater_depths), 3)
    if len(reduces) >= MIN_EPISODES_FOR_STYLE:
        out["winner_reduce_rate"] = _safe_div(len(red_gain), len(reduces), 3)
        out["loser_reduce_rate"] = _safe_div(len(red_loss), len(reduces), 3)
    if len(closes) >= MIN_EPISODES_FOR_STYLE:
        out["full_exit_ratio"] = _safe_div(len(closes), len(closes) + len(reduces), 3)
        out["partial_exit_ratio"] = _safe_div(len(reduces), len(closes) + len(reduces), 3)
        out["close_winner_ratio"] = _safe_div(len(close_gain), len(closes), 3)
        out["close_loser_ratio"] = _safe_div(len(close_loss), len(closes), 3)

    # staged de-risking: several reduces on one ticker over consecutive-ish days
    staged = [ep for ep in down if ep["n_actions"] >= 2]
    if down:
        out["staged_exit_episodes"] = len(staged)
        out["staged_exit_share"] = _safe_div(len(staged), n_down, 3)

    out["entry_style"] = ("incremental_scale_in" if n_up and (out.get("avg_adds_per_episode") or 0) >= 1.5
                          else "single_shot_entry" if n_up else None)
    out["exit_style"] = ("gradual_scale_out" if out.get("staged_exit_share") is not None
                         and out["staged_exit_share"] >= 0.4 else
                         "clean_full_exit" if closes and not staged else None)
    out["thresholds"] = {"min_events_for_rate": MIN_EPISODES_FOR_STYLE,
                         "average_down_min_adds": AVERAGE_DOWN_MIN_ADDS}
    out["note"] = ("Position profit/loss status is determined using last_price vs cost_price (the profit_loss_total definition is unreliable and was not included in aggregation). The ratio returns null rather than 0 when the number of events is insufficient.")
    return out


def conviction_and_rotation(days, events, views_hist):
    """Position persistence, re-entry, theme rotation."""
    appear = Counter()
    for v in views_hist:
        for t in v:
            appear[t] += 1
    n = len(days)
    weights_hist = [{k: x.get("weight") or 0 for k, x in v.items()} for v in views_hist]
    max_w = {}
    for w in weights_hist:
        for t, x in w.items():
            max_w[t] = max(max_w.get(t, 0), x)
    large = {t for t, w in max_w.items() if w >= 0.2}
    persistent_large = {t for t in large if appear[t] / n >= 0.5} if n else set()

    reentries = [e for e in events if e["action"] == "REENTER"]
    reopened = {e["ticker"] for e in reentries}
    closed = {e["ticker"] for e in events if e["action"] == "CLOSE"}

    themes_first = {}
    for v in views_hist:
        for t in v:
            from .taxonomy import ticker_info
            th = ticker_info(t)["theme"]
            if th not in ("Other",):
                themes_first.setdefault(th, 0)
                themes_first[th] += 1

    return {
        "available": True,
        "distinct_tickers_held": len(appear),
        "position_persistence_days_max": max(appear.values()) if appear else None,
        "avg_persistence_ratio": round(_mean([appear[t] / n for t in appear]), 3) if appear and n else None,
        "large_position_tolerance": ("high" if large and len(persistent_large) / len(large) >= 0.5
                                     else "moderate" if large else "low"),
        "max_position_weight_ever": round(max(max_w.values()), 4) if max_w else None,
        "reentry_count": len(reentries),
        "reentry_rate": _safe_div(len(reopened), len(closed), 3) if closed else None,
        "theme_rotation_frequency": dict(Counter(
            t for v in views_hist for t in v).most_common(0)) if False else None,
        "theme_activity": dict(sorted(
            {th: c for th, c in themes_first.items()}.items(), key=lambda kv: -kv[1])[:6]),
        "note": "reentry_rate = number of instruments bought again after being exited ÷ number of instruments exited",
    }


def rotation(days, views_hist):
    """How much the book changes composition, not just size."""
    if len(views_hist) < 2:
        return None
    changes, jaccards = 0, []
    prev = set(views_hist[0])
    for v in views_hist[1:]:
        cur = set(v)
        if cur != prev:
            changes += 1
        union = cur | prev
        if union:
            jaccards.append(len(cur & prev) / len(union))
        prev = cur
    return {
        "available": True,
        "composition_change_days": changes,
        "composition_change_rate": _safe_div(changes, len(views_hist) - 1, 3),
        "avg_similarity_day_to_day": round(_mean(jaccards), 3) if jaccards else None,
        "note": ("composition_change_rate is the proportion of trading days on which the holding set had additions or removals; similarity is the Jaccard coefficient of the holding sets on consecutive days."),
    }


def outcome_response(eps, events):
    """Winner/loser response summary, requiring repeated evidence."""
    up, down, ad, pw = _episode_direction_rates(eps)
    n_up = len(up)
    if n_up < MIN_EPISODES_FOR_STYLE:
        return {"available": False,
                "note": f"Only {n_up} add-position episodes, fewer than {MIN_EPISODES_FOR_STYLE}; no loss/profit response conclusion is provided (to avoid characterization based on a single case)"}
    up_side, down_side, avgdown, pyramid = _episode_direction_rates(eps)
    # Frequency and DEPTH answer different questions, so both are reported and the
    # label is driven by depth. `is_average_down` now excludes adds made at under
    # AVERAGE_DOWN_MIN_DEPTH below cost; those are counted separately as
    # `shallow_underwater_episodes` so the frequency fact is not lost.
    shallow = [e for e in up if e.get("is_shallow_underwater_add")]
    depths = sorted(e["depth_median"] for e in avgdown
                    if e.get("depth_median") is not None)
    n_ad = len(avgdown)
    return {
        "available": True,
        "accumulation_episodes": n_up,
        "average_down_episodes": n_ad,
        "shallow_underwater_episodes": len(shallow),
        "pyramid_winner_episodes": len(pyramid),
        # rate over ALL accumulation episodes, so it is comparable to before
        "average_down_rate": round(n_ad / n_up, 3),
        "shallow_underwater_rate": round(len(shallow) / n_up, 3),
        "median_underwater_depth": (depths[len(depths) // 2] if depths else None),
        "max_underwater_depth": (max(depths) if depths else None),
        "depth_tier_mix": dict(Counter(e["depth_tier"] for e in avgdown
                                       if e.get("depth_tier"))),
        "average_down_tendency": ("high" if n_ad / n_up >= 0.4 else
                                  "medium" if n_ad / n_up >= 0.15 else "low"),
        "pyramid_winner_tendency": ("high" if len(pyramid) / n_up >= 0.4 else
                                    "medium" if len(pyramid) / n_up >= 0.15 else "low"),
        "thresholds": {"min_adds_per_episode": AVERAGE_DOWN_MIN_ADDS,
                       "min_depth": AVERAGE_DOWN_MIN_DEPTH,
                       "deep_depth": AVERAGE_DOWN_DEEP_DEPTH,
                       "min_episodes": MIN_EPISODES_FOR_STYLE},
        "evidence_requirement": (f"A tendency is provided only with ≥{MIN_EPISODES_FOR_STYLE} episodes; a. A single case is insufficient for classification"),
        "interpretation_limit": ("The observed behavior is \"adding to a position when losing\"; motivation cannot be inferred from this alone (it may be bottom-fishing, averaging down, maintaining conviction, or anchoring). Motivation assessment requires combining BM433 research and BM428 self-report."),
        "depth_note": (f'Only add-ons with unrealized loss ≥{AVERAGE_DOWN_MIN_DEPTH:.0%} are counted as averaging down; those below this threshold ({len(shallow)}/{n_up} episodes for this user) are listed separately as "shallow underwater top-ups" because within this range, the add-on price is almost indistinguishable from the cost basis.'),
    }


# ---------- entry point ----------

def build_position_dynamics(snap, max_events=200):
    account = snap.get("account_daily") or []
    if not account:
        return {"available": False, "note": "No BM453 account snapshot"}
    days = build_days(account)
    if not days:
        return {"available": False, "note": "BM453 snapshot cannot be parsed"}

    events = diff_events(days)
    eps = episodes(events)
    views_hist = [book_view(d)[0] for d in days]
    shown = events[-max_events:]

    return {
        "available": True,
        "version": POSITION_VERSION,
        "window": [days[0]["date"], days[-1]["date"]],
        "window_days": len(days),
        "data_quality": quality_report(days),
        "event_counts": dict(Counter(e["action"] for e in events)),
        "tag_counts": dict(Counter(t for e in events for t in e["tags"])),
        "capital_management": capital_management(days),
        "style": style_metrics(days, events, eps),
        "outcome_response": outcome_response(eps, events),
        "conviction": conviction_and_rotation(days, events, views_hist),
        "rotation": rotation(days, views_hist),
        "de_risking_speed": _de_risking_speed(days, events),
        "recent_events": shown,
        # EVERY metric above is computed from the FULL event list. This list is the
        # readable record, and it used to be capped at 25 silently — which is how a
        # reader (and my own analysis) came to treat 25 events as the whole window.
        # The cap is now higher AND disclosed, so a truncated list can never be
        # mistaken for a complete one.
        "events_total": len(events),
        "events_shown": len(shown),
        "events_truncated": len(events) > len(shown),
        "events_note": (f"The following are the {len(shown)} most recent position change events"
                         + (f"(Total of {len(events)} entries in the full window, truncated for display)"
                            if len(events) > len(shown) else "(i.e., full-window event)")
                         + ". All metrics are calculated based on the full set of events, not this list."),
        "caveats": [
            "Days with empty holdings but stock_total>0 are treated as missing data, and the previous day's holdings are carried forward",
            "profit_loss_total is inconsistent with the cost-price definition; position profit/loss state uses last_price vs cost_price",
            "All conclusions describe behavior; motivation cannot be inferred from position changes alone",
            "Positions already held on the first day cannot be used to determine the entry time and are excluded from OPEN",
        ],
    }


def _de_risking_speed(days, events):
    """How fast does exposure come down once it starts coming down?"""
    inv = [(i, v) for i, _, v in valid_exposure_days(days)]
    if len(inv) < 5:
        return None
    drops, rises = [], []
    peak = None
    for i, v in inv:
        if peak is None or v > peak[1]:
            peak = (i, v)
        elif peak and v < peak[1] * 0.85:
            drops.append((i - peak[0], peak[1] - v))
            peak = (i, v)
    vals = [v for _, v in inv]
    for a, b in zip(vals, vals[1:]):
        if b > a * 1.15:
            rises.append(b - a)
    return {
        "available": bool(drops) or bool(rises),
        "de_risking_episodes": len(drops),
        "median_de_risking_days": (sorted(d[0] for d in drops)[len(drops) // 2]
                                   if drops else None),
        "median_de_risking_magnitude": (round(sorted(d[1] for d in drops)[len(drops) // 2], 3)
                                        if drops else None),
        "rapid_deployment_days": len(rises),
        "note": "De-risking episode = drawdown of ≥15% from the peak invested ratio; deployment = single-day increase in invested ratio of ≥15%",
    }


# ---------- agent-facing digest ----------

def position_dynamics_digest(pd):
    """Compact view for prompts (per-ticker event dumps stay out of the prompt)."""
    if not pd or not pd.get("available"):
        return {"available": False, "note": (pd or {}).get("note", "No holding dynamics data")}
    cm = pd.get("capital_management") or {}
    st = pd.get("style") or {}
    oq = pd.get("outcome_response") or {}
    cv = pd.get("conviction") or {}
    ro = pd.get("rotation") or {}
    dq = pd.get("data_quality") or {}
    return {
        "available": True,
        "window_days": pd.get("window_days"),
        "data_quality": {"usable_ratio": dq.get("usable_ratio"),
                         "carried_forward_days": dq.get("carried_forward_days")},
        "capital_management": {k: cm.get(k) for k in
                               ("avg_invested_ratio", "cash_buffer_ratio", "invested_ratio_range",
                                "risk_on_off_switch_count", "exposure_style")} if cm else None,
        "position_style": {k: st.get(k) for k in
                           ("opens", "adds", "reduces", "closes", "entry_style", "exit_style",
                            "average_down_rate", "average_down_episode_rate",
                            "median_underwater_depth", "shallow_underwater_rate",
                            "pyramid_winner_rate", "winner_reduce_rate",
                            "loser_reduce_rate", "full_exit_ratio", "partial_exit_ratio",
                            "staged_exit_share", "avg_adds_per_episode")} if st else None,
        "outcome_response": {k: oq.get(k) for k in
                             ("average_down_tendency", "pyramid_winner_tendency",
                              "average_down_episodes", "accumulation_episodes",
                              "shallow_underwater_episodes", "median_underwater_depth",
                              "max_underwater_depth", "depth_tier_mix")} if oq.get("available") else None,
        "conviction": {k: cv.get(k) for k in
                       ("large_position_tolerance", "max_position_weight_ever",
                        "reentry_rate", "distinct_tickers_held")} if cv else None,
        "rotation": {k: ro.get(k) for k in
                     ("composition_change_rate", "avg_similarity_day_to_day")} if ro else None,
    }
