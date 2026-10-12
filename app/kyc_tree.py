"""Assemble the complete L1 -> L2 -> L3 KYC tree for one user.

Merges three sources into one tree:
  * deterministic values   (app/kyc_features.py)  — computed, no LLM
  * LLM-judged values      (app/kyc_agents.py)    — grounded in cited evidence
  * explicit gaps          (app/kyc_catalog.py)   — insufficient_data + what is needed

Every L3 leaf always appears. A gap is reported as a gap, with `data_needed`
spelling out which questionnaire item or upstream feed would close it. That gap
list is a deliverable in its own right: it is the advisor's follow-up checklist.
"""
from datetime import datetime, timezone

from .kyc_agents import build_kyc_llm_features
from .kyc_behavior import build_behavior_signals
from .kyc_catalog import CATALOG
from .kyc_features import build_deterministic_features
from .kyc_proxy import build_proxy_features

TREE_VERSION = "kyc-tree-en-v3"


def _deterministic_by_key(det):
    return dict(det or {})


def build_kyc_tree(snap, evidence, intent=None, use_llm=True):
    """Full tree dict, ready to attach to a profile as `kyc_tree`."""
    det = build_deterministic_features(snap, intent)
    proxy = build_proxy_features(snap, intent)
    if use_llm:
        llm_feats, l2_narr, l1_narr, sources = build_kyc_llm_features(evidence, det)
    else:
        llm_feats, l2_narr, l1_narr, sources = {}, {}, {}, {}

    l1_out = {}
    stats = {"filled": 0, "proxy": 0, "gap": 0, "total": 0}

    for l1, node in CATALOG.items():
        l2_out = {}
        l1_filled, l1_gap = 0, 0
        for l2, l2node in node["l2"].items():
            leaves = {}
            for key, spec in l2node["features"].items():
                label, cov, streams, needed = spec
                stats["total"] += 1
                if cov == "deterministic":
                    r = det.get(key) or {}
                    value = r.get("value")
                    leaf = {
                        "label": label, "coverage": "deterministic", "source": "computed",
                        "data_sources": streams,
                        "value": value,
                        "display": r.get("display") or "",
                        "evidence": r.get("evidence") or [],
                        "confidence": r.get("confidence", 0.0),
                    }
                    if value in (None, {}, []):
                        leaf["status"] = "gap"
                        leaf["data_needed"] = needed or ["Current data is insufficient for calculation"]
                        stats["gap"] += 1
                        l1_gap += 1
                    else:
                        leaf["status"] = "filled"
                        stats["filled"] += 1
                        l1_filled += 1
                elif cov == "llm":
                    r = llm_feats.get(key) or {}
                    val = r.get("value", "insufficient_data")
                    is_gap = val in ("insufficient_data", None, "")
                    leaf = {
                        "label": label, "coverage": "llm", "source": "llm" if not is_gap else "gap",
                        "data_sources": streams,
                        "value": val,
                        "display": r.get("narrative") or ("Insufficient existing evidence to determine" if is_gap else ""),
                        "evidence": r.get("evidence") or [],
                        "confidence": r.get("confidence", 0.0),
                    }
                    if is_gap:
                        leaf["status"] = "gap"
                        leaf["data_needed"] = ([r["missing"]] if r.get("missing") else []) + (needed or [])
                        stats["gap"] += 1
                        l1_gap += 1
                    else:
                        leaf["status"] = "filled"
                        stats["filled"] += 1
                        l1_filled += 1
                elif cov == "proxy":
                    r = proxy.get(key) or {}
                    value = r.get("value")
                    # A proxy counts as coverage only when it yields a real
                    # estimate. A bounded scope statement ("nothing found within
                    # what we can see") is still recorded — so the advisor knows
                    # we looked — but it does not answer the KYC question, and
                    # counting it as covered would overstate the profile.
                    is_estimate = r.get("estimate", True)
                    if value in (None, {}, []) or r.get("confidence", 0) == 0 or not is_estimate:
                        leaf = {
                            "label": label, "coverage": "proxy", "source": "gap",
                            "data_sources": streams,
                            "value": "insufficient_data",
                            "display": r.get("display") or "Existing data cannot support indirect inference",
                            "evidence": r.get("evidence") or [],
                            "confidence": r.get("confidence", 0.0),
                            "status": "gap",
                            "is_proxy": True,
                            "scope_note": r.get("display"),
                            "proxy_basis": r.get("proxy_basis"),
                            "caveat": r.get("caveat"),
                            "data_needed": needed or ["Additional data collection required"],
                        }
                        stats["gap"] += 1
                        l1_gap += 1
                    else:
                        leaf = {
                            "label": label, "coverage": "proxy", "source": "proxy",
                            "data_sources": streams,
                            "value": value,
                            "display": r.get("display") or "",
                            "evidence": r.get("evidence") or [],
                            "confidence": r.get("confidence", 0.0),
                            "status": "proxy",
                            "is_proxy": True,
                            "proxy_basis": r.get("proxy_basis"),
                            "caveat": r.get("caveat"),
                            "data_needed": needed or [],
                        }
                        stats["proxy"] += 1
                        l1_filled += 1
                else:  # unavailable
                    leaf = {
                        "label": label, "coverage": "unavailable", "source": "gap",
                        "data_sources": streams,
                        "value": "insufficient_data",
                        "display": "Cannot determine from existing data; additional data collection required",
                        "evidence": [],
                        "confidence": 0.0,
                        "status": "gap",
                        "data_needed": needed or ["Additional data collection required"],
                    }
                    stats["gap"] += 1
                    l1_gap += 1
                leaves[key] = leaf
            filled_here = sum(1 for v in leaves.values() if v["status"] in ("filled", "proxy"))
            l2_out[l2] = {
                "label": l2node["label"],
                "status": "filled" if filled_here == len(leaves) else
                          ("gap" if filled_here == 0 else "partial"),
                "filled": filled_here, "total": len(leaves),
                "narrative": l2_narr.get(l2) or _compose_l2(leaves),
                "features": leaves,
            }
        total_l1 = sum(v["total"] for v in l2_out.values())
        l1_proxy = sum(1 for l2 in l2_out.values() for v in l2["features"].values()
                       if v["status"] == "proxy")
        l1_out[l1] = {
            "label": node["label"],
            "status": "filled" if l1_filled == total_l1 else ("gap" if l1_filled == 0 else "partial"),
            "filled": l1_filled, "proxy": l1_proxy, "total": total_l1,
            "narrative": l1_narr.get(l1) or _compose_l1(node["label"], l2_out, l1_filled, total_l1),
            "l2": l2_out,
        }

    covered = stats["filled"] + stats["proxy"]
    stats["covered"] = covered
    stats["fill_rate"] = round(covered / stats["total"], 3) if stats["total"] else 0
    stats["direct_fill_rate"] = round(stats["filled"] / stats["total"], 3) if stats["total"] else 0
    stats["note"] = (f"{stats['filled']} items derived directly from data, {stats['proxy']} items are evidence-based indirect inferences, {stats['gap']} items genuinely lack data (KYC questionnaire)")
    gap_keys = sorted(k for l1 in l1_out.values() for l2 in l1["l2"].values()
                      for k, v in l2["features"].items() if v["status"] == "gap")
    # B-class observation signals: separate block, excluded from the 60-feature
    # count on purpose (they describe platform behaviour, not KYC facts)
    try:
        behavior_signals = build_behavior_signals(snap, intent, evidence)
    except Exception as e:                                        # noqa: BLE001
        behavior_signals = {"count": 0, "filled": 0,
                            "note": f"Calculation failed: {type(e).__name__}", "signals": {}}
    tree = {
        "version": TREE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": stats,
        "llm_sources": sources,
        "l1": l1_out,
        "gap_list": gap_keys,
        "behavior_signals": behavior_signals,
    }
    from .english_output import englishize
    return englishize(tree)


def _compose_l2(leaves):
    """Fallback L2 text when the LLM did not supply one: join what we do know."""
    known = [v for v in leaves.values() if v["status"] in ("filled", "proxy")]
    gaps = sum(1 for v in leaves.values() if v["status"] == "gap")
    if not known:
        return f"All {len(leaves)} items under this group lack usable data."
    body = "；".join(f"{v['label']}：{v['display']}" for v in known[:3])
    tail = f" An additional {gaps} items could not be assessed due to missing data." if gaps else ""
    return body + tail


def _compose_l1(label, l2_out, filled, total):
    if filled == 0:
        return f"{label}: existing data cannot cover this dimension, with {total} items pending collection."
    partial = [v["label"] for v in l2_out.values() if v["status"] != "filled"]
    tail = f";{('、'.join(partial))} remains incomplete" if partial else ""
    return f"{label}: {filled}/{total} items determined{tail}."
