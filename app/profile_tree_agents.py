"""Model-written L1/L2 summaries for the profile tree.

Kept separate from `profile_tree.py` on purpose: assembling the tree is pure
Python and must stay that way (deterministic, auditable, no token cost). This
module adds the only part that genuinely needs a model — turning a list of facts
into 2-3 readable sentences.

Cost control
------------
11 L1 and ~34 L2 summaries would be 11 calls if grouped per L1. They are grouped
into 4 calls (`SUMMARY_GROUPS`), which is the same order of magnitude as the KYC
layer it replaces. Each call returns every L1 summary AND every L2 summary in its
group, so the model can keep the two levels consistent with each other.

Rules enforced in the prompt
----------------------------
* `status="gap"` facts must be described as UNKNOWN, never as a value. A missing
  KYC fact is the single easiest thing to accidentally invent.
* proxy facts must be named as inference.
* No advice, no suitability conclusion, no prediction.
* Do not repeat the fact list verbatim; say what it means and what is missing.
"""
import json

PROFILE_TREE_PROMPT_VERSION = "profile-tree-summary-v3-english"

SYSTEM = (
    "You are a financial user-profile analyst. Turn the supplied structured facts "
    "into readable English for an investment adviser, using only those facts and "
    "never inventing information.\nCore rules:"
    "\n1) facts contains only evidence-supported conclusions: status=filled is directly "
    "derived from data and status=proxy is a supported indirect inference. A dimension "
    "with no facts has no supported conclusion; state that explicitly rather than "
    "guessing, completing it, or treating absence as a value."
    "\n2) Identify every status=proxy item as an indirect inference and state its basis."
    "\n3) Do not list field names or copy facts verbatim; explain what they mean together."
    "\n4) Do not provide advice, suitability conclusions, or predictions. Behavioral "
    "observations are not psychological diagnoses."
    "\n5) Keep stated information separate from revealed behavior."
    "\n6) Every natural-language output value must be in English. Return only JSON that "
    "conforms to the schema."
)


def _schema_for(group, tree):
    """Build a per-group schema: one key per L1, each with summary + its L2 keys.

    Declared explicitly (not a bare dict) because structured output needs concrete
    properties, and because constraining the keys stops the model inventing L2s
    that do not exist in the tree.
    """
    from pydantic import create_model

    fields = {}
    for l1_key in group:
        l1 = (tree.get("l1") or {}).get(l1_key) or {}
        sub_keys = list(l1.get("l2") or {})
        sub = create_model(
            f"L2_{l1_key}",
            **{k: (str, "") for k in sub_keys},
            __config__=None,
        )
        fields[l1_key] = (create_model(
            f"L1_{l1_key}",
            summary=(str, ""),
            l2=(sub, sub()),
            __config__=None,
        ), None)
    # every L1 in the group is required so a group cannot come back half-empty
    return create_model("ProfileTreeSummaries", **{k: (v[0], ...) for k, v in fields.items()})


def summarize(tree, timeout_note=""):
    """Return (payload, source). `payload` is ready for `apply_summaries`.

    Never raises: on any failure the caller keeps the deterministic tree (which
    already carries the source narratives as L2 fallbacks), so a model outage
    degrades prose quality rather than breaking the profile.
    """
    from .orchestrator import agent_view  # noqa: F401  (kept for symmetry of imports)
    from .profilers import _client, resolve_model
    from .token_meter import record

    client = _client()
    if client is None:
        return {}, "rule:no_api_key"
    model = resolve_model("logic")     # summarising facts is a reasoning task

    from .profile_tree import SUMMARY_GROUPS, tree_for_prompt
    compact = tree_for_prompt(tree)
    merged, sources = {}, []
    for group in SUMMARY_GROUPS:
        payload = {k: compact[k] for k in group if k in compact}
        if not payload:
            continue
        try:
            schema = _schema_for(group, tree)
            raw_run = client.with_structured_output(schema, include_raw=True)
            res = raw_run.invoke([
                ("system", SYSTEM),
                ("user", json.dumps({"l1_groups": payload}, ensure_ascii=False) + timeout_note),
            ])
            record("profile_tree", model, getattr(res.get("raw"), "usage_metadata", None))
            out = res.get("parsed")
            if out is None:
                sources.append("rule:llm_parse_failed")
                continue
            d = out.model_dump()
            for l1_key, got in d.items():
                if l1_key not in group:
                    continue                     # model invented a key: ignore
                merged[l1_key] = {
                    "summary": got.get("summary") or "",
                    "l2": {k: v for k, v in (got.get("l2") or {}).items()
                           if k in ((tree["l1"][l1_key].get("l2")) or {})},
                }
            sources.append(f"llm:{model}")
        except Exception as e:                                            # noqa: BLE001
            sources.append(f"rule:llm_error:{type(e).__name__}")
            continue

    src = sources[0] if sources and all(s == sources[0] for s in sources) else ",".join(sorted(set(sources)))
    return merged, src
