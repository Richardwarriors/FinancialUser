"""Freshness of DERIVED (presentation) components, independent of the dimension gate.

Why this module exists
----------------------
Harness v2 made the expensive path conditional: a dimension is only recomputed
when its evidence actually moved. That is a cost win, but it created a second,
quieter problem. `portrait`, `kyc_tree` and `profile_tree` are *derived* from the
dimensions, and their rebuild decision was buried inside the same gate check that
protected the dimensions:

    narrate:        if decision in ("keep_old", "skip", "no_change"): return {}
    write_profile:  if decision in ("keep_old", "no_change"): <write nothing>

Once a user's dimensions went stable — which is precisely the steady state v2 is
built to reach — the gate returned `no_change`, `narrate` bailed out before
touching the derived layer, and `write_profile` discarded whatever it produced.
So a derived component could never:

  * be built for the first time (existing users had no tree at all),
  * migrate to a new version (bumping PROFILE_TREE_VERSION reached nobody who
    did not also happen to change a dimension in the same run),
  * or be refreshed when its TTL expired.

A presentation-layer change was therefore effectively unreachable in production.
This was observed, not theorised: the tree had to be back-filled by hand.

The fix is to stop conflating two different questions:

  dimension  — "did the underlying facts change?"
  derived    — "is what I am serving still built by the current code, from the
                 current upstream?"

They are evaluated independently here. A derived-only rebuild is persisted
in place (`SQLiteProfileStore.update_latest`) so refreshing prose does not inflate
the business version history with rows whose numbers are byte-identical.

A third case, found later and by the same mistake
The five DETERMINISTIC metric blocks (`trade_metrics`, `portfolio_metrics`,
`asset`, `position_dynamics`, `attention_capital`) are recomputed from the
snapshot on every refresh and attached in `assemble`. They answer to neither the
dimension gate nor the prose contract, and they were wired to the wrong one:
`assemble` bailed out when the gate said `no_change`, and `write_profile` then
discarded the freshly computed blocks. So a metric-definition fix
(`POSITION_VERSION`, `ATTENTION_CAPITAL_VERSION`) could not reach a user whose
dimensions happened to be quiet — which is the steady state.

Measured, not theorised: on a 12-user sweep, 2 users returned `no_change` and kept
the previous blocks, including a `depth_note` that said 「本类用户里」 about one
user's own numbers. They are treated here as one component (`metrics`) whose stamp
is the composite of the five blocks actually written, so bumping ANY engine
invalidates it.
"""

# Served-freshness windows for the derived layer. Kept as a bounded safety net:
# a rebuild triggered ONLY by expiry costs at most one rebuild per component per
# window per user, which is the same discipline `DIM_TTL_DAYS` applies to
# dimensions. Treating this as a per-refresh trigger is what v2 removed.
DERIVED_TTL_DAYS = {
    "portrait": 7,
    "kyc_tree": 30,
    "profile_tree": 30,
}

# Rebuild order matters: profile_tree consumes the kyc_tree output, so it must be
# able to see that kyc_tree was rebuilt in the same pass.
ORDER = ("kyc_tree", "portrait", "profile_tree", "metrics")

# The five deterministic metric blocks, in stamp order. They are stored as five
# top-level objects (no single `metrics` key exists in a profile), so this tuple is
# what turns them into one freshness question.
METRIC_BLOCKS = ("trade_metrics", "portfolio_metrics", "asset",
                 "position_dynamics", "attention_capital", "action_forecast")

# The keys a block may carry and still be nothing but a "not applicable" stub.
# Anything beyond these means the block holds real content and must be compared.
_NON_APPLICABLE_KEYS = {"available", "note"}

# portrait is a pure function of the five dimensions, so it is stale exactly when
# a dimension was recomputed (or its own prompt version moved).
# `metrics` is False: it is computed from the SNAPSHOT, so a dimension move does
# not make it stale, and its own version stamp lives in the block itself.
DEPENDS_ON_DIMS = {"portrait": True, "kyc_tree": False, "profile_tree": True,
                   "metrics": False}

# kyc_tree -> profile_tree: the tree embeds KYC-derived L2 content verbatim, so a
# rebuilt KYC tree invalidates the tree even when no dimension moved.
DEPENDS_ON = {"kyc_tree": (), "portrait": (), "profile_tree": ("kyc_tree",),
              "metrics": ()}


def current_metric_version(keys=METRIC_BLOCKS):
    """Running composite version stamp of the deterministic metric blocks.

    Built from the engines themselves, in `METRIC_BLOCKS` order, so it is directly
    comparable to the composite read off a stored profile.

    `keys` narrows the comparison to the blocks that can actually exist for this
    user. That matters: a user with no trades and no positions produces NONE of
    the five, and comparing their (empty) stored stamp against all five would
    report `metrics` stale on every single refresh forever — the same
    non-convergence trap as a fingerprint that can never be satisfied.
    """
    from .action_forecast import FORECAST_VERSION
    from .attention_capital import ATTENTION_CAPITAL_VERSION
    from .evidence import ASSET_EVIDENCE_VERSION
    from .portfolio_metrics import METRIC_VERSION as _PM
    from .position_events import POSITION_VERSION
    from .trade_metrics import METRIC_VERSION as _TM
    stamps = {"trade_metrics": _TM, "portfolio_metrics": _PM,
              "asset": ASSET_EVIDENCE_VERSION, "position_dynamics": POSITION_VERSION,
              "attention_capital": ATTENTION_CAPITAL_VERSION,
              "action_forecast": FORECAST_VERSION}
    return "|".join(stamps[k] for k in keys)


def applicable_metric_keys(profile):
    """Which metric blocks actually SAY something for this user.

    A block stored as `{available: False, note: ...}` is a statement of\n    NON-applicability (\"this user has no trades\"), not a metric. Counting it would
    make the composite stamp unsatisfiable forever, because the stub never carries
    a version: the comparison could never come back clean, the profile would be
    rewritten on every single refresh, and it would never converge. That is the
    same non-convergence trap as a fingerprint that can never be satisfied.

    The test is CONTENT, not the flag alone. A block that carries anything beyond
    the non-applicability markers qualifies even without a version, so a legacy
    block written before stamping existed is still compared, reported stale, and
    migrated to the stamped form. Ignoring it would leave it stale forever.
    """
    out = []
    for key in METRIC_BLOCKS:
        block = (profile or {}).get(key)
        if not isinstance(block, dict) or not block:
            continue
        if block.get("available") is False and set(block) <= _NON_APPLICABLE_KEYS:
            continue
        out.append(key)
    return tuple(out)


def stored_metric_version(profile, keys=METRIC_BLOCKS):
    """Composite stamp read off a stored profile's metric blocks.

    A block that is absent or unstamped reads as the literal `None`, which never
    equals the running stamp — an unstamped block must not be assumed current.
    """
    parts = []
    for key in keys:
        block = (profile or {}).get(key)
        parts.append(str(block.get("version")) if isinstance(block, dict) else "None")
    return "|".join(parts)


def _versions(metric_keys=METRIC_BLOCKS):
    """Current version stamp of each derived component, read from its own module.

    `profile_tree` is stamped with structure + prompt, because its content is both.
    `metrics` is stamped with the engines of the blocks this user can actually have.
    """
    from .kyc_tree import TREE_VERSION
    from .narrative import PORTRAIT_PROMPT_VERSION
    from .profile_tree import tree_version_stamp
    return {"portrait": PORTRAIT_PROMPT_VERSION,
            "kyc_tree": TREE_VERSION,
            "profile_tree": tree_version_stamp(),
            "metrics": current_metric_version(metric_keys)}


def stored_version(name, block, metric_keys=METRIC_BLOCKS):
    """Version recorded on a stored component, normalised across key names.

    The portrait records `prompt_version` (it is one LLM call, not a structure);
    the trees record `version`. Callers should not have to know that.

    `metrics` is the exception that proves the rule: no `metrics` object is stored,
    so `block` is the WHOLE profile and the stamp is the composite of the metric
    blocks inside it.
    """
    if not isinstance(block, dict):
        return None
    if name == "metrics":
        return stored_metric_version(block, metric_keys)
    return block.get("version") or block.get("prompt_version")


def _block_for(name, existing):
    """The stored object a component's freshness is judged against.

    Only `metrics` differs: its five blocks are stored at the top level rather than
    nested under one key.
    """
    return existing if name == "metrics" else existing.get(name)


def stale_components(existing, as_of=None, recomputed_dims=(), ttl_days=None,
                     metric_keys=None):
    """Which derived components may NOT be served as-is, and why.

    Returns `{name: reason}`. An empty dict means everything already stored is
    current for the running code — the common case, and it costs no LLM calls.

    `metric_keys` is the set of deterministic metric blocks that can exist for
    this user. When omitted it is DERIVED from the profile by
    `applicable_metric_keys`, because the stored blocks already say which metrics
    apply. Passing an EMPTY tuple means nothing applies, and the component is then
    not applicable at all — never "stale".

    Defaulting to all five instead would be a footgun: every caller that forgot the
    argument would report `metrics` stale forever for a user with no trades.

    A component is stale when ANY of these holds, checked in this order so the
    reported reason is the most authoritative one:
      missing   — never built (e.g. an existing user predating the component)
      version   — the producing code changed since it was written
      upstream  — a dimension it summarises was recomputed in this run
      upstream  — a component it embeds was itself rebuilt in this run
      expired   — older than its TTL window

    `metrics` deliberately has NO TTL. The five blocks are recomputed on every
    refresh and attached for free, so a TTL would buy no correctness: their code
    freshness is fully described by the version stamp, and their data freshness
    flows in through the dimension fingerprints (any snapshot change that matters
    to a dimension moves it, which is not `no_change`). A TTL here would force a
    write on every idle user for no gain.
    """
    from .lifecycle import age_days

    existing = existing or {}
    ttl = {**DERIVED_TTL_DAYS, **(ttl_days or {})}
    keys = (applicable_metric_keys(existing) if metric_keys is None
            else tuple(metric_keys))
    versions = _versions(keys)
    age = age_days(existing, as_of)
    moved = set(recomputed_dims or ())
    out = {}
    for name in ORDER:
        if name == "metrics" and not keys:
            # Nothing to refresh and nothing stored: not a component for this user.
            continue
        block = _block_for(name, existing)
        current = versions[name]
        found = stored_version(name, block, keys)
        reason = None
        if not isinstance(block, dict) or not block:
            reason = "missing"
        elif name == "metrics":
            # Every block is judged on its own stamp, so the reason names which one
            # moved instead of reporting an opaque composite mismatch.
            if found != current:
                reason = f"version {found!r} != {current!r}"
        elif found != current:
            reason = f"version {found!r} != {current!r}"
        elif DEPENDS_ON_DIMS[name] and moved:
            reason = f"dimensions recomputed: {sorted(moved)}"
        elif set(DEPENDS_ON[name]) & set(out):
            reason = f"upstream rebuilt: {sorted(set(DEPENDS_ON[name]) & set(out))}"
        else:
            limit = ttl.get(name)
            if limit is not None and (age is None or age > limit):
                # A missing timestamp reads as `unknown`, never the literal `None`:
                # these reasons are written into the audit log and read by people.
                seen = "unknown" if age is None else f"{age}d"
                reason = f"expired age={seen} > ttl={limit}d"
        if reason:
            out[name] = reason
    return out


def freshness_report(existing, as_of=None, recomputed_dims=(), ttl_days=None,
                     metric_keys=None):
    """Every component with `stale`/`reason`/`age_days` — for audit and the viewer."""
    from .lifecycle import age_days

    existing = existing or {}
    keys = (applicable_metric_keys(existing) if metric_keys is None
            else tuple(metric_keys))
    stale = stale_components(existing, as_of, recomputed_dims, ttl_days, keys)
    versions = _versions(keys)
    age = age_days(existing, as_of)
    return {name: {"present": isinstance(_block_for(name, existing), dict),
                   "version": stored_version(name, _block_for(name, existing), keys),
                   "current_version": versions[name],
                   "age_days": age,
                   "stale": name in stale,
                   "reason": stale.get(name) or "current"}
            for name in ORDER if name in versions and (name != "metrics" or keys)}
