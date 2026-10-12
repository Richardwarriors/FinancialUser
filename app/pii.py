"""Person-identifying content must not reach a user profile.

The profile is an investment document. A person's name, nationality or address has
no analytical role in it, and carrying them creates a real harm: they are personal
identifiers stored in `profile_versions`, served to every reader of the viewer, and
fed into LLM prompts across every dimension.

Where this came from
--------------------
`users.base_profile_summary` was ingested as a `("stated",)` block and mounted in
the tree as `自述原文` under two labels (年龄与语言 and 用户自述), then passed to all
five profilers through `POLICY`. That text is overwhelmingly demographic:

    国籍 94/157 · 语言 96/157 · 年龄 39/157 users

and in many cases it is explicit — "I am a 33-year-old Chinese national", "I hold
Chinese and Singaporean nationality" — or simply useless ("I have not provided
details about my financial experience"). The upstream file it comes from
(`base_profiles_en`) is third-person facts including a full
`United States-California-Los Angeles` locality, which an upstream LLM then rewrote
into first person. So the field was never authentic self-report either.

The channel is removed at the source (see `evidence.stated_evidence`). This module
is the **boundary guard**, and it exists because removing one channel does not stop
the next one: the same defect (person data presented as a profile fact) can arrive
from any new source or from a model writing prose. The guard is deliberately
source-agnostic.

Precision matters more than recall here
---------------------------------------
Over-blocking silently deletes legitimate analysis, and a false positive on
"美国上市证券" (US-listed securities — a portfolio fact about listing venue, not a
nationality) would be its own kind of wrong. So the patterns target **person-level
descriptors**, never bare country or language words:

    flagged:  "I am a 55-year-old American…"   (first-person demographic intro)
    flagged:  "Chinese national", "持有…国籍"   (explicit nationality)
    flagged:  "living in Hong Kong", "住所"     (address / locality)
    kept:     "持仓以美国上市证券为主"            (listing venue — a portfolio fact)
    kept:     "用户偏好以中文与金融工具互动"        (language preference — a service fact)
"""
import re

# First-person demographic self-introduction. This is the shape the raw channel
# had, so it catches that whole class even when no category word is recognised.
_SELF_INTRO = re.compile(
    r"^\s*(?:i\s*(?:am|'m)\b|my\s+name\b|i\s+hold\s+\w+\s+nationality|"
    r"我是|我叫|本人|我今年)", re.I)

# Nationality, including demonym + "national"/"nationality" and the Chinese
# equivalents. A bare demonym ("Chinese") is NOT matched on its own: it is
# overwhelmingly a language reference in this dataset (96/157 users), and language
# preference is a legitimate service fact.
_NATIONALITY = re.compile(
    r"(?:\b(?:american|chinese|japanese|korean|singaporean|indian|german|french|"
    r"british|canadian|australian|brazilian|vietnamese|thai|indonesian|malaysian|"
    r"filipino|russian|italian|spanish|mexican|taiwanese|dutch|swedish|swiss)\s+"
    r"(?:national|nationality|citizen)\b"
    r"|\bnationality\b"
    r"|\bhold(?:s|ing)?\s+\w+\s+and\s+\w+\s+nationality\b"
    r"|国籍|美国籍|中国籍|美国公民|中国公民"
    r"|\b(?:美国人|中国人|日本人|韩国人|新加坡人|印度人)\b)", re.I)

# Address / locality, matched literally because `Country-State-City` is the upstream
# field's own format. This pattern is CASE-SENSITIVE on purpose and has no `re.I`:
# the capitalisation IS the signal. With case-insensitivity the hyphenated-word
# shape "low-turnover ETF-oriented" (a real `investment_logic.primary_style` value)
# matched and would have silently deleted a legitimate style fact.
_ADDRESS_LOCALE = re.compile(
    r"[A-Z][A-Za-z]*(?:[ ][A-Z][A-Za-z]*)*"
    r"(?:-[A-Z][A-Za-z]+(?:[ ][A-Z][A-Za-z]*)*){2,}")

# Address _words_. Kept separate so the word alternatives can stay case-insensitive
# without weakening the locale pattern above. Two omissions are deliberate:
#   * a bare `居住` is not matched, because `税务居住地` is tax jurisdiction — a
#     compliance concept, not an address;
#   * `居住地` is not matched either, even though it reads as one. `法定居住地` and
#     `税务居住地` are KYC FIELD NAMES holding no locality value, and flagging them
#     made the fleet scan permanently red with 156 hits — a check nobody can fix is
#     a check people stop reading. An actual locality is caught by the locale
#     pattern above.
_ADDRESS_WORDS = re.compile(
    r"(?:\b(?:living|based|residing|located)\s+in\s+[A-Z]"
    r"|\baddress\b|\bpostal\s+code\b|\bzip\s+code\b"
    r"|地址|住所|住在)", re.I)

# Explicit name fields. A free-form name is not reliably detectable, which is why
# the channel is removed rather than filtered — this only catches the labelled case.
_NAME = re.compile(r"(\bmy\s+name\s+is\b|\bname\s*[:：]\s*\S|\b姓\s*名[:：]|"
                   r"\b(?:Mr|Ms|Mrs|Dr)\.\s*[A-Z])", re.I)

# Age stated as a personal attribute. Two forms, because both were observed:
#   * English self-description: "55-year-old", "age: 55";
#   * Chinese narrative, which does NOT start the string — the models write
#     "用户自述为55岁美国人" and "自述为53岁". Anchoring the Chinese form to `^` (an
#     earlier version did) missed exactly that, so a prose scrubber returned the
#     sentence untouched. `岁` is safe to match bare in this domain: holding periods
#     are expressed in 天/周/月, never 岁.
# The `(?![\d.])` guard stops `age 0.01d` — a freshness string the harness writes
# itself — from being read as an age of 0.
_AGE = re.compile(r"(?:\b\d{1,3}[\s-]*year[\s-]*old\b"
                  r"|\d{1,3}\s*岁"
                  r"|\bage\s*[:：]?\s*\d{1,3}(?![\d.])"
                  r"|\bmy\s+age\b|年龄为?\s*\d{1,3})", re.I)

_RULES = (("self_intro", _SELF_INTRO), ("nationality", _NATIONALITY),
          ("address", _ADDRESS_LOCALE), ("address", _ADDRESS_WORDS),
          ("name", _NAME), ("age", _AGE))


def find_pii(value):
    """Return the reason this value carries person-identifying content, or None.

    Accepts a string or any JSON value; containers are inspected recursively so a
    nested mapping cannot smuggle a name past a top-level check.
    """
    for text in _iter_strings(value):
        for reason, rx in _RULES:
            if rx.search(text):
                return reason
    return None


def _iter_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _iter_strings(k)
            yield from _iter_strings(v)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _iter_strings(v)
    elif isinstance(value, (int, float, bool)) or value is None:
        return
    else:
        yield str(value)


def is_pii(value):
    return find_pii(value) is not None


def scrub(value):
    """Best-effort removal of person data from a string, for use before a prompt.

    NOT the primary defence — the channel is removed upstream and the boundary
    guard drops the containing fact. This exists only so that a stray occurrence
    cannot reach a model, and it is intentionally conservative: it drops whole
    sentences that match, because a partially redacted sentence ("I am a
    55-year-old [REDACTED]") still reads as a personal description.
    """
    if not isinstance(value, str):
        return value
    if not find_pii(value):
        return value
    kept = [s for s in re.split(r"(?<=[.。;；!?！？])\s*", value)
            if s.strip() and not find_pii(s)]
    return " ".join(kept).strip()
