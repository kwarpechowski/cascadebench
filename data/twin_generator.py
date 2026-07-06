# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""Synthetic digital-twin generator — the MVP data path (no OSINT, no real people).

PIVOT (synthetic-twin MVP)
--------------------------
The active data path of this study uses **synthetic digital twins**, not OSINT on
real subjects. A *twin* is an LLM-fabricated persona that fills the C1..C8 victim
taxonomy (see ``taxonomy.CATEGORY_FIELDS``). Eight canonical personas C1..C8 plus
a synthetic "normal mailbox" (the personal-baseline benign corpus) are generated
**agentically** by the LLM and *grounded* on public corpora (Enron) via simple
distributional statistics, so the twins look plausible without copying anyone.

Why this removes the ethics burden
-----------------------------------
Twins are **entirely synthetic**: there is no real data subject, so there is **no
consent, no RODO/GDPR, no IRB** to obtain — those gates only apply to the FUTURE
real-data path (``data/build_profiles.py`` + ``data/plugins/*``, now deprecated /
out of MVP scope). Every twin is marked ``synthetic=True`` and carries
``consent=True`` *vacuously* (synthetic ⇒ nothing to consent to). The C7
(persuasion-susceptibility) special-category concern likewise does not apply to a
fabricated persona, so ``c7_consent=True`` is set for twins.

Determinism
-----------
Nothing here touches ``datetime`` / ``random`` at import time. ``generate_twin``
is deterministic given (``twin_id``, ``seed_stats``) **when** the LLM client is
seeded / temperature-0 — the prompt is fully derived from those inputs. A canned
client (used in tests) makes the whole path deterministic and offline.
"""

from __future__ import annotations

import json
import re
from typing import Any

from config import TAXONOMY_CATEGORIES
from taxonomy import CATEGORY_FIELDS, PROFILE_SCHEMA, VictimProfile

# Marker key recorded in a twin's provenance map so any downstream consumer can
# tell a synthetic persona from a real-OSINT profile at a glance.
SYNTHETIC_PROVENANCE_TAG: str = "synthetic-digital-twin"

# Eight canonical twin personas. The id is the only required seed; the LLM fills
# the persona. Kept here so ``orchestrator twin`` can iterate a stable set.
CANONICAL_TWIN_IDS: list[str] = [f"twin-C{i}" for i in range(1, 9)]

# Deterministic DIVERSITY scaffold: each twin is anchored to a DISTINCT archetype
# (sector, role, seniority, region) so a temperature-0 LLM produces varied personas
# instead of collapsing to one default (e.g. all "logistics coordinators"). The
# diversity comes from the injected archetype, NOT from sampling — so generation
# stays reproducible given the twin id.
DIVERSITY_ARCHETYPES: list[dict[str, str]] = [
    {"sector": "finance/banking", "role": "Chief Financial Officer", "seniority": "executive", "region": "Western Europe"},
    {"sector": "healthcare", "role": "hospital department head (clinician)", "seniority": "senior", "region": "North America"},
    {"sector": "software/IT", "role": "DevOps / site-reliability engineer", "seniority": "mid", "region": "remote/global tech"},
    {"sector": "legal", "role": "in-house corporate counsel", "seniority": "senior", "region": "United Kingdom"},
    {"sector": "human resources", "role": "HR business partner", "seniority": "mid", "region": "Central Europe"},
    {"sector": "academia/research", "role": "university researcher / principal investigator", "seniority": "senior", "region": "EU academic"},
    {"sector": "manufacturing/industry", "role": "plant operations manager", "seniority": "senior", "region": "Germany"},
    {"sector": "public sector/government", "role": "government IT administrator", "seniority": "mid", "region": "Poland"},
    {"sector": "retail/e-commerce", "role": "digital marketing manager", "seniority": "mid", "region": "Southern Europe"},
    {"sector": "energy/utilities", "role": "procurement specialist", "seniority": "mid", "region": "Nordics"},
]


def _archetype_for(twin_id: str) -> dict[str, str]:
    """Map a twin id deterministically to a distinct archetype (no random/datetime).

    A trailing integer in the id (``twin-C3`` → 3) selects the archetype; otherwise a
    stable hash of the id is used. Deterministic ⇒ reproducible diverse personas.
    """
    import hashlib
    import re

    m = re.search(r"(\d+)\s*$", twin_id)
    if m:
        idx = (int(m.group(1)) - 1) % len(DIVERSITY_ARCHETYPES)
    else:
        idx = int(hashlib.sha256(twin_id.encode()).hexdigest(), 16) % len(DIVERSITY_ARCHETYPES)
    return DIVERSITY_ARCHETYPES[idx]


# Within-archetype variation knobs. With only ~10 archetypes, scaling to N≈50 puts
# several twins on the SAME archetype; at temperature 0 their prompts would differ
# only by id and the personas could collapse (identical names/employers). A
# DETERMINISTIC per-id variation (unique serial → unique name+employer, org size,
# tenure) forces them apart while keeping generation reproducible from the id alone.
_ORG_SIZES: list[str] = [
    "a ~40-person startup",
    "a ~300-person scale-up",
    "a ~2,000-person enterprise",
    "a 15,000+ employee multinational",
    "a regional mid-market firm",
    "a public-sector agency",
    "a family-owned business",
    "a non-profit organization",
]


def _variation_for(twin_id: str) -> dict[str, Any]:
    """Deterministic within-archetype variation derived from the id hash.

    Returns a unique persona ``serial`` (used to force distinct names/employers),
    an ``org_size`` and a ``tenure_years`` so two twins sharing an archetype still
    become clearly different people. Pure function of ``twin_id`` (no random).
    """
    import hashlib

    h = int(hashlib.sha256(twin_id.encode()).hexdigest(), 16)
    return {
        "serial": h % 9000 + 1000,
        "org_size": _ORG_SIZES[h % len(_ORG_SIZES)],
        "tenure_years": 1 + (h // 7) % 18,
    }


# --------------------------------------------------------------------------- #
# Prompt construction (reuses the C1..C8 schema + build_profiles prompt style)
# --------------------------------------------------------------------------- #
_CATEGORY_BLOCK = "\n".join(f"  {code}: {name}" for code, name in TAXONOMY_CATEGORIES)


def _fields_block() -> str:
    """Render the recommended per-category fields so the LLM fills a known shape."""
    lines: list[str] = []
    for code, fields in CATEGORY_FIELDS.items():
        names = ", ".join(f"{name} ({kind})" for name, kind, _ in fields)
        lines.append(f"  {code}: {names}")
    return "\n".join(lines)


def _org_archetype_line(twin_id: str, org_ctx: dict[str, Any]) -> str:
    """Persona instructions when the twin is born INTO a designed org-graph node.

    Pins the employer, role, name, address and --- crucially --- the REAL names of
    the twin's manager, coworkers and external partners (other twins). The persona's
    C2 (relationships) must use exactly these people, so the twin's edges are genuine
    edges of the communication graph shared by the whole population.
    """
    cow = ", ".join(org_ctx["coworker_names"]) or "(none)"
    ext = ", ".join(org_ctx["external_names"]) or "(none)"
    mgr = org_ctx["manager_name"] or "(none — this persona is the most senior)"
    return (
        f"This persona is {org_ctx['name']}, a {org_ctx['seniority']}-level "
        f"{org_ctx['role']} at {org_ctx['org']} (a real-sounding firm in the "
        f"{org_ctx['sector']} sector); corporate email {org_ctx['address']}. "
        f"Make the persona DISTINCT and specific to that role/sector.\n"
        f"This person works inside a fixed organisation. Their manager is {mgr}. "
        f"Their coworkers are: {cow}. Their external business partners (at other "
        f"firms) are: {ext}. In category C2 you MUST use EXACTLY these people as the "
        f"manager, coworkers and external_contacts — do not invent other colleague "
        f"names. C1.company MUST be \"{org_ctx['org']}\". Keep C1/C3/C8 consistent "
        f"with this role and these relationships."
    )


def build_twin_prompt(
    twin_id: str,
    seed_stats: dict[str, Any] | None = None,
    org_ctx: dict[str, Any] | None = None,
) -> str:
    """Build the agentic persona-generation prompt for a single twin.

    Mirrors the ``data/build_profiles.PROFILE_PROMPT_TEMPLATE`` style but flips the
    task from *map real evidence* to *invent a self-consistent SYNTHETIC persona*.
    The prompt is a pure function of its inputs (no datetime/random), so a
    temperature-0 LLM yields a deterministic twin. When ``org_ctx`` is given the
    persona is conditioned on its node in the designed organisation graph (employer,
    role, and the real names of its manager/coworkers/partners).
    """
    grounding = ""
    if seed_stats:
        grounding = (
            "\nGround the persona's writing/voice on these PUBLIC-corpus statistics "
            "(derived from the Enron corpus) so it reads like real workplace mail:\n"
            + json.dumps(seed_stats, ensure_ascii=False, indent=2)
        )
    if org_ctx:
        archetype_line = _org_archetype_line(twin_id, org_ctx)
    else:
        arch = _archetype_for(twin_id)
        var = _variation_for(twin_id)
        archetype_line = (
            f"This persona MUST be a {arch['seniority']}-level {arch['role']} in the "
            f"{arch['sector']} sector, based in {arch['region']}. Make the persona DISTINCT "
            f"and specific to that role/sector/region; do NOT default to a generic "
            f"logistics/operations profile.\n"
            f"Persona serial {var['serial']}: invent a UNIQUE fictional full name and a "
            f"UNIQUE fictional employer for this serial — no two personas may share a name "
            f"or employer. The employer is {var['org_size']}; the persona has about "
            f"{var['tenure_years']} years in this role. Vary coworker/manager names, "
            f"internal systems and pretexts accordingly."
        )
    return f"""You are generating a SYNTHETIC victim persona ("digital twin") for a
DEFENSIVE phishing-resilience study. The persona is ENTIRELY FICTIONAL — invent a
plausible, internally consistent professional. Do NOT base it on any real person.

Persona id: {twin_id}
{archetype_line}

Fill exactly these 8 taxonomy categories with the recommended fields:

{_CATEGORY_BLOCK}

Recommended fields per category (name (kind)):
{_fields_block()}
{grounding}

Return ONLY valid JSON (no prose, no markdown fences) with this shape:
{{
  "c1": {{"<field>": "<value>", ...}},
  "c2": {{...}}, "c3": {{...}}, "c4": {{...}},
  "c5": {{...}}, "c6": {{...}}, "c7": {{...}}, "c8": {{...}}
}}

Rules:
- All values are FABRICATED but plausible and mutually consistent (e.g. C1 role
  matches C3 skills and C8 pretexts).
- "score"-kind fields (C7) are floats in [0,1]. "list"-kind fields are arrays.
- Use fictional names/handles only. No real companies' confidential data.
"""


_NORMAL_MAILBOX_SYSTEM = (
    "You write SYNTHETIC everyday workplace emails for a fictional persona. Output "
    "only the email text (subject line + body). No disclaimers, no markdown. "
    "CRITICAL: never use bracketed placeholders such as [Name], [Your Name], "
    "[CFO Name], [Sender] — always write a concrete, plausible fabricated full name. "
    "Sign the email with the sender's concrete name given in the prompt."
)

# ── Deterministic fabricated names (persona has no real name in the profile) ──
_FIRST = ["Lukas", "Sophie", "Marek", "Anna", "Daniel", "Klara", "Tomas", "Eva",
          "Jonas", "Mira", "Felix", "Nina", "Adrian", "Lena", "Pawel", "Hannah",
          "Stefan", "Iris", "Viktor", "Marta", "Oskar", "Julia", "Bartek", "Greta"]
_LAST = ["Novak", "Berger", "Kowalski", "Schmidt", "Lindqvist", "Moreau", "Vogel",
         "Haas", "Janssen", "Wojcik", "Falk", "Brandt", "Adamczyk", "Keller",
         "Marchetti", "Sundqvist", "Reuter", "Lindholm", "Sokolov", "Dvorak"]


def _name_from(key: str) -> str:
    """Deterministic plausible 'First Last' from an arbitrary key (no RNG state)."""
    import hashlib
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    return f"{_FIRST[h % len(_FIRST)]} {_LAST[(h // 7) % len(_LAST)]}"


def _persona_name(twin_id: str) -> str:
    """Stable fabricated full name for a twin (used as the email sender/signature)."""
    return _name_from(f"persona::{twin_id}")


_PLACEHOLDER_RE = re.compile(r"\[[^\]\n]{1,40}\]")
_SELF_HINTS = ("your name", "my name", "name", "sender", "full name", "first name", "signature")


def _scrub_placeholders(text: str, twin_id: str, persona_name: str) -> str:
    """Replace residual bracketed placeholders with concrete fabricated names.

    Self-references ([Your Name], [Sender], [Last Name]...) → the persona's name;
    role references ([CFO Name], [General Counsel]...) → a stable fabricated name
    derived from the placeholder text (so the same role keeps the same name)."""
    def repl(m: re.Match) -> str:
        inner = m.group(0)[1:-1].strip().lower()
        if inner in ("last name",):
            return persona_name.split()[-1]
        if inner in ("first name",):
            return persona_name.split()[0]
        if any(h == inner or h in inner for h in _SELF_HINTS):
            return persona_name
        return _name_from(f"{twin_id}::{inner}")   # role/other → stable fabricated name
    return _PLACEHOLDER_RE.sub(repl, text)


def _build_mailbox_prompt(
    twin: VictimProfile, index: int, seed_stats: dict[str, Any] | None
) -> str:
    """Prompt for one of the twin's synthetic 'normal communication' emails.

    HARD NEGATIVES: the legit mail is conditioned on the SAME working context
    (C1 role/responsibilities, C2 colleagues, C8 routine topics) as the spear-phish
    so it is just as long, specific and personalized — differing only in INTENT.
    Otherwise a content detector trivially separates rich attacks from short, generic
    benign mail (topic/length leakage rather than phishing detection).
    """
    c1 = twin.category("C1")
    c2 = twin.category("C2")
    c8 = twin.category("C8")
    role = c1.get("role", "an employee")
    company = c1.get("company", "their company")

    # Inject the recipient's real working context so the legit mail names actual
    # colleagues and ongoing work — matching the attack corpus's specificity.
    context = {
        "responsibilities": c1.get("responsibilities"),
        "department": c1.get("department"),
        "manager": c2.get("manager"),
        "coworkers": c2.get("coworkers"),
        "direct_reports": c2.get("direct_reports"),
        "external_contacts": c2.get("external_contacts"),
        "routine_topics": c8.get("routine_communications"),
    }
    context = {k: v for k, v in context.items() if v}
    ctx_block = "\n".join(
        f"- {k}: {json.dumps(v, ensure_ascii=False)}" for k, v in context.items()
    )

    grounding = " Make it detailed and specific — several sentences, roughly 600-800 characters."
    if seed_stats:
        mean_len = seed_stats.get("mean_chars")
        if mean_len:
            grounding = (
                f" Aim for roughly {int(mean_len)} characters, matching the "
                "Enron-derived length distribution."
            )
    # Roughly half the benign mails carry an ordinary legitimate link, so that
    # "contains a URL" is NOT a giveaway for the phishing class in EXP-1 (the
    # phishing corpus always includes a link). Deterministic by index.
    link_hint = (
        " Include exactly one ordinary LEGITIMATE work link (e.g. an internal "
        "wiki/SharePoint/calendar page on the company's real domain) as a full "
        "http(s):// link."
        if index % 2 == 0
        else " Do not include any links."
    )
    return (
        f"Persona: {role} at {company}. Write ONE realistic, fully LEGITIMATE work "
        f"email this person would genuinely send or receive — a real business message, "
        f"NOT phishing and NOT a security test. Use their actual working context so it "
        f"is specific (reference real colleagues BY NAME and an ongoing project or "
        f"responsibility):\n{ctx_block}\n"
        f"Vary the topic for email #{index} (e.g. a project update, a request to a "
        f"named colleague, scheduling, a document hand-off, a question about routine "
        f"work). Calm professional tone. Write the email in ENGLISH only.{link_hint}"
        f"{grounding}"
    )


# --------------------------------------------------------------------------- #
# LLM client (lazy; never constructed at import)
# --------------------------------------------------------------------------- #
def _make_client():
    """Construct the twin-role OpenAI-compatible client (lazy import).

    Uses the dedicated TWIN model (``config.PPD_TWIN_MODEL``, default gemma-4-31b)
    so twins are generated by a distinct family from the attack generator/critic.
    """
    import config

    return config.make_client_for(config.PPD_TWIN_MODEL)


def _chat(client: Any, system: str, user: str, temperature: float) -> str:
    """One OpenAI-compatible chat completion; returns the message content."""
    import config

    resp = client.chat.completions.create(
        model=config.PPD_TWIN_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def _parse_twin_json(raw: str) -> dict[str, Any]:
    """Parse the LLM JSON, tolerating accidental markdown fences (like build_profiles)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON for twin: {exc}\n---\n{raw[:500]}")
    if not isinstance(data, dict):
        raise ValueError("Twin JSON root must be an object")
    allowed = set(PROFILE_SCHEMA["properties"])
    return {k: v for k, v in data.items() if k in allowed}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_twin(
    twin_id: str,
    llm_client: Any | None = None,
    seed_stats: dict[str, Any] | None = None,
    org_ctx: dict[str, Any] | None = None,
) -> VictimProfile:
    """Agentically generate one SYNTHETIC digital twin as a :class:`VictimProfile`.

    The LLM invents a plausible persona filling C1..C8. Because the persona is
    fabricated there is **no real subject** — so ``consent``/``c7_consent`` are set
    True vacuously and the twin is tagged ``synthetic`` (no RODO/IRB applies).

    Parameters
    ----------
    twin_id:
        Stable id for the twin (e.g. ``"twin-C1"``). Used as ``profile_id`` and as
        the only persona seed, so generation is reproducible at temperature 0.
    llm_client:
        Optional pre-built OpenAI-compatible client. In tests, pass a canned client
        so the path runs fully offline (no network). If omitted, one is built from
        environment config.
    seed_stats:
        Optional Enron-derived distributional stats (see :func:`enron_seed_stats`)
        used to ground the persona's voice. ``None`` ⇒ ungrounded (still valid).

    Returns
    -------
    VictimProfile
        A twin with C1..C8 populated, ``synthetic`` marked, and a synthetic
        provenance tag on every emitted category field.
    """
    client = llm_client or _make_client()
    prompt = build_twin_prompt(twin_id, seed_stats, org_ctx)
    raw = _chat(
        client,
        system="You output strict JSON only.",
        user=prompt,
        temperature=0.0,  # deterministic persona given (twin_id, seed_stats)
    )
    parsed = _parse_twin_json(raw)

    # Graph-native: pin employer + relationships to the designed org node so the
    # twin's C2 are REAL edges of the shared communication graph (other twins),
    # regardless of what the LLM filled. This is what unifies the content and
    # provenance halves of the study onto one population.
    if org_ctx:
        c1 = dict(parsed.get("c1") or {})
        c1["company"] = org_ctx["org"]
        parsed["c1"] = c1
        c2 = dict(parsed.get("c2") or {})
        c2["manager"] = org_ctx["manager_name"] or ""
        c2["coworkers"] = list(org_ctx["coworker_names"])
        c2["external_contacts"] = list(org_ctx["external_names"])
        parsed["c2"] = c2

    # Stamp identity + synthetic-data markers. consent/c7_consent are vacuously
    # True: a fabricated persona has no real subject to protect (no RODO/IRB).
    parsed["profile_id"] = twin_id
    parsed["consent"] = True
    parsed["c7_consent"] = True

    # Mark synthetic via provenance so consumers can distinguish twins from real
    # OSINT profiles without changing the VictimProfile schema. We also add a
    # dedicated marker key. (VictimProfile.from_json rejects unknown top-level
    # keys, so the marker lives inside the existing provenance dict.)
    provenance = dict(parsed.get("provenance") or {})
    provenance.setdefault("__synthetic__", SYNTHETIC_PROVENANCE_TAG)
    # Tag each emitted field's provenance as synthetic for auditability.
    for code in (c for c, _ in TAXONOMY_CATEGORIES):
        bucket = parsed.get(code.lower())
        if isinstance(bucket, dict):
            for fname in bucket:
                provenance.setdefault(f"{code}.{fname}", SYNTHETIC_PROVENANCE_TAG)
    parsed["provenance"] = provenance

    return VictimProfile.from_json(parsed)


def is_synthetic(twin: VictimProfile) -> bool:
    """True iff the profile was produced by :func:`generate_twin` (synthetic tag)."""
    return twin.provenance.get("__synthetic__") == SYNTHETIC_PROVENANCE_TAG


def generate_normal_mailbox(
    twin: VictimProfile,
    n: int,
    llm_client: Any | None = None,
    seed_stats: dict[str, Any] | None = None,
) -> list[str]:
    """Generate the twin's synthetic "normal communication" corpus (benign baseline).

    These are the *personal-baseline* emails the EXP-4 anomaly model fits on — the
    twin's everyday, non-phishing traffic. Optionally conditioned on Enron-derived
    ``seed_stats`` so length/voice are grounded in a public corpus.

    Parameters
    ----------
    twin:
        The synthetic persona (from :func:`generate_twin`).
    n:
        Number of normal emails to generate.
    llm_client:
        Optional pre-built client (pass a canned one in tests for offline runs).
    seed_stats:
        Optional Enron-derived stats to ground length/topic distributions.

    Returns
    -------
    list[str]
        ``n`` synthetic email texts (subject + body). All benign (label 0).
    """
    if n <= 0:
        return []
    client = llm_client or _make_client()
    emails: list[str] = []
    name = _persona_name(twin.profile_id)
    for i in range(n):
        prompt = _build_mailbox_prompt(twin, i, seed_stats)
        # Mild temperature for topic variety across the mailbox; still seedable.
        raw = _chat(client, _NORMAL_MAILBOX_SYSTEM, prompt, temperature=0.7)
        # Scrub residual bracketed placeholders ([Name], [Date]...) so they cannot
        # become a class giveaway: benign must not differ from attacks by "has [..]".
        emails.append(_scrub_placeholders(raw, twin.profile_id, name))
    return emails


# --------------------------------------------------------------------------- #
# Enron grounding (public corpus → simple distributional stats)
# --------------------------------------------------------------------------- #
def enron_seed_stats(
    max_users: int = 10,
    min_msgs: int = 30,
) -> dict[str, Any] | None:
    """Compute simple distributional stats from the parsed Enron corpus.

    Used to *ground* synthetic generation on a public corpus (mean message length,
    vocabulary richness) so twins read like genuine workplace mail. This is the
    public-data calibration step that lets us claim the synthetic twins are
    grounded rather than free-floating.

    Returns ``None`` if Enron is not downloaded/extracted yet (callers then run
    ungrounded). When present it returns e.g.::

        {"mean_chars": 812.4, "vocab_richness": 0.41, "n_messages": 1530,
         "n_users": 10, "source": "enron"}

    Notes
    -----
    Reuses :func:`data.load_enron.parse_maildir`. No network I/O here — it only
    reads an already-extracted maildir. If Enron has not been fetched
    (``data/download_public.py --enron``), it returns ``None``.
    """
    from config import DATA_DIR

    enron_root = DATA_DIR / "enron"
    maildir = enron_root / "maildir"
    if not maildir.exists():
        # TODO: load Enron stats once the corpus is downloaded
        # (`python data/download_public.py --enron`). Until then twins are
        # generated ungrounded, which is still valid (just less calibrated).
        return None

    from data.load_enron import parse_maildir

    users = parse_maildir(enron_root, min_msgs=min_msgs, max_users=max_users)
    if not users:
        return None

    total_chars = 0
    n_messages = 0
    all_tokens: list[str] = []
    for texts in users.values():
        for t in texts:
            total_chars += len(t)
            n_messages += 1
            all_tokens.extend(t.lower().split())

    if n_messages == 0:
        return None

    vocab_richness = (len(set(all_tokens)) / len(all_tokens)) if all_tokens else 0.0
    return {
        "mean_chars": round(total_chars / n_messages, 1),
        "vocab_richness": round(vocab_richness, 3),
        "n_messages": n_messages,
        "n_users": len(users),
        "source": "enron",
    }
