# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""Graph-first design of the synthetic population: organizations -> people -> twins.

Instead of generating isolated personas and linking them post hoc, we FIRST design an
organizational communication graph and then generate each twin INTO its node. The graph
is the backbone of the whole study: the same population is attacked at the content level
(personalized phishing) and provides the sender--recipient graph for the provenance/BEC
experiment. A twin's C2 (relationships) are therefore REAL other twins, with real names
and corporate addresses, not free-floating fabrications.

Structure (deterministic, no random):
  * ``ceil(N / ORG_SIZE)`` organizations (distinct sectors), ~ORG_SIZE people each.
    The default population is N=150 twins to MATCH the Enron maildir's 150 users, so
    the synthetic and real communication graphs are compared at the same scale.
  * Per organization: 1 director, ~1 manager per 4 staff, rest specialists/analysts.
  * Intra-org edges: everyone in an org corresponds with everyone else (small office).
  * Inter-org edges: a ring of business partnerships -- each org's director knows the
    next org's director, and one specialist has a counterpart in the partner org.

Each twin node carries: org/domain/sector, role/seniority, persona name, corporate
address, manager (twin id), coworkers (twin ids), and external contacts (twin ids in
partner orgs). ``contacts(t)`` = coworkers + external = the twin's communication graph.
"""

from __future__ import annotations

import math

# Firm pool (name, domain, sector). Large enough for N up to ~160 twins at ORG_SIZE=8
# (20 firms). Distinct sectors so the LLM-generated personas stay diverse.
FIRMS = [
    ("Northwind Capital Partners", "northwind-capital.com", "finance/banking"),
    ("Meridian Health Group", "meridian-health.org", "healthcare"),
    ("Cobalt Systems", "cobaltsystems.io", "software/IT"),
    ("Lex Harbor Legal", "lexharbor.com", "legal"),
    ("Greenfield Industries", "greenfield-ind.com", "manufacturing/energy"),
    ("Arcadia Logistics", "arcadia-logistics.com", "logistics/supply-chain"),
    ("Helvetia Insurance", "helvetia-insure.com", "insurance"),
    ("Brightpath Education", "brightpath-edu.org", "education/academia"),
    ("Vantage Retail Group", "vantage-retail.com", "retail/e-commerce"),
    ("Aurora Media", "aurora-media.tv", "media/advertising"),
    ("Sterling Construction", "sterling-build.com", "construction/real-estate"),
    ("Polaris Pharma", "polaris-pharma.com", "pharmaceutical/biotech"),
    ("Ironclad Security", "ironclad-sec.com", "cybersecurity/defense"),
    ("Continental Foods", "continental-foods.com", "food/agriculture"),
    ("Zephyr Airlines Services", "zephyr-air.com", "aviation/travel"),
    ("Quantis Consulting", "quantis-consult.com", "management consulting"),
    ("Atlas Mining Corporation", "atlas-mining.com", "mining/resources"),
    ("Marlowe Public Office", "marlowe.gov", "public sector/government"),
    ("Beacon Telecom", "beacon-telecom.net", "telecommunications"),
    ("Solenne Automotive", "solenne-auto.com", "automotive"),
]
ORG_SIZE = 8  # target people per organization (a "small office")

# Procedural firms for scale (>len(FIRMS)). The first len(FIRMS) indices return
# EXACTLY FIRMS (N=150 reproduces 1:1); higher indices are generated deterministically
# from the index (no LLM, no randomness). The index in the name/domain guarantees uniqueness.
_FIRM_PREF = ["Apex", "Vertex", "Summit", "Cedar", "Onyx", "Pioneer", "Crest", "Nimbus",
              "Halcyon", "Verde", "Titan", "Lumen", "Forge", "Harbor", "Kestrel", "Bastion"]
_FIRM_SUFF = ["Group", "Systems", "Partners", "Industries", "Holdings", "Labs", "Works",
              "Networks", "Solutions", "Dynamics"]
_SECTORS = ["finance/banking", "healthcare", "software/IT", "legal", "manufacturing/energy",
            "logistics/supply-chain", "insurance", "education/academia", "retail/e-commerce",
            "media/advertising", "construction/real-estate", "pharmaceutical/biotech",
            "cybersecurity/defense", "food/agriculture", "aviation/travel",
            "management consulting", "mining/resources", "public sector/government",
            "telecommunications", "automotive"]


def _firm(o: int) -> tuple[str, str, str]:
    """Firm for organization o: fixed FIRMS for o<len(FIRMS), otherwise procedural."""
    if o < len(FIRMS):
        return FIRMS[o]
    pref = _FIRM_PREF[o % len(_FIRM_PREF)]
    suff = _FIRM_SUFF[(o // len(_FIRM_PREF)) % len(_FIRM_SUFF)]
    sec = _SECTORS[o % len(_SECTORS)]
    return (f"{pref} {suff} {o}", f"{pref.lower()}-{suff.lower()}-{o}.com", sec)


def _roles_for(size: int) -> list[tuple[str, str]]:
    """Seniority scaffold for an org of ``size`` people (1 director, ~1 mgr / 4 staff)."""
    roles: list[tuple[str, str]] = [("Managing Director", "executive")]
    n_mgr = max(1, (size - 1) // 4)
    roles += [("Senior Manager", "senior")] * n_mgr
    for j in range(size - len(roles)):
        roles.append(("Specialist", "mid") if j % 2 == 0 else ("Analyst", "junior"))
    return roles[:size]


def _persona_name(twin_id: str) -> str:
    from data.twin_generator import _persona_name as pn
    return pn(twin_id)


def _email(name: str, domain: str) -> str:
    import re
    parts = re.findall(r"[a-z]+", name.lower())
    local = f"{parts[0]}.{parts[-1]}" if len(parts) >= 2 else (parts[0] if parts else "user")
    return f"{local}@{domain}"


def build(n_twins: int = 150) -> dict[str, dict]:
    """Return ``{twin_id: node}`` for ``n_twins`` twins laid out on the org graph.

    Twins are split into ``ceil(n_twins/ORG_SIZE)`` balanced organizations (sizes
    differ by at most one). Default ``n_twins=150`` matches the Enron maildir.
    """
    n_orgs = max(1, math.ceil(n_twins / ORG_SIZE))
    # >len(FIRMS) organizations: firms generated procedurally by _firm() (for scale)
    # balanced org sizes: first (n_twins % n_orgs) orgs get one extra person
    base, extra = divmod(n_twins, n_orgs)
    sizes = [base + (1 if o < extra else 0) for o in range(n_orgs)]

    nodes: dict[str, dict] = {}
    members: list[list[str]] = []
    cursor = 0
    for o in range(n_orgs):
        mem = [f"twin-{cursor + j + 1:04d}" for j in range(sizes[o])]
        cursor += sizes[o]
        members.append(mem)

    for o in range(n_orgs):
        oname, odom, osec = _firm(o)
        mem = members[o]
        roles = _roles_for(len(mem))
        director = mem[0] if mem else None
        managers = [mem[k] for k in range(1, len(mem)) if roles[k][1] == "senior"]
        for j, tid in enumerate(mem):
            role, sen = roles[j]
            if j == 0:
                manager = None
            elif sen == "senior":
                manager = director
            else:
                manager = managers[j % len(managers)] if managers else director
            name = _persona_name(tid)
            nodes[tid] = {
                "org": oname, "domain": odom, "sector": osec, "role": role,
                "seniority": sen, "name": name, "address": _email(name, odom),
                "manager": manager, "coworkers": [m for m in mem if m != tid],
                "external": [],
            }

    # Guarantee unique corporate addresses: deterministic names can collide (two
    # "Greta Dvorak" in one firm), which would make a sender address ambiguous and
    # break provenance. On collision, disambiguate the local-part with the twin's
    # serial (e.g. greta.dvorak.0036@...) — real orgs do exactly this.
    seen: set[str] = set()
    for tid in (f"twin-{i+1:04d}" for i in range(n_twins)):
        if tid not in nodes:
            continue
        addr = nodes[tid]["address"]
        if addr in seen:
            local, _, dom = addr.partition("@")
            serial = tid.split("-")[-1]
            addr = f"{local}.{serial}@{dom}"
        seen.add(addr)
        nodes[tid]["address"] = addr

    # inter-org partnerships: ring of directors + one specialist counterpart per edge
    for o in range(n_orgs):
        nxt = (o + 1) % n_orgs
        if not members[o] or not members[nxt]:
            continue
        a_dir, b_dir = members[o][0], members[nxt][0]
        nodes[a_dir]["external"].append(b_dir)
        nodes[b_dir]["external"].append(a_dir)
        if len(members[o]) > 3 and len(members[nxt]) > 3:    # specialist counterparts
            a_sp, b_sp = members[o][3], members[nxt][3]
            nodes[a_sp]["external"].append(b_sp)
            nodes[b_sp]["external"].append(a_sp)
    return nodes


def build_structural(n_twins: int) -> dict[str, dict]:
    """Lightweight builder for SCALE: only {twin_id: {org, coworkers, external}} without
    names/addresses (does not import twin_generator/config/LLM). Just the structure the
    cascade graph needs. Org/contact logic identical to build()."""
    n_orgs = max(1, math.ceil(n_twins / ORG_SIZE))
    base, extra = divmod(n_twins, n_orgs)
    sizes = [base + (1 if o < extra else 0) for o in range(n_orgs)]
    members, cursor = [], 0
    for o in range(n_orgs):
        members.append([f"twin-{cursor + j + 1:04d}" for j in range(sizes[o])])
        cursor += sizes[o]
    nodes: dict[str, dict] = {}
    for o in range(n_orgs):
        oname = _firm(o)[0]
        for tid in members[o]:
            nodes[tid] = {"org": oname, "coworkers": [m for m in members[o] if m != tid],
                          "external": []}
    for o in range(n_orgs):
        nxt = (o + 1) % n_orgs
        if not members[o] or not members[nxt]:
            continue
        a_dir, b_dir = members[o][0], members[nxt][0]
        nodes[a_dir]["external"].append(b_dir); nodes[b_dir]["external"].append(a_dir)
        if len(members[o]) > 3 and len(members[nxt]) > 3:
            a_sp, b_sp = members[o][3], members[nxt][3]
            nodes[a_sp]["external"].append(b_sp); nodes[b_sp]["external"].append(a_sp)
    return nodes


def contacts(node: dict) -> list[str]:
    """Twin ids this twin corresponds with (coworkers + external partners)."""
    return list(dict.fromkeys(node["coworkers"] + node["external"]))


def org_context_for(twin_id: str, nodes: dict[str, dict]) -> dict:
    """Context block fed to the twin generator so the persona is born INTO its node.

    Carries the org/role/address plus the REAL names of the twin's manager,
    coworkers and external partners (other twins) --- the generator conditions C1..C8
    on these and overwrites C2 with them, so a twin's relationships are genuine edges
    in the communication graph rather than free-floating fabrications.
    """
    nd = nodes[twin_id]
    mgr = nodes[nd["manager"]]["name"] if nd["manager"] else None
    return {
        "org": nd["org"], "domain": nd["domain"], "sector": nd["sector"],
        "role": nd["role"], "seniority": nd["seniority"], "name": nd["name"],
        "address": nd["address"], "manager_name": mgr,
        "coworker_names": [nodes[c]["name"] for c in nd["coworkers"]],
        "external_names": [nodes[c]["name"] for c in nd["external"]],
    }


def write_csvs(nodes: dict[str, dict]) -> tuple:
    """Write results/twin_self.csv and results/twin_network.csv from the graph."""
    import csv

    from config import RESULTS_DIR
    self_path = RESULTS_DIR / "twin_self.csv"
    with self_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["twin_id", "name", "address", "domain", "org", "role"])
        for t, nd in nodes.items():
            w.writerow([t, nd["name"], nd["address"], nd["domain"], nd["org"], nd["role"]])
    net_path = RESULTS_DIR / "twin_network.csv"
    with net_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["twin_id", "contact_twin_id", "contact_name", "contact_address"])
        for t, nd in nodes.items():
            for u in contacts(nd):
                w.writerow([t, u, nodes[u]["name"], nodes[u]["address"]])
    return self_path, net_path


if __name__ == "__main__":
    nd = build()
    write_csvs(nd)
    degs = [len(contacts(n)) for n in nd.values()]
    n_orgs = len({n["org"] for n in nd.values()})
    print(f"[org-graph] {len(nd)} twins in {n_orgs} orgs | "
          f"mean degree {sum(degs)/len(degs):.1f}")
    # sample
    for t in list(nd)[:2]:
        n = nd[t]
        print(f"  {t}: {n['name']} | {n['role']} @ {n['org']} | mgr={n['manager']} "
              f"| {len(contacts(n))} contacts")
