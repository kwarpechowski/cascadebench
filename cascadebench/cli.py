# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Kamil Warpechowski, Bogdan Ksiezopolski.
"""cascadebench.cli — command-line interface for the testbed.

    python -m cascadebench demo
    python -m cascadebench pareto   --graph synthetic:600 --seeds 3
    python -m cascadebench adaptive --graph synthetic:600 --fanout 5
    python -m cascadebench panel     --graph email-Eu-core
    python -m cascadebench leak-audit --graph enron
"""
from __future__ import annotations

import argparse

from . import experiments as E
from . import load, CascadeStrategy, build_scenario, get_panel, evaluate


def _demo(args):
    g = load(args.graph)
    print(f"graph: {g.name}, {len(g)} nodes")
    sc = build_scenario(g, CascadeStrategy(fanout=args.fanout, spread=args.spread), 0)
    print(f"scenario: {len(sc.events)} events, {sc.n_attacks} attacks, {sc.infected_frac:.0%} infected")
    for n, m in evaluate(sc, get_panel(), 0).items():
        print(f"  {n:18s} AUC={m['auc']:.3f}  R@1%={m['recall@0.01']:.3f}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cascadebench",
                                description="Leak-aware adversarial testbed for graph-based phishing detectors")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="quick demo on a single scenario")
    d.add_argument("--graph", default="synthetic:500"); d.add_argument("--fanout", type=int, default=8)
    d.add_argument("--spread", type=int, default=1); d.set_defaults(fn=_demo)

    from .search import arms_race
    ar = sub.add_parser("arms-race", help="automatic attack<->defense arms race (red/blue co-search)")
    ar.add_argument("--graph", default="synthetic:400"); ar.add_argument("--rounds", type=int, default=3)
    ar.add_argument("--budget", type=int, default=6); ar.add_argument("--reach-min", type=float, default=0.5)
    ar.set_defaults(fn=lambda a: arms_race(a.graph, rounds=a.rounds, budget=a.budget, reach_min=a.reach_min))

    pp = sub.add_parser("predictivity", help="predictivity probe: detector rank transfer across topologies")
    pp.add_argument("--seeds", type=int, default=2)
    pp.set_defaults(fn=lambda a: E.predictivity_probe(seeds=a.seeds))

    fb = sub.add_parser("fabrication", help="(D) edge-fabricating attack (OSINT) vs panel")
    fb.add_argument("--graph", default="synthetic:600"); fb.add_argument("--seeds", type=int, default=3)
    fb.set_defaults(fn=lambda a: E.fabrication_sweep(a.graph, a.seeds))

    mc = sub.add_parser("mechanism", help="(C) ablation: GCN inflation ~ clique structure vs modularity")
    mc.add_argument("--seeds", type=int, default=3)
    mc.set_defaults(fn=lambda a: E.mechanism_ablation(seeds=a.seeds))

    rg = sub.add_parser("robustness-grid", help="(A) finding stability vs design choices")
    rg.set_defaults(fn=lambda a: E.robustness_grid())

    for name, fn in (("pareto", E.pareto), ("adaptive", E.adaptive),
                     ("panel", E.panel), ("leak-audit", E.leak_audit)):
        sp = sub.add_parser(name, help=fn.__doc__.splitlines()[0] if fn.__doc__ else name)
        sp.add_argument("--graph", default="enron" if name == "leak-audit" else "synthetic:600")
        sp.add_argument("--seeds", type=int, default=3)
        if name == "adaptive":
            sp.add_argument("--fanout", type=int, default=5)
        sp.set_defaults(fn=lambda a, _fn=fn: _fn(a.graph, a.seeds, **({"fanout": a.fanout}
                                                                      if hasattr(a, "fanout") and a.cmd == "adaptive" else {})))

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
