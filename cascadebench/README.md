# cascadebench — a leak-aware adversarial testbed for graph-based phishing detectors

A controlled, synthetic **test environment** for evaluating graph-based phishing and lateral
movement detectors against **adaptive attackers**, with ground-truth labels and **leak-aware**
evaluation. Real lateral-phishing data is proprietary and full of confounds (rhythm/degree leaks);
the testbed provides controllability, ground truth, and confound control that real corpora cannot.

## Installation
Requires: `numpy`, `scikit-learn`, `lightgbm`, `torch` (CPU OK). Run through the project venv.
```bash
./venv/Scripts/python -m cascadebench demo
```

## Quick start (API)
```python
import cascadebench as cb

g  = cb.Graph.synthetic(800)                      # or cb.load("email-Eu-core") / "enron"
sc = cb.build_scenario(g, cb.CascadeStrategy(fanout=8, spread=2), seed=0)
res = cb.evaluate(sc, cb.get_panel(), seed=0)     # AUC + Recall@{0.1%,0.5%,1%,5%}
```

## Components
| Module | Role |
|------|------|
| `graph.py` | graphs: `Graph.synthetic(N)` (procedural), `Graph.from_edgelist` (SNAP), `Graph.enron` |
| `attack.py` | `CascadeStrategy` (fan-out=reach; spread/mimicry/fabrication=evasion), `build_scenario` |
| `detect.py` | detector panel with a common interface (1-hop, COMPA, GCN-static, context, temporal) |
| `features.py` / `models.py` | features and lightweight graph models (pure torch) |
| `evaluate.py` | **leak-aware**: victim/org-level split, Recall@low-FPR, shuffle control, off-hours audit |
| `experiments.py` | reproducible: `pareto`, `adaptive`, `panel`, `leak_audit` |

## CLI
```bash
python -m cascadebench pareto    --graph synthetic:600 --seeds 3   # evasion<->reach front
python -m cascadebench adaptive  --graph synthetic:600 --fanout 5  # adaptive attack at fixed reach
python -m cascadebench panel     --graph email-Eu-core            # panel under strategies
python -m cascadebench leak-audit --graph enron                   # rhythm-leak audit
```

## Extending
A new detector = a `Detector` subclass with `fit_score(scenario, train_mask, test_mask, seed)`.
A new attack strategy = fields on `CascadeStrategy` (or a custom event-generating function).
A new graph = `Graph.from_edgelist(path)` (format `src dst [timestamp]`).

## Leak-aware (how this differs from existing benchmarks)
- **matched-design benign** — same endpoint/degree distribution as the attack (removes the identity/degree confound),
- **off-hours audit** — checks whether the result rides on the assumption that "benign is always in rhythm",
- **shuffle control** — shuffling time must destroy the advantage (proof of causality),
- **operational metrics** — Recall at low FPR, not just AUC.
