# headless_hibs

A research codebase exploring **Liquid Time-Constant Networks (LTC)** with a
fiber-bundle geometric backbone and a small ecosystem for training, exporting,
and inspecting models. The project is best read as an evolving notebook of
"what works and what doesn't" for a ~50K-parameter active-reasoning system
that learns online inside a memory-equipped virtual world.

This repository is published as **source only**. Tests, datasets, experiment
artifacts, internal logs, and AI-tooling directories are intentionally excluded
— see `.gitignore` for the full exclusion list.

## Repository layout

```
headless_hibs/
├── liquid_net/              # Core liquid neural network library
│   ├── models/              # LiquidNet, LTC cell, sparse LTC cell
│   ├── solvers/             # Euler / RK4 / generic ODE solver
│   ├── training/            # Training loop and losses
│   ├── analysis/            # Dynamics analysis utilities
│   └── configs/             # Default model configuration
├── hibs_studio/             # Model browser / trainer UI helpers
├── hibs_export/             # Export scripts (mobile / on-device formats)
├── twistor_studio/          # Studio utilities for twistor-based models
├── twistor_LMT/             # Twistor-based liquid time-constant package
├── scripts/                 # Training & smoke-test entry points
├── main.py                  # Top-level entry point
├── README.md
└── .gitignore
```

## Installation

Targets **Python 3.12+** and depends on PyTorch / NumPy. No `requirements.txt`
is shipped in the public release — install the runtime dependencies you need
for the path you take:

```bash
pip install torch numpy matplotlib pyyaml
```

## Quick start

```bash
# Smoke test the install
python scripts/smoke_test.py

# Train a small LiquidNet
python scripts/train_hibs_0_16.py

# Top-level entry point
python main.py
```

## Library usage

The `liquid_net` package exposes the model, solvers, and training utilities
directly:

```python
from liquid_net.models import LiquidNet
from liquid_net.solvers import rk4
from liquid_net.training import train

model = LiquidNet(input_dim=1, hidden_dim=16, output_dim=1)
```

## Research trajectory — V18 → V20.3

The most recent work turned the static inference stack into an **active,
continually-learning reasoner**. The headline numbers (full detail in
`docs/v16_5_to_v20_3_synthesis.md`):

| Version | Idea | Result |
| :-----: | :--- | :----- |
| V18.0 | Naive online BPTT (EFE action selection) | Diverges — PPL → ∞, drift 3.60 |
| V18.1 | Wave-driven local update | Stable, NLL −4.5 nats over 8K tokens |
| V18.5 | True epistemic value (IG via logvar) | Action match 6.0% (vs 2.2% random) |
| V18.7 | Internal-signal-only update (no backward) | Stable, KL 7.04→1.52, but NLL flat |
| V18.9 | Three-level world (text / FS / shell) | 0% feedback density → no learning |
| V19 | Discrete primitives (K=5: READ/WRITE/SEARCH/EDIT/ERROR) | 100% feedback density, world_loss 4.84→3.32 |
| V20.0 | Notebook World + uniform EFE | Element balance 17–23%, world_loss 3.56 |
| V20.1 | Irreversible "tombstone" tokens | TOME colludes at 98% — new shortcut |
| V20.2 | Decay + cooldown + weighted loss | Weighted path works (world_loss 1.80); hard cooldown breaks |
| **V20.3** | **Two-stage training + probabilistic cooldown** | **PPL 1e52 → 388, world_loss 0.09, WRITE→READ pair 99.5%** |

### Seven lessons from the V16.5 → V20.3 sweep

1. **Feedback density is the prerequisite for continual learning.** V18–V18.9
   spent six iterations confirming that until actions get 100% feedback,
   nothing else matters.
2. **EFE signals are real but biased.** Action selection via Expected Free
   Energy beats random by 1.57×, but "what to learn" and "how to learn" are
   separate problems.
3. **Physically correct ≠ engineering efficient.** Wave-driven updates
   converge but at ~10⁵–10⁶ tokens, not 10⁴.
4. **Action spaces must be small.** Going from 109⁴ token sequences to K=5
   discrete primitives was the unlock.
5. **Weighted loss and cooldown mechanisms pull in opposite directions.**
   Letting tombstones become learnable makes them stop triggering cooldown.
6. **Procedural and declarative causality are separable.** V20.3 learns
   "WRITE then READ" (99.5%) but not "what READ returns" — analogous to a
   baby who knows to press a button without knowing what the screen shows.
7. **Decoupling language and world learning is structural.** Stage-1 offline
   LM pretraining (lr=3e-4, 2 epoch) followed by Stage-2 frozen-LM WorldHead
   training (lr=1e-4, 1000 steps) cut PPL by 50 orders of magnitude and
   boosted world learning 20×.

### What survived into V20.3

| Component | Introduced | Status |
| :--------- | :--------: | :----: |
| Fiber-bundle backbone (`TwistFiberBundle`) | V16.6 | ✅ kept |
| Variational SSM scan with KL | V17 | ✅ kept |
| Structured prior (`log σ_p² − α·log(1+‖κ‖)`) | V17.1 | ✅ kept |
| True epistemic value (IG via logvar) | V18.5 | ✅ kept |
| Discrete K=5 primitives | V19 | ✅ kept |
| Uniform EFE pragmatic term | V20 | ✅ kept |
| Weighted WorldHead loss (`w_tome=0.3`) | V20.2 | ✅ kept |
| Two-stage training | V20.3 | ✅ kept |

### Roadmap

- **V20.4** — resolve the weighting-vs-cooldown tension (TOME back to 0.1,
  hard cap on ERASE usage, `N_RESPONSE=32`).
- **V21** — long-horizon continual learning (2K+ steps, push world_loss
  toward 0.01).
- **V22** — transfer to real-world interaction (real file I/O).
- **V23** — multi-agent coordination (two V20.3 models sharing a notebook).

The full report trail lives under `docs/` in the working tree; the public
release omits the full document set.

## License

MIT — see source headers for individual module licenses.
