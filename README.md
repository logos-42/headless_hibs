# headless_hibs

> **Languages / 语言:** [English](README.md) · [简体中文](README.zh-CN.md)

A research codebase exploring **Liquid Time-Constant Networks (LTC)** with a
fiber-bundle geometric backbone and a small ecosystem for training, exporting,
and inspecting models. The project is best read as an evolving notebook of
"what works and what doesn't" for a ~50K-parameter active-reasoning system
that learns online inside a memory-equipped virtual world.

This repository is published as **source only**. Tests, datasets, experiment
artifacts, internal logs, and AI-tooling directories are intentionally excluded
— see `.gitignore` for the full exclusion list.

---

## Table of contents

- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Library usage](#library-usage)
- [Research trajectory: V18 to V20.3](#research-trajectory-v18-to-v203)
  - [Seven lessons from the V16.5 to V20.3 sweep](#seven-lessons-from-the-v165-to-v203-sweep)
  - [What survived into V20.3](#what-survived-into-v203)
  - [Roadmap](#roadmap)
- [Documentation index](#documentation-index)
- [License](#license)

---

## Repository layout

```text
headless_hibs/
├── liquid_net/              # Core liquid neural network library
│   ├── models/              # LiquidNet, LTC cell, sparse LTC cell
│   ├── solvers/             # Euler / RK4 / generic ODE solver
│   ├── training/            # Training loop and losses
│   ├── analysis/            # Dynamics analysis utilities
│   └── configs/             # Default model configuration
├── hibs_studio/             # Model browser / trainer UI helpers
├── hibs_export/             # Export scripts (mobile / on-device formats)
├── hibs_cli/                # CLI interface for HIBS
├── twistor_studio/          # Studio utilities for twistor-based models
├── twistor_LMT/             # Twistor-based liquid time-constant package
├── headless/                # Headless bridge and training modules
├── models/                  # Pre-trained model checkpoints (.pt)
├── poc/                     # Proof-of-concept experiments
├── scripts/                 # Training & smoke-test entry points
├── main.py                  # Top-level entry point
├── README.md                # English README (this file)
├── README.zh-CN.md          # 简体中文 README
└── .gitignore
```

## 🔧 Installation

Targets **Python 3.12+** and depends on PyTorch / NumPy. No `requirements.txt`
is shipped in the public release — install the runtime dependencies you need
for the path you take:

```bash
pip install torch numpy matplotlib pyyaml
```

## ▶️ Quick start

```bash
# Smoke test the install
python scripts/smoke_test.py

# Train a small LiquidNet
python scripts/train_hibs_0_16.py

# Top-level entry point
python main.py
```

## 📚 Library usage

The `liquid_net` package exposes the model, solvers, and training utilities
directly:

```python
from liquid_net.models import LiquidNet
from liquid_net.solvers import rk4
from liquid_net.training import train

model = LiquidNet(input_dim=1, hidden_dim=16, output_dim=1)
```

## Research trajectory: V18 to V20.3

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
| V20.5 | Two-stage + healthy world (V20.4b-4) | PPL 544, content 84.5%, tombstone 1.9%, first dual win |
| V21 | Causal world with position binding | 87% set_decl but only 17.4% pos_decl (false positive) |
| V21.1 | Strict position evaluation | Exposed V21's 87% as set-theoretic false positive |
| **V22** | **WRITE freeze + big model + position weight** | **pos_decl 17.2% → 40.0% (+22.8%), first true declarative causality** |

### Seven lessons from the V16.5 to V20.3 sweep

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

- **V22.1** — better trained model (longer pretraining, PPL < 50)
- **V22.2** — better world (healthier content, lower tombstone)
- **V22.3** — multi-position learning (beyond position 0)
- **V23** — transfer to real-world interaction (real file I/O)
- **V24** — multi-agent coordination (two V20.3+ models sharing a notebook)

The full report trail lives under `docs/` in the working tree; the public
release omits the full document set.

## 📖 Documentation index

The public release ships only the source tree. The full research notebook
(test scripts, datasets, experiment reports, plans) is excluded by
`.gitignore` and lives in the working tree locally. When cloned with the
full set, the key entries are:

| Document | Topic |
| :------- | :----- |
| `docs/v16_5_to_v20_3_synthesis.md` | Seven-lesson synthesis of the whole sweep |
| `docs/v18_series_summary_report.md` | V18.0 → V18.8 evolution (online self-update) |
| `docs/v19_primitive_action_report.md` | V19 discrete K=5 primitives, 100% feedback |
| `docs/v20_3_two_stage_report.md` | V20.3 two-stage training, PPL 1e52 → 388 |
| `docs/v20_5_two_stage_notebook_report.md` | V20.5 two-stage + healthy world, first dual win |
| `docs/v21_1_proper_evaluation_report.md` | V21.1 strict position evaluation, exposed false positive |
| `docs/v22_frozen_world_report.md` | V22 WRITE freeze breakthrough, 40% declarative causality |
| `docs/v18_v20_5_comprehensive_report.md` | Comprehensive report from V18 to V20.5 |
| `docs/v20_4_healthier_world_plan.md` | V20.4 healthier world design plan |
| `docs/v20_4a_healthier_world_report.md` | V20.4a healthier world report |
| `docs/v20_4b_active_inference_report.md` | V20.4b active inference report |
| `docs/v20_4c_ablation_report.md` | V20.4c ablation study report |
| `docs/v21_causal_world_report.md` | V21 causal world report |
| `docs/v22_1_better_trained_report.md` | V22.1 better trained model report |
| `docs/v22_2_better_world_report.md` | V22.2 better world report |
| `docs/v22_frozen_world_plan.md` | V22 frozen world design plan |
| `docs/README_HIBS_0_16.md` / `docs/README_USER.md` | Library & user-facing overviews |
| `docs/FINAL_REPORT.md` | End-of-sprint consolidated report |

Source-of-truth modules worth reading first:

| Path | Why |
| :--- | :--- |
| `liquid_net/models/ltc_cell.py` | Core LTC cell (replaces V18-era SSM) |
| `liquid_net/models/sparse_ltc_cell.py` | Sparse variant for memory efficiency |
| `liquid_net/solvers/rk4.py` | Default ODE solver |
| `liquid_net/training/train.py` | Training loop and loss wiring |
| `scripts/train_hibs_0_16.py` | End-to-end training entry point |
| `scripts/smoke_test.py` | Minimal-install verification |

## 📄 License

MIT — see source headers for individual module licenses.
