# headless_hibs

A research codebase for **Liquid Time-Constant Networks (LTC)** and supporting
training infrastructure. The project focuses on compact, ODE-based recurrent
models with stable long-horizon dynamics, plus a small ecosystem for training,
exporting, and inspecting models.

This repository is published as **source only**. Tests, datasets, experiment
artifacts, and large generated files are intentionally excluded — see
`.gitignore` for the full exclusion list.

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

The project targets **Python 3.12+** and depends on PyTorch / NumPy. No
`requirements.txt` is shipped in the public release — install the runtime
dependencies you need for the path you take:

```bash
pip install torch numpy matplotlib pyyaml
```

## Quick start

### Smoke test

```bash
python scripts/smoke_test.py
```

### Train a small LiquidNet

```bash
python scripts/train_hibs_0_16.py
```

### Run the top-level entry point

```bash
python main.py
```

## Library usage

The `liquid_net` package exposes the model, solvers, and training utilities
directly:

```python
from liquid_net.models import LiquidNet
from liquid_net.solvers import rk4
from liquid_net.training import train

model = LiquidNet(
    input_dim=1,
    hidden_dim=16,
    output_dim=1,
)
```

## License

MIT — see source headers for individual module licenses.
