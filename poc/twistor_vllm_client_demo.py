"""
Twistor-LMT OpenAI-style client demo
=====================================

Demonstrates driving the Twistor-LMT via the OpenAI-compatible HTTP API.
Shows three flows:
  1. /v1/chat/completions  - text prompt -> action vector
  2. /twistor/act         - raw numeric observation -> action
  3. /twistor/predict     - multi-step forecast

Assumes the server is running on http://127.0.0.1:8000.
"""
import json
import os
import sys
import time

import numpy as np
import requests

BASE = os.environ.get("TWISTOR_BASE", "http://127.0.0.1:8765")


def main():
    # 1) Health
    r = requests.get(f"{BASE}/health", timeout=5)
    r.raise_for_status()
    print("== /health ==")
    print(json.dumps(r.json(), indent=2))

    # 2) List models
    r = requests.get(f"{BASE}/v1/models", timeout=5)
    r.raise_for_status()
    models = r.json()["data"]
    if not models:
        print("No models loaded on the server", file=sys.stderr)
        sys.exit(1)
    model_id = models[0]["id"]
    in_dim = models[0]["input_dim"]
    out_dim = models[0]["output_dim"]
    print(f"\n== /v1/models ==  using model_id={model_id}  in_dim={in_dim}  out_dim={out_dim}")

    # 3) /v1/chat/completions with text prompt
    print("\n== /v1/chat/completions (text prompt) ==")
    r = requests.post(
        f"{BASE}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": [
                {"role": "system", "content": "twistor control plane"},
                {"role": "user", "content": "balance pole, tilt = -0.1 rad, ang vel = 0.05"},
            ],
        },
        timeout=10,
    )
    r.raise_for_status()
    j = r.json()
    print("reply:", j["choices"][0]["message"]["content"])
    print("action:", j["x_twistor"]["action"])

    # 4) /twistor/act with raw observation
    print("\n== /twistor/act (raw obs) ==")
    rng = np.random.default_rng(0)
    obs = rng.standard_normal(in_dim).tolist()
    r = requests.post(
        f"{BASE}/twistor/act",
        json={"observation": obs, "reset": True},
        timeout=10,
    )
    r.raise_for_status()
    j = r.json()
    print(f"obs[0..3]    = {[f'{x:+.3f}' for x in obs[:3]]} ...")
    print(f"action       = {[f'{x:+.4f}' for x in j['action']]}")

    # 5) /twistor/predict with multi-step forecast
    print("\n== /twistor/predict (multi-step forecast) ==")
    seq = rng.standard_normal((20, in_dim)).tolist()
    r = requests.post(
        f"{BASE}/twistor/predict",
        json={"sequence": seq, "horizon": 5},
        timeout=10,
    )
    r.raise_for_status()
    j = r.json()
    print(f"n_input={j['n_input']}  horizon={j['horizon']}")
    print(f"history[0]   = {[f'{x:+.3f}' for x in j['history_outputs'][0]]}")
    print(f"history[-1]  = {[f'{x:+.3f}' for x in j['history_outputs'][-1]]}")
    print(f"forecast[0]  = {[f'{x:+.3f}' for x in j['forecast'][0]]}")
    print(f"forecast[-1] = {[f'{x:+.3f}' for x in j['forecast'][-1]]}")

    print("\n[OK] All endpoints responded.")


if __name__ == "__main__":
    main()
