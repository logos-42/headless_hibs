"""Smoke test for twistor_vllm_server on Windows Python 3.12."""
import sys
import numpy as np

sys.path.insert(0, ".")

from poc.twistor_vllm_server import load_twistor_model, select_device

dev = select_device("cpu")
print(f"[smoke] device={dev}")
agent, meta = load_twistor_model("models/twistor_quick.pt", dev)
print(f"[smoke] meta={meta}")
agent.reset()
obs = np.random.randn(meta.input_dim).astype(np.float32)
action = agent.act(obs)
print(f"[smoke] obs[:4]={obs[:4]}")
print(f"[smoke] action={action}")
print("[smoke] SUCCESS")
