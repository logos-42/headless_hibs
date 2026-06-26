"""Inspect all model checkpoints to learn the format."""
import sys
import torch

sys.path.insert(0, ".")

candidates = [
    "models/twistor_quick.pt",
    "models/twistor_trained.pt",
    "models/twistor_128_5000.pt",
    "models/twistor_growable_128.pt",
    "models/twistor_growable_v3.pt",
    "models/twistor_growable_receptor.pt",
    "models/twistor_growable_receptor_final.pt",
    "models/twistor_gpu_grow.pt",
]

for path in candidates:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"\n{path}: LOAD FAIL -> {type(e).__name__}: {e}")
        continue
    print(f"\n{path}: {type(obj).__name__}", end="")
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"  keys={keys[:8]}{' ...' if len(keys) > 8 else ''}")
        # peek at a few
        for k in keys[:4]:
            v = obj[k]
            print(f"    {k!r}: {type(v).__name__} {getattr(v, 'shape', '')}")
    elif hasattr(obj, "state_dict"):
        sd_keys = list(obj.state_dict().keys())
        print(f"  (nn.Module) state_dict[:5]={sd_keys[:5]}")
    else:
        print(f"  value={obj!r}")
