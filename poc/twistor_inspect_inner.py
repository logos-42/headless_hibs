"""Drill into model_state_dict keys to learn the actual layer naming."""
import torch

path = "models/twistor_quick.pt"
obj = torch.load(path, map_location="cpu", weights_only=False)
sd = obj["model_state_dict"]
print("config:", obj.get("model_config"))
print("vocab_size:", obj.get("vocab_size"))
print()
print("state_dict keys:")
for k, v in sd.items():
    print(f"  {k!r:50s} {tuple(v.shape)}")
