"""
Twistor-LMT Inference Server (OpenAI-compatible + native endpoints)
====================================================================

A vLLM-style HTTP server that drives your Twistor-Liquid Neural Network
using a PyTorch backend. The API surface mirrors OpenAI's `/v1/chat/completions`
and `/v1/completions` for drop-in compatibility with vLLM clients, plus a few
native endpoints specific to the continuous-time liquid model:

  GET  /health
  GET  /v1/models
  POST /v1/chat/completions        (text in -> action vector out, as JSON)
  POST /v1/completions             (numeric sequence in -> next-step predictions)
  POST /twistor/act                (raw observation -> raw action)
  POST /twistor/predict            (multi-step forecast)

Inference is CPU-first with optional CUDA. The LMT is small (~85 KB - 870 KB),
so CPU is perfectly adequate for interactive use.

Usage:
    python twistor_vllm_server.py --model models/twistor_quick.pt --port 8000
    python twistor_vllm_server.py --model models/twistor_quick.pt --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

# Ensure local package is importable regardless of CWD
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from twistor_LMT.growable import GrowableTwistorLMT, create_growable_twistor_LMT  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
import uvicorn  # noqa: E402


# ---------------------------------------------------------------------------
# Device selection: CPU first, optional CUDA fallback
# ---------------------------------------------------------------------------

def select_device(prefer: str = "cpu") -> str:
    """Select inference device.

    prefer='cpu'  -> always return 'cpu' (recommended for LMTs)
    prefer='cuda' -> return 'cuda' if available, else fallback to 'cpu'
    prefer='auto' -> 'cuda' if available else 'cpu'
    """
    prefer = prefer.lower()
    cuda_ok = torch.cuda.is_available()
    if prefer == "cuda":
        return "cuda" if cuda_ok else "cpu"
    if prefer == "auto":
        return "cuda" if cuda_ok else "cpu"
    return "cpu"


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

@dataclass
class ModelMeta:
    name: str
    path: str
    input_dim: int
    hidden_dim: int
    output_dim: int
    num_params: int
    device: str


def load_twistor_model(ckpt_path: str, device: str) -> tuple[TwistorAgent, ModelMeta]:
    """Load a TwistorLMT checkpoint into a TwistorAgent.

    Supports checkpoints saved by `twistor_LMT.growable.GrowableTwistorLMT`
    (the architecture used in this project: W_amplitude + manifold_theta +
    Möbius / resonance). Two checkpoint shapes are accepted:

      1. dict with 'model_state_dict' + 'model_config' (project standard)
      2. plain state_dict, or dict with 'state_dict'
      3. full nn.Module (legacy)

    Dims are inferred from the actual weight tensors when present, with
    model_config used only as a hint (the saved weights are authoritative).
    """
    ckpt_path = str(ckpt_path)
    obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Normalize to state_dict + optional config
    cfg: dict = {}
    sd = None
    if isinstance(obj, GrowableTwistorLMT) or (
        hasattr(obj, "state_dict") and not isinstance(obj, dict)
    ):
        sd = obj.state_dict()
    elif isinstance(obj, dict):
        if "model_state_dict" in obj:
            sd = obj["model_state_dict"]
            cfg = dict(obj.get("model_config") or {})
        elif "state_dict" in obj:
            sd = obj["state_dict"]
        else:
            sd = obj  # already a flat state_dict
    else:
        raise ValueError(f"Unrecognised checkpoint format: {type(obj)}")

    # --- Architecture inference from the saved tensors (authoritative) ---
    # GrowableTwistorLMT layout:
    #   manifold_theta: (max_h, 3)
    #   W_amplitude:    (max_h, max_h)
    #   U.weight:       (max_h, input_dim)
    #   out.weight:     (output_dim, max_h)
    if "W_amplitude" in sd:
        max_h = sd["W_amplitude"].shape[0]
    elif "W_real.weight" in sd:
        max_h = sd["W_real.weight"].shape[0]
    else:
        raise ValueError(
            "Checkpoint has neither 'W_amplitude' nor 'W_real.weight' — "
            "is this a TwistorLMT?"
        )

    if "U.weight" not in sd or "out.weight" not in sd:
        raise ValueError("Checkpoint missing 'U.weight' or 'out.weight'")

    input_dim = sd["U.weight"].shape[1]
    output_dim = sd["out.weight"].shape[0]

    # `hidden_dim` is the *active* number of liquid neurons. With the
    # growable model the *allocated* max is max_h; for inference we set
    # hidden_dim = max_h so the loaded weights are fully used. (Some
    # checkpoints in this project store a misleading small hidden_dim in
    # model_config; the actual parameter count from state_dict is the
    # source of truth.)
    hidden_dim = max_h

    enable_mobius = bool(cfg.get("enable_mobius", True))
    enable_growth = bool(cfg.get("enable_growth", True))
    enable_resonance = bool(cfg.get("enable_resonance", True))

    # Build the matching model. The `max_h` from state_dict must match
    # the growth_config.max_hidden_dim, otherwise the parameter buffers
    # are allocated at the wrong size and load_state_dict fails.
    from twistor_LMT.growable import GrowthConfig
    growth_config = GrowthConfig(
        min_hidden_dim=0,
        max_hidden_dim=max_h,
        enable_developmental_schedule=False,
    )

    # Build the matching model
    model = GrowableTwistorLMT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        enable_growth=enable_growth,
        enable_mobius=enable_mobius,
        enable_resonance=enable_resonance,
        growth_config=growth_config,
    )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    nontrivial_missing = [
        k for k in missing if not k.endswith((".num_batches_tracked",))
    ]
    if nontrivial_missing:
        print(
            f"[warn] {len(nontrivial_missing)} missing keys "
            f"(first 5: {nontrivial_missing[:5]})"
        )
    if unexpected:
        print(
            f"[warn] {len(unexpected)} unexpected keys "
            f"(first 5: {unexpected[:5]})"
        )

    model.eval()
    model.to(device)

    # TwistorAgent isn't strictly required (model.step is enough), but the
    # API is convenient for state management.
    class _TwistorRunner:
        def __init__(self, model, device):
            self.model = model
            self.device = device
            self.z = None
            self.input_dim = model.input_dim
            self.hidden_dim = model.hidden_dim
            self.output_dim = model.output_dim

        def reset(self, batch_size: int = 1):
            self.z = self.model.reset_state(batch_size, self.device)
            return self.z

        def act(self, obs: np.ndarray) -> np.ndarray:
            if self.z is None:
                self.reset()
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float()
            obs = obs.to(self.device)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
                single = True
            else:
                single = False
            with torch.no_grad():
                self.z, action = self.model.step(self.z, obs)
            action = action.cpu().numpy()
            if single:
                action = action[0]
            return action

    runner = _TwistorRunner(model, device)
    runner.reset()

    num_params = sum(p.numel() for p in model.parameters())
    meta = ModelMeta(
        name=Path(ckpt_path).stem,
        path=os.path.abspath(ckpt_path),
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_params=num_params,
        device=device,
    )
    # Patch: the global expects an "agent" with .act() / .reset() — _TwistorRunner matches.
    return runner, meta


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Twistor-LMT Inference Server",
    description=(
        "OpenAI-compatible HTTP API for a Twistor-inspired Liquid Neural Network. "
        "The underlying model is a continuous-time recurrent network (LTC) solved "
        "with RK4; vLLM 0.22.0 is installed in this environment for transformer-LLM "
        "inference alongside it."
    ),
    version="0.1.0",
)

# Globals populated at startup
AGENT: Optional[TwistorAgent] = None
META: Optional[ModelMeta] = None
STARTED_AT: float = time.time()


# --- Request / response models ---

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    temperature: float = 1.0
    max_tokens: int = 64
    stream: bool = False
    # Custom: numeric input mode and state-reset
    numeric_input: Optional[List[float]] = None
    reset: bool = False


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    max_tokens: int = 32
    temperature: float = 1.0
    stream: bool = False
    numeric_prompt: Optional[List[float]] = None


class ActRequest(BaseModel):
    observation: List[float]
    reset: bool = False


class PredictRequest(BaseModel):
    sequence: List[List[float]]
    horizon: int = 10


# --- Routes ---

@app.get("/health")
def health():
    return {
        "status": "ok",
        "uptime_s": round(time.time() - STARTED_AT, 2),
        "model": META.name if META else None,
        "device": META.device if META else None,
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": META.name,
                "object": "model",
                "created": int(STARTED_AT),
                "owned_by": "twistor-LMT",
                "input_dim": META.input_dim,
                "hidden_dim": META.hidden_dim,
                "output_dim": META.output_dim,
                "num_params": META.num_params,
                "device": META.device,
            }
        ] if META else [],
    }


def _coerce_to_obs(text_or_list: Union[str, List, None], input_dim: int) -> np.ndarray:
    """Convert a string or list into a numeric observation vector.

    - If a list of floats, use directly (truncate or pad to input_dim).
    - If a string, embed deterministically via SHA-256 to length input_dim.
    - The result is a row vector that exactly matches `model.U.weight`'s
      in_features so the linear projection can run end-to-end.
    """
    if text_or_list is None:
        return np.zeros(input_dim, dtype=np.float32)
    if isinstance(text_or_list, list):
        arr = np.array(text_or_list, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return np.zeros(input_dim, dtype=np.float32)
        if arr.size < input_dim:
            # tile to reach input_dim
            reps = (input_dim + arr.size - 1) // arr.size
            arr = np.tile(arr, reps)[:input_dim]
        elif arr.size > input_dim:
            arr = arr[:input_dim]
        return arr
    if isinstance(text_or_list, str):
        # Deterministic: SHA-256 stream, normalized to [-1, 1], tiled to input_dim.
        import hashlib
        h = hashlib.sha256(text_or_list.encode("utf-8")).digest()
        # Repeat the 32-byte digest to cover input_dim
        repeats = (input_dim + 31) // 32
        buf = np.frombuffer(h * repeats, dtype=np.uint8)[:input_dim].astype(np.float32)
        buf = (buf - 127.5) / 127.5
        return buf
    raise HTTPException(400, f"unsupported prompt type: {type(text_or_list)}")


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest):
    if AGENT is None or META is None:
        raise HTTPException(503, "model not loaded")
    if req.stream:
        raise HTTPException(400, "streaming not implemented for LMT backend")

    if req.reset:
        AGENT.reset()

    # Pick the last user message as the "prompt"
    user_msg = next(
        (m for m in reversed(req.messages) if m.role == "user"), None
    )
    if user_msg is None and req.numeric_input is None:
        raise HTTPException(400, "no user message and no numeric_input")

    obs = _coerce_to_obs(
        req.numeric_input if req.numeric_input is not None else user_msg.content,
        META.input_dim,
    )

    action = AGENT.act(obs)  # (output_dim,)
    # 'temperature' is a no-op for continuous action, kept for API parity
    # 'max_tokens' is also a no-op (single step)

    # Format the action as a chat reply
    action_list = action.tolist()
    text = (
        f"action=[{', '.join(f'{x:.6f}' for x in action_list)}] "
        f"shape=({META.output_dim},) step={META.name}"
    )

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": META.name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": META.output_dim,
            "total_tokens": 1 + META.output_dim,
        },
        "x_twistor": {
            "action": action_list,
            "observation_in": obs.tolist(),
            "device": META.device,
        },
    }


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    if AGENT is None or META is None:
        raise HTTPException(503, "model not loaded")
    if req.stream:
        raise HTTPException(400, "streaming not implemented")

    AGENT.reset()

    if req.numeric_prompt is not None:
        obs = _coerce_to_obs(req.numeric_prompt, META.input_dim)
    else:
        obs = _coerce_to_obs(
            req.prompt[0] if isinstance(req.prompt, list) else req.prompt,
            META.input_dim,
        )

    action = AGENT.act(obs)
    action_list = action.tolist()
    text = json.dumps({
        "action": action_list,
        "model": META.name,
        "device": META.device,
    })
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": META.name,
        "choices": [
            {
                "text": text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": META.output_dim,
            "total_tokens": 1 + META.output_dim,
        },
    }


@app.post("/twistor/act")
def act_native(req: ActRequest):
    """Native LMT endpoint: observation -> action (no chat wrapping)."""
    if AGENT is None or META is None:
        raise HTTPException(503, "model not loaded")
    if req.reset:
        AGENT.reset()
    obs = _coerce_to_obs(req.observation, META.input_dim)
    action = AGENT.act(obs)
    return {
        "action": action.tolist(),
        "observation": obs.tolist(),
        "model": META.name,
    }


@app.post("/twistor/predict")
def predict_native(req: PredictRequest):
    """Multi-step forecast: feed a sequence of observations, return horizon predictions."""
    if AGENT is None or META is None:
        raise HTTPException(503, "model not loaded")

    AGENT.reset()
    x = np.asarray(req.sequence, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape(-1, META.input_dim)
    if x.shape[1] != META.input_dim:
        raise HTTPException(
            400,
            f"sequence feature dim {x.shape[1]} != model input_dim {META.input_dim}",
        )

    outputs: List[List[float]] = []
    with torch.no_grad():
        for t in range(x.shape[0]):
            obs_t = torch.from_numpy(x[t]).unsqueeze(0)
            AGENT.z, y = AGENT.model.step(AGENT.z, obs_t)
            outputs.append(y.squeeze(0).cpu().numpy().tolist())

        # Roll forward for `horizon` more steps using last output as next input
        last_y = np.array(outputs[-1], dtype=np.float32)
        for _ in range(req.horizon):
            obs_t = torch.from_numpy(last_y).unsqueeze(0)
            AGENT.z, y = AGENT.model.step(AGENT.z, obs_t)
            last_y = y.squeeze(0).cpu().numpy()
            outputs.append(last_y.tolist())

    return {
        "model": META.name,
        "history_outputs": outputs[: x.shape[0]],
        "forecast": outputs[x.shape[0]:],
        "n_input": int(x.shape[0]),
        "horizon": req.horizon,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", "-m",
        default="models/twistor_quick.pt",
        help="Path to Twistor-LMT checkpoint (.pt)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "auto"],
        help="Inference device. Default 'cpu' (LMTs are tiny; CPU is fine).",
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Just load the model, print meta, and exit (no HTTP server).",
    )
    args = parser.parse_args()

    device = select_device(args.device)
    print(f"[twistor-vllm] device={device}  loading {args.model} ...")
    global AGENT, META
    AGENT, META = load_twistor_model(args.model, device)
    print(f"[twistor-vllm] loaded: name={META.name}  "
          f"in={META.input_dim} hidden={META.hidden_dim} out={META.output_dim}  "
          f"params={META.num_params}")

    if args.smoke_test:
        # Run a tiny smoke inference
        AGENT.reset()
        obs = np.random.randn(META.input_dim).astype(np.float32)
        action = AGENT.act(obs)
        print(f"[twistor-vllm] smoke-test obs={obs}  action={action}")
        return

    print(f"[twistor-vllm] serving on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
