"""
Minimal vLLM smoke test in WSL — verifies vLLM 0.22.0 is functional on
this machine with the GTX 1650 Ti (4GB VRAM, CC=7.5).

Strategy: use the smallest viable model (Qwen2.5-0.5B base) with very
short context to fit in 4GB, and skip CUDA graphs (enforce_eager) so we
don't spend 30+ min on first-launch compile.
"""
import os
import sys
import time

os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ.setdefault("VLLM_USE_V1", "1")
# Pin threads to avoid thrashing the small GPU
os.environ.setdefault("OMP_NUM_THREADS", "2")

from vllm import LLM, SamplingParams


def main():
    print(f"[demo] importing vllm ...", flush=True)
    import vllm
    import torch
    print(f"[demo] vllm={vllm.__version__}  torch={torch.__version__}", flush=True)
    print(f"[demo] cuda={torch.cuda.is_available()}, dev={torch.cuda.get_device_name(0)}", flush=True)

    # Use a tiny, fast-loading model
    model_id = "Qwen/Qwen2.5-0.5B"
    t0 = time.time()
    print(f"[demo] loading {model_id} (dtype=float16, gpu_mem=0.70, eager)...", flush=True)
    llm = LLM(
        model=model_id,
        gpu_memory_utilization=0.70,
        max_model_len=512,
        dtype="float16",
        enforce_eager=True,
        download_dir=os.path.expanduser("~/.cache/huggingface"),
    )
    print(f"[demo] LLM init OK in {time.time()-t0:.1f}s", flush=True)

    sp = SamplingParams(temperature=0.0, max_tokens=24)

    prompts = [
        "The capital of France is",
        "Q: 2+2=? A:",
    ]

    t1 = time.time()
    outputs = llm.generate(prompts, sp)
    print(f"[demo] generated in {time.time()-t1:.2f}s", flush=True)
    for i, o in enumerate(outputs):
        print(f"[demo] prompt {i}: {o.prompt!r}", flush=True)
        print(f"[demo] answer  {i}: {o.outputs[0].text!r}", flush=True)


if __name__ == "__main__":
    main()
