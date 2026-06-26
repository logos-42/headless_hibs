"""
Training bridge — trains a personal model from JSONL data,
writing JSON progress lines to stdout for the TS caller to consume.

Usage:
    python bridge/train.py --data workspace/user_data/train.jsonl \\
        --epochs 5 --preset "standard" --lr 3e-4

Each stdout line: {"progress": 0.0-1.0, "message": "..."}
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / ".."))


def emit(progress: float, message: str):
    print(json.dumps({"progress": round(progress, 3), "message": message}, ensure_ascii=False))
    sys.stdout.flush()


PRESETS = {
    "quick":  {"d_model": 256, "n_layers": 4,  "d_state": 8,  "max_seq_len": 256},
    "standard": {"d_model": 768, "n_layers": 12, "d_state": 16, "max_seq_len": 512},
    "deep":   {"d_model": 1024, "n_layers": 16, "d_state": 16, "max_seq_len": 1024},
}


def main():
    parser = argparse.ArgumentParser(description="Train a personal model")
    parser.add_argument("--data", required=True, help="JSONL training data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--preset", default="standard", choices=list(PRESETS.keys()))
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    device = "cuda" if torch.cuda.is_available() else "cpu"
    preset_cfg = PRESETS[args.preset]
    vocab_size = 256  # UTF-8 bytes

    emit(0.05, f"Device: {device}, Preset: {args.preset} ({preset_cfg['d_model']}d)")

    emit(0.10, "Loading data...")
    texts = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
            except Exception:
                continue

    if not texts:
        emit(0.0, "ERROR: No training data found")
        sys.exit(1)

    emit(0.15, f"Data: {len(texts)} segments, {sum(len(t) for t in texts) / 1024:.1f} KB")

    # UTF-8 byte encoding
    all_bytes = []
    for t in texts:
        all_bytes.extend(t.encode("utf-8", errors="ignore"))
        all_bytes.append(0)

    data = torch.tensor(all_bytes, dtype=torch.long)

    emit(0.20, "Building model...")
    from hibs_lnn.v16_6_50m import Hibs_0_16_50M, Hibs_0_16_50M_Config

    cfg = Hibs_0_16_50M_Config(
        vocab_size=vocab_size,
        d_model=preset_cfg["d_model"],
        n_layers=preset_cfg["n_layers"],
        d_state=preset_cfg["d_state"],
        max_seq_len=preset_cfg["max_seq_len"],
    )
    model = Hibs_0_16_50M(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    emit(0.25, f"Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    seq_len = preset_cfg["max_seq_len"]
    n_seqs = max(1, (len(data) - 1) // (seq_len + 1))
    total_steps = n_seqs * args.epochs

    emit(0.30, f"Training: {args.epochs} epochs x {n_seqs} steps = {total_steps} steps")

    model.train()
    global_step = 0
    t_start = time.time()
    avg_loss = 0.0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        max_start = len(data) - seq_len - 1
        start = (epoch * 17) % max_start if max_start > 0 else 0

        for i in range(n_seqs):
            idx = start + i * (seq_len + 1)
            if idx + seq_len + 1 > len(data):
                break

            x = data[idx:idx + seq_len].unsqueeze(0).to(device)
            y = data[idx + 1:idx + seq_len + 1].unsqueeze(0).to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % 5 == 0:
                p = 0.30 + 0.60 * (global_step / total_steps)
                elapsed = time.time() - t_start
                eta = elapsed / max(1, global_step) * (total_steps - global_step)
                emit(p, f"Ep{epoch + 1}/{args.epochs} Step {global_step}/{total_steps} loss={loss.item():.3f} ETA {eta / 60:.0f}min")

        avg_loss = epoch_loss / max(1, n_seqs)
        emit(0.30 + 0.60 * ((epoch + 1) / args.epochs), f"Epoch {epoch + 1} done, avg loss={avg_loss:.4f}")

    emit(0.95, "Saving checkpoint...")
    ckpt_dir = ROOT / ".." / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    model_path = ckpt_dir / "user_model_best.pt"

    torch.save({
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "epochs": args.epochs,
        "final_loss": avg_loss,
        "training_data_size": len(texts),
    }, model_path)

    emit(1.0, f"Done! Model saved to {model_path}")


if __name__ == "__main__":
    main()
