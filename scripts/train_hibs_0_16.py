"""
hibs 0.16 (基于 hibs 0.16) 训练脚本
==================

特性:
  - AMP (BF16) 混合精度
  - 梯度累积 (模拟大 batch)
  - 梯度检查点 (节省显存)
  - Cosine LR schedule + warmup
  - 周期性保存 checkpoint
  - TensorBoard 日志 (可选)

使用方法:
    python scripts/train_v16_6_50m.py --config configs/v16_6_50m.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from hibs_LMT.v16_6_50m import Hibs_0_16_50M, Hibs_0_16_50M_Config


# ============================================================
# 数据集 (JSONL 格式, 每行 {"text": "..."})
# ============================================================
class TextDataset(Dataset):
    """简单 JSONL 文本数据集, 使用预训练 tokenizer 编码"""

    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.texts = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text") or obj.get("content") or ""
                    if text:
                        self.texts.append(text)
                except json.JSONDecodeError:
                    # 容错: 整行作为文本
                    self.texts.append(line)

        print(f"  加载 {len(self.texts)} 条文本 from {jsonl_path}")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.tokenizer.encode(text)
        ids = ids[:self.max_seq_len + 1]

        # 填充到固定长度 (用 0)
        if len(ids) < self.max_seq_len + 1:
            ids = ids + [0] * (self.max_seq_len + 1 - len(ids))

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)
        return x, y


# ============================================================
# Tokenizer (BPE 包装)
# ============================================================
class BPETokenizerWrapper:
    """
    BPE tokenizer 包装, 支持 tiktoken / sentencepiece / 自定义 JSON。
    """

    def __init__(self, tokenizer_path: str = None, vocab_size: int = 8192):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.kind = None

        if tokenizer_path and os.path.exists(tokenizer_path):
            try:
                import tiktoken
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.kind = "tiktoken"
                self.vocab_size = self.tokenizer.max_token_value + 1
                print(f"  使用 tiktoken cl100k_base, vocab={self.vocab_size}")
                return
            except ImportError:
                pass

            # 尝试 sentencepiece
            try:
                import sentencepiece as spm
                self.tokenizer = spm.SentencePieceProcessor()
                self.tokenizer.Load(tokenizer_path)
                self.kind = "sentencepiece"
                print(f"  使用 sentencepiece: {tokenizer_path}")
                return
            except Exception:
                pass

        # Fallback: 字符级 tokenizer (与 hibs 0.16 一致)
        print("  ⚠️ 未找到 BPE tokenizer, 使用字符级 (vocab=256)")
        self.kind = "char"
        self.vocab_size = 256
        self.char_to_id = {}
        self.id_to_char = {}

    def encode(self, text: str) -> list:
        if self.kind == "tiktoken":
            return self.tokenizer.encode(text)
        elif self.kind == "sentencepiece":
            return self.tokenizer.EncodeAsIds(text)
        else:
            # 字符级 fallback
            ids = []
            for c in text:
                if c not in self.char_to_id:
                    self.char_to_id[c] = len(self.char_to_id)
                    self.id_to_char[len(self.id_to_char)] = c
                ids.append(self.char_to_id[c])
            return ids

    def decode(self, ids: list) -> str:
        if self.kind == "tiktoken":
            return self.tokenizer.decode(ids)
        elif self.kind == "sentencepiece":
            return self.tokenizer.DecodeIds(ids)
        else:
            return "".join(self.id_to_char.get(i, "?") for i in ids)


def load_tokenizer(config: dict) -> BPETokenizerWrapper:
    tok_path = config.get("data", {}).get("tokenizer_path")
    return BPETokenizerWrapper(
        tokenizer_path=tok_path,
        vocab_size=config.get("model", {}).get("vocab_size", 8192),
    )


# ============================================================
# 学习率调度器
# ============================================================
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, current_step: int):
        if current_step < self.warmup_steps:
            factor = current_step / max(1, self.warmup_steps)
        else:
            progress = (current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = max(factor, self.min_lr / self.base_lrs[0])

        for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            g["lr"] = base_lr * factor


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    return math.exp(sum(losses) / max(1, len(losses)))


# ============================================================
# 主训练循环
# ============================================================
def train(config_path: str):
    # 加载配置
    with open(config_path, "r") as f:
        cfg = json.load(f)

    model_cfg = Hibs_0_16_50M_Config(
        vocab_size=cfg["model"]["vocab_size"],
        d_model=cfg["model"]["d_model"],
        n_layers=cfg["model"]["n_layers"],
        d_state=cfg["model"]["d_state"],
        max_seq_len=cfg["model"]["max_seq_len"],
        conv_kernel=cfg["model"].get("conv_kernel", 4),
    )

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # Tokenizer
    tokenizer = load_tokenizer(cfg)

    # 数据
    train_ds = TextDataset(data_cfg["train_path"], tokenizer, model_cfg.max_seq_len)
    val_ds = TextDataset(data_cfg["val_path"], tokenizer, model_cfg.max_seq_len) \
        if os.path.exists(data_cfg["val_path"]) else None

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=1
    ) if val_ds else None

    # 模型
    model = Hibs_0_16_50M(model_cfg).to(device)
    n_params = model.num_params()
    print(f"\n模型参数: {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"层数: {model_cfg.n_layers} | d_model: {model_cfg.d_model} | d_state: {model_cfg.d_state}")

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.05),
        betas=(0.9, 0.95),
    )

    # 调度器
    total_steps = len(train_loader) * train_cfg["epochs"] // train_cfg.get("grad_accum", 1)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=train_cfg.get("warmup_steps", 2000),
        total_steps=total_steps,
    )

    # AMP (BF16)
    amp_dtype = torch.bfloat16 if train_cfg.get("amp_dtype") == "bf16" else torch.float16
    use_amp = device == "cuda"

    # 训练循环
    print(f"\n开始训练: {train_cfg['epochs']} epochs, {total_steps} total steps")
    print("=" * 70)

    global_step = 0
    accum_count = 0
    best_val_ppl = float("inf")
    t_start = time.time()

    for epoch in range(train_cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                ) / train_cfg.get("grad_accum", 1)

            loss.backward()
            accum_count += 1

            if accum_count >= train_cfg.get("grad_accum", 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.get("grad_clip", 1.0))
                scheduler.step(global_step)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                accum_count = 0

            epoch_loss += loss.item() * train_cfg.get("grad_accum", 1)
            n_batches += 1

            if batch_idx % 50 == 0:
                elapsed = time.time() - t_start
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Ep{epoch+1}/{train_cfg['epochs']} "
                    f"Step {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item()*train_cfg.get('grad_accum', 1):.4f} "
                    f"lr={lr:.2e} "
                    f"elapsed={elapsed:.0f}s",
                    flush=True,
                )

        avg_loss = epoch_loss / max(1, n_batches)
        train_ppl = math.exp(avg_loss)
        print(f"\nEpoch {epoch+1} train loss={avg_loss:.4f} ppl={train_ppl:.2f}")

        # 验证
        if val_loader is not None:
            val_ppl = evaluate(model, val_loader, device)
            print(f"Epoch {epoch+1} val ppl={val_ppl:.2f}")

            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                ckpt_path = ROOT / "checkpoints" / f"hibs_0_16_best.pt"
                ckpt_path.parent.mkdir(exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "config": model_cfg.__dict__,
                    "epoch": epoch + 1,
                    "val_ppl": val_ppl,
                }, ckpt_path)
                print(f"  ✅ 保存最佳 checkpoint: {ckpt_path}")

        # 周期性保存
        if (epoch + 1) % 1 == 0:
            ckpt_path = ROOT / "checkpoints" / f"hibs_0_16_epoch{epoch+1}.pt"
            ckpt_path.parent.mkdir(exist_ok=True)
            torch.save({
                "model": model.state_dict(),
                "config": model_cfg.__dict__,
                "epoch": epoch + 1,
            }, ckpt_path)

    print(f"\n训练完成! 最佳 val_ppl={best_val_ppl:.2f}")
    print(f"耗时: {(time.time()-t_start)/60:.1f} 分钟")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v16_6_50m.json")
    args = parser.parse_args()
    train(args.config)