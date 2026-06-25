"""
个人模型训练器 (供 Streamlit 调用)
=================================

简化版训练流程, 适配个人用户场景:
  - 自动检测 GPU/CPU
  - 根据数据量自动选择模型规模
  - 内置进度回调
"""
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from twistor_lnn.v16_6_50m import V16_6_50M, V16_6_50M_Config


# 预设配置
PRESETS = {
    "快速 (5M, 笔记本 10 分钟)": {
        "d_model": 256,
        "n_layers": 4,
        "d_state": 8,
        "max_seq_len": 256,
    },
    "标准 (50M, 笔记本 1-3 小时)": {
        "d_model": 768,
        "n_layers": 12,
        "d_state": 16,
        "max_seq_len": 512,
    },
    "深度 (100M, 需 GPU)": {
        "d_model": 1024,
        "n_layers": 16,
        "d_state": 16,
        "max_seq_len": 1024,
    },
}


def detect_optimal_preset(data_size_kb: float) -> str:
    """根据数据量自动选择预设"""
    if data_size_kb < 50:
        return "快速 (5M, 笔记本 10 分钟)"
    elif data_size_kb < 5000:  # 5MB
        return "标准 (50M, 笔记本 1-3 小时)"
    else:
        return "深度 (100M, 需 GPU)"


def train_personal_model(
    train_file: str,
    epochs: int = 5,
    preset: str = "标准 (50M, 笔记本 1-3 小时)",
    lr: float = 3e-4,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> str:
    """
    训练个人模型 (简化版).

    Args:
        train_file: JSONL 训练数据路径
        epochs: 训练轮数
        preset: 模型规模预设
        lr: 学习率
        progress_callback: 进度回调 (0-1, 消息)

    Returns:
        训练好的模型路径
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preset_cfg = PRESETS.get(preset, PRESETS["标准 (50M, 笔记本 1-3 小时)"])

    if progress_callback:
        progress_callback(0.05, f"🔧 设备: {device}, 预设: {preset}")

    # 构建字符级 vocab (简化: 使用 UTF-8 字节)
    vocab_size = 256

    # 加载数据
    if progress_callback:
        progress_callback(0.1, "📂 加载数据...")

    texts = []
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                import json
                obj = json.loads(line)
                texts.append(obj.get("text", ""))
            except Exception:
                continue

    if not texts:
        raise ValueError("训练数据为空")

    if progress_callback:
        progress_callback(0.15, f"📊 数据: {len(texts)} 段, {sum(len(t) for t in texts)/1024:.1f} KB")

    # 编码 (字符级)
    all_bytes = []
    for t in texts:
        all_bytes.extend(t.encode("utf-8", errors="ignore"))
        all_bytes.append(0)  # 段分隔

    data = torch.tensor(all_bytes, dtype=torch.long)

    # 构建模型
    if progress_callback:
        progress_callback(0.2, "🏗️ 构建模型...")

    cfg = V16_6_50M_Config(
        vocab_size=vocab_size,
        d_model=preset_cfg["d_model"],
        n_layers=preset_cfg["n_layers"],
        d_state=preset_cfg["d_state"],
        max_seq_len=preset_cfg["max_seq_len"],
    )
    model = V16_6_50M(cfg).to(device)
    n_params = model.num_params()

    if progress_callback:
        progress_callback(0.25, f"📐 模型参数: {n_params:,} ({n_params/1e6:.1f}M)")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)

    # 训练循环
    seq_len = preset_cfg["max_seq_len"]
    n_seqs = max(1, (len(data) - 1) // (seq_len + 1))
    total_steps = n_seqs * epochs

    if progress_callback:
        progress_callback(0.3, f"🏋️ 开始训练: {epochs} 轮 × {n_seqs} 步 = {total_steps} 步")

    model.train()
    global_step = 0
    t_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        # 随机偏移, 增加数据多样性
        max_start = len(data) - seq_len - 1
        if max_start > 0:
            start = (epoch * 17) % max_start
        else:
            start = 0

        for i in range(n_seqs):
            idx = start + i * (seq_len + 1)
            if idx + seq_len + 1 > len(data):
                break

            x = data[idx:idx+seq_len].unsqueeze(0).to(device)
            y = data[idx+1:idx+seq_len+1].unsqueeze(0).to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            # 进度回调
            if global_step % 5 == 0 and progress_callback:
                p = 0.3 + 0.6 * (global_step / total_steps)
                elapsed = time.time() - t_start
                eta = elapsed / max(1, global_step) * (total_steps - global_step)
                progress_callback(
                    p,
                    f"📚 Ep{epoch+1}/{epochs} Step {global_step}/{total_steps} "
                    f"loss={loss.item():.3f} ETA {eta/60:.0f}min"
                )

        avg_loss = epoch_loss / max(1, n_seqs)
        if progress_callback:
            progress_callback(
                0.3 + 0.6 * ((epoch + 1) / epochs),
                f"✅ Epoch {epoch+1} 完成, avg loss={avg_loss:.4f}"
            )

    # 保存
    if progress_callback:
        progress_callback(0.95, "💾 保存模型...")

    ckpt_dir = ROOT / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    model_path = ckpt_dir / "user_model_best.pt"

    torch.save({
        "model": model.state_dict(),
        "config": cfg.__dict__,
        "epochs": epochs,
        "final_loss": avg_loss,
        "training_data_size": len(texts),
    }, model_path)

    if progress_callback:
        progress_callback(1.0, f"🎉 训练完成! 模型: {model_path}")

    return str(model_path)