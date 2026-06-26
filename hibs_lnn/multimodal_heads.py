"""
多模态解码头 (Multimodal Heads)
==============================

设计目标:
- 主干零分支, 解码延后
- 文本 → 离散 vocab
- 视觉/音频 → 连续 patch/sample 重建
- 共享主干输出 z, 各 head 独立
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .multimodal_normalizer import (
    MODALITY_TEXT,
    MODALITY_VISION,
    MODALITY_AUDIO,
    SUPPORTED_MODALITIES,
)


# ----------------------------------------------------------------------
# 文本头: 离散 token
# ----------------------------------------------------------------------
class TextHead(nn.Module):
    """z → vocab logits (next-token prediction)"""

    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z 是复数, 取实部 (与现有 TwistorLMT.out 保持一致)
        return self.proj(z.real)


# ----------------------------------------------------------------------
# 连续重建头: 视觉 patch / 音频样本
# ----------------------------------------------------------------------
class ContinuousHead(nn.Module):
    """z → 连续 patch/sample (重建 / 预测)"""

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, out_dim)
        nn.init.orthogonal_(self.proj.weight, gain=0.5)
        nn.init.zeros_(self.proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z.real)


# ----------------------------------------------------------------------
# 容器
# ----------------------------------------------------------------------
class MultimodalHeads(nn.Module):
    """
    多模态解码头容器.

    用法:
        heads = MultimodalHeads(
            hidden_dim=64,
            head_specs={
                "text":  {"type": "text",  "out_dim": 32},
                "vision": {"type": "continuous", "out_dim": 16},
                "audio": {"type": "continuous", "out_dim": 16},
            },
        )
        out = heads.decode(z, modality="audio")
    """

    def __init__(
        self,
        hidden_dim: int,
        head_specs: Dict[str, Dict],
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = nn.ModuleDict()

        for m, spec in head_specs.items():
            t = spec["type"]
            od = spec["out_dim"]
            if t == "text":
                self.heads[m] = TextHead(hidden_dim, od)
            elif t == "continuous":
                self.heads[m] = ContinuousHead(hidden_dim, od)
            else:
                raise ValueError(f"unknown head type: {t}")

    def decode(self, z: torch.Tensor, modality: str) -> torch.Tensor:
        if modality not in self.heads:
            raise KeyError(f"head for modality '{modality}' not found")
        return self.heads[modality](z)

    def has_modality(self, modality: str) -> bool:
        return modality in self.heads
