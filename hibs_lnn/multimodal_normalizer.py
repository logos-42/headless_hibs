"""
多模态前端归一化 (Modality Normalizer)
======================================

设计目标:
- 不做 token 化, 只做"几何归一化"
- 把 RGB 像素 / 音频样本 / 文本 embedding 拉到统一的 Möbius 流形入口
- 极轻量: 模态特定参数 < 主干 1%

核心机制:
1. 每模态一个轻量正交投影 (Linear, no bias)
2. Per-modality max-norm (在进入主干前约束 z 模长)
3. Per-modality τ 缩放因子 ρ(m) (异步采样)

参考:
- docs/100M 参数版本设计.md
- plans/multimodal_mobius_plan.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


# 模态 ID 常量 (用字符串便于日志/调试)
MODALITY_TEXT = "text"
MODALITY_VISION = "vision"
MODALITY_AUDIO = "audio"
SUPPORTED_MODALITIES = (MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO)


# 每模态的默认 τ 缩放先验 (异步采样节奏)
# 文本: 慢; 视频帧: 中; 音频样本: 快
DEFAULT_TAU_SCALE = {
    MODALITY_TEXT: 1.0,
    MODALITY_VISION: 0.3,
    MODALITY_AUDIO: 0.01,
}


class ModalityNormalizer(nn.Module):
    """
    极轻量模态前端, 把不同模态的原始信号几何地拉到 hidden_dim 维空间.

    Args:
        modality: 模态 ID (text / vision / audio)
        raw_dim: 原始信号维度 (像素 / 样本 / token embedding 维度)
        hidden_dim: 主干隐藏维度
        tau_scale: 该模态的 τ 缩放先验
        max_norm: 该模态进入主干前的最大模长
    """

    def __init__(
        self,
        modality: str,
        raw_dim: int,
        hidden_dim: int,
        tau_scale: float = 1.0,
        max_norm: float = 5.0,
    ):
        super().__init__()
        assert modality in SUPPORTED_MODALITIES, f"unknown modality: {modality}"
        self.modality = modality
        self.raw_dim = raw_dim
        self.hidden_dim = hidden_dim
        self.max_norm = max_norm

        # 轻量正交投影 (no bias, 模态专属)
        self.proj = nn.Linear(raw_dim, hidden_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight, gain=0.5)

        # τ 缩放: 静态先验 + 可学习微调
        # 用 log space 保证正值
        self.log_tau_scale = nn.Parameter(torch.tensor(float(tau_scale)).log())

        # 模态 embedding (供 multimodal_bridge 注入)
        self.modality_embed = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    @property
    def tau_scale(self) -> torch.Tensor:
        return self.log_tau_scale.exp()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, raw_dim) 或 (B, raw_dim) - 模态原始信号
        Returns:
            (B, T, hidden_dim) - 归一化后的复数入口 (实部,虚部都是 proj(x))
        """
        h = self.proj(x)
        # max-norm 约束 (按最后一维)
        h = self._max_norm_clip(h)
        # 返回复数: 实部 = 虚部 = h (保持等幅)
        # 这样模态信号进入复数 z 时, 不偏向 real/imag 任一方
        return torch.complex(h, h)

    def _max_norm_clip(self, h: torch.Tensor) -> torch.Tensor:
        """对最后一维做 max-norm 约束 (梯度友好版)"""
        norm = h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (self.max_norm / norm).clamp(max=1.0)
        return h * scale


class MultimodalNormalizer(nn.Module):
    """
    多模态前端容器, 统一管理多种 ModalityNormalizer.

    用法:
        norm = MultimodalNormalizer(
            hidden_dim=64,
            modality_dims={"text": 32, "vision": 16, "audio": 16},
            tau_scales={"text": 1.0, "vision": 0.3, "audio": 0.01},
        )
        z_entry = norm.encode(x, modality="audio")
    """

    def __init__(
        self,
        hidden_dim: int,
        modality_dims: Dict[str, int],
        tau_scales: Optional[Dict[str, float]] = None,
        max_norm: float = 5.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        if tau_scales is None:
            tau_scales = DEFAULT_TAU_SCALE

        self.normalizers = nn.ModuleDict()
        for m, d in modality_dims.items():
            self.normalizers[m] = ModalityNormalizer(
                modality=m,
                raw_dim=d,
                hidden_dim=hidden_dim,
                tau_scale=tau_scales.get(m, 1.0),
                max_norm=max_norm,
            )

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        """把某模态信号编码为复数入口 z_entry ∈ ℂ^hidden"""
        if modality not in self.normalizers:
            raise KeyError(
                f"modality '{modality}' not registered. "
                f"available: {list(self.normalizers.keys())}"
            )
        return self.normalizers[modality](x)

    def tau_scale(self, modality: str) -> torch.Tensor:
        return self.normalizers[modality].tau_scale

    def modality_embed(self, modality: str) -> torch.Tensor:
        """取模态 embedding (供 multimodal_bridge 注入)"""
        return self.normalizers[modality].modality_embed

    def get_all_tau_scales(self) -> Dict[str, float]:
        return {m: float(n.tau_scale.item()) for m, n in self.normalizers.items()}

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, "
            f"modalities={list(self.normalizers.keys())}, "
            f"tau_scales={self.get_all_tau_scales()}"
        )


def create_multimodal_normalizer(
    hidden_dim: int,
    modality_dims: Optional[Dict[str, int]] = None,
    tau_scales: Optional[Dict[str, float]] = None,
    max_norm: float = 5.0,
) -> MultimodalNormalizer:
    """工厂函数: 创建默认的三模态 normalizer"""
    if modality_dims is None:
        modality_dims = {
            MODALITY_TEXT: hidden_dim,
            MODALITY_VISION: hidden_dim // 4,
            MODALITY_AUDIO: hidden_dim // 4,
        }
    return MultimodalNormalizer(
        hidden_dim=hidden_dim,
        modality_dims=modality_dims,
        tau_scales=tau_scales,
        max_norm=max_norm,
    )
