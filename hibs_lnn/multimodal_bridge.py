"""
多模态桥接与等距约束 (Multimodal Bridge & Isometry)
===================================================

设计目标:
- 在主干动力学上注入模态条件 (类似附加项, 不破坏复数 z 演化)
- 提供"模态等距"正则: 跨模态对在 Möbius 流形上测地距离一致
- 完全可拔插: 默认不启用时不影响现有动力学

核心机制:
1. ModalityInjector: 用低秩 Linear 把模态 embedding 注入 dz/dt
2. MobiusIsometryLoss: 配对样本在流形上的测地距离对齐
3. geod_distance: 简化的双曲距离 (球面投影)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List

from .multimodal_normalizer import (
    MODALITY_TEXT,
    MODALITY_VISION,
    MODALITY_AUDIO,
    SUPPORTED_MODALITIES,
)


# ----------------------------------------------------------------------
# 测地距离 (Möbius 流形上的简化近似)
# ----------------------------------------------------------------------
def mobius_geodesic_distance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    在 Möbius 流形 (简化为球面投影) 上的测地距离.

    Args:
        z1, z2: 复数张量 (..., D)
    Returns:
        距离张量 (...,), 范围 [0, π]
    """
    # 先按模长归一 (球面投影)
    n1 = z1 / (z1.abs().clamp(min=1e-6))
    n2 = z2 / (z2.abs().clamp(min=1e-6))
    # 内积 → 夹角
    inner = (n1 * n2.conj()).real.sum(dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.arccos(inner)


def mobius_raw_distance(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    复数 L2 (RAW) 距离 — V5 P2 v2 修复后使用的算法.

    绕过球面归一化塌缩 (V5 报告 §1.1).
    复数 z 的 L2 距离 = sqrt(sum |z1_i - z2_i|^2)
    """
    return (z1 - z2).abs().norm(dim=-1)


def mobius_retrieval_r1(
    z_query: torch.Tensor,
    z_gallery: torch.Tensor,
    distance_mode: str = "raw",
) -> float:
    """
    Möbius 流形上的跨模态检索 R@1.

    Args:
        z_query:   (B, D) 复数, 查询模态的 z
        z_gallery: (B, D) 复数, 候选模态的 z (配对样本在 i==j)
        distance_mode: "raw" (复数 L2) 或 "geo" (测地距离)
    Returns:
        R@1 = 配对样本最近邻命中率 (1.0=完美, 1/B=随机)

    用法: 测试 text→image 或 image→text 跨模态检索
    """
    if distance_mode == "raw":
        # (B_query, B_gallery) 距离矩阵
        # z_query (B, D), z_gallery (B, D)
        # diff: (B_q, B_g, D)
        diff = z_query.unsqueeze(1) - z_gallery.unsqueeze(0)  # (Bq, Bg, D)
        d = diff.abs().norm(dim=-1)  # (Bq, Bg)
    elif distance_mode == "geo":
        # 用测地距离
        B_q = z_query.size(0)
        B_g = z_gallery.size(0)
        d = torch.zeros(B_q, B_g, device=z_query.device)
        for i in range(B_q):
            d[i] = mobius_geodesic_distance(
                z_query[i:i+1].expand(B_g, -1), z_gallery
            )
    else:
        raise ValueError(f"unknown distance_mode: {distance_mode}")

    # R@1: argmin 是否在对角线 (i==j)
    nn_idx = d.argmin(dim=1)  # (B_q,)
    B = z_query.size(0)
    correct = (nn_idx == torch.arange(B, device=z_query.device)).float()
    return float(correct.mean().item())


# ----------------------------------------------------------------------
# 模态注入器: 把模态 embedding 注入 dz/dt
# ----------------------------------------------------------------------
class ModalityInjector(nn.Module):
    """
    极轻量模态注入器.

    在 compute_dzdt 末尾以附加项形式注入模态条件:
        dz/dt += inject(dzdt, modality_embed)

    Args:
        hidden_dim: 主干隐藏维度
        rank: 低秩瓶颈
        strength: 注入强度
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 8,
        strength: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.strength = nn.Parameter(torch.tensor(strength))

        # 低秩: hidden -> rank -> hidden
        self.down = nn.Linear(hidden_dim, rank, bias=False)
        self.up = nn.Linear(rank, hidden_dim, bias=False)

        nn.init.orthogonal_(self.down.weight, gain=0.3)
        nn.init.zeros_(self.up.weight)

    def forward(self, z: torch.Tensor, modality_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: 当前复数状态 (B, D) 或 (B, T, D)
            modality_embed: 模态 embedding (D,)
        Returns:
            dzdt 附加项 (B, D) 或 (B, T, D) (复数)
        """
        # 用 |z| 作为 modulator (模长驱动注入强度)
        z_amp = z.abs()
        h = self.down(z_amp)
        h = F.gelu(h)
        h = self.up(h)
        return torch.complex(h, torch.zeros_like(h)) * self.strength


# ----------------------------------------------------------------------
# 模态等距损失
# ----------------------------------------------------------------------
class MobiusIsometryLoss(nn.Module):
    """
    跨模态等距损失: 配对样本在 Möbius 流形上测地距离一致.

    用法:
        crit = MobiusIsometryLoss()
        loss = crit(z_dict, pairs)
        # z_dict: {modality: (B, D) 复数状态}
        # pairs: [(modality_a, modality_b, target_distance), ...]
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        z_dict: Dict[str, torch.Tensor],
        pairs: Optional[List] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_dict: {modality: (B, D) 复数状态}
            pairs: 自定义配对, None 则默认全配对 (target=0)
        Returns:
            标量 loss
        """
        mods = list(z_dict.keys())
        if len(mods) < 2:
            return torch.tensor(0.0, device=next(iter(z_dict.values())).device)

        if pairs is None:
            # 默认: 配对模态之间 target 距离 = 0 (同一时刻应该等距)
            pairs = []
            for i in range(len(mods)):
                for j in range(i + 1, len(mods)):
                    pairs.append((mods[i], mods[j], 0.0))

        loss = torch.tensor(0.0, device=next(iter(z_dict.values())).device)
        n = 0
        for m1, m2, target in pairs:
            if m1 not in z_dict or m2 not in z_dict:
                continue
            d = mobius_geodesic_distance(z_dict[m1], z_dict[m2])
            loss = loss + ((d - target) ** 2).mean() / self.temperature
            n += 1

        if n == 0:
            return torch.tensor(0.0, device=loss.device)
        return loss / n


# ----------------------------------------------------------------------
# 多模态桥接容器
# ----------------------------------------------------------------------
class MultimodalBridge(nn.Module):
    """
    多模态桥接容器, 统一管理 ModalityInjector 和 MobiusIsometryLoss.

    用法:
        bridge = MultimodalBridge(hidden_dim=64)
        # 在 compute_dzdt 末尾:
        dzdt = bridge.inject(dzdt, z, modality_id)
        # 训练时:
        loss_iso = bridge.isometry_loss(z_dict)
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 8,
        strength: float = 0.1,
        enable_injector: bool = True,
        enable_isometry: bool = True,
    ):
        super().__init__()
        self.enable_injector = enable_injector
        self.enable_isometry = enable_isometry

        if enable_injector:
            self.injector = ModalityInjector(
                hidden_dim=hidden_dim,
                rank=rank,
                strength=strength,
            )
        else:
            self.injector = None

        if enable_isometry:
            self.isometry_loss = MobiusIsometryLoss()
        else:
            self.isometry_loss = None

    def inject(
        self,
        dzdt: torch.Tensor,
        z: torch.Tensor,
        modality_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        把模态信息注入 dz/dt (附加项).

        Args:
            dzdt: 当前 dz/dt (复数, B,D 或 B,T,D)
            z: 当前 z 状态 (复数, 同形)
            modality_embed: 模态 embedding (D,)
        Returns:
            注入后的 dz/dt
        """
        if not self.enable_injector or self.injector is None:
            return dzdt
        # 拼接模态信息 (简单广播)
        emb = modality_embed  # (D,)
        return dzdt + self.injector(z, emb)

    def isometry(
        self,
        z_dict: Dict[str, torch.Tensor],
        pairs: Optional[List] = None,
    ) -> torch.Tensor:
        if not self.enable_isometry or self.isometry_loss is None:
            return torch.tensor(0.0, device=next(iter(z_dict.values())).device)
        return self.isometry_loss(z_dict, pairs)
