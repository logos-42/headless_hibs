"""
扭量共振注意力模块 (Twistor Resonance Attention)
=================================================
核心思想:
利用复数状态的相位差计算"共振分数"，实现类似 attention 的全局交互效果

共振机制:
  R_ij = |z_i| · |z_j| · |cos(φ_i - φ_j)|

- 相位相近 → 强共振 → 类似 attention 高权重
- 相位相反 → 弱共振 → 类似被忽略
- 振幅调制 → 类似 value 强度

稀疏模式:
  由莫比乌斯拓扑距离控制，只计算邻近神经元的共振
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TwistorResonance(nn.Module):
    """
    扭量共振注意力

    不使用 Transformer 的 QKV 机制，而是利用复数状态本身的
    振幅和相位信息计算共振分数
    """

    def __init__(
        self,
        hidden_dim: int,
        resonance_strength: float = 0.1,
        sparse_mode: bool = True,
        sparse_k: int = 8,
        use_topology_mask: bool = True,
        device: str = "cpu",
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.resonance_strength = nn.Parameter(torch.tensor(resonance_strength))
        self.sparse_mode = sparse_mode
        self.sparse_k = sparse_k
        self.use_topology_mask = use_topology_mask
        self.device = device

        self.kernel_scale = nn.Parameter(torch.tensor(1.0))
        self.kernel_bias = nn.Parameter(torch.tensor(0.0))

        self.phase_sensitivity = nn.Parameter(torch.tensor(1.0))

        self.topology_threshold = nn.Parameter(torch.tensor(0.1))

        self._topo_cache = None
        self._topo_cache_N = 0

    def compute_amplitude(self, z: torch.Tensor) -> torch.Tensor:
        """计算振幅 |z|"""
        return torch.abs(z)

    def compute_phase(self, z: torch.Tensor) -> torch.Tensor:
        """计算相位 angle(z)"""
        return torch.angle(z)

    def compute_resonance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算全局共振分数矩阵 R ∈ ℝ^(B×N×N)

        R_ij = |z_i| · |z_j| · |cos(φ_i - φ_j)|^γ
        """
        B, N = z.shape

        amp = self.compute_amplitude(z)
        phi = self.compute_phase(z)

        delta_phi = phi.unsqueeze(2) - phi.unsqueeze(1)

        phase_term = torch.cos(delta_phi * self.phase_sensitivity).abs()
        exponent = F.softplus(self.kernel_scale) + F.softplus(self.kernel_bias) + 1e-6
        phase_term = phase_term.clamp(min=1e-10) ** exponent

        amp_outer = amp.unsqueeze(2) * amp.unsqueeze(1)

        R = amp_outer * phase_term

        return R

    def compute_sparse_resonance(
        self, z: torch.Tensor, topology_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算稀疏共振分数矩阵

        使用拓扑距离掩码，只保留邻近神经元的共振
        """
        B, N = z.shape

        R_full = self.compute_resonance_matrix(z)

        if topology_weights is not None and self.use_topology_mask:
            if self._topo_cache is None or self._topo_cache_N != N:
                mask = (topology_weights.abs() > self.topology_threshold.abs()).float()
                self._topo_cache = mask.detach()
                self._topo_cache_N = N
            else:
                mask = self._topo_cache
            R_sparse = R_full * mask.unsqueeze(0).expand(B, -1, -1)
        else:
            R_sparse = self._topk_sparsify(R_full, self.sparse_k)

        return R_sparse

    def _topk_sparsify(self, R: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k 稀疏化"""
        B, N, _ = R.shape
        R_flat = R.view(B, -1)
        total = R_flat.shape[1]

        k_val = min(max(k * N, 1), total - 1)
        threshold = R_flat.kthvalue(total - k_val, dim=1, keepdim=True).values

        mask = (R_flat >= threshold).float()
        R_sparse = R_flat * mask

        return R_sparse.view(B, N, N)

    def apply_resonance(
        self, z: torch.Tensor, R: torch.Tensor, mode: str = "additive"
    ) -> torch.Tensor:
        """
        将共振分数应用到状态更新

        模式:
          additive: dzdt += α · R @ z
          multiplicative: dzdt *= (1 + α · R_mean)
          gating: dzdt = σ(R @ 1) · dzdt
        """
        B, N = z.shape

        if mode == "additive":
            z_r = z.real.unsqueeze(-1)
            z_i = z.imag.unsqueeze(-1)
            out_r = torch.bmm(R, z_r).squeeze(-1)
            out_i = torch.bmm(R, z_i).squeeze(-1)
            resonance_effect = torch.complex(out_r, out_i)
            return self.resonance_strength * resonance_effect

        elif mode == "multiplicative":
            R_mean = R.mean(dim=-1)
            modulation = 1 + self.resonance_strength * R_mean
            return z * modulation

        elif mode == "gating":
            gate = torch.sigmoid(
                torch.bmm(R, torch.ones(B, N, 1, device=z.device)).squeeze(-1)
            )
            return gate * z

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward(
        self,
        z: torch.Tensor,
        topology_weights: Optional[torch.Tensor] = None,
        mode: str = "additive",
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            z: 复数状态 (B, N)
            topology_weights: 莫比乌斯拓扑权重 (N, N)，用于稀疏掩码
            mode: 应用模式

        Returns:
            dzdt_resonance: 共振驱动的状态变化
        """
        if self.sparse_mode:
            R = self.compute_sparse_resonance(z, topology_weights)
        else:
            R = self.compute_resonance_matrix(z)

        dzdt_resonance = self.apply_resonance(z, R, mode)

        return dzdt_resonance


class MultiHeadResonance(nn.Module):
    """
    多共振头注意力

    类似多头注意力，但每个头学习不同的共振核函数。
    每个头处理完整的输入维度，输出加权组合。
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.heads = nn.ModuleList(
            [TwistorResonance(hidden_dim, **kwargs) for _ in range(num_heads)]
        )

        self.head_weights = nn.Parameter(torch.ones(num_heads))

    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = []
        for head in self.heads:
            out_h = head(z, **kwargs)
            outputs.append(out_h)

        stacked = torch.stack(outputs, dim=0)
        weights = F.softmax(self.head_weights, dim=0)
        result = (weights.view(-1, 1, 1) * stacked).sum(dim=0)

        return result


def create_resonance(hidden_dim: int, multi_head: bool = False, **kwargs) -> nn.Module:
    """工厂函数创建共振注意力模块"""
    if multi_head:
        return MultiHeadResonance(hidden_dim, **kwargs)
    else:
        return TwistorResonance(hidden_dim, **kwargs)
