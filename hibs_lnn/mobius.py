"""
多维莫比乌斯环约束模块 (Mobius Manifold Constraint)
====================================================
核心特性:
1. 可学习升维 - 流形维度随神经元增长自动进化
2. 混合扭转模式 - 低维莫比乌斯 + 高维克莱因自动过渡
3. 状态投影约束 - 将复数状态约束到非定向流形上
4. 拓扑连接调制 - 连接权重受流形几何调制

流形维度进化路径:
  1-8 神经元   → 1D 莫比乌斯环 (半扭转 π)
  9-32 神经元  → 2D 莫比乌斯曲面 (双参数扭转)
  33-128 神经元 → 3D 莫比乌斯体 (三参数扭转)
  129-512 神经元 → 4D+ 高维非定向流形
  512+ 神经元  → 克莱因瓶型全局约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class MobiusConstraint(nn.Module):
    """
    多维莫比乌斯环约束层

    核心机制:
    1. 可学习的维度决策器 - 自动决定当前流形维度
    2. 混合扭转张量 - 莫比乌斯模式与克莱因模式混合
    3. 状态投影算子 - 将 z 投影到流形上
    4. 拓扑距离调制 - 用于连接权重和共振计算
    """

    def __init__(
        self,
        max_dim: int = 512,
        enable_learning: bool = True,
        mobius_weight: float = 1.0,
        klein_weight: float = 0.0,
        constraint_strength: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()

        self.max_dim = max_dim
        self.enable_learning = enable_learning
        self.constraint_strength = constraint_strength

        self.mobius_weight = nn.Parameter(torch.tensor(mobius_weight))
        self.klein_weight = nn.Parameter(torch.tensor(klein_weight))

        if enable_learning:
            self.dimension_decoder = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
            )

        self.current_manifold_dim = 1
        self._last_manifold_dim = 1

        self.register_buffer("positions", torch.zeros(max_dim))

        self._twist_tensor = None
        self._twist_dirty = True
        self._last_hidden_dim = 0
        self._last_weights = (self.mobius_weight.item(), self.klein_weight.item())

        self._topo_cache = None
        self._topo_cache_N = 0

        self.device = device

    def _get_device(self):
        """Dynamically get device from parameters"""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device(self.device)

    def compute_manifold_dimension(self, hidden_dim: int) -> int:
        """计算当前流形维度 - 可学习决定"""
        if not self.enable_learning:
            return self._rule_based_dimension(hidden_dim)

        x = torch.tensor(
            [[hidden_dim / self.max_dim]],
            dtype=torch.float32,
            device=self._get_device(),
        )
        logits = self.dimension_decoder(x)
        dim_idx = logits.argmax(dim=-1).item()
        manifold_dim = dim_idx + 1
        manifold_dim = min(manifold_dim, hidden_dim)
        manifold_dim = max(1, manifold_dim)
        return manifold_dim

    def _rule_based_dimension(self, hidden_dim: int) -> int:
        """规则式维度计算 (回退方案)"""
        if hidden_dim <= 8:
            return 1
        elif hidden_dim <= 32:
            return 2
        elif hidden_dim <= 128:
            return 3
        elif hidden_dim <= 512:
            return 4
        else:
            return min(hidden_dim // 128, 8)

    def compute_twist_tensor(self, hidden_dim: int, manifold_dim: int) -> torch.Tensor:
        """
        计算混合扭转张量 Θ ∈ ℝ^(N×N)

        混合模式:
          Θ = α · Θ_mobius + β · Θ_klein

        其中:
          Θ_mobius[i,j] = π · (i+j) / (2·N)  (莫比乌斯半扭转)
          Θ_klein[i,j] = 2π · (i·j) / N²     (克莱因全局扭转)
        """
        weights_changed = (
            abs(self.mobius_weight.item() - self._last_weights[0]) > 1e-6
            or abs(self.klein_weight.item() - self._last_weights[1]) > 1e-6
        )
        dim_changed = (
            hidden_dim != self._last_hidden_dim
            or manifold_dim != self._last_manifold_dim
        )

        if (
            self._twist_tensor is not None
            and not self._twist_dirty
            and not weights_changed
            and not dim_changed
        ):
            if self._twist_tensor.shape == (hidden_dim, hidden_dim):
                return self._twist_tensor

        N = hidden_dim
        d = manifold_dim

        i = torch.arange(N, dtype=torch.float32, device=self._get_device())
        j = torch.arange(N, dtype=torch.float32, device=self._get_device())
        I, J = torch.meshgrid(i, j, indexing="ij")

        twist_mobius = math.pi * (I + J) / (2 * N) * d

        twist_klein = 2 * math.pi * (I * J) / (N * N) * d

        alpha = torch.sigmoid(self.mobius_weight)
        beta = torch.sigmoid(self.klein_weight)
        total = alpha + beta + 1e-6
        alpha = alpha / total
        beta = beta / total

        twist = alpha * twist_mobius + beta * twist_klein

        self._twist_tensor = twist.detach()
        self._twist_dirty = False
        self._last_hidden_dim = hidden_dim
        self._last_manifold_dim = manifold_dim
        self._last_weights = (self.mobius_weight.item(), self.klein_weight.item())

        return twist

    def project_state(self, z: torch.Tensor) -> torch.Tensor:
        """
        将复数状态投影到莫比乌斯流形上

        投影算子:
          P_M(z) = z ⊙ exp(i · Θ · pos)

        支持 warmup: 通过 set_constraint_strength() 动态调整软约束强度.
        当 strength=0 时退化为恒等映射 (无约束).
        """
        B, N = z.shape
        # 短路: 强度 = 0 时直接返回 z (无 Möbius 约束)
        if float(self.constraint_strength) <= 0.0:
            return z
        manifold_dim = self.compute_manifold_dimension(N)

        self.positions[:N] = torch.linspace(0, 1, N, device=self._get_device())
        twist = self.compute_twist_tensor(N, manifold_dim)

        pos = torch.linspace(0, 1, N, device=self._get_device())
        twist_phase = twist @ pos

        twist_factor = torch.exp(1j * twist_phase)

        z_projected = z * twist_factor.unsqueeze(0).expand(B, -1)

        z_constrained = z + self.constraint_strength * (z_projected - z)

        return z_constrained

    def set_constraint_strength(self, strength: float) -> None:
        """动态设置软约束强度, 支持 warmup ramp.

        Args:
            strength: ∈ [0, 1]. 0=无约束, 1=完全投影.
        """
        # constraint_strength 默认是 float; 用 object.__setattr__ 绕过 nn.Module 的赋值检查
        object.__setattr__(self, "constraint_strength", float(strength))

    def topology_distance(self, i: int, j: int, N: int) -> float:
        """计算流形上的拓扑距离（环距离）"""
        d_ring = min(abs(i - j), N - abs(i - j))
        return d_ring / N

    def topology_weight_matrix(self, N: int) -> torch.Tensor:
        """
        计算拓扑距离调制矩阵 (带缓存)

        W_topology[i,j] = exp(-d(i,j)² / (2σ²)) · cos(π · d(i,j))
        """
        if self._topo_cache is not None and self._topo_cache_N == N:
            return self._topo_cache

        manifold_dim = self.compute_manifold_dimension(N)

        i = torch.arange(N, dtype=torch.float32, device=self._get_device())
        j = torch.arange(N, dtype=torch.float32, device=self._get_device())
        I, J = torch.meshgrid(i, j, indexing="ij")

        d_ring = torch.min(torch.abs(I - J), N - torch.abs(I - J))
        d_norm = d_ring / N

        sigma = 0.3  # Gaussian bandwidth for topology weight decay
        gaussian = torch.exp(-(d_norm**2) / (2 * sigma**2))

        mobius_modulation = torch.cos(math.pi * d_norm * manifold_dim)

        alpha = torch.sigmoid(self.mobius_weight)
        beta = torch.sigmoid(self.klein_weight)

        W_topo = gaussian * (
            alpha * mobius_modulation + beta * torch.cos(2 * math.pi * d_norm)
        )

        self._topo_cache = W_topo.detach()
        self._topo_cache_N = N

        return W_topo

    def on_dimension_change(self, new_hidden_dim: int):
        """当神经元数量变化时调用 (自生长触发)"""
        old_dim = self.current_manifold_dim
        new_manifold_dim = self.compute_manifold_dimension(new_hidden_dim)

        if new_manifold_dim != old_dim:
            self.current_manifold_dim = new_manifold_dim
            self._twist_dirty = True
            return True

        return False

    def get_manifold_info(self, hidden_dim: int) -> Dict:
        """获取当前流形状态信息"""
        manifold_dim = self.compute_manifold_dimension(hidden_dim)

        alpha = torch.sigmoid(self.mobius_weight).item()
        beta = torch.sigmoid(self.klein_weight).item()
        total = alpha + beta + 1e-6

        if alpha / total > 0.7:
            mode = "mobius"
        elif beta / total > 0.7:
            mode = "klein"
        else:
            mode = "mixed"

        return {
            "manifold_dim": manifold_dim,
            "hidden_dim": hidden_dim,
            "mode": mode,
            "mobius_weight": alpha / total,
            "klein_weight": beta / total,
            "constraint_strength": self.constraint_strength,
        }

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """前向: 状态投影到流形"""
        return self.project_state(z)


class AdaptiveMobiusConstraint(MobiusConstraint):
    """
    自适应莫比乌斯约束 - 增强版

    额外特性:
    1. 可学习的逐神经元约束强度
    2. 动态模式过渡 (训练过程中自动从莫比乌斯过渡到克莱因)
    3. 流形能量监控
    """

    def __init__(self, max_dim: int = 512, **kwargs):
        super().__init__(max_dim=max_dim, **kwargs)

        device = kwargs.get("device", "cpu")
        self.per_neuron_strength = nn.Parameter(torch.zeros(max_dim, device=device))
        self.transition_progress = nn.Parameter(torch.tensor(0.0, device=device))

    def update_transition(self, step: int, max_steps: int):
        """更新模式过渡进度"""
        progress = step / max(max_steps, 1)
        self.transition_progress.data = torch.tensor(progress, device=self.device)

        with torch.no_grad():
            self.mobius_weight.data = torch.tensor(
                1.0 - progress * 0.7, device=self.device
            )
            self.klein_weight.data = torch.tensor(progress * 0.7, device=self.device)

    def project_state(self, z: torch.Tensor) -> torch.Tensor:
        """增强版状态投影 - 逐神经元约束强度"""
        B, N = z.shape

        z_projected = super().project_state(z)

        strength = (
            torch.sigmoid(self.per_neuron_strength[:N]).unsqueeze(0).expand(B, -1)
        )
        strength = strength * self.constraint_strength

        z_constrained = z + strength * (z_projected - z)

        return z_constrained

    def compute_manifold_energy(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算流形能量 - 衡量状态偏离流形的程度
        """
        z_projected = super().project_state(z)
        energy = torch.abs(z - z_projected).mean()
        return energy


def create_mobius_constraint(
    hidden_dim: int, enable_learning: bool = True, adaptive: bool = False, **kwargs
) -> MobiusConstraint:
    """
    工厂函数创建莫比乌斯约束

    Args:
        hidden_dim: 初始神经元数量
        enable_learning: 是否使用可学习维度决策
        adaptive: 是否使用自适应版本
        **kwargs: 其他参数

    Returns:
        constraint: MobiusConstraint 实例
    """
    max_dim = kwargs.get("max_dim", max(hidden_dim * 4, 512))

    if adaptive:
        return AdaptiveMobiusConstraint(
            max_dim=max_dim, enable_learning=enable_learning, **kwargs
        )
    else:
        return MobiusConstraint(
            max_dim=max_dim, enable_learning=enable_learning, **kwargs
        )
