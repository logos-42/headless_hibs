"""
受体门控扭量液态神经网络 v2 (Receptor-Gated Twistor LMT)
=======================================================
核心架构:
  z ∈ ℂ^n          ← 神经元状态(复数向量)
  c ∈ ℝ^k          ← 通道/递质类型(动态,由输入决定)
  g = σ(W_r · c)   ← 受体门控(操作符,神经元特异性)
  A_gated = A * g  ← 门控振幅
  Θ' = Θ + ΔΘ(c)   ← 相位调制(递质控制路径)
  W = A_gated · exp(i·Θ')  ← 复数权重

关键特性:
  1. 动态递质: c = f(x), 不是静态参数
  2. 神经元特异性受体: 每个神经元对不同递质有不同敏感度
  3. 低秩振幅门控: g_row × g_col, 控制特定路径而非整个神经元
  4. 相位调制: 递质不仅控制强度,还控制信息路径
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import math


@dataclass
class GrowthConfig:
    min_hidden_dim: int = 0
    max_hidden_dim: int = 128
    prob_add_connection: float = 0.05
    prob_add_node: float = 0.03
    growth_interval: int = 10
    prune_interval: int = 25
    topology_penalty: float = 0.001
    enable_growth: bool = True


class ReceptorGatedTwistorLMT(nn.Module):
    """
    受体门控扭量液态神经网络 v2

    架构层次:
      1. 表示层: z ∈ ℂ^n (复数状态)
      2. 通道层: c ∈ ℝ^k (动态递质)
      3. 受体层: g = σ(W_r · c) (门控操作符)
      4. 权重层: W = A_gated · exp(i·Θ') (复数权重)
      5. 动力学: dz/dt = (-z + W·tanh(z) + Ux) / τ
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        n_channels: int = 3,
        n_receptor_types: int = 3,
        dt: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        dzdt_max: float = 10.0,
        z_max: float = 100.0,
        sparsity: float = 0.3,
        use_low_rank_gating: bool = True,
        use_phase_modulation: bool = True,
        growth_config: Optional[GrowthConfig] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_channels = n_channels
        self.n_receptor_types = n_receptor_types
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dzdt_max = dzdt_max
        self.z_max = z_max
        self.sparsity = sparsity
        self.use_low_rank_gating = use_low_rank_gating
        self.use_phase_modulation = use_phase_modulation

        self.growth_config = growth_config or GrowthConfig()

        # === 1. 动态通道网络 (递质生成) ===
        self.channel_net = nn.Sequential(
            nn.Linear(input_dim, n_channels * 2),
            nn.ReLU(),
            nn.Linear(n_channels * 2, n_channels),
        )

        # === 2. 神经元特异性受体 ===
        # 每个神经元对每种递质有不同敏感度
        self.receptor_weights = nn.Parameter(
            torch.randn(hidden_dim, n_receptor_types) * 0.5
        )

        # 递质到受体类型的映射
        self.channel_to_receptor = nn.Linear(n_channels, n_receptor_types)

        # === 3. 振幅矩阵 (可学习) ===
        self.W_amplitude = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.5)

        # 低秩门控投影 (如果启用)
        if use_low_rank_gating:
            self.gate_row_proj = nn.Linear(n_receptor_types, hidden_dim)
            self.gate_col_proj = nn.Linear(n_receptor_types, hidden_dim)

        # === 4. 相位调制网络 ===
        if use_phase_modulation:
            self.phase_net = nn.Linear(n_receptor_types, hidden_dim * hidden_dim)
        else:
            self.phase_net = None

        # 基础相位 (莫比乌斯扭转)
        self.manifold_theta = nn.Parameter(torch.randn(hidden_dim, 3) * 0.1)

        # === 5. 输入/输出层 ===
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        self.tau_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.b_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Linear(hidden_dim, output_dim)

        # 稀疏掩码
        self.sparse_mask = nn.Parameter(torch.ones(hidden_dim, hidden_dim) * -5)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau.weight, gain=0.1)
        nn.init.orthogonal_(self.W_amplitude, gain=0.5)

        # 初始化流形坐标 (均匀分布在环上)
        for i in range(self.hidden_dim):
            theta = 2 * math.pi * i / max(1, self.hidden_dim)
            self.manifold_theta.data[i, 0] = theta
            self.manifold_theta.data[i, 1] = 0.0
            self.manifold_theta.data[i, 2] = 0.0

        # 初始化稀疏掩码
        if self.sparsity > 0:
            with torch.no_grad():
                mask = (
                    torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity
                ).float()
                self.sparse_mask.data.copy_(mask * 4 - 5)

    def compute_channels(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算动态递质浓度

        Args:
            x: 输入 [batch, input_dim]

        Returns:
            c: 递质浓度 [batch, n_channels]
        """
        c = torch.sigmoid(self.channel_net(x))
        return c

    def compute_receptor_activation(self, c: torch.Tensor) -> torch.Tensor:
        """
        计算受体激活

        Args:
            c: 递质浓度 [batch, n_channels]

        Returns:
            receptor_act: 受体激活 [batch, n_receptor_types]
        """
        receptor_act = torch.tanh(self.channel_to_receptor(c))
        return receptor_act

    def compute_gate(
        self, receptor_act: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        计算门控信号

        Args:
            receptor_act: 受体激活 [batch, n_receptor_types]

        Returns:
            gate: 门控信号 [batch, hidden_dim] 或 [batch, hidden_dim, hidden_dim]
            gate_2d: 2D门控 (仅低秩模式)
        """
        if self.use_low_rank_gating:
            # 低秩门控: g_row × g_col
            g_row = torch.sigmoid(self.gate_row_proj(receptor_act))  # [batch, hidden]
            g_col = torch.sigmoid(self.gate_col_proj(receptor_act))  # [batch, hidden]
            return g_row, g_col
        else:
            # 简单门控
            gate = torch.sigmoid(self.gate_row_proj(receptor_act))  # [batch, hidden]
            return gate, None

    def compute_twist_phase(self) -> torch.Tensor:
        """
        计算基础莫比乌斯扭转相位

        Returns:
            Θ_base: 基础相位 [hidden, hidden]
        """
        n = self.hidden_dim
        theta = self.manifold_theta[:, 0]  # [hidden]

        i = theta.unsqueeze(1)  # [hidden, 1]
        j = theta.unsqueeze(0)  # [1, hidden]

        # 莫比乌斯半扭转
        twist_mobius = math.pi * (i + j) / (2 * max(n, 1))

        # 克莱因全局扭转
        twist_klein = 2 * math.pi * (i * j) / (max(n, 1) ** 2)

        # 混合 (默认 50/50)
        phase = 0.5 * twist_mobius + 0.5 * twist_klein

        return phase

    def compute_phase_modulation(self, receptor_act: torch.Tensor) -> torch.Tensor:
        """
        计算相位调制 (递质控制路径)

        Args:
            receptor_act: 受体激活 [batch, n_receptor_types]

        Returns:
            phase_shift: 相位偏移 [batch, hidden, hidden]
        """
        if not self.use_phase_modulation or self.phase_net is None:
            return torch.zeros(
                receptor_act.shape[0],
                self.hidden_dim,
                self.hidden_dim,
                device=receptor_act.device,
            )

        # 相位偏移 [batch, hidden*hidden]
        phase_shift_flat = self.phase_net(receptor_act)
        phase_shift = phase_shift_flat.view(-1, self.hidden_dim, self.hidden_dim)

        # 限制相位偏移范围 (避免 phase chaos)
        phase_shift = torch.tanh(phase_shift) * math.pi

        return phase_shift

    def get_complex_weight(self, c: torch.Tensor) -> torch.Tensor:
        """
        获取复数权重矩阵 W = A_gated · exp(i·Θ')

        Args:
            c: 递质浓度 [batch, n_channels]

        Returns:
            W: 复数权重 [batch, hidden, hidden]
        """
        batch_size = c.shape[0]
        n = self.hidden_dim

        # 1. 受体激活
        receptor_act = self.compute_receptor_activation(c)  # [batch, n_receptor_types]

        # 2. 门控信号
        gate_1d, gate_2d = self.compute_gate(receptor_act)

        # 3. 振幅门控
        if self.use_low_rank_gating and gate_2d is not None:
            # 低秩门控: A * g_row * g_col^T
            A = self.W_amplitude.unsqueeze(0)  # [1, hidden, hidden]
            g_row = gate_1d.unsqueeze(2)  # [batch, hidden, 1]
            g_col = gate_2d.unsqueeze(1)  # [batch, 1, hidden]
            A_gated = A * g_row * g_col  # [batch, hidden, hidden]
        else:
            # 简单门控
            A = self.W_amplitude.unsqueeze(0)  # [1, hidden, hidden]
            gate = gate_1d.unsqueeze(2)  # [batch, hidden, 1]
            A_gated = A * gate  # [batch, hidden, hidden]

        # 4. 相位计算
        Θ_base = self.compute_twist_phase()  # [hidden, hidden]
        Θ_base = Θ_base.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, hidden, hidden]

        # 5. 相位调制
        phase_shift = self.compute_phase_modulation(
            receptor_act
        )  # [batch, hidden, hidden]
        Θ = Θ_base + phase_shift

        # 6. 稀疏掩码
        mask = torch.sigmoid(self.sparse_mask)  # [hidden, hidden]
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, hidden, hidden]

        # 7. 复数权重
        W = A_gated * mask * torch.exp(1j * Θ)

        return W

    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """计算自适应时间尺度"""
        z_mod = torch.abs(z)
        tau = F.sigmoid(self.W_tau(z_mod))
        tau = tau + self.tau_bias.unsqueeze(0)
        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau + 1e-6

    def compute_dzdt(
        self, z: torch.Tensor, x: torch.Tensor, c: torch.Tensor
    ) -> torch.Tensor:
        """
        计算状态导数

        Args:
            z: 当前状态 [batch, hidden]
            x: 输入 [batch, input_dim]
            c: 递质浓度 [batch, n_channels]

        Returns:
            dzdt: 状态导数 [batch, hidden]
        """
        z_real = z.real
        z_imag = z.imag

        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        # 复数权重
        W = self.get_complex_weight(c)  # [batch, hidden, hidden]

        # 复数矩阵乘法
        W_real = W.real  # [batch, hidden, hidden]
        W_imag = W.imag  # [batch, hidden, hidden]

        # W @ tanh(z)
        W_tanh_real = torch.bmm(W_real, tanh_real.unsqueeze(2)).squeeze(2) - torch.bmm(
            W_imag, tanh_imag.unsqueeze(2)
        ).squeeze(2)
        W_tanh_imag = torch.bmm(W_real, tanh_imag.unsqueeze(2)).squeeze(2) + torch.bmm(
            W_imag, tanh_real.unsqueeze(2)
        ).squeeze(2)

        Ux = self.U(x)

        dz_real = -z_real + W_tanh_real + Ux + self.b_real.unsqueeze(0)
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag.unsqueeze(0)

        tau = self.compute_tau(z)
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        dzdt_real = torch.clamp(dzdt.real, -self.dzdt_max, self.dzdt_max)
        dzdt_imag = torch.clamp(dzdt.imag, -self.dzdt_max, self.dzdt_max)
        dzdt = torch.complex(dzdt_real, dzdt_imag)

        return dzdt

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
        return_channels: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            x: 输入 [seq_len, batch, input_dim]
            return_states: 是否返回所有状态
            return_channels: 是否返回递质浓度

        Returns:
            y: 输出 [seq_len, batch, output_dim]
            states: 状态列表 (可选)
            channels: 递质浓度列表 (可选)
        """
        T, B, _ = x.shape

        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []
        channels = []

        for t in range(T):
            x_t = x[t]

            # 计算递质浓度
            c = self.compute_channels(x_t)  # [B, n_channels]

            # 计算状态导数
            dzdt = self.compute_dzdt(z, x_t, c)

            # 状态更新
            z = z + self.dt * dzdt

            # 限制状态范围
            z = torch.complex(
                torch.clamp(z.real, -self.z_max, self.z_max),
                torch.clamp(z.imag, -self.z_max, self.z_max),
            )

            # 输出
            y_t = self.out(z.real)
            outputs.append(y_t)

            if return_states:
                states.append(z.detach().cpu())
            if return_channels:
                channels.append(c.detach().cpu())

        y = torch.stack(outputs, dim=0)

        if return_states and return_channels:
            return y, states, channels
        elif return_states:
            return y, states
        elif return_channels:
            return y, channels
        else:
            return y

    def compute_amplitude_regularization(
        self, l1_weight: float = 0.01, l2_weight: float = 0.001
    ) -> torch.Tensor:
        """振幅正则化"""
        l1_loss = self.W_amplitude.abs().mean()
        l2_loss = self.W_amplitude.pow(2).mean()
        return l1_weight * l1_loss + l2_weight * l2_loss

    def get_diagnostics(self) -> Dict:
        """获取模型诊断信息"""
        diag = {
            "hidden_dim": self.hidden_dim,
            "n_channels": self.n_channels,
            "n_receptor_types": self.n_receptor_types,
            "use_low_rank_gating": self.use_low_rank_gating,
            "use_phase_modulation": self.use_phase_modulation,
        }

        if self.hidden_dim > 0:
            W = self.get_complex_weight(torch.ones(1, self.n_channels))
            diag["amp_mean"] = W.abs().mean().item()
            diag["amp_std"] = W.abs().std().item()
            diag["phase_std"] = W.angle().std().item()
            diag["weight_sparse_ratio"] = (W.abs() < 0.01).float().mean().item()

        return diag

    # ============ 反思模块接口 ============

    def reset_state(self, batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
        """重置隐藏状态"""
        return torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.complex64, device=device
        )

    def get_overloaded_neurons(self) -> list:
        """
        获取过载神经元索引

        用于反思模块决定是否需要增长

        Returns:
            overloaded: 过载神经元索引列表
        """
        overloaded = []

        if not hasattr(self, '_activation_buffer') or len(self._activation_buffer) < 10:
            # 没有足够数据，返回随机选择
            if self.hidden_dim > 0:
                return [np.random.randint(self.hidden_dim)]
            return overloaded

        # 分析激活方差
        valid_buffer = [
            b for b in self._activation_buffer
            if b.shape[0] == self.hidden_dim
        ]
        if len(valid_buffer) < 5:
            if self.hidden_dim > 0:
                return [np.random.randint(self.hidden_dim)]
            return overloaded

        buffer = torch.stack(valid_buffer[-50:])
        variance = buffer.var(dim=0)

        # 找出方差最大的神经元
        threshold = variance.quantile(0.8).item()
        overloaded = (variance > threshold).nonzero(as_tuple=True)[0].tolist()

        if not overloaded and self.hidden_dim > 0:
            overloaded = [np.random.randint(self.hidden_dim)]

        return overloaded

    def split_neuron(self, parent_idx: int) -> int:
        """
        分裂神经元

        Args:
            parent_idx: 父神经元索引

        Returns:
            new_idx: 新神经元索引，失败返回-1
        """
        if self.hidden_dim >= self.growth_config.max_hidden_dim:
            return -1

        if parent_idx < 0 or parent_idx >= self.hidden_dim:
            return -1

        new_idx = self.hidden_dim

        with torch.no_grad():
            # 复制父神经元的权重
            self.W_amplitude.data[new_idx] = self.W_amplitude.data[parent_idx] * 0.5
            self.W_amplitude.data[parent_idx] = self.W_amplitude.data[parent_idx] * 0.5

            self.sparse_mask.data[new_idx] = self.sparse_mask.data[parent_idx]

            self.manifold_theta.data[new_idx] = (
                self.manifold_theta.data[parent_idx] + torch.randn(3) * 0.1
            )

            self.b_real.data[new_idx] = self.b_real.data[parent_idx]
            self.b_imag.data[new_idx] = self.b_imag.data[parent_idx]

            if hasattr(self, 'tau_bias') and self.tau_bias is not None:
                self.tau_bias.data[new_idx] = self.tau_bias.data[parent_idx]

            # 输入权重
            if hasattr(self, 'U'):
                self.U.weight.data[new_idx] = self.U.weight.data[parent_idx] * 0.5
                self.U.weight.data[parent_idx] = self.U.weight.data[parent_idx] * 0.5

            # 输出权重
            if hasattr(self, 'out'):
                self.out.weight.data[:, new_idx] = self.out.weight.data[:, parent_idx] * 0.5
                self.out.weight.data[:, parent_idx] = self.out.weight.data[:, parent_idx] * 0.5

        self.hidden_dim += 1

        return new_idx

    def prune_neurons(self) -> int:
        """
        剪枝不重要的神经元

        Returns:
            n_pruned: 被剪枝的神经元数量
        """
        if self.hidden_dim <= self.growth_config.min_hidden_dim:
            return 0

        pruned = 0

        if not hasattr(self, '_activation_buffer') or len(self._activation_buffer) < 10:
            # 没有足够数据，随机剪枝一个
            if self.hidden_dim > self.growth_config.min_hidden_dim:
                prune_idx = self.hidden_dim - 1
                self._prune_neuron_at(prune_idx)
                pruned = 1
            return pruned

        # 分析激活强度
        valid_buffer = [
            b for b in self._activation_buffer
            if b.shape[0] == self.hidden_dim
        ]
        if len(valid_buffer) < 5:
            return 0

        buffer = torch.stack(valid_buffer[-50:])
        mean_activation = buffer.mean(dim=0)

        # 找出最不活跃的神经元
        threshold = mean_activation.quantile(0.2).item()
        candidates = (mean_activation < threshold).nonzero(as_tuple=True)[0].tolist()

        for idx in candidates:
            if self.hidden_dim <= self.growth_config.min_hidden_dim:
                break
            self._prune_neuron_at(idx)
            pruned += 1

        return pruned

    def _prune_neuron_at(self, idx: int):
        """剪枝指定索引的神经元"""
        with torch.no_grad():
            # 将权重置零
            self.W_amplitude.data[idx] = 0
            self.sparse_mask.data[idx] = -10

            if hasattr(self, 'tau_bias') and self.tau_bias is not None:
                self.tau_bias.data[idx] = 0

        self.hidden_dim -= 1

    def compute_tau_stats(self) -> dict:
        """计算时间常数统计"""
        tau = self.compute_tau(
            torch.zeros(1, self.hidden_dim, device=next(self.parameters()).device)
        )
        return {
            'tau_mean': tau.mean().item(),
            'tau_std': tau.std().item(),
            'tau_min': tau.min().item(),
            'tau_max': tau.max().item(),
        }
