"""
可增长扭量液态神经网络 (Growable Twistor-LMT)
==============================================
完全模仿NEAT (NeuroEvolution of Augmenting Topologies)机制：
1. Add Node Mutation - 分裂神经元，禁用原连接，插入新节点
2. Add Connection Mutation - 添加新连接
3. Disable Connection - 禁用连接(剪枝)
4. 从最小结构开始(0隐藏节点)
5. 扭量复数状态空间
6. 流形几何约束 - 权重和生长在莫比乌斯-克莱因流形上
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
import math


@dataclass
class ConnectionGene:
    """连接基因 - NEAT风格的基因表示"""

    in_node: int
    out_node: int
    weight: float
    enabled: bool = True
    innovation: int = 0


@dataclass
class NeuronState:
    index: int
    neuron_type: str = "hidden"  # input, output, hidden
    active: bool = True
    birth_step: int = 0
    activation_variance: float = 0.0
    activation_mean: float = 0.0
    importance_score: float = 0.0
    usage_count: int = 0

    consolidation_score: float = 0.0
    decay_counter: float = 0.0
    survival_threshold: float = 0.0
    recent_activity: float = 0.0
    life_span: int = 0
    peak_activation: float = 0.0


@dataclass
class DevelopmentalPhase:
    """发育阶段定义 - 模拟人类大脑发育"""

    name: str
    start_step: int
    end_step: int
    target_neurons: int
    target_connections_per_neuron: int
    growth_rate: float
    prune_rate: float
    plasticity: float
    description: str


@dataclass
class GrowthConfig:
    min_hidden_dim: int = 0
    max_hidden_dim: int = 8192

    prob_add_connection: float = 0.05
    prob_add_node: float = 0.03
    prob_disable_connection: float = 0.1

    prune_threshold: float = 0.05
    connection_threshold: float = 0.01

    growth_interval: int = 10
    prune_interval: int = 25

    topology_penalty: float = 0.001

    aggressive_growth: bool = False
    aggressive_growth_steps: int = 5
    aggressive_growth_target: int = 8

    consolidation_rate: float = 0.02
    decay_rate: float = 0.01
    survival_threshold: float = 0.15

    developmental_phases: List[DevelopmentalPhase] = None
    enable_developmental_schedule: bool = True

    def __post_init__(self):
        if self.developmental_phases is None and self.enable_developmental_schedule:
            self.developmental_phases = self._create_brain_development_schedule()

    def _create_brain_development_schedule(self) -> List[DevelopmentalPhase]:
        """创建人类大脑发育时间表 (×100量级)

        5个阶段:
        1. 胎儿期: 结构生成 (6400神经元)
        2. 0-2岁: 连接爆炸 (600连接/神经元, 成人2倍)
        3. 3-10岁: 修剪优化 (300连接/神经元)
        4. 青春期: 系统重构 (400连接/神经元)
        5. 20-30岁: 整合巅峰 (300连接/神经元, 收敛)
        """
        return [
            DevelopmentalPhase(
                name="fetal",
                start_step=0,
                end_step=100,
                target_neurons=6400,
                target_connections_per_neuron=2,
                growth_rate=0.8,
                prune_rate=0.0,
                plasticity=1.0,
                description="胎儿期: 6400神经元生成,硬件搭建",
            ),
            DevelopmentalPhase(
                name="infant",
                start_step=100,
                end_step=300,
                target_neurons=6400,
                target_connections_per_neuron=600,
                growth_rate=0.1,
                prune_rate=0.02,
                plasticity=1.0,
                description="婴儿期(0-2岁): 连接爆炸,600连接/神经元(成人2倍)",
            ),
            DevelopmentalPhase(
                name="child",
                start_step=300,
                end_step=600,
                target_neurons=4800,
                target_connections_per_neuron=300,
                growth_rate=0.05,
                prune_rate=0.3,
                plasticity=0.7,
                description="儿童期(3-10岁): 修剪优化,300连接/神经元",
            ),
            DevelopmentalPhase(
                name="adolescent",
                start_step=600,
                end_step=900,
                target_neurons=6400,
                target_connections_per_neuron=400,
                growth_rate=0.4,
                prune_rate=0.15,
                plasticity=0.8,
                description="青春期(10-20岁): 系统重构,400连接/神经元",
            ),
            DevelopmentalPhase(
                name="adult",
                start_step=900,
                end_step=1500,
                target_neurons=4800,
                target_connections_per_neuron=300,
                growth_rate=0.05,
                prune_rate=0.1,
                plasticity=0.3,
                description="成年期(20-30岁): 整合巅峰,300连接/神经元,收敛",
            ),
        ]


class GrowableTwistorLMT(nn.Module):
    """
    可增长扭量液态神经网络 - NEAT风格

    核心机制:
    - Add Node Mutation: 分裂神经元，禁用原连接
    - Add Connection Mutation: 添加新连接
    - Disable Connection: 剪枝不重要连接
    - 从最小结构开始，逐步增长
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 0,
        output_dim: int = 1,
        sparsity: float = 0.3,
        multi_scale_tau: bool = True,
        dt: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        dzdt_max: float = 10.0,
        z_max: float = 100.0,
        growth_config: Optional[GrowthConfig] = None,
        enable_growth: bool = True,
        enable_mobius: bool = False,
        enable_resonance: bool = False,
        mobius_strength: float = 0.1,
        resonance_strength: float = 0.1,
        learn_manifold_dim: bool = True,
        sparse_resonance: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.multi_scale_tau = multi_scale_tau

        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dzdt_max = dzdt_max
        self.z_max = z_max

        self.growth_config = growth_config or GrowthConfig(
            min_hidden_dim=0, max_hidden_dim=128
        )
        self.enable_growth = enable_growth

        self._next_innovation = 1
        self.connection_genes: List[ConnectionGene] = []

        self.mobius = None
        self.resonance = None
        self._resonance_mode = "additive"

        self.manifold_geometry = None
        self.weight_initializer = None
        self.growth_planner = None

        self._preallocate_parameters()

        if enable_mobius or enable_resonance:
            self._init_mobius_resonance(
                enable_mobius=enable_mobius,
                enable_resonance=enable_resonance,
                mobius_strength=mobius_strength,
                resonance_strength=resonance_strength,
                learn_manifold_dim=learn_manifold_dim,
                sparse_resonance=sparse_resonance,
            )

        self._init_manifold_geometry()

        self.neuron_states: List[NeuronState] = []
        for i in range(input_dim):
            self.neuron_states.append(NeuronState(index=i, neuron_type="input"))
        for i in range(output_dim):
            self.neuron_states.append(
                NeuronState(index=input_dim + i, neuron_type="output")
            )
        for i in range(hidden_dim):
            self.neuron_states.append(
                NeuronState(index=input_dim + output_dim + i, neuron_type="hidden")
            )

        self.active_neurons = list(range(input_dim + output_dim))
        if hidden_dim > 0:
            self.active_neurons.extend(
                range(input_dim + output_dim, input_dim + output_dim + hidden_dim)
            )

        self.training_step = 0
        self._activation_buffer = []
        self._max_buffer_size = 100

    def _preallocate_parameters(self):
        """预分配参数 - 振幅+相位(流形约束)表示

        权重表示:
          W_ij = A_ij · exp(i·Θ_ij)
          A_ij ∈ ℝ+  ← 振幅(自由学习)
          Θ_ij = twist(θ_i, θ_j)  ← 相位(由流形坐标决定)

        优化: θ (N×3) + A (N×N)
        约束: 相位自动受莫比乌斯几何约束
        """
        max_h = self.growth_config.max_hidden_dim

        # 流形坐标 (每个神经元 3 维: θ, φ, twist_offset)
        self.manifold_theta = nn.Parameter(torch.randn(max_h, 3) * 0.1)

        # 振幅矩阵 (替代 W_real + W_imag)
        self.W_amplitude = nn.Parameter(torch.randn(max_h, max_h) * 0.5)

        # 输入/输出层 (保持实数)
        self.U = nn.Linear(self.input_dim, max_h)
        self.W_tau = nn.Linear(max_h, max_h)

        self.sparse_mask = nn.Parameter(torch.ones(max_h, max_h) * -5)

        if self.multi_scale_tau:
            self.tau_bias = nn.Parameter(torch.zeros(max_h))
        else:
            self.register_parameter("tau_bias", None)

        self.b_real = nn.Parameter(torch.zeros(max_h))
        self.b_imag = nn.Parameter(torch.zeros(max_h))

        self.out = nn.Linear(max_h, self.output_dim)

        if self.hidden_dim > 0:
            nn.init.orthogonal_(self.U.weight[: self.hidden_dim, :], gain=0.5)
            nn.init.orthogonal_(
                self.W_tau.weight[: self.hidden_dim, : self.hidden_dim], gain=0.1
            )
        else:
            nn.init.orthogonal_(self.U.weight, gain=0.5)

        nn.init.zeros_(self.b_real)
        nn.init.zeros_(self.b_imag)

        # 初始化流形坐标 (均匀分布在环上)
        if self.hidden_dim > 0:
            for i in range(self.hidden_dim):
                theta = 2 * math.pi * i / max(1, self.hidden_dim)
                self.manifold_theta.data[i, 0] = theta
                self.manifold_theta.data[i, 1] = 0.0
                self.manifold_theta.data[i, 2] = 0.0

    def _init_parameters(self):
        """兼容旧接口 - 使用预分配"""
        self._preallocate_parameters()

    def _init_weights(self):
        """初始化权重 - 振幅+相位"""
        if self.hidden_dim > 0:
            nn.init.orthogonal_(self.U.weight[: self.hidden_dim, :], gain=0.5)
            nn.init.orthogonal_(
                self.W_tau.weight[: self.hidden_dim, : self.hidden_dim], gain=0.1
            )

            # 初始化振幅 (orthogonal)
            amp = self.W_amplitude[: self.hidden_dim, : self.hidden_dim]
            nn.init.orthogonal_(amp, gain=0.5)

        nn.init.zeros_(self.b_real[: self.hidden_dim])
        nn.init.zeros_(self.b_imag[: self.hidden_dim])

        if self.sparsity > 0 and self.hidden_dim > 0:
            with torch.no_grad():
                mask = (
                    torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity
                ).float()
                self.sparse_mask.data[: self.hidden_dim, : self.hidden_dim].copy_(
                    mask * 4 - 5
                )

    def compute_twist_phase(self, n: int) -> torch.Tensor:
        """
        计算莫比乌斯扭转相位矩阵 Θ ∈ ℝ^(n×n)

        Θ_ij = twist_rate · (θ_i + θ_j) / (2n) + klein_mix · (θ_i · θ_j) / n²

        其中 θ_i 是神经元 i 的流形坐标
        """
        theta = self.manifold_theta[:n, 0]  # 取第一个维度作为角度

        i = theta.unsqueeze(1)  # (n, 1)
        j = theta.unsqueeze(0)  # (1, n)

        # 莫比乌斯半扭转
        twist_mobius = math.pi * (i + j) / (2 * max(n, 1))

        # 克莱因全局扭转
        twist_klein = 2 * math.pi * (i * j) / (max(n, 1) ** 2)

        # 混合
        if self.mobius is not None:
            alpha = torch.sigmoid(self.mobius.mobius_weight)
            beta = torch.sigmoid(self.mobius.klein_weight)
        else:
            alpha = torch.tensor(0.5)
            beta = torch.tensor(0.5)

        total = alpha + beta + 1e-6
        phase = (alpha / total) * twist_mobius + (beta / total) * twist_klein

        return phase

    def get_complex_weight(self) -> torch.Tensor:
        """
        获取复数权重矩阵 W = A · exp(i·Θ)

        W ∈ ℂ^(N×N), 由振幅和相位生成
        """
        n = self.hidden_dim
        if n == 0:
            return torch.zeros(0, 0, dtype=torch.complex64, device=self.b_real.device)

        amp = self.W_amplitude[:n, :n]
        phase = self.compute_twist_phase(n)

        # 稀疏掩码
        mask = torch.sigmoid(self.sparse_mask[:n, :n])

        # 复数权重 = 振幅 × exp(i·相位) × 稀疏掩码
        W_complex = amp * mask * torch.exp(1j * phase)

        return W_complex

    def compute_amplitude_regularization(
        self, l1_weight: float = 0.01, l2_weight: float = 0.001
    ) -> torch.Tensor:
        """
        振幅正则化 - 鼓励稀疏化

        L1: 鼓励振幅趋零 (稀疏性)
        L2: 防止振幅过大 (稳定性)

        返回: regularization loss
        """
        if self.hidden_dim == 0:
            return torch.tensor(0.0, device=self.b_real.device)

        amp = self.W_amplitude[: self.hidden_dim, : self.hidden_dim]

        # L1 正则化 (稀疏性)
        l1_loss = amp.abs().mean()

        # L2 正则化 (稳定性)
        l2_loss = amp.pow(2).mean()

        return l1_weight * l1_loss + l2_weight * l2_loss

    def _init_mobius_resonance(
        self,
        enable_mobius: bool = True,
        enable_resonance: bool = True,
        mobius_strength: float = 0.1,
        resonance_strength: float = 0.1,
        learn_manifold_dim: bool = True,
        sparse_resonance: bool = True,
    ):
        """初始化莫比乌斯约束和共振注意力"""
        if enable_mobius:
            from .mobius import MobiusConstraint

            max_h = self.growth_config.max_hidden_dim
            self.mobius = MobiusConstraint(
                max_dim=max(max_h * 4, 512),
                constraint_strength=mobius_strength,
                enable_learning=learn_manifold_dim,
                device=str(self.U.weight.device),
            )

        if enable_resonance:
            from .resonance import TwistorResonance

            self.resonance = TwistorResonance(
                hidden_dim=max(1, self.hidden_dim),
                resonance_strength=resonance_strength,
                sparse_mode=sparse_resonance,
                device=str(self.U.weight.device),
            )

    def _init_manifold_geometry(self):
        """初始化流形几何约束系统"""
        from .manifold_geometry import (
            ManifoldGeometry,
            ManifoldWeightInitializer,
            GeodesicGrowthPlanner,
        )

        max_h = self.growth_config.max_hidden_dim
        twist_rate = math.pi
        klein_mix = 0.0

        if self.mobius is not None:
            twist_rate = (
                self.mobius.twist_rate.item()
                if hasattr(self.mobius, "twist_rate")
                else math.pi
            )
            klein_mix = torch.sigmoid(self.mobius.klein_weight).item()

        self.manifold_geometry = ManifoldGeometry(
            max_dim=max_h,
            manifold_radius=1.0,
            twist_rate=twist_rate,
            klein_mix=klein_mix,
        )

        twist_tensor = None
        if self.mobius is not None:
            try:
                twist_tensor = self.mobius.compute_twist_tensor(
                    min(max_h, 64),
                    self.mobius.compute_manifold_dimension(min(max_h, 64)),
                )
            except Exception:
                pass

        self.weight_initializer = ManifoldWeightInitializer(
            self.manifold_geometry, twist_tensor
        )

        self.growth_planner = GeodesicGrowthPlanner(self.manifold_geometry)

    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        if self.hidden_dim == 0:
            return torch.ones_like(z.real) * self.tau_min

        z_mod = torch.abs(z)[:, : self.hidden_dim]

        tau = F.sigmoid(
            F.linear(
                z_mod,
                self.W_tau.weight[: self.hidden_dim, : self.hidden_dim],
                self.W_tau.bias[: self.hidden_dim],
            )
        )

        if self.multi_scale_tau and self.tau_bias is not None:
            tau = tau + self.tau_bias[: self.hidden_dim].unsqueeze(0)

        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau + 1e-6

    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.hidden_dim == 0:
            return torch.zeros_like(z)

        z_real = z.real
        z_imag = z.imag

        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        # 复数权重 W = A · exp(i·Θ)
        W_complex = self.get_complex_weight()

        # 复数矩阵乘法: W @ tanh(z)
        # (Wr + iWi) @ (tr + i·ti) = (Wr@tr - Wi@ti) + i(Wr@ti + Wi@tr)
        W_real = W_complex.real
        W_imag = W_complex.imag

        W_tanh_real = F.linear(tanh_real, W_real) - F.linear(tanh_imag, W_imag)
        W_tanh_imag = F.linear(tanh_real, W_imag) + F.linear(tanh_imag, W_real)

        Ux = self.U(x)[:, : self.hidden_dim]

        dz_real = -z_real + W_tanh_real + Ux + self.b_real[: self.hidden_dim]
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag[: self.hidden_dim]

        tau = self.compute_tau(z)
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        if self.resonance is not None and self.hidden_dim > 0:
            topo_weights = None
            if self.mobius is not None:
                topo_weights = self.mobius.topology_weight_matrix(self.hidden_dim)
            dzdt_resonance = self.resonance(
                z, topology_weights=topo_weights, mode=self._resonance_mode
            )
            dzdt = dzdt + dzdt_resonance

        dzdt_real = torch.clamp(dzdt.real, -self.dzdt_max, self.dzdt_max)
        dzdt_imag = torch.clamp(dzdt.imag, -self.dzdt_max, self.dzdt_max)
        dzdt = torch.complex(dzdt_real, dzdt_imag)

        return dzdt

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        T, B, _ = x.shape

        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []

        for t in range(T):
            x_t = x[t]
            dzdt = self.compute_dzdt(z, x_t)

            z = z + self.dt * dzdt

            if self.mobius is not None and self.hidden_dim > 0:
                z = self.mobius.project_state(z)

            z = torch.complex(
                torch.clamp(z.real, -self.z_max, self.z_max),
                torch.clamp(z.imag, -self.z_max, self.z_max),
            )

            if self.training and self.enable_growth and self.hidden_dim > 0:
                if len(self._activation_buffer) >= self._max_buffer_size:
                    self._activation_buffer.pop(0)
                self._activation_buffer.append(torch.abs(z).mean(dim=0).detach().cpu())

            y_t = (
                F.linear(z.real, self.out.weight[:, : self.hidden_dim], self.out.bias)
                if self.hidden_dim > 0
                else torch.zeros(B, self.output_dim, device=x.device)
            )
            outputs.append(y_t)

            if return_states:
                states.append(z)

        y = torch.stack(outputs, dim=0)

        if return_states:
            return y, torch.stack(states, dim=0)
        return y

    def _update_neuron_stats(self):
        if self.hidden_dim == 0 or len(self._activation_buffer) < 10:
            return

        valid_buffer = [
            b for b in self._activation_buffer if b.shape[0] == self.hidden_dim
        ]
        if len(valid_buffer) < 5:
            return

        buffer = torch.stack(valid_buffer[-50:])

        input_offset = self.input_dim
        output_offset = self.input_dim + self.output_dim

        for i in range(self.hidden_dim):
            state_idx = output_offset + i
            if state_idx >= len(self.neuron_states):
                continue
            acts = buffer[:, i]
            self.neuron_states[state_idx].activation_mean = acts.mean().item()
            self.neuron_states[state_idx].activation_variance = acts.var().item()
            self.neuron_states[state_idx].usage_count += len(acts)
            current_peak = acts.max().item()
            if current_peak > self.neuron_states[state_idx].peak_activation:
                self.neuron_states[state_idx].peak_activation = current_peak
            self.neuron_states[state_idx].recent_activity = (
                acts[-1].item() if len(acts) > 0 else 0.0
            )

    def _update_neuron_decay_consolidation(self):
        if self.hidden_dim == 0:
            return

        input_offset = self.input_dim
        output_offset = self.input_dim + self.output_dim
        phase = self._get_current_developmental_phase()

        for i in range(self.hidden_dim):
            state_idx = output_offset + i
            if state_idx >= len(self.neuron_states):
                continue
            state = self.neuron_states[state_idx]
            if not state.active:
                continue

            state.life_span += 1

            activity_signal = state.activation_mean * 0.6 + state.recent_activity * 0.4
            if activity_signal > 0.05:
                state.consolidation_score += (
                    self.growth_config.consolidation_rate
                    * activity_signal
                    * phase.plasticity
                )
                state.decay_counter = max(
                    0, state.decay_counter - phase.growth_rate * 0.5
                )
            else:
                state.decay_counter += phase.prune_rate * 0.1
                if phase.prune_rate > 0.1:
                    state.consolidation_score *= 0.95

            state.consolidation_score = min(1.0, max(0.0, state.consolidation_score))
            state.decay_counter = max(0.0, state.decay_counter)

    def _get_current_developmental_phase(self) -> DevelopmentalPhase:
        if (
            not self.growth_config.enable_developmental_schedule
            or not self.growth_config.developmental_phases
        ):
            return DevelopmentalPhase(
                name="default",
                start_step=0,
                end_step=99999,
                target_neurons=32,
                target_connections_per_neuron=3,
                growth_rate=0.1,
                prune_rate=0.05,
                plasticity=0.5,
                description="默认阶段",
            )

        for phase in self.growth_config.developmental_phases:
            if phase.start_step <= self.training_step < phase.end_step:
                return phase

        return self.growth_config.developmental_phases[-1]

    def _get_phase_growth_multiplier(self) -> float:
        phase = self._get_current_developmental_phase()
        return phase.growth_rate

    def compute_importance_scores(self) -> torch.Tensor:
        scores = torch.zeros(self.hidden_dim)

        input_offset = self.input_dim
        output_offset = self.input_dim + self.output_dim

        for i in range(self.hidden_dim):
            state_idx = output_offset + i
            if state_idx >= len(self.neuron_states):
                continue
            state = self.neuron_states[state_idx]
            if not state.active:
                scores[i] = 0.0
                continue

            act_score = torch.sigmoid(torch.tensor(state.activation_mean))
            var_score = torch.sigmoid(torch.tensor(state.activation_variance))
            usage_score = torch.sigmoid(
                torch.tensor(state.usage_count / max(1, self.training_step))
            )

            importance = 0.5 * act_score + 0.3 * var_score + 0.2 * usage_score
            scores[i] = importance.item()
            state.importance_score = importance.item()

        return scores

    def get_overloaded_neurons(self) -> List[int]:
        overloaded = []

        input_offset = self.input_dim
        output_offset = self.input_dim + self.output_dim
        phase = self._get_current_developmental_phase()
        variance_threshold = 0.15 if phase.plasticity > 0.7 else 0.3

        for i in range(self.hidden_dim):
            state_idx = output_offset + i
            if state_idx >= len(self.neuron_states):
                continue
            state = self.neuron_states[state_idx]
            if not state.active:
                continue

            if state.activation_variance > variance_threshold:
                overloaded.append(i)

        return overloaded

    def _get_next_innovation(self) -> int:
        """获取下一个创新编号"""
        self._next_innovation += 1
        return self._next_innovation - 1

    def _expand_parameters(self, new_dim: int):
        """O(1)操作 - 参数已预分配,只需更新hidden_dim计数"""
        pass

    def add_connection(
        self, in_node: int, out_node: int, manifold_weight: Optional[float] = None
    ) -> bool:
        """添加新连接 - NEAT Add Connection Mutation (流形约束版本)"""
        if in_node < 0 or out_node < 0:
            return False
        if in_node >= self.input_dim + self.output_dim + self.hidden_dim:
            return False
        if out_node >= self.input_dim + self.output_dim + self.hidden_dim:
            return False

        hidden_in_offset = self.input_dim + self.output_dim
        hidden_out_offset = self.input_dim + self.output_dim

        for gene in self.connection_genes:
            if gene.in_node == in_node and gene.out_node == out_node:
                return False

        if in_node >= hidden_in_offset and out_node >= hidden_in_offset:
            in_idx = in_node - hidden_in_offset
            out_idx = out_node - hidden_out_offset
            if (
                in_idx >= 0
                and in_idx < self.hidden_dim
                and out_idx >= 0
                and out_idx < self.hidden_dim
            ):
                pass
        elif in_node < self.input_dim and out_node >= hidden_in_offset:
            pass
        elif (
            in_node >= hidden_in_offset and out_node >= self.input_dim + self.output_dim
        ):
            pass
        else:
            return False

        if manifold_weight is not None:
            weight = manifold_weight
        else:
            weight = torch.randn(1).item() * 0.5

        gene = ConnectionGene(
            in_node=in_node,
            out_node=out_node,
            weight=weight,
            enabled=True,
            innovation=self._get_next_innovation(),
        )
        self.connection_genes.append(gene)

        self._apply_connection_gene(gene)

        return True

    def _apply_connection_gene(self, gene: ConnectionGene):
        """应用连接基因到振幅矩阵"""
        hidden_in_offset = self.input_dim + self.output_dim

        if gene.in_node >= hidden_in_offset and gene.out_node >= hidden_in_offset:
            in_idx = gene.in_node - hidden_in_offset
            out_idx = gene.out_node - hidden_in_offset

            if (
                in_idx >= 0
                and in_idx < self.hidden_dim
                and out_idx >= 0
                and out_idx < self.hidden_dim
            ):
                self.W_amplitude.data[out_idx, in_idx] = abs(gene.weight)
                self.sparse_mask.data[out_idx, in_idx] = 2.0

        elif gene.in_node < self.input_dim and gene.out_node >= hidden_in_offset:
            out_idx = gene.out_node - hidden_in_offset
            in_idx = gene.in_node

            if (
                out_idx >= 0
                and out_idx < self.hidden_dim
                and in_idx >= 0
                and in_idx < self.input_dim
            ):
                self.U.weight.data[out_idx, in_idx] = gene.weight

    def split_neuron(self, parent_idx: int) -> int:
        """分裂神经元 - NEAT Add Node Mutation

        NEAT机制:
        1. 找到父神经元的输入连接
        2. 禁用原连接
        3. 在原连接位置插入新节点
        4. in->new 权重=1.0, new->out 权重=原权重

        扭量扩展:
        - 两个子神经元分别继承父神经元的实部和虚部特性
        - 保持复数状态表示
        """
        if self.hidden_dim >= self.growth_config.max_hidden_dim:
            return -1

        if parent_idx < 0 or parent_idx >= self.hidden_dim:
            return -1

        new_idx = self.hidden_dim
        old_dim = self.hidden_dim
        new_dim = old_dim + 1

        input_offset = self.input_dim + self.output_dim
        parent_node = input_offset + parent_idx

        parent_connections = []
        for i, gene in enumerate(self.connection_genes):
            if gene.out_node == parent_node and gene.enabled:
                parent_connections.append((i, gene))

        if len(parent_connections) == 0:
            for i, gene in enumerate(self.connection_genes):
                if gene.in_node == parent_node and gene.enabled:
                    parent_connections.append((i, gene))

        if len(parent_connections) == 0:
            if self.hidden_dim == 1:
                gene = ConnectionGene(
                    in_node=torch.randint(self.input_dim, (1,)).item(),
                    out_node=parent_node,
                    weight=torch.randn(1).item() * 0.5,
                    enabled=True,
                    innovation=self._get_next_innovation(),
                )
                self.connection_genes.append(gene)
                parent_connections.append((len(self.connection_genes) - 1, gene))
            else:
                return -1

        gene_idx, parent_gene = parent_connections[0]

        parent_gene.enabled = False

        new_node = input_offset + new_idx

        in_to_new = ConnectionGene(
            in_node=parent_gene.in_node,
            out_node=new_node,
            weight=1.0,
            enabled=True,
            innovation=self._get_next_innovation(),
        )

        new_to_out = ConnectionGene(
            in_node=new_node,
            out_node=parent_node,
            weight=parent_gene.weight,
            enabled=True,
            innovation=self._get_next_innovation(),
        )

        self.connection_genes.append(in_to_new)
        self.connection_genes.append(new_to_out)

        with torch.no_grad():
            self._expand_parameters(new_dim)

            parent_state = self._get_neuron_manifold_state(parent_idx)

            in_idx = parent_gene.in_node - input_offset
            if in_idx >= 0 and in_idx < old_dim:
                if self.growth_planner is not None:
                    new_state = self.growth_planner.plan_new_neuron(
                        self._get_all_neuron_states(), parent_idx
                    )
                    manifold_weight = new_state.norm().item()
                    init_scale = max(0.1, min(manifold_weight, 1.0))
                else:
                    init_scale = abs(parent_gene.weight)

                self.W_amplitude.data[new_idx, in_idx] = init_scale
                self.sparse_mask.data[new_idx, in_idx] = 2.0

                out_idx = parent_idx
                self.W_amplitude.data[out_idx, new_idx] = abs(parent_gene.weight)
                self.sparse_mask.data[out_idx, new_idx] = 2.0

                # 流形坐标: 新神经元在父神经元附近
                self.manifold_theta.data[new_idx] = (
                    self.manifold_theta.data[parent_idx] + torch.randn(3) * 0.1
                )
            else:
                in_input_idx = parent_gene.in_node
                if in_input_idx < self.input_dim:
                    if self.weight_initializer is not None and parent_state is not None:
                        weight, _ = self.weight_initializer.init_connection_weight(
                            parent_state, parent_state, device=str(parent_state.device)
                        )
                        init_val = weight.norm().item() * 0.5
                    else:
                        init_val = abs(parent_gene.weight)
                    self.U.weight.data[new_idx, in_input_idx] = init_val
                    self.out.weight.data[:, new_idx] = torch.tensor(
                        [parent_gene.weight] * self.output_dim
                    )

            if parent_state is not None:
                self.b_real.data[new_idx] = self.b_real.data[parent_idx] + (
                    torch.randn(1).item()
                    * 0.05
                    * self.manifold_geometry.manifold_radius.abs().item()
                )
                self.b_imag.data[new_idx] = self.b_imag.data[parent_idx] + (
                    torch.randn(1).item()
                    * 0.05
                    * self.manifold_geometry.manifold_radius.abs().item()
                )
            else:
                self.b_real.data[new_idx] = (
                    self.b_real.data[parent_idx] + torch.randn(1).item() * 0.1
                )
                self.b_imag.data[new_idx] = (
                    self.b_imag.data[parent_idx] + torch.randn(1).item() * 0.1
                )

            if self.tau_bias is not None:
                self.tau_bias.data[new_idx] = self.tau_bias.data[parent_idx]

        self.neuron_states.append(
            NeuronState(
                index=new_node,
                neuron_type="hidden",
                active=True,
                birth_step=self.training_step,
            )
        )

        self.active_neurons.append(new_node)
        self.hidden_dim = new_dim

        return new_idx

    def add_random_connection(self) -> bool:
        """随机添加连接"""
        if self.hidden_dim == 0:
            return False

        input_offset = self.input_dim + self.output_dim
        output_offset = self.input_dim + self.output_dim

        candidates = []

        for in_node in range(self.input_dim):
            for out_node in range(output_offset, output_offset + self.hidden_dim):
                exists = any(
                    g.in_node == in_node and g.out_node == out_node and g.enabled
                    for g in self.connection_genes
                )
                if not exists:
                    candidates.append((in_node, out_node))

        for in_node in range(input_offset, input_offset + self.hidden_dim):
            for out_node in range(input_offset, input_offset + self.hidden_dim):
                if in_node != out_node:
                    exists = any(
                        g.in_node == in_node and g.out_node == out_node and g.enabled
                        for g in self.connection_genes
                    )
                    if not exists:
                        candidates.append((in_node, out_node))

        for in_node in range(input_offset, input_offset + self.hidden_dim):
            for out_node in range(self.input_dim, self.input_dim + self.output_dim):
                exists = any(
                    g.in_node == in_node and g.out_node == out_node and g.enabled
                    for g in self.connection_genes
                )
                if not exists:
                    candidates.append((in_node, out_node))

        if not candidates:
            return False

        in_node, out_node = candidates[torch.randint(len(candidates), (1,)).item()]
        return self.add_connection(in_node, out_node)

    def add_batch_connections(self, n: int) -> int:
        """批量添加连接 - 用于连接爆炸期

        流形约束版本: 权重沿测地线方向初始化
        """
        if self.hidden_dim == 0:
            return 0

        input_offset = self.input_dim + self.output_dim
        added = 0

        all_states = self._get_all_neuron_states()

        for _ in range(n * 3):
            if added >= n:
                break

            conn_type = torch.randint(3, (1,)).item()

            if conn_type == 0:
                in_node = torch.randint(self.input_dim, (1,)).item()
                out_node = input_offset + torch.randint(self.hidden_dim, (1,)).item()
            elif conn_type == 1:
                in_node = input_offset + torch.randint(self.hidden_dim, (1,)).item()
                out_node = input_offset + torch.randint(self.hidden_dim, (1,)).item()
                if in_node == out_node:
                    continue
            else:
                in_node = input_offset + torch.randint(self.hidden_dim, (1,)).item()
                out_node = self.input_dim + torch.randint(self.output_dim, (1,)).item()

            exists = any(
                g.in_node == in_node and g.out_node == out_node and g.enabled
                for g in self.connection_genes
            )
            if not exists:
                weight = None
                if self.weight_initializer is not None and all_states is not None:
                    in_idx = in_node - input_offset if in_node >= input_offset else 0
                    out_idx = out_node - input_offset if out_node >= input_offset else 0
                    if 0 <= in_idx < len(all_states) and 0 <= out_idx < len(all_states):
                        w, _ = self.weight_initializer.init_connection_weight(
                            all_states[in_idx],
                            all_states[out_idx],
                            device=str(all_states[in_idx].device),
                        )
                        weight = w.norm().item() * 0.5

                if self.add_connection(in_node, out_node, manifold_weight=weight):
                    added += 1

        return added

    def _get_neuron_manifold_state(self, neuron_idx: int) -> Optional[torch.Tensor]:
        """获取神经元在流形上的状态表示"""
        if self.hidden_dim == 0 or neuron_idx >= self.hidden_dim:
            return None

        r = self.manifold_geometry.manifold_radius.abs().item()
        theta = 2 * math.pi * neuron_idx / max(1, self.hidden_dim)
        phi = self.b_real.data[neuron_idx].item() * 0.1

        state = torch.tensor(
            [
                r * math.cos(theta),
                r * math.sin(theta),
                phi,
            ],
            dtype=torch.float32,
        )

        return state

    def _get_all_neuron_states(self) -> Optional[torch.Tensor]:
        """获取所有神经元的流形状态"""
        if self.hidden_dim == 0:
            return None

        states = []
        for i in range(self.hidden_dim):
            s = self._get_neuron_manifold_state(i)
            if s is not None:
                states.append(s)

        if not states:
            return None

        return torch.stack(states, dim=0)

    def disable_random_connection(self) -> bool:
        """随机禁用连接 - NEAT Disable Connection Mutation"""
        enabled_connections = [
            i for i, g in enumerate(self.connection_genes) if g.enabled
        ]
        if not enabled_connections:
            return False

        idx = torch.randint(len(enabled_connections), (1,)).item()
        gene_idx = enabled_connections[idx]
        self.connection_genes[gene_idx].enabled = False

        gene = self.connection_genes[gene_idx]

        hidden_in_offset = self.input_dim + self.output_dim
        hidden_out_offset = self.input_dim + self.output_dim

        if gene.in_node >= hidden_in_offset and gene.out_node >= hidden_in_offset:
            in_idx = gene.in_node - hidden_in_offset
            out_idx = gene.out_node - hidden_in_offset

            if (
                in_idx >= 0
                and in_idx < self.hidden_dim
                and out_idx >= 0
                and out_idx < self.hidden_dim
            ):
                self.sparse_mask.data[out_idx, in_idx] = -10

        return True

    def prune_neurons(self) -> int:
        """剪枝不重要的神经元 - 基于衰减和巩固分数"""
        if self.hidden_dim == 0:
            return 0

        phase = self._get_current_developmental_phase()
        if phase.prune_rate < 0.05:
            return 0

        input_offset = self.input_dim + self.output_dim
        pruned = 0

        candidates = []
        for i in range(self.hidden_dim):
            state_idx = input_offset + i
            if state_idx >= len(self.neuron_states):
                continue
            state = self.neuron_states[state_idx]
            if not state.active:
                continue

            survival_score = state.consolidation_score * 0.5 - state.decay_counter * 0.5
            state.survival_threshold = survival_score
            candidates.append((i, survival_score))

        candidates.sort(key=lambda x: x[1])

        n_prune = max(1, int(self.hidden_dim * phase.prune_rate))
        n_prune = min(n_prune, self.hidden_dim - self.growth_config.min_hidden_dim)

        if n_prune <= 0:
            return 0

        n_prune = min(n_prune, len(candidates))

        for i in range(n_prune):
            idx, score = candidates[i]
            if score < self.growth_config.survival_threshold:
                state_idx = input_offset + idx
                self.neuron_states[state_idx].active = False
                pruned += 1

        return pruned

    def prune_connections(self) -> int:
        """剪枝不重要的连接"""
        if self.hidden_dim == 0:
            return 0

        with torch.no_grad():
            mask = torch.sigmoid(self.sparse_mask[: self.hidden_dim, : self.hidden_dim])

            pruned = (mask < self.growth_config.connection_threshold).sum().item()

            self.sparse_mask.data[: self.hidden_dim, : self.hidden_dim][
                mask < self.growth_config.connection_threshold
            ] = -10

        return int(pruned)

    def compute_topology_penalty(self) -> torch.Tensor:
        """计算拓扑惩罚"""
        penalty = torch.tensor(
            0.0, device=self.b_real.device if hasattr(self, "b_real") else "cpu"
        )

        if hasattr(self, "sparse_mask") and self.hidden_dim > 0:
            mask_sum = self.sparse_mask[: self.hidden_dim, : self.hidden_dim].sum()
            target = self.hidden_dim * self.hidden_dim * 0.5
            penalty = (
                penalty + abs(mask_sum - target) * self.growth_config.topology_penalty
            )

        return penalty

    def add_first_neuron(self) -> int:
        """添加第一个神经元 - 从最小结构开始的关键步骤"""
        if self.hidden_dim != 0:
            return -1

        with torch.no_grad():
            self.W_amplitude.data[0, 0] = 0.0
            self.sparse_mask.data[0, 0] = 2.0
            self.manifold_theta.data[0, 0] = 0.0
            self.manifold_theta.data[0, 1] = 0.0
            self.manifold_theta.data[0, 2] = 0.0
            self.W_tau.weight.data[0, 0] = 0.1
            self.b_real.data[0] = 0.0
            self.b_imag.data[0] = 0.0
            if self.tau_bias is not None:
                self.tau_bias.data[0] = 0.0

        input_offset = self.input_dim + self.output_dim

        U_weight = torch.randn(1, self.input_dim) * 0.5
        self.U.weight.data[0, :] = U_weight
        self.U.bias.data[0] = 0.0

        out_weight = torch.randn(self.output_dim, 1) * 0.5
        self.out.weight.data[:, 0] = out_weight.squeeze()
        self.out.bias.data[:] = 0.0

        for in_node in range(self.input_dim):
            for out_node in range(self.input_dim, self.input_dim + self.output_dim):
                gene = ConnectionGene(
                    in_node=in_node,
                    out_node=out_node,
                    weight=out_weight[out_node - self.input_dim, 0].item(),
                    enabled=True,
                    innovation=self._get_next_innovation(),
                )
                self.connection_genes.append(gene)

                out_idx = out_node - self.input_dim
                self.out.weight.data[out_idx, 0] = gene.weight

        for in_node in range(self.input_dim):
            gene = ConnectionGene(
                in_node=in_node,
                out_node=input_offset,
                weight=U_weight[0, in_node].item(),
                enabled=True,
                innovation=self._get_next_innovation(),
            )
            self.connection_genes.append(gene)

        self.neuron_states.append(
            NeuronState(
                index=input_offset,
                neuron_type="hidden",
                active=True,
                birth_step=self.training_step,
            )
        )

        self.active_neurons.append(input_offset)
        self.hidden_dim = 1

        return 0

    def growth_step(self):
        """执行一步增长/剪枝 - 大脑发育时间表驱动"""
        if not self.enable_growth:
            return {"action": "disabled", "changes": 0}

        self.training_step += 1
        self._update_neuron_stats()
        self._update_neuron_decay_consolidation()

        phase = self._get_current_developmental_phase()
        effective_growth_interval = max(
            1, int(self.growth_config.growth_interval / max(0.05, phase.growth_rate))
        )
        effective_prune_interval = max(
            1, int(self.growth_config.prune_interval / max(0.05, phase.prune_rate))
        )

        if self.training_step % effective_growth_interval == 0:
            action_taken = None
            changes = 0

            init_threshold = 10 if self.growth_config.aggressive_growth else 50
            if self.hidden_dim == 0 and self.training_step > init_threshold:
                new_idx = self.add_first_neuron()
                if new_idx >= 0:
                    return {
                        "action": "init",
                        "phase": phase.name,
                        "count": 1,
                        "new_dim": self.hidden_dim,
                    }

            if self.hidden_dim > 0:
                input_offset = self.input_dim + self.output_dim
                active_hidden_count = sum(
                    1
                    for i in range(self.hidden_dim)
                    if input_offset + i < len(self.neuron_states)
                    and self.neuron_states[input_offset + i].active
                )

                neuron_gap = phase.target_neurons - active_hidden_count
                burst_count = max(
                    1,
                    min(
                        int(neuron_gap * 0.3),
                        max(active_hidden_count * 2, 10),
                        self.growth_config.max_hidden_dim - self.hidden_dim,
                    ),
                )

                if neuron_gap > 0 and phase.growth_rate > 0.1:
                    for _ in range(burst_count):
                        if self.hidden_dim >= self.growth_config.max_hidden_dim:
                            break
                        overloaded = self.get_overloaded_neurons()
                        if overloaded:
                            parent = overloaded[
                                torch.randint(len(overloaded), (1,)).item()
                            ]
                            new_idx = self.split_neuron(parent)
                            if new_idx >= 0:
                                action_taken = "split_burst"
                                changes += 1
                        else:
                            active_hidden = [
                                i
                                for i in range(self.hidden_dim)
                                if input_offset + i < len(self.neuron_states)
                                and self.neuron_states[input_offset + i].active
                            ]
                            if active_hidden:
                                parent = active_hidden[
                                    torch.randint(len(active_hidden), (1,)).item()
                                ]
                                new_idx = self.split_neuron(parent)
                                if new_idx >= 0:
                                    action_taken = "split_dev"
                                    changes += 1

                conn_per_neuron = self._get_avg_connections_per_neuron()
                conn_gap = phase.target_connections_per_neuron - conn_per_neuron
                if conn_gap > 0 and phase.growth_rate > 0:
                    max_conn = self.hidden_dim * min(
                        phase.target_connections_per_neuron, 20
                    )
                    current_conn = sum(1 for g in self.connection_genes if g.enabled)
                    if current_conn < max_conn:
                        conn_burst = max(
                            1, int(conn_gap * phase.growth_rate * self.hidden_dim * 0.3)
                        )
                        conn_burst = min(conn_burst, 200)
                        added = self.add_batch_connections(conn_burst)
                        if added > 0:
                            if action_taken is None:
                                action_taken = "add_connection"
                            changes += added

            if action_taken:
                if self.mobius is not None:
                    self.mobius.on_dimension_change(self.hidden_dim)
                return {
                    "action": action_taken,
                    "phase": phase.name,
                    "count": changes,
                    "new_dim": self.hidden_dim,
                }

        if self.training_step % effective_prune_interval == 0:
            n_pruned = self.prune_neurons()
            n_connections = (
                self.prune_connections()
                if self.growth_config.connection_threshold > 0
                else 0
            )

            if n_pruned > 0 or n_connections > 0:
                return {
                    "action": "prune",
                    "phase": phase.name,
                    "neurons": n_pruned,
                    "connections": n_connections,
                }

        return {"action": "none", "phase": phase.name, "changes": 0}

    def _get_avg_connections_per_neuron(self) -> float:
        if self.hidden_dim == 0:
            return 0.0
        enabled = sum(1 for g in self.connection_genes if g.enabled)
        return enabled / max(1, self.hidden_dim)

    def force_grow_to(self, target_dim: int):
        """
        强制增长到目标维度 (用于预生长)
        绕过概率和过载检查，直接分裂神经元
        """
        while (
            self.hidden_dim < target_dim
            and self.hidden_dim < self.growth_config.max_hidden_dim
        ):
            if self.hidden_dim == 0:
                idx = self.add_first_neuron()
                if idx < 0:
                    break
            else:
                overloaded = self.get_overloaded_neurons()
                if overloaded:
                    parent_idx = overloaded[torch.randint(len(overloaded), (1,)).item()]
                else:
                    active_hidden = [
                        s.index - self.input_dim - self.output_dim
                        for s in self.neuron_states
                        if s.active and s.neuron_type == "hidden"
                    ]
                    if not active_hidden:
                        break
                    parent_idx = active_hidden[
                        torch.randint(len(active_hidden), (1,)).item()
                    ]

                new_idx = self.split_neuron(parent_idx)
                if new_idx < 0:
                    break

            if self.mobius is not None:
                self.mobius.on_dimension_change(self.hidden_dim)

    def get_diagnostics(self) -> Dict:
        importance = self.compute_importance_scores()

        active_importance = (
            importance[importance > 0].tolist() if self.hidden_dim > 0 else []
        )

        manifold_info = {}
        if self.manifold_geometry is not None:
            manifold_info = {
                "manifold_radius": self.manifold_geometry.manifold_radius.item(),
                "twist_rate": self.manifold_geometry.twist_rate.item(),
                "klein_mix": self.manifold_geometry.klein_mix.item(),
            }

        if self.mobius is not None:
            mobius_info = self.mobius.get_manifold_info(self.hidden_dim)
            manifold_info.update(mobius_info)

        return {
            "hidden_dim": self.hidden_dim,
            "active_count": len([s for s in self.neuron_states if s.active]),
            "training_step": self.training_step,
            "connection_count": len([g for g in self.connection_genes if g.enabled]),
            "importance_mean": np.mean(active_importance) if active_importance else 0.0,
            "enable_growth": self.enable_growth,
            **manifold_info,
        }

    def create_riemannian_optimizer(
        self, optimizer_class=torch.optim.Adam, **optimizer_kwargs
    ):
        """创建黎曼优化器 - 梯度自动投影到流形切空间"""
        from .manifold_geometry import RiemannianOptimizer

        optimizer = optimizer_class(self.parameters(), **optimizer_kwargs)
        return RiemannianOptimizer(optimizer, self.manifold_geometry)

    def reset_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(
            batch_size, max(1, self.hidden_dim), dtype=torch.complex64, device=device
        )

    def step(
        self, z: torch.Tensor, x: torch.Tensor, dt: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if dt is None:
            dt = self.dt

        if self.hidden_dim == 0:
            output = self.out.weight.sum() * torch.ones(
                x.shape[0], self.output_dim, device=x.device
            )
            z_empty = torch.zeros(x.shape[0], 1, dtype=torch.complex64, device=x.device)
            return z_empty, output

        dzdt = self.compute_dzdt(z, x)
        z_new = z + dt * dzdt

        if self.mobius is not None:
            z_new = self.mobius.project_state(z_new)

        z_new = torch.complex(
            torch.clamp(z_new.real, -self.z_max, self.z_max),
            torch.clamp(z_new.imag, -self.z_max, self.z_max),
        )

        output = (
            self.out(z_new.real[:, : self.output_dim])
            if self.output_dim <= z_new.shape[1]
            else self.out(z_new.real)
        )
        return z_new, output

    def get_mobius_info(self) -> Optional[Dict]:
        """获取莫比乌斯流形当前状态"""
        if self.mobius is None:
            return None
        return self.mobius.get_manifold_info(self.hidden_dim)


def create_growable_twistor_LMT(
    input_dim: int, hidden_dim: int = 0, output_dim: int = 1, **kwargs
) -> GrowableTwistorLMT:
    return GrowableTwistorLMT(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs
    )
