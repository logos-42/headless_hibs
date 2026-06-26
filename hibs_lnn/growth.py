"""
Growable Twistor-LMT: 可增长扭量神经网络
========================================
核心机制：
1. 神经元负载监控
2. 神经元分裂（基于负载触发）
3. 神经元剪枝
4. 集合拓扑约束

设计原则：
- 扭量复数空间作为状态表示
- 动态结构而非静态架构
- 生物启发的生长/分化/修剪机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np
from copy import deepcopy


@dataclass
class NeuronState:
    """单个神经元状态"""

    index: int
    active: bool = True
    birth_epoch: int = 0

    gradient_variance: float = 0.0
    activation_variance: float = 0.0
    activation_mean: float = 0.0
    loss_sensitivity: float = 0.0

    importance_score: float = 0.0
    usage_count: int = 0

    split_count: int = 0
    parent_index: Optional[int] = None


@dataclass
class GrowthConfig:
    """增长配置"""

    min_hidden_dim: int = 8
    max_hidden_dim: int = 512

    split_threshold_gradient_var: float = 0.5
    split_threshold_activation_var: float = 0.3
    split_threshold_sensitivity: float = 0.5

    prune_threshold_importance: float = 0.01
    prune_ratio: float = 0.1

    growth_interval: int = 100
    prune_interval: int = 50

    noise_scale: float = 0.1
    topology_penalty: float = 0.001

    max_growth_per_step: int = 4
    max_prune_per_step: int = 4


class NeuronMonitor:
    """
    神经元负载监控器
    ===============
    追踪每个神经元的:
    - 梯度方差
    - 激活分布
    - Loss敏感度
    - 使用频率
    """

    def __init__(self, hidden_dim: int, device: str = "cpu"):
        self.hidden_dim = hidden_dim
        self.device = device

        self.states = [NeuronState(index=i) for i in range(hidden_dim)]

        self.activation_buffer = []
        self.gradient_buffer = []

        self.epoch_count = 0

    def update_activations(self, z: torch.Tensor):
        """更新激活统计"""
        z_abs = torch.abs(z).detach()

        for i in range(min(len(self.states), z_abs.shape[1])):
            if self.states[i].active:
                act = z_abs[:, i].cpu().numpy()
                self.states[i].activation_mean = float(np.mean(act))
                self.states[i].activation_variance = float(np.var(act))
                self.states[i].usage_count += len(act)

    def update_gradients(self, model: nn.Module):
        """更新梯度统计"""
        self.epoch_count += 1

        for i, state in enumerate(self.states):
            if not state.active:
                continue

            param_name = f"_growth_W_real"
            if hasattr(model, param_name):
                param = getattr(model, param_name)
                if param.grad is not None:
                    grad = param.grad[:, i].abs().mean().item()
                    state.gradient_variance = grad

    def compute_loss_sensitivity(self, model, x: torch.Tensor, y: torch.Tensor):
        """计算Loss对每个神经元的敏感度"""
        model.eval()

        sensitivities = []

        with torch.no_grad():
            for i, state in enumerate(self.states):
                if not state.active:
                    sensitivities.append(0.0)
                    continue

                base_pred = model(x)
                base_loss = F.mse_loss(base_pred, y).item()

                perturb_factor = 0.01

                for param_name in ["W_real", "W_imag"]:
                    if hasattr(model, param_name):
                        param = getattr(model, param_name)
                        if param.weight.shape[1] > i:
                            original = param.weight[:, i].clone()
                            param.weight[:, i] += perturb_factor

                            pred_perturbed = model(x)
                            loss_perturbed = F.mse_loss(pred_perturbed, y).item()

                            param.weight[:, i] = original

                            sensitivity = (
                                abs(loss_perturbed - base_loss) / perturb_factor
                            )
                            sensitivities.append(sensitivity)
                            break
                else:
                    sensitivities.append(0.0)

        for i, sens in enumerate(sensitivities):
            if i < len(self.states):
                self.states[i].loss_sensitivity = sens

        model.train()
        return sensitivities

    def compute_importance_scores(self) -> torch.Tensor:
        """计算每个神经元的综合重要性分数"""
        scores = torch.zeros(len(self.states))

        for i, state in enumerate(self.states):
            if not state.active:
                scores[i] = 0.0
                continue

            act_score = torch.sigmoid(torch.tensor(state.activation_mean))
            grad_score = torch.sigmoid(torch.tensor(state.gradient_variance))
            sens_score = torch.sigmoid(torch.tensor(state.loss_sensitivity))
            usage_score = torch.sigmoid(
                torch.tensor(state.usage_count / max(1, self.epoch_count))
            )

            importance = (
                0.3 * act_score
                + 0.3 * grad_score
                + 0.25 * sens_score
                + 0.15 * usage_score
            )

            scores[i] = importance.item()
            state.importance_score = importance.item()

        return scores

    def get_overloaded_neurons(self, config: GrowthConfig) -> List[int]:
        """获取负载过重的神经元索引"""
        overloaded = []

        for i, state in enumerate(self.states):
            if not state.active:
                continue

            criteria_met = 0

            if state.gradient_variance > config.split_threshold_gradient_var:
                criteria_met += 1
            if state.activation_variance > config.split_threshold_activation_var:
                criteria_met += 1
            if state.loss_sensitivity > config.split_threshold_sensitivity:
                criteria_met += 1

            if criteria_met >= 2:
                overloaded.append(i)

        return overloaded

    def add_neuron(self, index: int):
        """添加新神经元状态"""
        while len(self.states) <= index:
            self.states.append(NeuronState(index=len(self.states)))

        self.states[index] = NeuronState(
            index=index, active=True, birth_epoch=self.epoch_count
        )

    def mark_inactive(self, indices: List[int]):
        """标记神经元为非活跃"""
        for i in indices:
            if i < len(self.states):
                self.states[i].active = False


class TopologyConstraint:
    """
    集合拓扑约束
    =============
    控制网络结构:
    - 节点数量限制
    - 连接稀疏度
    - 结构正则化
    """

    def __init__(self, config: GrowthConfig):
        self.config = config

        self.current_dim = config.min_hidden_dim
        self.history = []

    def compute_topology_penalty(self, model: nn.Module) -> torch.Tensor:
        """计算拓扑惩罚项"""
        penalty = 0.0

        if hasattr(model, "sparse_mask_real"):
            mask_sum = model.sparse_mask_real.sum() + model.sparse_mask_imag.sum()
            target = model.hidden_dim * model.hidden_dim * 0.5
            sparsity_penalty = (mask_sum - target).abs() * self.config.topology_penalty
            penalty += sparsity_penalty

        if hasattr(model, "hidden_dim"):
            dim_penalty = (
                abs(model.hidden_dim - self.current_dim) * self.config.topology_penalty
            )
            penalty += dim_penalty

        return torch.tensor(penalty, requires_grad=True)

    def should_grow(self, current_dim: int) -> bool:
        """判断是否可以增长"""
        return current_dim < self.config.max_hidden_dim

    def should_prune(self, current_dim: int) -> bool:
        """判断是否可以修剪"""
        return current_dim > self.config.min_hidden_dim

    def get_valid_indices(self, active_count: int, n_requested: int) -> int:
        """获取有效的增长/修剪数量"""
        if n_requested <= 0:
            return 0

        grow_available = self.config.max_hidden_dim - active_count
        prune_available = active_count - self.config.min_hidden_dim

        return min(n_requested, grow_available, prune_available)


class NeuronSplitter:
    """
    神经元分裂器
    ============
    核心机制:
    1. 复制父神经元权重
    2. 添加对称扰动: w1 = w + ε, w2 = w - ε
    3. 学习路由权重
    """

    def __init__(self, config: GrowthConfig):
        self.config = config

    def split_neuron(self, model: nn.Module, parent_idx: int, new_idx: int):
        """分裂单个神经元"""
        noise = (
            torch.randn_like(model.W_real.weight[:, parent_idx : parent_idx + 1])
            * self.config.noise_scale
        )

        if model.W_real.weight.shape[1] > parent_idx:
            model.W_real.weight[:, new_idx] = (
                model.W_real.weight[:, parent_idx] + noise.squeeze()
            )
            model.W_imag.weight[:, new_idx] = (
                model.W_imag.weight[:, parent_idx] - noise.squeeze()
            )

            if (
                model.W_real.bias is not None
                and parent_idx < model.W_real.weight.shape[0]
            ):
                model.W_real.bias[new_idx] = model.W_real.bias[parent_idx]
                model.W_imag.bias[new_idx] = model.W_imag.bias[parent_idx]

        if (
            hasattr(model, "sparse_mask_real")
            and model.sparse_mask_real.shape[1] > new_idx
        ):
            model.sparse_mask_real[:, new_idx] = model.sparse_mask_real[:, parent_idx]
            model.sparse_mask_imag[:, new_idx] = model.sparse_mask_imag[:, parent_idx]

        if hasattr(model, "tau_bias") and model.tau_bias is not None:
            if model.tau_bias.shape[0] > new_idx:
                model.tau_bias[new_idx] = model.tau_bias[parent_idx]

        return {
            "parent_idx": parent_idx,
            "new_idx": new_idx,
            "noise_added": noise.abs().mean().item(),
        }


class NeuronPruner:
    """
    神经元剪枝器
    ============
    基于重要性评分删除:
    - 不重要的连接
    - 不活跃的神经元
    """

    def __init__(self, config: GrowthConfig):
        self.config = config

    def prune_by_importance(
        self, model: nn.Module, importance_scores: torch.Tensor, active_mask: List[bool]
    ) -> Tuple[nn.Module, List[int]]:
        """根据重要性剪枝神经元"""
        n_neurons = len(active_mask)
        active_indices = [i for i, a in enumerate(active_mask) if a]

        if len(active_indices) <= self.config.min_hidden_dim:
            return model, []

        sorted_indices = sorted(
            active_indices, key=lambda i: importance_scores[i].item()
        )

        n_prune = min(
            self.config.max_prune_per_step,
            len(active_indices) - self.config.min_hidden_dim,
        )

        prune_indices = sorted_indices[:n_prune]

        for idx in prune_indices:
            if idx < model.W_real.weight.shape[1]:
                model.W_real.weight.data[:, idx] = 0
                model.W_imag.weight.data[:, idx] = 0

            if hasattr(model, "sparse_mask_real"):
                model.sparse_mask_real.data[:, idx] = 0
                model.sparse_mask_imag.data[:, idx] = 0

        return model, prune_indices

    def prune_connections(self, model: nn.Module, threshold: float = 0.01) -> int:
        """剪枝不重要的连接"""
        if not hasattr(model, "sparse_mask_real"):
            return 0

        with torch.no_grad():
            mask_real_sig = torch.sigmoid(model.sparse_mask_real)
            mask_imag_sig = torch.sigmoid(model.sparse_mask_imag)

            pruned_real = (mask_real_sig < threshold).sum().item()
            pruned_imag = (mask_imag_sig < threshold).sum().item()

            model.sparse_mask_real.data[mask_real_sig < threshold] = -10
            model.sparse_mask_imag.data[mask_imag_sig < threshold] = -10

        return pruned_real + pruned_imag


class GrowableTwistorLMT(nn.Module):
    """
    可增长扭量神经网络
    ==================
    整合所有增长机制的核心类
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        output_dim: int = 1,
        sparsity: float = 0.3,
        multi_scale_tau: bool = True,
        dt: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        growth_config: Optional[GrowthConfig] = None,
        device: str = "cpu",
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

        self.growth_config = growth_config or GrowthConfig(min_hidden_dim=hidden_dim)
        self.device = device

        self._init_weights()

        self.monitor = NeuronMonitor(hidden_dim, device)
        self.topology = TopologyConstraint(self.growth_config)
        self.splitter = NeuronSplitter(self.growth_config)
        self.pruner = NeuronPruner(self.growth_config)

        self.active_neurons = list(range(hidden_dim))
        self.training_step = 0

        self.growth_enabled = True
        self.prune_enabled = True

    def _init_weights(self):
        """初始化权重"""
        self.W_real = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_imag = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.U = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_tau = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.sparse_mask_real = nn.Parameter(
            torch.ones(self.hidden_dim, self.hidden_dim)
        )
        self.sparse_mask_imag = nn.Parameter(
            torch.ones(self.hidden_dim, self.hidden_dim)
        )

        if self.multi_scale_tau:
            self.tau_bias = nn.Parameter(torch.zeros(self.hidden_dim))
        else:
            self.tau_bias = None

        self.b_real = nn.Parameter(torch.zeros(self.hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(self.hidden_dim))

        self.out = nn.Linear(self.hidden_dim, self.output_dim)

        nn.init.orthogonal_(self.W_real.weight, gain=0.5)
        nn.init.orthogonal_(self.W_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau.weight, gain=0.1)

        if self.sparsity > 0:
            with torch.no_grad():
                mask_real = (
                    torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity
                ).float()
                mask_imag = (
                    torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity
                ).float()
                self.sparse_mask_real.copy_(mask_real)
                self.sparse_mask_imag.copy_(mask_imag)

    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """计算状态依赖时间常数"""
        z_mod = torch.abs(z)
        tau = F.sigmoid(self.W_tau(z_mod))

        if self.multi_scale_tau and self.tau_bias is not None:
            tau = tau + self.tau_bias.unsqueeze(0)

        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau + 1e-6

    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """计算时间导数"""
        z_real = z.real
        z_imag = z.imag

        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        W_real_sparse = self.W_real.weight * torch.sigmoid(self.sparse_mask_real)
        W_imag_sparse = self.W_imag.weight * torch.sigmoid(self.sparse_mask_imag)

        W_tanh_real = F.linear(tanh_real, W_real_sparse, self.W_real.bias)
        W_tanh_imag = F.linear(tanh_imag, W_imag_sparse, self.W_imag.bias)

        Ux = self.U(x)

        dz_real = -z_real + W_tanh_real + Ux + self.b_real
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag

        tau = self.compute_tau(z)
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        return dzdt

    def forward(
        self, x: torch.Tensor, return_states: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """前向传播"""
        T, B, _ = x.shape

        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []

        for t in range(T):
            x_t = x[t]
            dzdt = self.compute_dzdt(z, x_t)

            z = z + self.dt * dzdt

            z = torch.complex(
                torch.clamp(z.real, -100, 100),
                torch.clamp(z.imag, -100, 100),
            )

            if self.training and self.monitor is not None:
                self.monitor.update_activations(z)

            y_t = self.out(z.real)
            outputs.append(y_t)

            if return_states:
                states.append(z)

        y = torch.stack(outputs, dim=0)

        if return_states:
            return y, torch.stack(states, dim=0)
        return y

    def grow_if_needed(self):
        """执行神经元分裂（如果需要）"""
        if not self.growth_enabled:
            return

        self.training_step += 1

        if self.training_step % self.growth_config.growth_interval != 0:
            return

        overloaded = self.monitor.get_overloaded_neurons(self.growth_config)

        if not overloaded:
            return

        if self.hidden_dim >= self.growth_config.max_hidden_dim:
            return

        actual_growth = 0

        for parent_idx in overloaded[: self.growth_config.max_growth_per_step]:
            if self.hidden_dim >= self.growth_config.max_hidden_dim:
                break

            new_idx = self.hidden_dim

            self.splitter.split_neuron(self, parent_idx, new_idx)

            self._expand_parameters(new_idx + 1)

            self.monitor.add_neuron(new_idx)
            self.active_neurons.append(new_idx)

            actual_growth += 1
            self.hidden_dim = new_idx + 1

        if actual_growth > 0:
            print(f"[Growth] Split {actual_growth} neurons. New dim: {self.hidden_dim}")

    def prune_if_needed(self):
        """执行神经元剪枝（如果需要）"""
        if not self.prune_enabled:
            return

        if self.training_step % self.growth_config.prune_interval != 0:
            return

        importance = self.monitor.compute_importance_scores()

        active_mask = [i in self.active_neurons for i in range(len(importance))]

        self.pruner.prune_by_importance(self, importance, active_mask)

        self.pruner.prune_connections(
            self, self.growth_config.prune_threshold_importance
        )

    def _expand_parameters(self, new_dim: int):
        """扩展参数以适应新神经元"""
        with torch.no_grad():
            new_W_real = torch.zeros(new_dim, new_dim)
            new_W_imag = torch.zeros(new_dim, new_dim)
            new_mask_real = torch.ones(new_dim, new_dim) * -5
            new_mask_imag = torch.ones(new_dim, new_dim) * -5

            new_W_real[: self.hidden_dim, : self.hidden_dim] = self.W_real.weight
            new_W_imag[: self.hidden_dim, : self.hidden_dim] = self.W_imag.weight
            new_mask_real[: self.hidden_dim, : self.hidden_dim] = self.sparse_mask_real
            new_mask_imag[: self.hidden_dim, : self.hidden_dim] = self.sparse_mask_imag

            self.W_real = nn.Linear(new_dim, new_dim)
            self.W_imag = nn.Linear(new_dim, new_dim)
            self.sparse_mask_real = nn.Parameter(new_mask_real)
            self.sparse_mask_imag = nn.Parameter(new_mask_imag)

            self.W_real.weight.data[: self.hidden_dim, : self.hidden_dim] = new_W_real[
                : self.hidden_dim, : self.hidden_dim
            ]
            self.W_imag.weight.data[: self.hidden_dim, : self.hidden_dim] = new_W_imag[
                : self.hidden_dim, : self.hidden_dim
            ]

            new_b_real = torch.zeros(new_dim)
            new_b_imag = torch.zeros(new_dim)
            new_b_real[: self.hidden_dim] = self.b_real
            new_b_imag[: self.hidden_dim] = self.b_imag
            self.b_real = nn.Parameter(new_b_real)
            self.b_imag = nn.Parameter(new_b_imag)

            if self.tau_bias is not None:
                new_tau_bias = torch.zeros(new_dim)
                new_tau_bias[: self.hidden_dim] = self.tau_bias
                self.tau_bias = nn.Parameter(new_tau_bias)

            new_out = nn.Linear(new_dim, self.output_dim)
            new_out.weight.data[: self.output_dim, : self.hidden_dim] = self.out.weight
            new_out.bias.data = self.out.bias
            self.out = new_out

            self.hidden_dim = new_dim

    def step(self):
        """单步增长/剪枝检查"""
        self.grow_if_needed()
        self.prune_if_needed()

    def get_diagnostics(self) -> Dict:
        """获取诊断信息"""
        importance = self.monitor.compute_importance_scores()

        return {
            "hidden_dim": self.hidden_dim,
            "active_neurons": len(self.active_neurons),
            "training_step": self.training_step,
            "importance_mean": importance.mean().item(),
            "importance_std": importance.std().item(),
            "growth_enabled": self.growth_enabled,
            "prune_enabled": self.prune_enabled,
        }

    def reset_state(self, batch_size: int = 1) -> torch.Tensor:
        """重置隐藏状态"""
        return torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.complex64, device=self.device
        )


def create_growth_model(
    input_dim: int, hidden_dim: int = 16, output_dim: int = 1, **kwargs
) -> GrowableTwistorLMT:
    """创建可增长模型"""
    config = GrowthConfig(
        min_hidden_dim=hidden_dim,
        max_hidden_dim=kwargs.get("max_hidden_dim", 256),
    )

    return GrowableTwistorLMT(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        growth_config=config,
        **kwargs,
    )
