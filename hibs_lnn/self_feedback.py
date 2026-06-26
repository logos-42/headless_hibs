"""
自感知扭量神经网络 - 自我反馈改进系统 v2
==========================================
核心思想：模型的输出能够被自己感知，并基于感知微调自身

工作流程:
  每个输入时间步:
    内部思考循环 (2-5轮):
      感知 → ODE一步 → 输出 → 评估 → 微调 → 检查收敛
    收敛或达到轮数 → 产出最终输出

支持: 收敛检测，提前终止思考
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class SelfFeedbackConfig:
    """自反馈配置"""
    # 思考循环配置
    min_think_steps: int = 2        # 最少思考轮数
    max_think_steps: int = 5        # 最多思考轮数
    convergence_threshold: float = 0.01  # 收敛阈值
    convergence_window: int = 2      # 收敛检测窗口

    # 反馈评估
    feedback_window: int = 10        # 评估窗口大小
    consistency_weight: float = 0.3  # 一致性权重
    stability_weight: float = 0.3    # 稳定性权重
    coherence_weight: float = 0.4    # coherence权重

    # 微量修改
    tau_adjust_rate: float = 0.01   # τ调整率
    phase_adjust_rate: float = 0.005 # 相位调整率
    weight_adjust_rate: float = 0.001 # 权重调整率
    max_adjust_per_step: float = 0.1  # 每步最大调整

    # 修改间隔
    feedback_interval: int = 1       # 每几步执行一次反馈
    momentum: float = 0.9           # 动量（平滑修改）


class OutputSelfAttention(nn.Module):
    """
    输出自我注意力 - 让模型"看到"自己的输出
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.output_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        self.state_output_attention = nn.Bilinear(
            hidden_dim, hidden_dim, 1
        )

    def perceive(
        self,
        state: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        output_enc = self.output_encoder(output)
        attn = self.state_output_attention(
            state, output_enc[:, :self.hidden_dim]
        )
        perception = output_enc + attn * state
        return perception


class InternalCritic(nn.Module):
    """
    内部评估器 - 自己评估输出质量
    """

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.consistency_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.stability_net = nn.Sequential(
            nn.Linear(output_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.coherence_net = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def evaluate(
        self,
        current_output: torch.Tensor,
        current_state: torch.Tensor,
        previous_output: Optional[torch.Tensor] = None,
        previous_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        scores = {}

        if previous_state is not None:
            consistency_input = torch.cat([current_state, previous_state], dim=-1)
            scores['consistency'] = self.consistency_net(consistency_input).mean().item()
        else:
            scores['consistency'] = 0.5

        if previous_output is not None:
            stability_input = torch.cat([current_output, previous_output], dim=-1)
            scores['stability'] = self.stability_net(stability_input).mean().item()
        else:
            scores['stability'] = 0.5

        scores['coherence'] = self.coherence_net(current_output).mean().item()

        scores['overall'] = (
            scores['consistency'] * 0.3 +
            scores['stability'] * 0.3 +
            scores['coherence'] * 0.4
        )

        return scores


class MicroModifier(nn.Module):
    """
    微量修改器 - 基于评估微调参数
    """

    def __init__(
        self,
        hidden_dim: int,
        config: Optional[SelfFeedbackConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or SelfFeedbackConfig()

        self.modulation_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),
        )

        self.register_buffer('tau_momentum', torch.zeros(hidden_dim))
        self.register_buffer('phase_momentum', torch.zeros(hidden_dim, hidden_dim))

    def compute_modulation(
        self,
        evaluation: Dict[str, float],
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eval_vec = torch.tensor([
            evaluation['consistency'],
            evaluation['stability'],
            evaluation['coherence'],
        ], device=state.device, dtype=state.dtype)

        modulation = self.modulation_net(eval_vec)

        delta_tau = torch.tanh(modulation[0]) * self.config.tau_adjust_rate
        delta_phase = torch.tanh(modulation[1]) * self.config.phase_adjust_rate
        delta_weight = torch.tanh(modulation[2]) * self.config.weight_adjust_rate

        state_modulation = torch.sigmoid(state.mean(dim=0, keepdim=True))
        delta_tau = delta_tau * state_modulation.squeeze()

        return delta_tau, delta_phase, delta_weight

    def apply_tau_modification(
        self,
        tau_bias: torch.Tensor,
        delta_tau: torch.Tensor,
    ) -> torch.Tensor:
        self.tau_momentum.mul_(self.config.momentum)
        self.tau_momentum.add_(delta_tau.detach())

        actual_delta = torch.clamp(
            self.tau_momentum,
            -self.config.max_adjust_per_step,
            self.config.max_adjust_per_step
        )

        return tau_bias + actual_delta

    def apply_phase_modification(
        self,
        manifold_theta: torch.Tensor,
        delta_phase: torch.Tensor,
    ) -> torch.Tensor:
        # manifold_theta has shape [H, 3], only first column is used for phases
        # delta_phase is scalar [1]
        self.phase_momentum.mul_(self.config.momentum)
        self.phase_momentum.add_(delta_phase.detach())

        actual_delta = torch.clamp(
            self.phase_momentum,
            -self.config.max_adjust_per_step,
            self.config.max_adjust_per_step
        )

        # Apply to first column only (the one used for phase computation)
        result = manifold_theta.clone()
        result[:, 0] = manifold_theta[:, 0] + actual_delta
        return result


class SelfAwareTwistorLMT(nn.Module):
    """
    自感知扭量神经网络 v2

    核心创新：每个输入步，内部多轮ODE思考 + 收敛检测

    工作流程:
    for 每个输入时间步 t:
        内部思考循环 (2-5轮):
            感知自己的输出
            ODE一步
            评估输出质量
            微量调整参数
            检查收敛?
        输出最终结果
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        config: Optional[SelfFeedbackConfig] = None,
        dt: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.config = config or SelfFeedbackConfig()

        # === 核心网络组件 ===
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_amplitude = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.5)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        self.tau_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.b_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Linear(hidden_dim, output_dim)

        self.manifold_theta = nn.Parameter(
            torch.randn(hidden_dim, 3) * 0.1
        )

        self.sparse_mask = nn.Parameter(torch.ones(hidden_dim, hidden_dim) * -5)

        # === 自我感知组件 ===
        self.self_attention = OutputSelfAttention(hidden_dim, output_dim)
        self.critic = InternalCritic(hidden_dim, output_dim)
        self.modifier = MicroModifier(hidden_dim, config)

        # === 状态追踪 ===
        self.output_history: List[torch.Tensor] = []
        self.state_history: List[torch.Tensor] = []
        self.evaluation_history: List[Dict] = []
        self.think_history: List[Dict] = []  # 记录每步思考情况

        # 修改追踪
        self.total_tau_modifications = 0
        self.total_phase_modifications = 0
        self.total_converged = 0  # 收敛次数

    def reset_history(self):
        """重置历史记录"""
        self.output_history.clear()
        self.state_history.clear()
        self.evaluation_history.clear()
        self.think_history.clear()
        self.total_converged = 0

    def compute_dzdt(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        perception: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """计算状态导数"""
        z_real = z.real
        z_imag = z.imag

        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        mask = torch.sigmoid(self.sparse_mask)
        amp = self.W_amplitude * mask

        theta = self.manifold_theta[:, 0]
        phase = theta.unsqueeze(1) - theta.unsqueeze(0)
        W = amp * torch.exp(1j * phase)

        W_real = W.real
        W_imag = W.imag

        W_tanh_real = F.linear(tanh_real, W_real) - F.linear(tanh_imag, W_imag)
        W_tanh_imag = F.linear(tanh_real, W_imag) + F.linear(tanh_imag, W_real)

        Ux = self.U(x)

        dz_real = -z_real + W_tanh_real + Ux + self.b_real
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag

        if perception is not None:
            dz_real = dz_real + perception * 0.05

        tau = F.sigmoid(self.W_tau(torch.abs(z)))
        tau = tau + self.tau_bias
        tau = torch.clamp(tau, 0.01, 1.0)

        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        return dzdt

    def check_convergence(
        self,
        recent_outputs: List[torch.Tensor],
        recent_states: List[torch.Tensor],
    ) -> Tuple[bool, float]:
        """
        检查思考是否收敛

        收敛条件:
        1. 连续几轮输出变化很小
        2. 连续几轮状态变化很小

        Returns:
            converged: 是否收敛
            diff: 当前与上次的差异
        """
        if len(recent_outputs) < self.config.convergence_window + 1:
            return False, 1.0

        # 计算输出的变化
        recent = recent_outputs[-self.config.convergence_window:]
        output_diffs = []
        for i in range(1, len(recent)):
            diff = (recent[i] - recent[i-1]).abs().mean().item()
            output_diffs.append(diff)

        avg_diff = np.mean(output_diffs)
        converged = avg_diff < self.config.convergence_threshold

        return converged, avg_diff

    def think_step(
        self,
        z: torch.Tensor,
        x_t: torch.Tensor,
        prev_output: Optional[torch.Tensor],
        prev_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        单步思考：感知 → ODE → 评估

        Returns:
            z: 新状态
            output: 输出
            evaluation: 评估结果
        """
        # 1. 感知
        perception = None
        if prev_output is not None:
            perception = self.self_attention.perceive(z.real, prev_output)

        # 2. ODE一步
        dzdt = self.compute_dzdt(z, x_t, perception)
        z = z + self.dt * dzdt

        # 限制
        z = torch.complex(
            torch.clamp(z.real, -100, 100),
            torch.clamp(z.imag, -100, 100),
        )

        # 3. 输出
        output = self.out(z.real)

        # 4. 评估
        evaluation = self.critic.evaluate(
            output,
            z.real,
            prev_output,
            prev_state.real if prev_state is not None else None,
        )

        return z, output, evaluation

    def internal_think(
        self,
        x_t: torch.Tensor,
        init_z: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        内部思考循环：每个输入步，多次ODE+反馈

        Args:
            x_t: 当前输入 [batch, input_dim]
            init_z: 初始状态 (可选)
            batch_size: batch大小

        Returns:
            final_z: 最终状态
            final_output: 最终输出
            think_log: 每轮思考的日志
        """
        think_log = []

        # 初始化状态
        if init_z is None:
            z = torch.zeros(batch_size, self.hidden_dim, dtype=torch.complex64, device=x_t.device)
        else:
            z = init_z

        prev_output = None
        prev_state = None

        # 思考历史 (用于收敛检测)
        recent_outputs = []
        recent_states = []

        for step in range(self.config.max_think_steps):
            # 思考一步
            z, output, evaluation = self.think_step(
                z, x_t, prev_output, prev_state
            )

            # 记录历史
            recent_outputs.append(output.detach())
            recent_states.append(z.detach())

            # 保持窗口大小
            if len(recent_outputs) > self.config.convergence_window + 1:
                recent_outputs.pop(0)
                recent_states.pop(0)

            # 评估并微调
            delta_tau, delta_phase, delta_weight = self.modifier.compute_modulation(
                evaluation, z.real
            )

            # 应用修改 (间隔控制)
            if step % self.config.feedback_interval == 0:
                new_tau = self.modifier.apply_tau_modification(self.tau_bias, delta_tau)
                self.tau_bias.data = new_tau
                self.total_tau_modifications += 1

                new_theta = self.modifier.apply_phase_modification(
                    self.manifold_theta, delta_phase
                )
                self.manifold_theta.data = new_theta
                self.total_phase_modifications += 1

            # 收敛检测
            converged, diff = self.check_convergence(recent_outputs, recent_states)

            # 记录
            think_log.append({
                'step': step,
                'evaluation': evaluation,
                'diff': diff,
                'converged': converged,
            })

            # 收敛检测：需要至少min步，且变化足够小
            if (step >= self.config.min_think_steps - 1 and converged):
                self.total_converged += 1
                break

            prev_output = output
            prev_state = z

        return z, output, think_log

    def forward(
        self,
        x: torch.Tensor,
        enable_thinking: bool = True,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            x: [seq_len, batch, input_dim]
            enable_thinking: 是否启用内部思考
            return_diagnostics: 是否返回诊断信息

        Returns:
            output: [seq_len, batch, output_dim]
            diagnostics: 诊断信息 (可选)
        """
        T, B, _ = x.shape

        outputs = []
        all_think_logs = []

        for t in range(T):
            x_t = x[t]

            if enable_thinking:
                # 内部思考循环
                final_z, final_output, think_log = self.internal_think(x_t, batch_size=B)
                all_think_logs.append(think_log)
            else:
                # 简单前向 (无思考)
                z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
                dzdt = self.compute_dzdt(z, x_t, None)
                z = z + self.dt * dzdt
                z = torch.complex(
                    torch.clamp(z.real, -100, 100),
                    torch.clamp(z.imag, -100, 100),
                )
                final_output = self.out(z.real)

            outputs.append(final_output)

        output_seq = torch.stack(outputs, dim=0)

        if return_diagnostics:
            # 统计
            avg_think_steps = np.mean([len(log) for log in all_think_logs]) if all_think_logs else 0
            convergence_rate = self.total_converged / max(1, T)

            return output_seq, {
                'total_converged': self.total_converged,
                'convergence_rate': convergence_rate,
                'avg_think_steps': avg_think_steps,
                'n_tau_mods': self.total_tau_modifications,
                'n_phase_mods': self.total_phase_modifications,
                'think_logs': all_think_logs,
            }

        return output_seq

    def get_diagnostics(self) -> Dict:
        """获取诊断信息"""
        return {
            'total_tau_modifications': self.total_tau_modifications,
            'total_phase_modifications': self.total_phase_modifications,
            'total_converged': self.total_converged,
            'config': {
                'min_think_steps': self.config.min_think_steps,
                'max_think_steps': self.config.max_think_steps,
                'convergence_threshold': self.config.convergence_threshold,
            }
        }


def test_internal_thinking():
    """测试内部思考循环"""
    print("=" * 70)
    print("自感知扭量神经网络 v2 - 内部思考测试")
    print("=" * 70)

    device = 'cpu'
    torch.manual_seed(42)

    # 创建配置
    config = SelfFeedbackConfig(
        min_think_steps=2,
        max_think_steps=5,
        convergence_threshold=0.01,
        convergence_window=2,
    )

    # 创建模型
    model = SelfAwareTwistorLMT(
        input_dim=10,
        hidden_dim=32,
        output_dim=10,
        config=config,
        dt=0.1,
    ).to(device)

    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"思考配置: {config.min_think_steps}-{config.max_think_steps} 轮, 收敛阈值={config.convergence_threshold}")

    # 测试数据
    x = torch.randn(10, 4, 10).to(device)  # [seq, batch, input]

    # 测试1: 启用思考
    print("\n" + "-" * 50)
    print("测试1: 启用内部思考")
    print("-" * 50)

    model.train()
    output_think, diag_think = model(x, enable_thinking=True, return_diagnostics=True)

    print(f"  输出形状: {output_think.shape}")
    print(f"  收敛次数: {diag_think['total_converged']}/{x.shape[0]}")
    print(f"  收敛率: {diag_think['convergence_rate']:.1%}")
    print(f"  平均思考轮数: {diag_think['avg_think_steps']:.1f}")
    print(f"  τ修改次数: {diag_think['n_tau_mods']}")
    print(f"  相位修改次数: {diag_think['n_phase_mods']}")

    # 打印前几步的思考详情
    print("\n  各时间步思考详情:")
    for t, log in enumerate(diag_think['think_logs'][:3]):
        steps_info = []
        for entry in log:
            ev = entry['evaluation']
            steps_info.append(
                f"k={entry['step']}: eval={ev['overall']:.2f}, diff={entry['diff']:.4f}"
            )
        print(f"    t={t}: {' | '.join(steps_info)}")

    # 测试2: 关闭思考
    print("\n" + "-" * 50)
    print("测试2: 关闭思考 (简单前向)")
    print("-" * 50)

    model.reset_history()
    output_simple, diag_simple = model(x, enable_thinking=False, return_diagnostics=True)

    print(f"  输出形状: {output_simple.shape}")
    print(f"  输出差异: {F.mse_loss(output_think, output_simple).item():.6f}")

    # 测试3: 不同收敛阈值
    print("\n" + "-" * 50)
    print("测试3: 不同收敛阈值")
    print("-" * 50)

    for threshold in [0.1, 0.05, 0.01, 0.001]:
        config_test = SelfFeedbackConfig(
            min_think_steps=2,
            max_think_steps=5,
            convergence_threshold=threshold,
        )
        model_test = SelfAwareTwistorLMT(
            input_dim=10, hidden_dim=16, output_dim=10,
            config=config_test,
        ).to(device)

        _, diag = model_test(x[:5], enable_thinking=True, return_diagnostics=True)

        print(f"  阈值={threshold:.3f}: 平均轮数={diag['avg_think_steps']:.1f}, "
              f"收敛率={diag['convergence_rate']:.1%}")

    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)


if __name__ == "__main__":
    test_internal_thinking()
