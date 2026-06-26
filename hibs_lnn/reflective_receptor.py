"""
反思型受体门控液态神经网络
==========================
整合: ReceptorGatedTwistorLMT + ReflectionModule + DifferentiableMemory

这个模块实现了完整的"思考-行动-反思-修改"循环:
1. 液态神经网络负责常规的信息处理
2. 外部记忆增强长程依赖建模
3. 反思模块周期性评估并修改网络结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from .receptor_gated_LMT import ReceptorGatedTwistorLMT, GrowthConfig
from .reflection import ReflectionModule, ReflectionConfig, ReflectiveTrainer
from .memory import DifferentiableMemory, MemoryConfig


@dataclass
class ReflectiveConfig:
    """反思型配置"""
    # 反思配置
    reflection_interval: int = 50     # 反思间隔
    think_steps: int = 10            # 思考步数
    grow_threshold: float = 0.3     # 增长阈值
    prune_threshold: float = 0.9     # 剪枝阈值

    # 记忆配置
    memory_size: int = 128           # 记忆槽位
    read_strength: float = 0.1       # 读取强度
    write_strength: float = 0.1      # 写入强度

    # 生长配置
    min_hidden_dim: int = 16        # 最小隐藏维度
    max_hidden_dim: int = 128       # 最大隐藏维度
    enable_growth: bool = True       # 启用生长


class ReflectiveReceptorLMT(nn.Module):
    """
    反思型受体门控液态神经网络

    整合架构:
    ┌─────────────────────────────────────────────────────────────┐
    │                    反思控制器 (ReflectionModule)              │
    │  Think → Evaluate → Decide → Modify                          │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              外部记忆 (DifferentiableMemory)                    │
    │  ┌─────────────────────────────────────────────────────┐   │
    │  │  M ∈ ℝ^(128 × hidden_dim)                          │   │
    │  │  Read Head ←→ Write Head                           │   │
    │  └─────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              核心液态网络 (ReceptorGatedTwistorLMT)           │
    │  dz/dt = (-z + W·tanh(z) + U·x + Memory_Read) / τ(z)     │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        n_channels: int = 3,
        n_receptor_types: int = 3,
        config: Optional[ReflectiveConfig] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.config = config or ReflectiveConfig()

        # 1. 核心网络 - 受体门控液态网络
        growth_config = GrowthConfig(
            min_hidden_dim=self.config.min_hidden_dim,
            max_hidden_dim=self.config.max_hidden_dim,
            enable_growth=self.config.enable_growth,
        )

        self.core = ReceptorGatedTwistorLMT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            n_channels=n_channels,
            n_receptor_types=n_receptor_types,
            growth_config=growth_config,
            use_low_rank_gating=True,
            use_phase_modulation=True,
        )

        # 2. 外部记忆
        self.memory = DifferentiableMemory(
            memory_size=self.config.memory_size,
            key_dim=hidden_dim,
            value_dim=hidden_dim,
            config=MemoryConfig(
                read_strength=self.config.read_strength,
                write_strength=self.config.write_strength,
            ),
        )

        # 3. 反思模块
        reflection_config = ReflectionConfig(
            reflection_interval=self.config.reflection_interval,
            think_steps=self.config.think_steps,
            grow_threshold=self.config.grow_threshold,
            prune_threshold=self.config.prune_threshold,
        )

        self.reflection = ReflectionModule(
            hidden_dim=hidden_dim,
            config=reflection_config,
        )

        # 4. 记忆接口
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # 5. 训练状态
        self.training_step = 0
        self.loss_history: List[float] = []
        self.grad_history: List[float] = []

        # 诊断
        self.diagnostics_history = []

    def read_memory(self, z_real: torch.Tensor) -> torch.Tensor:
        """从外部记忆读取"""
        query = self.query_proj(z_real)
        result = self.memory(query)
        return result['read_content']

    def write_memory(self, z_real: torch.Tensor, importance: float = 1.0):
        """写入外部记忆"""
        key = self.key_proj(z_real)
        value = self.value_proj(z_real)

        # 重要性加权
        weighted_value = value * importance

        self.memory.write(key, weighted_value)

    def compute_dzdt_with_memory(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """计算带记忆的状态导数"""
        # 基础导数
        dzdt = self.core.compute_dzdt(z, x, c)

        # 记忆读取
        memory_read = self.read_memory(z.real)

        # 融合到导数
        memory_complex = torch.complex(
            memory_read,
            torch.zeros_like(memory_read),
        )
        dzdt = dzdt + self.config.read_strength * memory_complex

        return dzdt

    def forward(
        self,
        x: torch.Tensor,
        write_memory: bool = True,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            x: [seq_len, batch, input_dim]
            write_memory: 是否写入记忆
            return_diagnostics: 是否返回诊断信息

        Returns:
            y: [seq_len, batch, output_dim]
            diagnostics: 诊断信息 (可选)
        """
        T, B, _ = x.shape

        # 初始化状态
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        prev_read = torch.zeros(B, self.hidden_dim, device=x.device)

        outputs = []
        states = []
        read_weights_history = []
        memory_access_count = 0

        for t in range(T):
            x_t = x[t]

            # 1. 计算递质浓度
            c = self.core.compute_channels(x_t)

            # 2. 读取记忆
            memory_result = self.memory(
                self.query_proj(z.real),
                prev_read=prev_read,
            )
            memory_read = memory_result['read_content']
            read_weights_history.append(memory_result['read_weights'])

            # 3. 计算状态导数
            dzdt = self.core.compute_dzdt(z, x_t, c)

            # 4. 记忆增强
            memory_complex = torch.complex(
                memory_read,
                torch.zeros_like(memory_read),
            )
            dzdt = dzdt + self.config.read_strength * memory_complex

            # 5. 状态更新
            z = z + self.core.dt * dzdt

            # 6. 限制状态范围
            z = torch.complex(
                torch.clamp(z.real, -self.core.z_max, self.core.z_max),
                torch.clamp(z.imag, -self.core.z_max, self.core.z_max),
            )

            # 7. 写入记忆
            if write_memory and t % 5 == 0:
                importance = torch.sigmoid(
                    torch.abs(z).mean(dim=-1, keepdim=True)
                ).squeeze(-1)
                self.write_memory(z.real, importance.mean().item())
                memory_access_count += 1

            # 8. 更新prev_read
            prev_read = memory_read

            # 9. 输出
            y_t = self.core.out(z.real)
            outputs.append(y_t)

            if return_diagnostics:
                states.append(z.detach().cpu())

        y = torch.stack(outputs, dim=0)

        if return_diagnostics:
            diagnostics = {
                'states': torch.stack(states) if states else None,
                'read_weights': torch.stack(read_weights_history) if read_weights_history else None,
                'memory_access_count': memory_access_count,
                'hidden_dim': self.hidden_dim,
            }
            return y, diagnostics

        return y

    def think(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        思考阶段

        Args:
            x: 输入
            y: 目标 (可选)

        Returns:
            thought_summary: 思考总结
            evaluation: 评估结果
        """
        thought_summary, thoughts = self.reflection.think(
            self.core, x, y,
            n_steps=self.config.think_steps,
        )

        evaluation = self.reflection.evaluate(
            self.core,
            self.loss_history,
            self.grad_history,
        )

        return thought_summary, evaluation

    def reflect(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        完整反思

        Args:
            x: 输入
            y: 目标 (可选)

        Returns:
            result: 反思结果
        """
        result = self.reflection.reflect(
            self.core, x, y,
            self.loss_history,
            self.grad_history,
            self.training_step,
        )

        self.training_step += 1

        return result

    def should_reflect(self) -> bool:
        """判断是否应该反思"""
        return self.reflection.should_reflect(self.training_step)

    def get_diagnostics(self) -> Dict:
        """获取诊断信息"""
        return {
            'hidden_dim': self.hidden_dim,
            'training_step': self.training_step,
            'reflection_stats': self.reflection.get_stats(),
            'memory_stats': self.memory.get_stats(),
            'current_confidence': self.reflection.confidence_ema.item(),
        }

    def reset_state(self, batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
        """重置隐藏状态"""
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.complex64, device=device)


class ReflectiveTrainerV2:
    """
    反思型训练器 V2

    整合反思机制和外部记忆的训练循环
    """

    def __init__(
        self,
        model: ReflectiveReceptorLMT,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.train_history: List[Dict] = []

    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Dict:
        """
        单步训练

        Args:
            x: [seq_len, batch, input_dim]
            y: [seq_len, batch, output_dim]

        Returns:
            result: 训练结果
        """
        self.model.train()
        self.model.training_step += 1

        # 1. 常规训练
        self.optimizer.zero_grad()

        y_pred = self.model(x, write_memory=True)

        # 调整维度
        if y_pred.dim() == 3:
            y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])
        else:
            y_pred_flat = y_pred
            y_flat = y

        loss = self.criterion(y_pred_flat, y_flat)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        ).item()

        self.optimizer.step()

        # 记录历史
        self.model.loss_history.append(loss.item())
        self.model.grad_history.append(grad_norm)

        result = {
            'step': self.model.training_step,
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'hidden_dim': self.model.hidden_dim,
        }

        # 2. 反思检查
        if self.model.should_reflect():
            reflection_result = self.model.reflect(x, y)

            result['reflection'] = reflection_result
            result['decision'] = reflection_result['decision']['action']
            result['n_mods'] = reflection_result['n_modifications']
            result['confidence'] = reflection_result['evaluation']['confidence']

        self.train_history.append(result)
        return result

    def train_loop(
        self,
        dataloader,
        n_steps: int,
        print_every: int = 50,
    ) -> Dict:
        """
        训练循环

        Args:
            dataloader: 数据加载器
            n_steps: 训练步数
            print_every: 打印间隔

        Returns:
            history: 训练历史
        """
        print("=" * 70)
        print("反思型受体门控液态神经网络训练")
        print("=" * 70)
        print(f"反思间隔: {self.model.config.reflection_interval} 步")
        print(f"记忆槽位: {self.model.config.memory_size}")
        print(f"初始隐藏维度: {self.model.config.min_hidden_dim}")
        print(f"最大隐藏维度: {self.model.config.max_hidden_dim}")
        print("=" * 70)

        for step, (x, y) in enumerate(dataloader):
            if step >= n_steps:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            result = self.train_step(x, y)

            if (step + 1) % print_every == 0:
                stats = self.model.get_diagnostics()
                print(
                    f"Step {step+1:5d} | "
                    f"Loss: {result['loss']:.4f} | "
                    f"Dim: {result['hidden_dim']:3d} | "
                    f"Conf: {result.get('confidence', stats['current_confidence']):.3f} | "
                    f"Action: {result.get('decision', 'keep'):5s} | "
                    f"Mem: {stats['memory_stats']['n_reads']:4d}/{stats['memory_stats']['n_writes']}"
                )

        print("=" * 70)
        print("训练完成!")
        print("-" * 70)
        print(f"总步数: {self.model.training_step}")
        print(f"最终隐藏维度: {self.model.hidden_dim}")
        stats = self.model.get_diagnostics()
        print(f"总反思次数: {stats['reflection_stats']['total_thinks']}")
        print(f"总增长次数: {stats['reflection_stats']['total_grows']}")
        print(f"总剪枝次数: {stats['reflection_stats']['total_prunes']}")
        print("=" * 70)

        return {
            'train_history': self.train_history,
            'final_diagnostics': self.model.get_diagnostics(),
        }
