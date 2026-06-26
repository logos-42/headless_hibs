"""
反思模块 - Think-Act-Reflect-Modify 循环的核心
============================================
让液态神经网络能够在思考后自动评估并修改自身结构

核心工作流程:
1. Think: 运行ODE多步收集信息
2. Evaluate: 分析损失趋势、梯度稳定性
3. Decide: 决定是否修改、修改什么
4. Modify: 执行结构修改
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time


@dataclass
class ReflectionConfig:
    """反思配置"""
    think_steps: int = 10          # 思考步数
    reflection_interval: int = 50   # 反思间隔
    grow_threshold: float = 0.3     # 增长阈值
    prune_threshold: float = 0.9    # 剪枝阈值
    min_steps_before_grow: int = 100 # 增长前最小步数
    min_steps_before_prune: int = 200 # 剪枝前最小步数
    max_modifications_per_reflection: int = 2  # 每次反思最大修改次数
    confidence_smoothing: float = 0.9  # EMA平滑系数


@dataclass
class ModificationRecord:
    """修改记录"""
    step: int
    action: str  # 'grow', 'prune', 'keep'
    target_idx: Optional[int] = None
    new_idx: Optional[int] = None
    confidence_before: float = 0.0
    confidence_after: float = 0.0
    loss_trend: float = 0.0


class ReflectionModule(nn.Module):
    """
    反思模块 - 核心创新点

    这个模块赋予液态神经网络"自反思"能力：
    - 周期性暂停训练进行深度思考
    - 分析训练趋势做出智能决策
    - 在合适的时机修改网络结构
    """

    def __init__(
        self,
        hidden_dim: int,
        config: Optional[ReflectionConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or ReflectionConfig()

        # 评估网络: 分析训练状态
        # 输入: [loss_trend, grad_stable, tau_diverse, confidence]
        # 输出: [grow_prob, prune_prob, keep_prob]
        self.evaluator = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        # 置信度EMA平滑
        self.register_buffer('confidence_ema', torch.tensor(0.5))

        # 反思缓冲区 - 存储历史评估
        self.reflection_buffer: List[Dict] = []

        # 修改历史
        self.modification_history: List[ModificationRecord] = []

        # 统计
        self.total_thinks = 0
        self.total_grows = 0
        self.total_prunes = 0

    def reset(self):
        """重置反思状态"""
        self.reflection_buffer.clear()
        self.modification_history.clear()
        self.total_thinks = 0
        self.total_grows = 0
        self.total_prunes = 0
        self.confidence_ema.fill_(0.5)

    def think(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        思考阶段: 多步ODE推理收集信息

        Args:
            model: 液态神经网络模型
            x: 输入 [seq_len, batch, input_dim]
            y: 目标 [seq_len, batch, output_dim]
            n_steps: 思考步数

        Returns:
            thought_summary: 思考总结向量
            thoughts: 所有思考向量列表
        """
        if n_steps is None:
            n_steps = self.config.think_steps

        model.eval()
        with torch.no_grad():
            B = x.shape[1]

            # 初始化隐藏状态
            if hasattr(model, 'reset_state'):
                z = model.reset_state(B, x.device)
            elif hasattr(model, 'hidden_dim'):
                z = torch.zeros(B, model.hidden_dim, dtype=torch.complex64, device=x.device)
            else:
                z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

            thoughts = []
            thought_states = []

            for step in range(n_steps):
                # 准备思考输入 - 使用第一个时间步作为上下文
                x_context = x[0] if x.shape[0] > 0 else x.squeeze(0)

                # 单步ODE更新
                if hasattr(model, 'compute_dzdt'):
                    dzdt = model.compute_dzdt(z, x_context)
                elif hasattr(model, 'LMT') and hasattr(model.LMT, 'compute_dzdt'):
                    dzdt = model.LMT.compute_dzdt(z, x_context)
                else:
                    # 简单线性更新作为后备
                    if hasattr(model, 'U'):
                        dzdt = -z + model.U(x_context.unsqueeze(0)).squeeze(0)
                    else:
                        break

                z = z + (model.dt if hasattr(model, 'dt') else 0.1) * dzdt

                # 限制状态范围
                z = torch.complex(
                    torch.clamp(z.real, -100, 100),
                    torch.clamp(z.imag, -100, 100),
                )

                thoughts.append(z.clone())
                thought_states.append(z.real)  # 使用实部作为思考向量

                # 思考收敛检测
                if step >= 2 and len(thoughts) >= 3:
                    recent_variance = torch.stack(thoughts[-3:]).var(dim=0).mean().item()
                    if recent_variance < 1e-4:
                        break

            # 汇总思考结果
            if thoughts:
                thought_summary = torch.stack(thoughts).mean(dim=0)
            else:
                thought_summary = torch.zeros(B, self.hidden_dim, device=x.device)

            self.total_thinks += 1

        return thought_summary, thoughts

    def evaluate(
        self,
        model: nn.Module,
        loss_history: List[float],
        grad_norms: List[float],
    ) -> Dict[str, float]:
        """
        评估阶段: 分析训练趋势

        Args:
            model: 模型
            loss_history: 损失历史
            grad_norms: 梯度范数历史

        Returns:
            evaluation: 评估结果字典
        """
        # 损失趋势
        if len(loss_history) >= 20:
            recent = loss_history[-20:]
            old = loss_history[max(0, len(loss_history) - 100):len(loss_history) - 20]
            if len(old) > 0:
                trend = (np.mean(recent) - np.mean(old)) / (np.mean(old) + 1e-6)
            else:
                trend = 0.0
        else:
            trend = 0.0

        # 梯度稳定性
        if len(grad_norms) >= 20:
            grad_std = np.std(grad_norms[-20:])
            grad_mean = np.mean(grad_norms[-20:]) + 1e-6
            grad_stable = 1.0 / (grad_std / grad_mean + 1.0)
        else:
            grad_stable = 1.0

        # τ分布多样性
        try:
            if hasattr(model, 'compute_tau'):
                tau_stats = model.compute_tau(torch.zeros(1, model.hidden_dim, device=next(model.parameters()).device))
                tau_diverse = tau_stats.std().item()
            elif hasattr(model, 'LMT') and hasattr(model.LMT, 'compute_tau'):
                tau_stats = model.LMT.compute_tau(torch.zeros(1, model.LMT.hidden_dim, device=next(model.parameters()).device))
                tau_diverse = tau_stats.std().item()
            else:
                tau_diverse = 0.1
        except:
            tau_diverse = 0.1

        # 计算置信度
        confidence_raw = 0.5 - trend * 0.5 + grad_stable * 0.3 + tau_diverse * 0.2
        confidence_raw = max(0.0, min(1.0, confidence_raw))

        # EMA平滑
        confidence = self.confidence_ema.item() * self.config.confidence_smoothing + \
                     confidence_raw * (1 - self.config.confidence_smoothing)
        self.confidence_ema.fill_(confidence)

        return {
            'loss_trend': trend,
            'grad_stable': grad_stable,
            'tau_diverse': tau_diverse,
            'confidence': confidence,
            'confidence_raw': confidence_raw,
        }

    def decide(self, evaluation: Dict[str, float], training_step: int) -> Dict:
        """
        决策阶段: 决定修改策略

        Args:
            evaluation: 评估结果
            training_step: 当前训练步数

        Returns:
            decision: 决策字典
        """
        confidence = evaluation['confidence']

        # 检查是否满足修改前置条件
        can_grow = training_step >= self.config.min_steps_before_grow
        can_prune = training_step >= self.config.min_steps_before_prune

        if confidence < self.config.grow_threshold and can_grow:
            # 低置信度: 需要结构改变 -> 增长
            decision = {
                'action': 'grow',
                'strength': self.config.grow_threshold - confidence,
                'reason': 'low_confidence'
            }
        elif confidence > self.config.prune_threshold and can_prune:
            # 高置信度: 可能过度拟合 -> 剪枝
            decision = {
                'action': 'prune',
                'strength': confidence - self.config.prune_threshold,
                'reason': 'potential_overfitting'
            }
        else:
            decision = {
                'action': 'keep',
                'strength': 0.0,
                'reason': 'stable'
            }

        return decision

    def modify(
        self,
        model: nn.Module,
        decision: Dict,
        training_step: int,
    ) -> Tuple[int, Optional[ModificationRecord]]:
        """
        执行修改

        Args:
            model: 模型
            decision: 决策
            training_step: 当前步数

        Returns:
            n_modifications: 修改次数
            record: 修改记录
        """
        action = decision['action']
        confidence_before = self.confidence_ema.item()

        if action == 'grow':
            # 尝试增长神经元
            n_added = 0

            for _ in range(self.config.max_modifications_per_reflection):
                if hasattr(model, 'split_neuron'):
                    # Growable模型
                    overloaded = []
                    if hasattr(model, 'get_overloaded_neurons'):
                        overloaded = model.get_overloaded_neurons()

                    if overloaded:
                        parent = overloaded[0]
                        new_idx = model.split_neuron(parent)
                        if new_idx >= 0:
                            n_added += 1
                    else:
                        # 随机选择活跃神经元
                        if model.hidden_dim > 0:
                            parent = np.random.randint(model.hidden_dim)
                            new_idx = model.split_neuron(parent)
                            if new_idx >= 0:
                                n_added += 1
                elif hasattr(model, 'LMT') and hasattr(model.LMT, 'split_neuron'):
                    # 包装模型
                    overloaded = model.LMT.get_overloaded_neurons() if hasattr(model.LMT, 'get_overloaded_neurons') else []
                    if overloaded:
                        new_idx = model.LMT.split_neuron(overloaded[0])
                        if new_idx >= 0:
                            n_added += 1

            self.total_grows += n_added

            record = ModificationRecord(
                step=training_step,
                action='grow',
                new_idx=new_idx if n_added > 0 else None,
                confidence_before=confidence_before,
                confidence_after=self.confidence_ema.item(),
                loss_trend=0.0,  # 暂时设为0
            )

            return n_added, record

        elif action == 'prune':
            # 尝试剪枝
            n_pruned = 0

            if hasattr(model, 'prune_neurons'):
                n_pruned = model.prune_neurons()
            elif hasattr(model, 'LMT') and hasattr(model.LMT, 'prune_neurons'):
                n_pruned = model.LMT.prune_neurons()

            self.total_prunes += n_pruned

            record = ModificationRecord(
                step=training_step,
                action='prune',
                target_idx=None,
                confidence_before=confidence_before,
                confidence_after=self.confidence_ema.item(),
                loss_trend=0.0,
            )

            return n_pruned, record

        else:
            # keep - 不修改
            return 0, None

    def reflect(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_history: List[float],
        grad_norms: List[float],
        training_step: int,
    ) -> Dict:
        """
        完整的反思过程

        Args:
            model: 模型
            x: 输入
            y: 目标
            loss_history: 损失历史
            grad_norms: 梯度范数历史
            training_step: 当前步数

        Returns:
            result: 反思结果
        """
        # 1. Think
        thought_summary, thoughts = self.think(model, x, y)

        # 2. Evaluate
        evaluation = self.evaluate(model, loss_history, grad_norms)

        # 3. Decide
        decision = self.decide(evaluation, training_step)

        # 4. Modify
        n_mods, record = self.modify(model, decision, training_step)

        # 记录
        self.reflection_buffer.append({
            'step': training_step,
            'thought_summary': thought_summary.detach(),
            'evaluation': evaluation,
            'decision': decision,
            'n_modifications': n_mods,
        })

        if record:
            self.modification_history.append(record)

        return {
            'thought_summary': thought_summary,
            'evaluation': evaluation,
            'decision': decision,
            'n_modifications': n_mods,
        }

    def should_reflect(self, training_step: int) -> bool:
        """判断是否应该反思"""
        return training_step > 0 and training_step % self.config.reflection_interval == 0

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_thinks': self.total_thinks,
            'total_grows': self.total_grows,
            'total_prunes': self.total_prunes,
            'total_reflections': len(self.reflection_buffer),
            'current_confidence': self.confidence_ema.item(),
            'modification_rate': (
                (self.total_grows + self.total_prunes) / max(1, self.total_thinks)
            ),
        }


class ReflectiveTrainer:
    """
    反思训练器 - 整合反思机制到训练循环
    """

    def __init__(
        self,
        model: nn.Module,
        reflection_module: ReflectionModule,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
    ):
        self.model = model
        self.reflection = reflection_module
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.loss_history: List[float] = []
        self.grad_history: List[float] = []
        self.training_step = 0

        self.train_history: List[Dict] = []

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> Dict:
        """
        单步训练

        Args:
            x: 输入 [seq_len, batch, input_dim]
            y: 目标 [seq_len, batch, output_dim]

        Returns:
            result: 训练结果
        """
        self.model.train()
        self.training_step += 1

        # 1. 常规训练步骤
        self.optimizer.zero_grad()

        y_pred = self.model(x)

        # 处理维度
        if y_pred.dim() == 3:
            y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
            y_flat = y.reshape(-1, y.shape[-1])
        else:
            y_pred_flat = y_pred
            y_flat = y

        loss = self.criterion(y_pred_flat, y_flat)
        loss.backward()

        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        ).item()

        self.optimizer.step()

        # 记录历史
        self.loss_history.append(loss.item())
        self.grad_history.append(grad_norm)

        result = {
            'step': self.training_step,
            'loss': loss.item(),
            'grad_norm': grad_norm,
            'reflection_triggered': False,
        }

        # 2. 反思检查
        if self.reflection.should_reflect(self.training_step):
            reflection_result = self.reflection.reflect(
                self.model, x, y,
                self.loss_history,
                self.grad_history,
                self.training_step,
            )

            result['reflection_triggered'] = True
            result['reflection'] = reflection_result

            # 更新结果
            result['decision'] = reflection_result['decision']['action']
            result['n_mods'] = reflection_result['n_modifications']

        self.train_history.append(result)
        return result

    def train_loop(
        self,
        dataloader,
        n_steps: int,
        print_every: int = 100,
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
        print(f"开始反思训练，共 {n_steps} 步...")
        print(f"反思间隔: {self.reflection.config.reflection_interval} 步")
        print("-" * 60)

        for step, (x, y) in enumerate(dataloader):
            if step >= n_steps:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            result = self.train_step(x, y)

            if (step + 1) % print_every == 0:
                stats = self.reflection.get_stats()
                print(
                    f"Step {step+1:4d} | "
                    f"Loss: {result['loss']:.4f} | "
                    f"Grad: {result['grad_norm']:.4f} | "
                    f"Conf: {stats['current_confidence']:.3f} | "
                    f"Action: {result.get('decision', 'keep'):5s} | "
                    f"Mods: {stats['total_grows']}/{stats['total_prunes']}"
                )

        print("-" * 60)
        print("训练完成!")
        print(f"总反思次数: {stats['total_thinks']}")
        print(f"总增长次数: {stats['total_grows']}")
        print(f"总剪枝次数: {stats['total_prunes']}")

        return {
            'train_history': self.train_history,
            'stats': stats,
        }
