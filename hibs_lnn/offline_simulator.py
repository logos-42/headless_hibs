"""
离线模拟器 - Offline Simulator
===============================
用于反事实推理和想象推演的引擎：

1. 离线模拟 - 不更新真实权重的模型运行
2. 反事实推演 - 假设性"如果...会怎样"推理
3. 想象模拟 - 生成未曾发生的轨迹
4. 因果效应评估 - 量化干预的效果

核心思想：在"精神空间"中进行实验，不需要真实干预
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass, field
import copy


@dataclass
class SimulationConfig:
    """模拟配置"""
    horizon: int = 20
    num_rollouts: int = 5
    intervention_duration: int = 5
    
    imagination_temperature: float = 1.0
    imagination_noise_scale: float = 0.1
    
    use_rejection: bool = True
    rejection_threshold: float = 0.5
    
    
@dataclass
class CounterfactualResult:
    """反事实结果"""
    factual_trajectory: torch.Tensor
    counterfactual_trajectory: torch.Tensor
    causal_effect: torch.Tensor
    intervention_points: List[int]
    effect_magnitude: float
    confidence: float


@dataclass
class ImaginedTrajectory:
    """想象的轨迹"""
    trajectory: torch.Tensor
    start_state: torch.Tensor
    intervention_applied: bool
    imagined_outcome: torch.Tensor
    novelty_score: float


class ModelCloner:
    """
    模型克隆器 - 创建模型的浅拷贝用于离线模拟
    
    关键：不复制梯度，只复制参数
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.cloned_state = None
        self.original_state = None
        
    def clone_parameters(self) -> Dict[str, torch.Tensor]:
        """复制模型参数"""
        return {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
        }
        
    def apply_parameters(self, parameters: Dict[str, torch.Tensor]):
        """应用参数到模型（不更新原始模型）"""
        for name, param in self.model.named_parameters():
            if name in parameters:
                param.data.copy_(parameters[name])
                
    def restore_parameters(self, original_parameters: Dict[str, torch.Tensor]):
        """恢复原始参数"""
        self.apply_parameters(original_parameters)


class OfflineSimulator(nn.Module):
    """
    离线模拟器
    
    功能：
    1. 冻结模型运行 - 不更新权重的模拟
    2. 反事实推理 - 应用干预并比较结果
    3. 想象推演 - 生成假设性轨迹
    4. 因果效应量化
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[SimulationConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or SimulationConfig()
        
        self.register_buffer('simulation_count', torch.tensor(0))
        
        self.trajectory_buffer: List[torch.Tensor] = []
        
    def simulate_without_learning(
        self,
        model: nn.Module,
        initial_state: torch.Tensor,
        num_steps: int,
        input_sequence: Optional[torch.Tensor] = None,
        store_trajectory: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        不学习地模拟 - 冻结模型参数
        
        使用torch.no_grad()防止梯度计算和参数更新
        
        Args:
            model: 要模拟的模型（会被临时修改）
            initial_state: (batch, hidden_dim) 初始状态
            num_steps: 模拟步数
            input_sequence: (num_steps, input_dim) 输入序列
            store_trajectory: 是否存储完整轨迹
            
        Returns:
            final_state: 最终状态
            trajectory: 完整轨迹
        """
        model.eval()
        
        original_params = {
            name: param.clone()
            for name, param in model.named_parameters()
        }
        
        state = initial_state.detach().clone()
        trajectory = [state]
        
        for step in range(num_steps):
            with torch.no_grad():
                if input_sequence is not None and step < input_sequence.size(0):
                    x_t = input_sequence[step:step+1]
                else:
                    x_t = torch.zeros(1, model.input_dim if hasattr(model, 'input_dim') else self.hidden_dim, device=state.device)
                    
                if hasattr(model, 'compute_dzdt'):
                    dzdt = model.compute_dzdt(state, x_t)
                    dt = 0.1
                    state = state + dt * dzdt
                elif hasattr(model, 'forward'):
                    output = model.forward(state, x_t)
                    if isinstance(output, tuple):
                        state = output[0]
                    else:
                        state = output
                        
                state = torch.clamp(state, -10, 10)
                trajectory.append(state)
                
        for name, param in model.named_parameters():
            param.data.copy_(original_params[name])
            
        trajectory_tensor = torch.stack(trajectory, dim=0)
        
        if store_trajectory:
            self.trajectory_buffer.append(trajectory_tensor)
            
        self.simulation_count += 1
        
        return state, trajectory_tensor
        
    def counterfactual_rollout(
        self,
        model: nn.Module,
        initial_state: torch.Tensor,
        intervention_fn: Callable,
        num_steps: int = 20,
        num_rollouts: int = 5,
    ) -> CounterfactualResult:
        """
        反事实rollout - 比较干预和无干预的轨迹
        
        Args:
            model: 模型
            initial_state: 初始状态
            intervention_fn: 干预函数 (state, step) -> intervened_state
            num_steps: rollout步数
            num_rollouts: 蒙特卡洛rollouts数量
            
        Returns:
            CounterfactualResult包含事实和反事实轨迹
        """
        factual_rollouts = []
        cf_rollouts = []
        
        for rollout_idx in range(num_rollouts):
            state_factual = initial_state.detach().clone()
            state_cf = initial_state.detach().clone()
            
            factual_traj = [state_factual]
            cf_traj = [state_cf]
            
            for step in range(num_steps):
                with torch.no_grad():
                    if hasattr(model, 'compute_dzdt'):
                        dzdt_f = model.compute_dzdt(state_factual, torch.zeros(1, self.hidden_dim, device=state_factual.device))
                        dt = 0.1
                        state_factual = state_factual + dt * dzdt_f
                        
                        dzdt_cf = model.compute_dzdt(state_cf, torch.zeros(1, self.hidden_dim, device=state_cf.device))
                        state_cf = intervention_fn(state_cf, step)
                        state_cf = state_cf + dt * dzdt_cf
                        
                    state_factual = torch.clamp(state_factual, -10, 10)
                    state_cf = torch.clamp(state_cf, -10, 10)
                    
                    factual_traj.append(state_factual)
                    cf_traj.append(state_cf)
                    
            factual_rollouts.append(torch.stack(factual_traj, dim=0))
            cf_rollouts.append(torch.stack(cf_traj, dim=0))
            
        factual_avg = torch.stack(factual_rollouts).mean(dim=0)
        cf_avg = torch.stack(cf_rollouts).mean(dim=0)
        
        causal_effect = cf_avg - factual_avg
        
        effect_magnitude = torch.norm(causal_effect).item()
        
        result = CounterfactualResult(
            factual_trajectory=factual_avg,
            counterfactual_trajectory=cf_avg,
            causal_effect=causal_effect,
            intervention_points=list(range(num_steps)),
            effect_magnitude=effect_magnitude,
            confidence=1.0 / (1.0 + np.std([torch.norm(f - cf_avg).item() for f in factual_rollouts])),
        )
        
        return result
        
    def imagine_alternatives(
        self,
        model: nn.Module,
        initial_state: torch.Tensor,
        num_alternatives: int = 10,
        imagination_horizon: int = 15,
        intervention_strength: float = 2.0,
    ) -> List[ImaginedTrajectory]:
        """
        想象替代方案 - 生成多种假设性轨迹
        
        用于创造性问题和决策制定
        """
        imagined = []
        
        for alt_idx in range(num_alternatives):
            state = initial_state.detach().clone()
            trajectory = [state]
            
            intervention_time = np.random.randint(2, imagination_horizon // 2)
            
            for step in range(imagination_horizon):
                with torch.no_grad():
                    if step == intervention_time:
                        noise = torch.randn_like(state) * self.config.imagination_noise_scale
                        intervention = torch.tanh(state + noise) * intervention_strength
                        state = state + intervention
                        
                    if hasattr(model, 'compute_dzdt'):
                        dzdt = model.compute_dzdt(state, torch.zeros(1, self.hidden_dim, device=state.device))
                        dt = 0.1
                        state = state + dt * dzdt
                        
                    state = torch.clamp(state, -10, 10)
                    trajectory.append(state)
                    
            traj = torch.stack(trajectory, dim=0)
            
            novelty = torch.std(traj, dim=(0, 1)).mean().item()
            
            imagined.append(ImaginedTrajectory(
                trajectory=traj,
                start_state=initial_state,
                intervention_applied=True,
                imagined_outcome=traj[-1],
                novelty_score=novelty,
            ))
            
        imagined.sort(key=lambda x: x.novelty_score, reverse=True)
        
        return imagined
        
    def evaluate_causal_effect(
        self,
        model: nn.Module,
        state: torch.Tensor,
        intervention_targets: List[int],
        intervention_value: float,
        num_simulations: int = 20,
    ) -> Dict[str, float]:
        """
        评估因果效应大小
        
        Returns:
            包含各种因果效应指标的字典
        """
        def apply_intervention(state, step):
            intervened = state.clone()
            for target in intervention_targets:
                if target < state.size(-1):
                    intervened[:, target] = intervention_value
            return intervened
            
        results = self.counterfactual_rollout(
            model, state, apply_intervention,
            num_steps=10, num_rollouts=num_simulations
        )
        
        return {
            'mean_effect': results.effect_magnitude,
            'confidence': results.confidence,
            'final_state_diff': torch.norm(
                results.factual_trajectory[-1] - results.counterfactual_trajectory[-1]
            ).item(),
            'trajectory_divergence': torch.mean(torch.abs(
                results.factual_trajectory - results.counterfactual_trajectory
            )).item(),
        }
        
    def reset(self):
        """重置模拟器状态"""
        self.trajectory_buffer.clear()
        self.simulation_count.fill_(0)


class DreamCortex(nn.Module):
    """
    梦想-皮层系统 - 离线想象和创意生成
    
    模拟大脑在休息/睡眠期间的创意生成和因果推演
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.simulator = OfflineSimulator(hidden_dim)
        
        self.creative_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.novelty_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
    def dream(
        self,
        model: nn.Module,
        initial_states: List[torch.Tensor],
        num_dreams: int = 10,
    ) -> List[ImaginedTrajectory]:
        """
        做梦 - 从多个初始状态生成想象轨迹
        
        用于：
        1. 创意问题解决
        2. 计划模拟
        3. 因果结构发现
        """
        all_imagined = []
        
        for initial_state in initial_states:
            imagined = self.simulator.imagine_alternatives(
                model, initial_state,
                num_alternatives=num_dreams // len(initial_states),
            )
            all_imagined.extend(imagined)
            
        return all_imagined
        
    def counterfactual_dream(
        self,
        model: nn.Module,
        episode_states: torch.Tensor,
        intervention_fn: Callable,
    ) -> CounterfactualResult:
        """
        反事实做梦 - 从记忆片段进行反事实推理
        
        将情景记忆片段作为想象起点
        """
        initial_state = episode_states[0]
        
        result = self.simulator.counterfactual_rollout(
            model, initial_state, intervention_fn,
            num_steps=episode_states.size(0),
        )
        
        return result
        
    def creative_blend(
        self,
        trajectory_a: torch.Tensor,
        trajectory_b: torch.Tensor,
        blend_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        创意混合 - 结合两个轨迹创造新轨迹
        
        模拟人类的"创造性联想"
        """
        min_len = min(trajectory_a.size(0), trajectory_b.size(0))
        
        blended = trajectory_a[:min_len] * blend_ratio + trajectory_b[:min_len] * (1 - blend_ratio)
        
        return blended
        
    def forward(
        self,
        model: nn.Module,
        state: torch.Tensor,
        mode: str = 'dream',
    ) -> Dict[str, torch.Tensor]:
        """
        主前向传播
        """
        if mode == 'dream':
            imagined = self.simulator.imagine_alternatives(model, state)
            return {'imagined_trajectories': imagined}
        elif mode == 'counterfactual':
            result = self.simulator.counterfactual_rollout(
                model, state, lambda s, t: s + 0.5
            )
            return {
                'factual': result.factual_trajectory,
                'counterfactual': result.counterfactual_trajectory,
                'effect': result.causal_effect,
            }
        else:
            return {}


class CausalReasoningEngine(nn.Module):
    """
    因果推理引擎 - 整合所有组件
    
    提供统一的因果推理接口：
    1. 干预 (do)
    2. 反事实 (what if)
    3. 想象 (what could be)
    4. 因果发现 (what causes what)
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.simulator = OfflineSimulator(hidden_dim)
        self.dream_cortex = DreamCortex(hidden_dim)
        
        self.causal_strength_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
    def do_intervene(
        self,
        model: nn.Module,
        state: torch.Tensor,
        intervention_targets: List[int],
        intervention_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行干预并返回结果
        """
        intervened_state = state.clone()
        intervened_state[:, intervention_targets] = intervention_values
        
        result_state, trajectory = self.simulator.simulate_without_learning(
            model, intervened_state, num_steps=10
        )
        
        return result_state, trajectory
        
    def what_if(
        self,
        model: nn.Module,
        state: torch.Tensor,
        intervention_fn: Callable,
    ) -> CounterfactualResult:
        """
        反事实查询：what if ...?
        """
        result = self.simulator.counterfactual_rollout(
            model, state, intervention_fn,
            num_steps=15,
        )
        
        return result
        
    def why_because(
        self,
        model: nn.Module,
        state: torch.Tensor,
        outcome: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        因果归因：为什么是这样？
        
        返回最可能的因果路径和置信度
        """
        original_output, _ = self.simulator.simulate_without_learning(
            model, state, num_steps=1
        )
        
        diff = outcome - original_output
        
        causal_strength = self.causal_strength_net(
            torch.cat([state, diff], dim=-1)
        )
        
        return diff, causal_strength.mean().item()
        
    def what_causes(
        self,
        model: nn.Module,
        state: torch.Tensor,
        target_idx: int,
        num_tests: int = 20,
    ) -> Tuple[List[int], List[float]]:
        """
        因果发现：什么导致了这个变化？
        
        通过扰动每个维度来评估其对目标的影响
        """
        original_output, _ = self.simulator.simulate_without_learning(
            model, state, num_steps=5
        )
        
        effects = []
        
        for dim in range(min(target_idx + 10, self.hidden_dim)):
            perturbed = state.clone()
            perturbed[:, dim] += 1.0
            
            perturbed_output, _ = self.simulator.simulate_without_learning(
                model, perturbed, num_steps=5
            )
            
            effect = torch.norm(perturbed_output - original_output).item()
            effects.append(effect)
            
        top_indices = np.argsort(effects)[::-1][:5]
        
        return top_indices.tolist(), [effects[i] for i in top_indices]
