"""
干预执行器 - Intervention Executor
===================================
实现因果推理中的 do-演算语义：

1. 硬干预 (Hard Intervention): do(X = x₀) - 强制变量到某值，完全覆盖动力学
2. 软干预 (Soft Intervention): do(X := f(X)) - 修改变量更新规则
3. 机制干预 (Mechanism Intervention): 修改ODE中的特定项

支持：
- 状态干预 do(Z)
- 时间常数干预 do(τ)
- 连接权重干预 do(W)
- 组合干预

核心思想：干预不是修改系统，而是"接管"系统的控制权
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass, field
import copy


@dataclass
class InterventionTarget:
    """干预目标定义"""
    target_type: str  # 'z', 'tau', 'w', 'connection'
    indices: List[int]  # 目标索引
    value: Optional[torch.Tensor] = None
    modulator: Optional[torch.Tensor] = None
    
    
@dataclass
class InterventionResult:
    """干预执行结果"""
    original_state: torch.Tensor
    intervened_state: torch.Tensor
    intervention_cost: torch.Tensor
    causal_effect: torch.Tensor
    success: bool
    
    
@dataclass
class DoIntervention:
    """do-干预语义"""
    intervention_type: str  # 'hard', 'soft', 'modulate'
    target: InterventionTarget
    duration: int = 1  # 干预持续时间步数
    reversible: bool = True  # 是否可逆
    
    
class InterventionCostEstimator(nn.Module):
    """
    干预成本估计器
    评估干预的"侵入性"作为正则化项
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.cost_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        original: torch.Tensor,
        intervened: torch.Tensor,
        target_type: str,
    ) -> torch.Tensor:
        """
        计算干预成本
        
        成本 = ||original - intervened||² + 干预侵入性惩罚
        """
        state_diff = torch.norm(original - intervened, dim=-1)
        
        intervention_intrusion = self.cost_net(intervened).squeeze(-1)
        
        if target_type == 'hard':
            intrusion_penalty = intervention_intrusion * 2.0
        elif target_type == 'soft':
            intrusion_penalty = intervention_intrusion * 0.5
        else:
            intrusion_penalty = intervention_intrusion * 1.0
            
        cost = state_diff + intrusion_penalty
        
        return cost


class MechanismModifier(nn.Module):
    """
    机制修改器 - 修改ODE中的特定项
    用于软干预和机制干预
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.modulation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
    def forward(
        self,
        original_term: torch.Tensor,
        state: torch.Tensor,
        modulation_signal: torch.Tensor,
    ) -> torch.Tensor:
        """
        调制原始项
        
        modified = original * (1 + modulation_signal)
        """
        modulation = self.modulation_net(torch.cat([state, modulation_signal], dim=-1))
        
        modified = original_term * (1.0 + modulation)
        
        return modified


class InterventionExecutor(nn.Module):
    """
    干预执行器 - 核心do-演算实现
    
    支持的干预类型:
    1. do(Z_i = z₀) - 硬干预状态维度
    2. do(τ_i = τ₀) - 硬干预时间常数
    3. do(W_ij = w₀) - 硬干预连接权重
    4. do(Z_i := f(Z)) - 软干预状态更新
    5. do_mechanism(term) - 机制干预ODE项
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.cost_estimator = InterventionCostEstimator(hidden_dim)
        self.mechanism_modifier = MechanismModifier(hidden_dim)
        
        self.intervention_history: List[InterventionResult] = []
        
        self.register_buffer('intervention_count', torch.tensor(0))
        self.register_buffer('total_cost', torch.tensor(0.0))
        
    def do_hard_z(
        self,
        state: torch.Tensor,
        target_idx: int,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        硬干预状态: do(Z_target = value)
        
        完全覆盖目标维度的值，绕过ODE一步
        """
        intervened = state.clone()
        intervened[:, target_idx] = value
        
        cost = self.cost_estimator(state, intervened, 'hard')
        
        causal_effect = intervened - state
        
        self.intervention_count += 1
        self.total_cost += cost.mean().detach()
        
        return intervened, cost, causal_effect
        
    def do_hard_tau(
        self,
        tau: torch.Tensor,
        target_idx: int,
        value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        硬干预时间常数: do(τ_target = value)
        
        直接修改特定神经元的时间常数
        """
        intervened = tau.clone()
        intervened[:, target_idx] = value
        
        cost = torch.abs(tau - intervened).mean(dim=-1)
        
        return intervened, cost
        
    def do_hard_w(
        self,
        weight_matrix: torch.Tensor,
        source_idx: int,
        target_idx: int,
        value: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        硬干预连接权重: do(W_source_target = value)
        
        直接修改连接矩阵中的特定权重
        """
        intervened = weight_matrix.clone()
        intervened[source_idx, target_idx] = value
        
        cost = torch.abs(weight_matrix[source_idx, target_idx] - value)
        
        return intervened, cost
        
    def do_soft_z(
        self,
        state: torch.Tensor,
        target_indices: List[int],
        modulator: torch.Tensor,
        strength: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        软干预状态: do(Z_target := Z + modulator)
        
        通过添加调制信号影响状态，而不是覆盖
        """
        intervened = state.clone()
        
        modulator_signal = torch.tanh(modulator) * strength
        
        intervened[:, target_indices] += modulator_signal[:, target_indices]
        
        cost = self.cost_estimator(state, intervened, 'soft')
        
        causal_effect = intervened - state
        
        return intervened, cost, causal_effect
        
    def do_mechanism_z(
        self,
        dzdt: torch.Tensor,
        state: torch.Tensor,
        target_indices: List[int],
        modulation_signal: torch.Tensor,
    ) -> torch.Tensor:
        """
        机制干预ODE的dZ/dt项
        
        修改状态导数，而不是状态本身
        """
        modified_dzdt = self.mechanism_modifier(
            dzdt, state, modulation_signal
        )
        
        mask = torch.zeros_like(dzdt)
        mask[:, target_indices] = 1.0
        
        result = dzdt * (1 - mask) + modified_dzdt * mask
        
        return result
        
    def do_connection_cut(
        self,
        weight_matrix: torch.Tensor,
        source_idx: int,
        target_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        切断连接: do(W_ij = 0)
        
        心理学等价于"消除因果效应"
        """
        intervened, cost = self.do_hard_w(
            weight_matrix, source_idx, target_idx, 0.0
        )
        
        return intervened, cost
        
    def do_connection_strengthen(
        self,
        weight_matrix: torch.Tensor,
        source_idx: int,
        target_idx: int,
        factor: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        强化连接: do(W_ij = W_ij * factor)
        """
        intervened = weight_matrix.clone()
        intervened[source_idx, target_idx] *= factor
        
        cost = torch.abs(weight_matrix[source_idx, target_idx] * (factor - 1))
        
        return intervened, cost
        
    def execute_intervention(
        self,
        state: torch.Tensor,
        intervention: DoIntervention,
        current_tau: Optional[torch.Tensor] = None,
        current_w: Optional[torch.Tensor] = None,
    ) -> InterventionResult:
        """
        执行干预的主接口
        
        根据干预类型调用相应的do操作
        """
        original_state = state.clone()
        
        target = intervention.target
        
        if intervention.intervention_type == 'hard':
            if target.target_type == 'z':
                intervened, cost, causal_effect = self.do_hard_z(
                    state, target.indices[0], target.value
                )
            elif target.target_type == 'tau' and current_tau is not None:
                intervened, cost = self.do_hard_tau(
                    current_tau, target.indices[0], target.value.item()
                )
                causal_effect = current_tau - intervened
            else:
                intervened = state
                cost = torch.zeros(state.size(0), device=state.device)
                causal_effect = torch.zeros_like(state)
        elif intervention.intervention_type == 'soft':
            intervened, cost, causal_effect = self.do_soft_z(
                state, target.indices, target.modulator
            )
        else:
            intervened = state
            cost = torch.zeros(state.size(0), device=state.device)
            causal_effect = torch.zeros_like(state)
            
        result = InterventionResult(
            original_state=original_state,
            intervened_state=intervened,
            intervention_cost=cost,
            causal_effect=causal_effect,
            success=True,
        )
        
        self.intervention_history.append(result)
        
        return result
        
    def compute_causal_effect(
        self,
        factual_traj: torch.Tensor,
        counterfactual_traj: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算因果效应 = E[Y_t | do(X)] - E[Y]
        
        用于评估干预的效果
        """
        effect = counterfactual_traj.mean(dim=0) - factual_traj.mean(dim=0)
        
        return effect
        
    def get_intervention_statistics(self) -> Dict:
        """获取干预统计信息"""
        return {
            'total_interventions': self.intervention_count.item(),
            'average_cost': (self.total_cost / max(self.intervention_count.item(), 1)),
            'history_length': len(self.intervention_history),
        }
        
    def reset(self):
        """重置干预执行器状态"""
        self.intervention_history.clear()
        self.intervention_count.fill_(0)
        self.total_cost.fill_(0)


class CausalInterventionEngine(nn.Module):
    """
    因果干预引擎 - 高级接口
    
    整合干预执行、因果效应评估、反事实推理
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.executor = InterventionExecutor(hidden_dim)
        
    def do(
        self,
        state: torch.Tensor,
        target_type: str,
        target_indices: List[int],
        value: Optional[torch.Tensor] = None,
        intervention_type: str = 'hard',
        modulator: Optional[torch.Tensor] = None,
    ) -> InterventionResult:
        """
        简洁的do-操作接口
        
        Usage:
            intervened_state, cost = engine.do(
                state, 'z', [0], value=target_value
            )
        """
        target = InterventionTarget(
            target_type=target_type,
            indices=target_indices,
            value=value,
            modulator=modulator,
        )
        
        intervention = DoIntervention(
            intervention_type=intervention_type,
            target=target,
        )
        
        result = self.executor.execute_intervention(
            state, intervention
        )
        
        return result
        
    def counterfactual(
        self,
        factual_state: torch.Tensor,
        intervention: DoIntervention,
        simulate_fn: Callable,
        num_rollouts: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        反事实推理
        
        1. 假设干预已发生（cloned state）
        2. 模拟干预后的轨迹
        3. 返回反事实结果和因果效应
        """
        cf_state = factual_state.clone()
        
        rollouts = []
        for _ in range(num_rollouts):
            result = self.executor.execute_intervention(
                cf_state, intervention
            )
            
            next_state = simulate_fn(result.intervened_state)
            
            rollouts.append(next_state)
            
        cf_trajectory = torch.stack(rollouts, dim=0)
        
        factual_next = simulate_fn(factual_state)
        
        causal_effect = cf_trajectory.mean(dim=0) - factual_next
        
        return cf_trajectory, causal_effect
        
    def abduction_action_prediction(
        self,
        observed_outcome: torch.Tensor,
        state: torch.Tensor,
        intervention: DoIntervention,
        simulate_fn: Callable,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pearl的因果推理三层:
        1. Abduction: 推断潜在变量/背景因素
        2. Action: 应用干预
        3. Prediction: 预测结果
        
        Returns:
            latent_factors, cf_outcome, causal_effect
        """
        abduction_residual = observed_outcome - simulate_fn(state)
        
        modified_state = state + 0.1 * abduction_residual
        
        cf_result = self.executor.execute_intervention(
            modified_state, intervention
        )
        
        cf_outcome = simulate_fn(cf_result.intervened_state)
        
        factual_outcome = simulate_fn(state)
        
        causal_effect = cf_outcome - factual_outcome
        
        return abduction_residual, cf_outcome, causal_effect
