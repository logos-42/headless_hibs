"""
因果学习信号模块 - Causal Learning Signal Module
=================================================
类多巴胺的奖励预测误差系统，用于更新因果知识：

1. TD误差计算 - 奖励预测误差
2. 因果权重更新 - 基于TD误差更新连接强度
3. 神经调节信号 - 类多巴胺/血清素的全局调节
4. 可塑性调制 - 活动依赖的塑性阈值变化

核心思想：学习"什么导致了什么"需要错误驱动的信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class RewardSignal:
    """奖励信号"""
    extrinsic: torch.Tensor  # 外部环境奖励
    intrinsic: torch.Tensor  # 内在好奇心奖励
    social: torch.Tensor  # 社会奖励（如有）
    total: torch.Tensor  # 总奖励
    prediction: torch.Tensor  # 奖励预测
    td_error: torch.Tensor  # TD误差
    
    
@dataclass
class NeuromodulatorySignal:
    """神经调节信号"""
    dopamine: torch.Tensor  # 类多巴胺 - 奖励预测误差
    serotonin: torch.Tensor  # 类血清素 - 情感调节
    acetylcholine: torch.Tensor  # 类乙酰胆碱 - 注意力/可塑性
    norepinephrine: torch.Tensor  # 类去甲肾上腺素 - 唤醒/注意
    

@dataclass
class PlasticitySignal:
    """可塑性信号"""
    eligibility_trace: torch.Tensor  # 资格痕迹
    modulation: torch.Tensor  # 调节信号
    threshold: torch.Tensor  # 可塑性阈值


class EligibilityTrace(nn.Module):
    """
    资格痕迹 - 关联时序近的刺激和奖励
    
    关键：不是每个激活都导致学习，只有被奖励加强的激活才能学习
    """
    
    def __init__(self, hidden_dim: int, trace_decay: float = 0.9):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.trace_decay = trace_decay
        
        self.traces: Optional[torch.Tensor] = None
        
    def update(
        self,
        pre_synaptic: torch.Tensor,
        post_synaptic: torch.Tensor,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """
        更新资格痕迹
        
        e = decay * e + pre * post
        """
        if self.traces is None:
            self.traces = torch.zeros_like(pre_synaptic)
            
        new_trace = (
            self.trace_decay * self.traces +
            torch.outer(pre_synaptic, post_synaptic)
        )
        
        self.traces = new_trace
        
        return self.traces
        
    def decay(self):
        """衰减资格痕迹"""
        if self.traces is not None:
            self.traces = self.trace_decay * self.traces
            
    def reset(self):
        """重置"""
        self.traces = None


class TemporalDifferenceError(nn.Module):
    """
    时序差分误差计算器 - TD误差
    
    TD误差 = r_t + γ * V(s_{t+1}) - V(s_t)
    
    这就是类多巴胺信号
    """
    
    def __init__(
        self,
        hidden_dim: int,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.register_buffer('previous_value', torch.tensor(0.0))
        self.register_buffer('previous_state', torch.zeros(hidden_dim))
        
    def compute(
        self,
        state: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算TD误差
        
        Returns:
            td_error: (batch,) TD误差
            value: 当前状态价值
            next_value: 下一状态价值
        """
        value = self.value_net(state).squeeze(-1)
        next_value = self.value_net(next_state).squeeze(-1)
        
        target = reward + self.gamma * next_value.detach()
        
        td_error = target - value.detach()
        
        self.previous_value = value.detach()
        self.previous_state = state.detach()
        
        return td_error, value, next_value


class DopamineSignal(nn.Module):
    """
    类多巴胺信号 - 奖励预测误差的核心传达者
    
    功能：
    1. 计算TD误差
    2. 调制学习强度
    3. 决定哪些连接应该加强/减弱
    """
    
    def __init__(
        self,
        hidden_dim: int,
        dopamine_scale: float = 1.0,
        tau_dopamine: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dopamine_scale = dopamine_scale
        self.tau_dopamine = tau_dopamine
        
        self.td_error_calc = TemporalDifferenceError(hidden_dim)
        
        self.dopamine_trace: List[float] = []
        
        self.momentum = nn.Parameter(torch.tensor(0.0))
        
    def forward(
        self,
        state: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播计算多巴胺信号
        """
        td_error, value, next_value = self.td_error_calc.compute(
            state, reward, next_state
        )
        
        dopamine = torch.clamp(
            td_error * self.dopamine_scale,
            -2.0, 2.0
        )
        
        self.dopamine_trace.append(dopamine.mean().item())
        if len(self.dopamine_trace) > 1000:
            self.dopamine_trace.pop(0)
            
        info = {
            'td_error': td_error,
            'dopamine': dopamine,
            'value': value,
            'next_value': next_value,
        }
        
        return dopamine, info


class NeuromodulatorySystem(nn.Module):
    """
    神经调节系统 - 整合多种调节信号
    
    类比生物神经递质系统：
    - 多巴胺：奖励预测误差
    - 血清素：情感/价值持久性
    - 乙酰胆碱：注意力调节可塑性
    - 去甲肾上腺素：唤醒和 novelty检测
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.dopamine = DopamineSignal(hidden_dim)
        
        self.serotonin_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.acetylcholine_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
        self.norepinephrine_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.register_buffer(' novelty_bonus', torch.tensor(0.0))
        
    def forward(
        self,
        state: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        attention: Optional[torch.Tensor] = None,
    ) -> NeuromodulatorySignal:
        """
        计算所有神经调节信号
        """
        dopamine, dopamine_info = self.dopamine(state, reward, next_state)
        
        serotonin = self.serotonin_net(state).squeeze(-1)
        
        if attention is not None:
            acetylcholine = self.acetylcholine_net(
                torch.cat([state, attention], dim=-1)
            ).squeeze(-1)
        else:
            acetylcholine = torch.ones_like(serotonin) * 0.5
            
        norepinephrine = self.norepinephrine_net(state).squeeze(-1)
        
        return NeuromodulatorySignal(
            dopamine=dopamine,
            serotonin=serotonin,
            acetylcholine=acetylcholine,
            norepinephrine=norepinephrine,
        )


class MetaplasticityModule(nn.Module):
    """
    可塑性阈值调节模块 - 类比Bienenstock-Cooper-Munro (BCM) 规则
    
    关键思想：可塑性阈值不是固定的，而是根据使用情况动态调整
    频繁使用的连接变得难以改变（欣快感）
    """
    
    def __init__(
        self,
        hidden_dim: int,
        theta_init: float = 0.0,
        theta_lr: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.theta = nn.Parameter(torch.ones(hidden_dim) * theta_init)
        self.theta_lr = theta_lr
        
        self.activity_history: List[torch.Tensor] = []
        
    def compute_plasticity(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        neuromod: NeuromodulatorySignal,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算可塑性调整后的权重变化
        
        Δw = η * (post - theta) * pre * dopamine
        
        如果post > theta（活跃），可塑性增强
        如果post < theta（不活跃），可塑性减弱
        """
        co_activity = pre.unsqueeze(-1) * post.unsqueeze(0)
        
        post_mean = post.mean()
        
        plasticity_signal = F.relu(post_mean - self.theta)
        
        modulated_plasticity = (
            plasticity_signal *
            neuromod.dopamine.unsqueeze(-1) *
            neuromod.acetylcholine.unsqueeze(-1)
        )
        
        weight_change = modulated_plasticity * co_activity
        
        return weight_change, modulated_plasticity
        
    def update_threshold(
        self,
        post_activity: torch.Tensor,
        dopamine: torch.Tensor,
    ):
        """
        更新可塑性阈值
        
        theta = theta + lr * (post^2 - theta)
        """
        post_power = torch.mean(post_activity ** 2)
        
        theta_update = self.theta_lr * (post_power.detach() - self.theta.detach())
        
        self.theta.data = self.theta + theta_update * dopamine.mean()
        
        self.theta.data = torch.clamp(self.theta.data, -2.0, 2.0)


class CausalWeightUpdater(nn.Module):
    """
    因果权重更新器 - 基于TD误差更新连接强度
    
    核心：W_ij的强度反映i对j的因果影响力
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.eligibility_traces = EligibilityTrace(hidden_dim)
        
        self.learning_rate = nn.Parameter(torch.tensor(0.01))
        
        self.weight_scale = nn.Parameter(torch.tensor(1.0))
        
    def update_weights(
        self,
        weights: torch.Tensor,
        pre_synaptic: torch.Tensor,
        post_synaptic: torch.Tensor,
        dopamine: torch.Tensor,
    ) -> torch.Tensor:
        """
        更新权重矩阵
        
        ΔW = lr * dopamine * eligibility_trace
        """
        trace = self.eligibility_traces.update(
            pre_synaptic, post_synaptic
        )
        
        weight_delta = (
            self.learning_rate *
            dopamine.unsqueeze(-1).unsqueeze(-1) *
            trace.unsqueeze(0) *
            self.weight_scale
        )
        
        new_weights = weights + weight_delta
        
        new_weights = torch.clamp(new_weights, -5.0, 5.0)
        
        return new_weights


class CausalLearningSignal(nn.Module):
    """
    因果学习信号模块 - 整合所有组件
    
    主接口：
    1. 计算奖励和TD误差
    2. 生成神经调节信号
    3. 更新因果权重
    4. 调节可塑性阈值
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.neuromodulatory = NeuromodulatorySystem(hidden_dim)
        self.metaplasticity = MetaplasticityModule(hidden_dim)
        self.causal_updater = CausalWeightUpdater(hidden_dim)
        
        self.intrinsic_reward_weight = nn.Parameter(torch.tensor(0.5))
        self.extrinsic_reward_weight = nn.Parameter(torch.tensor(0.5))
        
        self.register_buffer('update_count', torch.tensor(0))
        
    def forward(
        self,
        state: torch.Tensor,
        extrinsic_reward: torch.Tensor,
        intrinsic_reward: torch.Tensor,
        next_state: torch.Tensor,
        weights: torch.Tensor,
        pre_synaptic: torch.Tensor,
        post_synaptic: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        主前向传播
        
        Returns:
            updated_weights: 更新后的权重
            info: 学习信号信息
        """
        total_reward = (
            self.extrinsic_reward_weight * extrinsic_reward +
            self.intrinsic_reward_weight * intrinsic_reward
        )
        
        neuromod = self.neuromodulatory(
            state, total_reward, next_state
        )
        
        weight_change, plasticity = self.metaplasticity.compute_plasticity(
            pre_synaptic, post_synaptic, neuromod
        )
        
        updated_weights = self.causal_updater.update_weights(
            weights, pre_synaptic, post_synaptic, neuromod.dopamine
        )
        
        self.metaplasticity.update_threshold(
            post_synaptic, neuromod.dopamine
        )
        
        self.eligibility_traces.decay()
        
        self.update_count += 1
        
        info = {
            'dopamine': neuromod.dopamine,
            'td_error': neuromod.dopamine,
            'serotonin': neuromod.serotonin,
            'acetylcholine': neuromod.acetylcholine,
            'norepinephrine': neuromod.norepinephrine,
            'plasticity': plasticity,
            'threshold': self.metaplasticity.theta,
            'update_count': self.update_count.item(),
        }
        
        return updated_weights, info
        
    def compute_reward_signal(
        self,
        extrinsic: torch.Tensor,
        intrinsic: torch.Tensor,
        predicted: torch.Tensor,
    ) -> RewardSignal:
        """
        计算完整奖励信号
        """
        total = (
            self.extrinsic_reward_weight * extrinsic +
            self.intrinsic_reward_weight * intrinsic
        )
        
        td_error = total - predicted.detach()
        
        return RewardSignal(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            social=torch.zeros_like(total),
            total=total,
            prediction=predicted,
            td_error=td_error,
        )
        
    def reset(self):
        """重置模块状态"""
        self.eligibility_traces.reset()
        self.update_count.fill_(0)


class CausalKnowledgeConsolidator(nn.Module):
    """
    因果知识整合器 - 跨经验整合因果结构
    
    模拟记忆巩固过程：
    1. 睡眠期间重演重要记忆
    2. 将短时因果知识整合到长期
    3. 抽象出更高层的因果规则
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self因果_rule_extractor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.long_term_memory: Dict[str, torch.Tensor] = {}
        
        self.consolidation_rate = 0.05
        
    def consolidate(
        self,
        short_term_causal_knowledge: Dict[str, torch.Tensor],
        replayed_trajectories: List[torch.Tensor],
    ):
        """
        整合因果知识
        """
        for trajectory in replayed_trajectories:
            for t in range(trajectory.size(0) - 1):
                state_pair = torch.cat([
                    trajectory[t],
                    trajectory[t + 1]
                ], dim=-1)
                
                causal_strength = self因果_rule_extractor(state_pair)
                
                key = f"rule_{t}_{hash(str(state_pair[:5].tolist()))}"
                
                if key in self.long_term_memory:
                    self.long_term_memory[key] = (
                        (1 - self.consolidation_rate) * self.long_term_memory[key] +
                        self.consolidation_rate * causal_strength.detach()
                    )
                else:
                    self.long_term_memory[key] = causal_strength.detach()
                    
    def retrieve_causal_rules(
        self,
        query_state: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        检索相关因果规则
        """
        similarities = []
        
        for key, rule in self.long_term_memory.items():
            sim = F.cosine_similarity(
                query_state.unsqueeze(0),
                rule.unsqueeze(0)
            ).item()
            similarities.append((key, sim))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
