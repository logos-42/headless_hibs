"""
内在动机模块 - Curiosity-Driven Intervention System
===================================================
生物启发的自主探索系统：
1. 信息增益驱动的好奇心 - 优先探索不确定性高的区域
2. 内在奖励信号 - 类多巴胺的奖励预测误差
3. 干预选择机制 - 基于价值选择干预点

核心思想：agent不是因为外部奖励而学习，而是因为"想要知道"而探索
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import math


@dataclass
class CuriosityConfig:
    """内在动机配置"""
    hidden_dim: int = 128
    memory_dim: int = 64
    
    curiosity_weight: float = 1.0
    entropy_weight: float = 0.1
    novelty_weight: float = 0.5
    
    uncertainty_threshold: float = 0.5
    novelty_threshold: float = 0.3
    
    exploration_bonus: float = 0.1
    intervention_cost_weight: float = 0.01
    
    prediction_horizon: int = 5
    ensemble_size: int = 3
    
    forget_rate: float = 0.01
    consolidation_rate: float = 0.05
    
    
@dataclass
class InterventionChoice:
    """干预选择"""
    target_type: str  # 'z', 'tau', 'w', 'connection'
    target_indices: List[int]
    intervention_value: torch.Tensor
    expected_information_gain: float
    expectedNovelty: float
    urgency: float
    confidence: float


class PredictiveModel(nn.Module):
    """
    预测模型 - 用于评估信息增益
    给定当前状态，预测干预后的结果
    """
    
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        self.transition = nn.GRU(
            hidden_dim, hidden_dim, num_layers=2, batch_first=True
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
    def forward(
        self, 
        current_state: torch.Tensor, 
        intervention: torch.Tensor,
        horizon: int = 5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测干预后的未来状态
        
        Args:
            current_state: (batch, hidden_dim) 当前状态
            intervention: (batch, hidden_dim) 干预向量
            horizon: 预测步数
            
        Returns:
            predicted_trajectory: (batch, horizon, hidden_dim)
            uncertainty: (batch, horizon) 预测不确定性
        """
        batch = current_state.size(0)
        
        state = current_state + intervention
        trajectory = [state]
        
        hidden = None
        for step in range(horizon):
            rnn_input = state.unsqueeze(1)
            output, hidden = self.transition(rnn_input, hidden)
            state = output.squeeze(1)
            trajectory.append(state)
        
        trajectory = torch.stack(trajectory, dim=1)
        
        predictions = self.decoder(trajectory)
        
        prediction_variance = torch.var(predictions, dim=1)
        
        return trajectory, prediction_variance


class UncertaintyEstimator(nn.Module):
    """
    不确定性估计器 - 评估状态/预测的不确定性
    使用dropout近似贝叶斯不确定性
    """
    
    def __init__(self, hidden_dim: int, num_samples: int = 10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        
        self.estimator_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        估计状态的不确定性
        
        Returns:
            uncertainty: (batch, hidden_dim) 每个维度的不确定性
            total_uncertainty: (batch,) 总体不确定性
        """
        samples = []
        for _ in range(self.num_samples):
            sample = self.estimator_net(state)
            samples.append(sample)
        
        samples = torch.stack(samples, dim=0)
        
        uncertainty = torch.var(samples, dim=0).squeeze(-1)
        total_uncertainty = torch.mean(uncertainty, dim=-1)
        
        return uncertainty, total_uncertainty


class NoveltyDetector(nn.Module):
    """
    新奇性检测器 - 检测与过去经验的差异
    基于情景记忆的熟悉度评估
    """
    
    def __init__(self, hidden_dim: int, memory_size: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.memory = None
        self.memory_ptr = 0
        self.memory_full = False
        
        self.novelty_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
    def init_memory(self, device):
        """初始化记忆缓冲区"""
        if self.memory is None:
            self.memory = torch.zeros(self.memory_size, self.hidden_dim, device=device)
            self.memory_ptr = 0
            self.memory_full = False
            
    def store(self, states: torch.Tensor):
        """存储状态到记忆"""
        batch_size = states.size(0)
        encoded = self.encoder(states)
        
        for i in range(batch_size):
            idx = self.memory_ptr % self.memory_size
            self.memory[idx] = encoded[i].detach()
            self.memory_ptr += 1
            if self.memory_ptr >= self.memory_size:
                self.memory_full = True
                
    def compute_novelty(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算新奇性分数
        
        Returns:
            novelty_scores: (batch, hidden_dim) 每个维度的新奇性
            total_novelty: (batch,) 总体新奇性
        """
        if self.memory_full or self.memory_ptr > 0:
            encoded = self.encoder(states)
            
            memory_valid = self.memory[:self.memory_ptr] if not self.memory_full else self.memory
            
            similarities = torch.matmul(encoded, memory_valid.T)
            
            max_similarity = torch.max(similarities, dim=-1)[0]
            
            novelty_scores = 1.0 - max_similarity
            
            total_novelty = torch.mean(novelty_scores, dim=-1)
        else:
            novelty_scores = torch.ones_like(states)
            total_novelty = torch.ones(states.size(0), device=states.device)
            
        return novelty_scores, total_novelty


class IntrinsicRewardCalculator(nn.Module):
    """
    内在奖励计算器 - 计算类多巴胺的奖励信号
    结合信息增益、好奇心和预测误差
    """
    
    def __init__(self, hidden_dim: int, config: CuriosityConfig):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        
        self.info_gain_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.curiosity_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.reward_scale = nn.Parameter(torch.tensor(1.0))
        
    def compute_intrinsic_reward(
        self,
        state: torch.Tensor,
        predicted_state: torch.Tensor,
        actual_state: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算内在奖励 = 信息增益 + 好奇心奖励 + 探索奖励
        
        Returns:
            total_reward: (batch,)
            reward_components: 奖励分解
        """
        info_gain_input = torch.cat([state, actual_state - predicted_state], dim=-1)
        info_gain = self.info_gain_net(info_gain_input).squeeze(-1)
        
        curiosity_input = torch.cat([state, uncertainty], dim=-1)
        curiosity_reward = self.curiosity_net(curiosity_input).squeeze(-1)
        
        prediction_error = torch.norm(actual_state - predicted_state, dim=-1)
        
        novelty_reward = novelty
        
        total_reward = (
            self.config.curiosity_weight * curiosity_reward +
            self.config.novelty_weight * novelty_reward +
            self.config.entropy_weight * prediction_error
        )
        
        components = {
            'curiosity': curiosity_reward,
            'novelty': novelty_reward,
            'prediction_error': prediction_error,
            'info_gain': info_gain,
            'total': total_reward,
        }
        
        return total_reward, components


class InterventionSelector(nn.Module):
    """
    干预选择器 - 基于价值选择干预点和干预方式
    评估每个可能干预的信息增益，选择最优干预
    """
    
    def __init__(self, hidden_dim: int, config: CuriosityConfig):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config
        
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.value_head_z = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.value_head_tau = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.value_head_connection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.cost_estimator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
    def evaluate_z_intervention(
        self, 
        state: torch.Tensor,
        target_indices: List[int],
        intervention_value: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估Z状态干预的价值
        """
        encoded = self.state_encoder(state)
        
        target_mask = torch.zeros_like(state)
        target_mask[:, target_indices] = 1.0
        
        intervention_impact = encoded * target_mask
        
        value = self.value_head_z(intervention_impact).squeeze(-1)
        
        intervention_cost = self.cost_estimator(torch.abs(intervention_value)).squeeze(-1)
        
        urgency = uncertainty.mean(dim=-1)
        
        net_value = (
            value + 
            self.config.novelty_threshold * novelty +
            self.config.exploration_bonus * urgency -
            self.config.intervention_cost_weight * intervention_cost
        )
        
        return net_value, value
        
    def select_intervention(
        self,
        state: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: torch.Tensor,
        num_candidates: int = 10,
    ) -> InterventionChoice:
        """
        选择最优干预
        
        基于不确定性高和新奇性高的区域选择干预点
        """
        batch_size = state.size(0)
        device = state.device
        
        uncertainty_scores = uncertainty.mean(dim=-1)
        novelty_scores = novelty.mean(dim=-1)
        
        combined_scores = uncertainty_scores + novelty_scores
        
        k = min(num_candidates, self.hidden_dim, combined_scores.numel())
        _, top_indices = torch.topk(combined_scores, k)
        
        if top_indices.dim() == 0:
            target_idx = top_indices.item()
        else:
            target_idx = top_indices[0].item()
        
        intervention_direction = torch.randn(batch_size, 1, device=device)
        intervention_direction = torch.tanh(intervention_direction)
        
        intervention_value = intervention_direction * 2.0
        
        expected_value, base_value = self.evaluate_z_intervention(
            state, [target_idx], intervention_value, uncertainty, novelty
        )
        
        choice = InterventionChoice(
            target_type='z',
            target_indices=[target_idx],
            intervention_value=intervention_value,
            expected_information_gain=expected_value.mean().item(),
            expectedNovelty=novelty_scores.mean().item(),
            urgency=uncertainty_scores.mean().item(),
            confidence=base_value.mean().item(),
        )
        
        return choice


class CuriosityModule(nn.Module):
    """
    内在动机模块 - 整合所有子模块
    
    工作流程:
    1. 估计当前状态的不确定性和新奇性
    2. 计算内在奖励
    3. 选择干预点
    4. 预测干预结果
    5. 更新预测误差信号（类多巴胺）
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[CuriosityConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or CuriosityConfig()
        self.config.hidden_dim = hidden_dim
        
        self.predictive_model = PredictiveModel(hidden_dim, hidden_dim)
        self.uncertainty_estimator = UncertaintyEstimator(hidden_dim)
        self.novelty_detector = NoveltyDetector(hidden_dim)
        self.intrinsic_reward = IntrinsicRewardCalculator(hidden_dim, self.config)
        self.intervention_selector = InterventionSelector(hidden_dim, self.config)
        
        self.register_buffer('baseline', torch.tensor(0.0))
        self.register_buffer('episodic_count', torch.tensor(0))
        
        self.dopamine_trace = []
        
    def init_memory(self, device):
        """初始化情景记忆"""
        self.novelty_detector.init_memory(device)
        
    def forward(
        self,
        state: torch.Tensor,
        intervention: Optional[torch.Tensor] = None,
        store_to_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        主 forward pass
        
        Args:
            state: (batch, hidden_dim) 当前状态
            intervention: 可选的预设干预
            store_to_memory: 是否存储到情景记忆
            
        Returns:
            result: 包含奖励、选择、不确定性等的字典
        """
        if store_to_memory:
            self.novelty_detector.store(state)
            
        uncertainty, total_uncertainty = self.uncertainty_estimator(state)
        novelty, total_novelty = self.novelty_detector.compute_novelty(state)
        
        if intervention is None:
            choice = self.intervention_selector.select_intervention(
                state, uncertainty, novelty
            )
            intervention = torch.zeros_like(state)
            intervention[:, choice.target_indices] = choice.intervention_value
            
        predicted_trajectory, prediction_variance = self.predictive_model(
            state, intervention, self.config.prediction_horizon
        )
        
        predicted_final = predicted_trajectory[:, -1, :]
        
        actual_next_state = state + intervention * 0.5
        
        intrinsic_reward, reward_components = self.intrinsic_reward.compute_intrinsic_reward(
            state, predicted_final, actual_next_state, uncertainty, novelty
        )
        
        self.update_baseline(intrinsic_reward.mean())
        
        td_error = intrinsic_reward - self.baseline.detach()
        
        result = {
            'intrinsic_reward': intrinsic_reward,
            'td_error': td_error,
            'reward_components': reward_components,
            'uncertainty': uncertainty,
            'total_uncertainty': total_uncertainty,
            'novelty': novelty,
            'total_novelty': total_novelty,
            'intervention': intervention,
            'choice': choice if intervention is None else None,
            'predicted_trajectory': predicted_trajectory,
            'prediction_variance': prediction_variance,
            'baseline': self.baseline,
        }
        
        return result
        
    def update_baseline(self, reward: torch.Tensor, momentum: float = 0.99):
        """更新奖励基线（EMA）"""
        self.baseline = momentum * self.baseline + (1 - momentum) * reward.detach()
        
    def compute_dopamine_signal(self, reward: torch.Tensor) -> torch.Tensor:
        """
        计算类多巴胺信号 = TD误差
        用于更新连接权重和价值函数
        """
        td_error = reward - self.baseline.detach()
        
        dopamine = torch.clamp(td_error, -1.0, 1.0)
        
        self.dopamine_trace.append(dopamine.mean().item())
        if len(self.dopamine_trace) > 1000:
            self.dopamine_trace.pop(0)
            
        return dopamine
        
    def reset(self):
        """重置模块状态"""
        self.baseline.fill_(0.0)
        self.episodic_count.fill_(0)
        self.dopamine_trace.clear()
