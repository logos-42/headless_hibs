"""
因果生长模块 - Causal Growth Module
=================================
增强版的生长模块，用神经调节信号替代固定的发育阶段：

1. 神经调节驱动的生长 - 多巴胺/血清素等信号调控生长决策
2. 好奇心激发的结构探索 - 基于信息增益决定生长位置
3. 可塑性依赖的修剪 - 活动依赖的突触修剪
4. 因果重要性评估 - 评估连接/神经元的因果重要性

替代原有的固定时间表发育阶段
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import copy

from .causal_learning_signal import CausalLearningSignal, NeuromodulatorySignal


@dataclass
class NeuromodulatedGrowthConfig:
    """神经调节生长配置"""
    hidden_dim: int = 128
    max_hidden_dim: int = 8192
    
    enable_neuromodulation: bool = True
    enable_curiosity_growth: bool = True
    enable_causal_importance: bool = True
    
    dopamine_growth_threshold: float = 0.3
    dopamine_prune_threshold: float = -0.3
    serotonin_persistence_bonus: float = 0.2
    
    growth_interval: int = 10
    prune_interval: int = 25
    
    curiosity_weight: float = 0.5
    importance_weight: float = 0.5
    
    prob_add_connection: float = 0.05
    prob_add_node: float = 0.03
    prob_disable_connection: float = 0.1
    
    consolidation_rate: float = 0.05
    decay_rate: float = 0.01
    survival_threshold: float = 0.15


@dataclass
class CausalImportanceScore:
    """因果重要性分数"""
    neuron_idx: int
    causal_influence: float
    information_gain: float
    neuromodulation_level: float
    total_score: float


class CausalImportanceEstimator(nn.Module):
    """
    因果重要性估计器
    
    评估每个神经元/连接的因果重要性
    基于其对输出的贡献和预测误差
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.importance_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        pre_synaptic: torch.Tensor,
        post_synaptic: torch.Tensor,
        activation: torch.Tensor,
    ) -> torch.Tensor:
        """
        评估神经元/连接的因果重要性
        
        Returns:
            importance: (hidden_dim,) 每个神经元的因果重要性
        """
        combined = torch.cat([
            pre_synaptic,
            activation,
        ], dim=-1)
        
        importance = self.importance_net(combined).squeeze(-1)
        
        return importance


class NeuromodulationDrivenGrowth(nn.Module):
    """
    神经调节驱动的生长系统
    
    基于神经调节信号（多巴胺、血清素等）动态调整：
    1. 生长速率
    2. 修剪阈值
    3. 可塑性水平
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[NeuromodulatedGrowthConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or NeuromodulatedGrowthConfig()
        self.config.hidden_dim = hidden_dim
        
        if self.config.enable_causal_importance:
            self.importance_estimator = CausalImportanceEstimator(hidden_dim)
            
        if self.config.enable_neuromodulation:
            self.causal_learning = CausalLearningSignal(hidden_dim)
            
        self.register_buffer('dopamine_level', torch.tensor(0.0))
        self.register_buffer('serotonin_level', torch.tensor(0.5))
        self.register_buffer('norepinephrine_level', torch.tensor(0.5))
        
        self.growth_rate = nn.Parameter(torch.tensor(0.1))
        self.prune_threshold = nn.Parameter(torch.tensor(0.15))
        self.plasticity = nn.Parameter(torch.tensor(0.5))
        
        self.neuron_importance_history: List[List[float]] = []
        
    def update_neuromodulation(
        self,
        neuromod: NeuromodulatorySignal,
    ):
        """更新神经调节水平"""
        self.dopamine_level = neuromod.dopamine.mean().detach()
        self.serotonin_level = neuromod.serotonin.mean().detach()
        self.norepinephrine_level = neuromod.norepinephrine.mean().detach()
        
        dopamine_norm = torch.sigmoid(self.dopamine_level)
        
        self.growth_rate.data = torch.clamp(
            self.growth_rate.data + dopamine_norm * 0.01,
            0.01, 0.5
        )
        
        if dopamine_norm < 0.3:
            self.plasticity.data = torch.clamp(
                self.plasticity.data - 0.01,
                0.1, 1.0
            )
        else:
            self.plasticity.data = torch.clamp(
                self.plasticity.data + 0.005,
                0.1, 1.0
            )
            
    def compute_causal_importance(
        self,
        states: torch.Tensor,
        outputs: torch.Tensor,
    ) -> List[CausalImportanceScore]:
        """
        计算每个神经元的因果重要性
        
        Returns:
            importance_scores: 每个神经元的因果重要性分数
        """
        if not self.config.enable_causal_importance:
            return []
            
        importance_scores = []
        
        for i in range(min(i + 1, states.size(-1))):
            pre = states[:, :-1] if states.size(1) > 1 else states
            post = states[:, 1:] if states.size(1) > 1 else states
            activation = states[:, i] if i < states.size(-1) else torch.zeros(states.size(0))
            
            importance = self.importance_estimator(
                pre, post, activation
            )
            
            info_gain = torch.std(states[:, i]).item() if i < states.size(-1) else 0.0
            
            neuromod_level = self.dopamine_level.item()
            
            total_score = (
                self.config.importance_weight * importance.item() +
                self.config.curiosity_weight * info_gain +
                neuromod_level * 0.2
            )
            
            importance_scores.append(CausalImportanceScore(
                neuron_idx=i,
                causal_influence=importance.item(),
                information_gain=info_gain,
                neuromodulation_level=neuromod_level,
                total_score=total_score,
            ))
            
        self.neuron_importance_history.append([
            s.total_score for s in importance_scores
        ])
        
        if len(self.neuron_importance_history) > 100:
            self.neuron_importance_history.pop(0)
            
        return importance_scores
        
    def should_grow(self) -> bool:
        """
        基于神经调节信号决定是否生长
        
        高多巴胺 + 低重要性神经元 → 应该生长
        """
        dopamine_norm = torch.sigmoid(self.dopamine_level)
        
        if dopamine_norm > self.config.dopamine_growth_threshold:
            return True
            
        if self.norepinephrine_level > 0.7:
            return True
            
        return False
        
    def should_prune(self) -> Tuple[bool, List[int]]:
        """
        基于神经调节信号决定是否修剪
        
        低多巴胺 + 低因果重要性 → 应该修剪
        """
        dopamine_norm = torch.sigmoid(self.dopamine_level)
        
        prune_indices = []
        
        if dopamine_norm < self.config.dopamine_prune_threshold:
            if self.neuron_importance_history:
                recent_scores = self.neuron_importance_history[-1]
                for i, score in enumerate(recent_scores):
                    if score < self.prune_threshold.item():
                        prune_indices.append(i)
                        
        return len(prune_indices) > 0, prune_indices
        
    def get_growth_candidates(
        self,
        importance_scores: List[CausalImportanceScore],
        num_candidates: int = 5,
    ) -> List[int]:
        """
        获取最佳生长候选神经元
        
        基于好奇心和信息增益选择
        """
        if not importance_scores:
            return []
            
        sorted_scores = sorted(
            importance_scores,
            key=lambda x: x.total_score,
            reverse=True
        )
        
        candidates = [s.neuron_idx for s in sorted_scores[:num_candidates]]
        
        return candidates
        
    def get_neuromodulation_summary(self) -> Dict:
        """获取神经调节状态摘要"""
        return {
            'dopamine': self.dopamine_level.item(),
            'serotonin': self.serotonin_level.item(),
            'norepinephrine': self.norepinephrine_level.item(),
            'growth_rate': self.growth_rate.item(),
            'prune_threshold': self.prune_threshold.item(),
            'plasticity': self.plasticity.item(),
        }


class CausalGrowableWrapper(nn.Module):
    """
    因果生长包装器
    
    将神经调节驱动的生长机制包装到现有的GrowableTwistorLMT中
    
    使用方法:
        wrapper = CausalGrowableWrapper(growable_model)
        wrapper.update_neuromodulation(neuromod_signal)
        if wrapper.should_grow():
            wrapper.grow()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[NeuromodulatedGrowthConfig] = None,
    ):
        super().__init__()
        self.model = model
        self.config = config or NeuromodulatedGrowthConfig()
        
        self.neuromod_growth = NeuromodulationDrivenGrowth(
            model.hidden_dim if hasattr(model, 'hidden_dim') else self.config.hidden_dim,
            self.config,
        )
        
        self.last_growth_step = 0
        self.last_prune_step = 0
        
    def update_neuromodulation(
        self,
        states: torch.Tensor,
        outputs: torch.Tensor,
        neuromod: NeuromodulatorySignal,
    ):
        """更新神经调节状态"""
        self.neuromod_growth.update_neuromodulation(neuromod)
        
        importance_scores = self.neuromod_growth.compute_causal_importance(
            states, outputs
        )
        
        return importance_scores
        
    def step(self, current_step: int) -> Dict:
        """
        执行生长/修剪步骤
        
        Returns:
            result: 操作结果
        """
        result = {
            'action': 'none',
            'n_grown': 0,
            'n_pruned': 0,
        }
        
        if self.neuromod_growth.should_grow():
            if current_step - self.last_growth_step >= self.config.growth_interval:
                importance_scores = self.neuromod_growth.neuron_importance_history[-1] if self.neuromod_growth.neuron_importance_history else []
                
                candidates = self.neuromod_growth.get_growth_candidates(
                    [CausalImportanceScore(i, s, s, 0, s) for i, s in enumerate(importance_scores)]
                )
                
                if candidates and hasattr(self.model, 'split_neuron'):
                    parent = candidates[0]
                    new_idx = self.model.split_neuron(parent)
                    if new_idx >= 0:
                        result['action'] = 'grow'
                        result['n_grown'] = 1
                        self.last_growth_step = current_step
                        
        should_prune, prune_indices = self.neuromod_growth.should_prune()
        if should_prune:
            if current_step - self.last_prune_step >= self.config.prune_interval:
                if hasattr(self.model, 'disable_connection'):
                    for idx in prune_indices[:3]:
                        self.model.disable_connection(idx)
                    result['action'] = 'prune'
                    result['n_pruned'] = len(prune_indices[:3])
                    self.last_prune_step = current_step
                    
        return result
        
    def get_summary(self) -> Dict:
        """获取生长系统摘要"""
        neuromod_summary = self.neuromod_growth.get_neuromodulation_summary()
        
        return {
            'neuromodulation': neuromod_summary,
            'last_growth_step': self.last_growth_step,
            'last_prune_step': self.last_prune_step,
            'hidden_dim': self.model.hidden_dim if hasattr(self.model, 'hidden_dim') else 0,
        }


class AdaptiveDevelopmentalSchedule(nn.Module):
    """
    自适应发育调度器
    
    用神经调节信号替代固定的发育时间表
    
    原有的固定阶段:
    - Fetal (0-100): 硬件搭建
    - Infant (100-300): 连接爆炸
    - Child (300-600): 修剪优化
    - Adolescent (600-900): 系统重构
    - Adult (900+): 整合收敛
    
    新的自适应方案:
    - 基于多巴胺水平动态调整生长/修剪速率
    - 基于信息增益决定何时进入新阶段
    - 基于去甲肾上腺素水平决定探索强度
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.register_buffer('phase', torch.tensor(0))
        self.register_buffer('exploration_intensity', torch.tensor(0.5))
        
        self.phase_transition_net = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Softmax(dim=-1),
        )
        
        self.phase_names = ['fetal', 'infant', 'child', 'adult']
        
        self.phase_thresholds = {
            'fetal': 0.0,
            'infant': 0.3,
            'child': 0.6,
            'adolescent': 0.8,
            'adult': 1.0,
        }
        
    def forward(
        self,
        states: torch.Tensor,
        neuromod: NeuromodulatorySignal,
    ) -> Dict:
        """
        根据神经调节信号和状态自适应调整发育阶段
        """
        combined = torch.cat([
            states.mean(dim=0),
            neuromod.dopamine.unsqueeze(0),
            neuromod.serotonin.unsqueeze(0),
            neuromod.norepinephrine.unsqueeze(0),
        ], dim=-1)
        
        phase_probs = self.phase_transition_net(combined)
        
        new_phase = phase_probs.argmax().item()
        phase_confidence = phase_probs[new_phase].item()
        
        dopamine_norm = torch.sigmoid(neuromod.dopamine.mean())
        
        if dopamine_norm > 0.7:
            self.exploration_intensity = torch.clamp(
                self.exploration_intensity + 0.05,
                0.0, 1.0
            )
        elif dopamine_norm < 0.3:
            self.exploration_intensity = torch.clamp(
                self.exploration_intensity - 0.02,
                0.0, 1.0
            )
            
        self.phase = new_phase
        
        return {
            'current_phase': self.phase_names[new_phase],
            'phase_confidence': phase_confidence,
            'exploration_intensity': self.exploration_intensity.item(),
            'phase_probs': phase_probs.tolist(),
        }
        
    def get_current_phase_params(self) -> Dict:
        """获取当前阶段的参数"""
        phase = self.phase.item()
        
        if phase == 0:
            return {
                'growth_rate': 0.8,
                'prune_rate': 0.0,
                'plasticity': 1.0,
                'target_connections': 2,
            }
        elif phase == 1:
            return {
                'growth_rate': 0.1,
                'prune_rate': 0.02,
                'plasticity': 1.0,
                'target_connections': 600,
            }
        elif phase == 2:
            return {
                'growth_rate': 0.05,
                'prune_rate': 0.3,
                'plasticity': 0.7,
                'target_connections': 300,
            }
        else:
            return {
                'growth_rate': 0.05,
                'prune_rate': 0.1,
                'plasticity': 0.3,
                'target_connections': 300,
            }
