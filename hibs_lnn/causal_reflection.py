"""
因果反思模块 - Causal Reflection Module
=======================================
增强版的反思模块，集成了因果推理能力：

1. 因果干预选择 - 基于好奇心和价值选择干预点
2. 反事实思考 - 在思考阶段进行反事实推演
3. 因果学习信号 - 使用类多巴胺信号指导结构修改
4. 神经调节整合 - 将神经调节信号整合到决策中

替换原有的固定阈值决策机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass, field
import copy

from .curiosity_module import CuriosityModule, CuriosityConfig
from .intervention_executor import CausalInterventionEngine, DoIntervention, InterventionTarget
from .episodic_memory import HippocampalSystem
from .offline_simulator import OfflineSimulator, SimulationConfig, DreamCortex
from .causal_learning_signal import CausalLearningSignal, NeuromodulatorySignal


@dataclass
class CausalReflectionConfig:
    """因果反思配置"""
    hidden_dim: int = 128
    
    enable_curiosity: bool = True
    enable_intervention: bool = True
    enable_counterfactual: bool = True
    enable_neuromodulation: bool = True
    
    reflection_interval: int = 50
    think_steps: int = 10
    
    grow_threshold: float = 0.3
    prune_threshold: float = 0.9
    min_steps_before_grow: int = 100
    min_steps_before_prune: int = 200
    
    max_modifications_per_reflection: int = 2
    confidence_smoothing: float = 0.9
    
    curiosity_weight: float = 1.0
    causal_weight: float = 0.5
    neuromodulation_weight: float = 0.3


@dataclass
class CausalDecision:
    """因果决策"""
    action: str  # 'grow', 'prune', 'intervene', 'keep'
    target_type: Optional[str] = None  # 'neuron', 'connection', 'z', 'tau'
    target_indices: Optional[List[int]] = None
    intervention_value: Optional[torch.Tensor] = None
    confidence: float = 0.0
    reasoning: str = ""


class CausalReflectionModule(nn.Module):
    """
    因果反思模块
    
    工作流程:
    1. Think with curiosity - 好奇心驱动的思考
    2. Counterfactual reasoning - 反事实推演
    3. Neuromodulation signals - 神经调节信号
    4. Causal decision - 因果决策（替代固定阈值）
    5. Execute modification - 执行修改
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[CausalReflectionConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or CausalReflectionConfig()
        self.config.hidden_dim = hidden_dim
        
        if self.config.enable_curiosity:
            self.curiosity = CuriosityModule(
                hidden_dim,
                CuriosityConfig(
                    hidden_dim=hidden_dim,
                    curiosity_weight=self.config.curiosity_weight,
                )
            )
            
        if self.config.enable_intervention:
            self.intervention_engine = CausalInterventionEngine(hidden_dim)
            
        if self.config.enable_counterfactual:
            self.simulator = OfflineSimulator(
                hidden_dim,
                SimulationConfig(
                    horizon=10,
                    num_rollouts=3,
                )
            )
            
        if self.config.enable_neuromodulation:
            self.causal_learning = CausalLearningSignal(hidden_dim)
            
        self.hippocampus = HippocampalSystem(hidden_dim)
        
        self.decision_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
            nn.Softmax(dim=-1),
        )
        
        self.register_buffer('confidence_ema', torch.tensor(0.5))
        self.register_buffer('dopamine_level', torch.tensor(0.0))
        
        self.reflection_buffer: List[Dict] = []
        self.modification_history: List[Dict] = []
        
        self.total_thinks = 0
        self.total_grows = 0
        self.total_prunes = 0
        self.total_interventions = 0
        
    def think_with_curiosity(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        n_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], Dict]:
        """
        好奇心驱动的思考阶段
        
        Returns:
            thought_summary: 思考总结
            thoughts: 所有思考向量
            curiosity_info: 好奇心信息
        """
        if n_steps is None:
            n_steps = self.config.think_steps
            
        model.eval()
        
        B = x.shape[1]
        
        if hasattr(model, 'hidden_dim'):
            z = torch.zeros(B, model.hidden_dim, dtype=torch.complex64, device=x.device)
        else:
            z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
            
        thoughts = []
        curiosity_info_list = []
        
        for step in range(n_steps):
            x_context = x[0] if x.shape[0] > 0 else x.squeeze(0)
            
            if hasattr(model, 'compute_dzdt'):
                dzdt = model.compute_dzdt(z, x_context)
            else:
                break
                
            z = z + (model.dt if hasattr(model, 'dt') else 0.1) * dzdt
            
            z = torch.complex(
                torch.clamp(z.real, -100, 100),
                torch.clamp(z.imag, -100, 100),
            )
            
            thoughts.append(z.clone())
            
            if self.config.enable_curiosity and step % 2 == 0:
                curiosity_result = self.curiosity(z.real)
                curiosity_info_list.append(curiosity_result)
                
            if step >= 2 and len(thoughts) >= 3:
                recent_variance = torch.stack(thoughts[-3:]).var(dim=0).mean().item()
                if recent_variance < 1e-4:
                    break
                    
        if thoughts:
            thought_summary = torch.stack(thoughts).mean(dim=0)
        else:
            thought_summary = torch.zeros(B, self.hidden_dim, device=x.device)
            
        curiosity_info = {
            'intrinsic_reward': torch.stack([c['intrinsic_reward'] for c in curiosity_info_list]).mean()
                if curiosity_info_list else torch.tensor(0.0),
            'total_uncertainty': torch.stack([c['total_uncertainty'] for c in curiosity_info_list]).mean()
                if curiosity_info_list else torch.tensor(0.0),
            'total_novelty': torch.stack([c['total_novelty'] for c in curiosity_info_list]).mean()
                if curiosity_info_list else torch.tensor(0.0),
        }
        
        self.total_thinks += 1
        
        return thought_summary, thoughts, curiosity_info
        
    def counterfactual_thinking(
        self,
        model: nn.Module,
        initial_state: torch.Tensor,
        num_alternatives: int = 5,
    ) -> Dict:
        """
        反事实思考 - 探索"如果...会怎样"
        
        Returns:
            cf_info: 反事实推理结果
        """
        cf_info = {
            'alternatives': [],
            'causal_effects': [],
        }
        
        if not self.config.enable_counterfactual:
            return cf_info
            
        for alt_idx in range(num_alternatives):
            def apply_intervention(state, step):
                noise = torch.randn_like(state) * 0.5
                return state + noise
                
            result = self.simulator.counterfactual_rollout(
                model, initial_state, apply_intervention,
                num_steps=5, num_rollouts=2
            )
            
            cf_info['alternatives'].append({
                'factual': result.factual_trajectory,
                'counterfactual': result.counterfactual_trajectory,
                'effect_magnitude': result.effect_magnitude,
            })
            cf_info['causal_effects'].append(result.effect_magnitude)
            
        if cf_info['causal_effects']:
            cf_info['avg_effect'] = np.mean(cf_info['causal_effects'])
            cf_info['max_effect'] = np.max(cf_info['causal_effects'])
        else:
            cf_info['avg_effect'] = 0.0
            cf_info['max_effect'] = 0.0
            
        return cf_info
        
    def compute_neuromodulation(
        self,
        state: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
    ) -> NeuromodulatorySignal:
        """
        计算神经调节信号
        """
        if not self.config.enable_neuromodulation:
            return NeuromodulatorySignal(
                dopamine=torch.zeros(1),
                serotonin=torch.zeros(1),
                acetylcholine=torch.zeros(1),
                norepinephrine=torch.zeros(1),
            )
            
        neuromod = self.causal_learning.neuromodulatory(
            state, reward, next_state
        )
        
        self.dopamine_level = neuromod.dopamine.mean().detach()
        
        return neuromod
        
    def causal_decision(
        self,
        thought_summary: torch.Tensor,
        curiosity_info: Dict,
        cf_info: Dict,
        neuromod: NeuromodulatorySignal,
        training_step: int,
    ) -> CausalDecision:
        """
        因果决策 - 基于多种信号做出决策
        
        替代原有的固定阈值决策
        """
        decision_input = torch.cat([
            thought_summary,
            thought_summary * curiosity_info.get('total_uncertainty', torch.tensor(0.0)).to(thought_summary.device),
            thought_summary * self.dopamine_level.to(thought_summary.device),
        ], dim=-1)
        
        action_probs = self.decision_net(decision_input)
        
        action_idx = action_probs.argmax().item()
        action_names = ['keep', 'grow', 'prune', 'intervene']
        action = action_names[action_idx]
        
        confidence = action_probs[action_idx].item()
        
        can_grow = training_step >= self.config.min_steps_before_grow
        can_prune = training_step >= self.config.min_steps_before_prune
        
        if action == 'grow' and not can_grow:
            action = 'keep'
            confidence *= 0.5
        elif action == 'prune' and not can_prune:
            action = 'keep'
            confidence *= 0.5
            
        target_indices = None
        intervention_value = None
        target_type = None
        
        if action == 'intervene':
            target_indices = [np.random.randint(self.hidden_dim)]
            intervention_value = torch.randn(1, self.hidden_dim) * 0.5
            target_type = 'z'
            
        reasoning = f"curiosity={curiosity_info.get('total_uncertainty', 0):.3f}, " \
                   f"cf_effect={cf_info.get('avg_effect', 0):.3f}, " \
                   f"dopamine={neuromod.dopamine.mean().item():.3f}"
        
        return CausalDecision(
            action=action,
            target_type=target_type,
            target_indices=target_indices,
            intervention_value=intervention_value,
            confidence=confidence,
            reasoning=reasoning,
        )
        
    def execute_modification(
        self,
        model: nn.Module,
        decision: CausalDecision,
        training_step: int,
    ) -> Tuple[int, Optional[Dict]]:
        """
        执行修改
        """
        action = decision.action
        confidence_before = self.confidence_ema.item()
        
        n_mods = 0
        record = None
        
        if action == 'grow':
            if hasattr(model, 'split_neuron'):
                new_idx = model.split_neuron(np.random.randint(model.hidden_dim))
                if new_idx >= 0:
                    n_mods = 1
                    self.total_grows += 1
                    
        elif action == 'prune':
            if hasattr(model, 'prune_neurons'):
                n_mods = model.prune_neurons()
                self.total_prunes += n_mods
                
        elif action == 'intervene':
            if hasattr(model, 'causal_cortex') and decision.target_indices is not None:
                intervened = model.causal_cortex.do(
                    model.z.real if hasattr(model, 'z') else torch.zeros(1, self.hidden_dim),
                    decision.target_type or 'z',
                    decision.target_indices,
                    decision.intervention_value.to(model.z.device) if decision.intervention_value is not None else torch.zeros(1, self.hidden_dim),
                )
                n_mods = 1
                self.total_interventions += 1
                
        record = {
            'step': training_step,
            'action': action,
            'confidence_before': confidence_before,
            'confidence_after': self.confidence_ema.item(),
            'decision_confidence': decision.confidence,
            'reasoning': decision.reasoning,
        }
        
        return n_mods, record
        
    def reflect(
        self,
        model: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        loss_history: List[float],
        training_step: int,
    ) -> Dict:
        """
        完整的因果反思过程
        """
        thought_summary, thoughts, curiosity_info = self.think_with_curiosity(
            model, x, y
        )
        
        initial_state = thought_summary
        
        cf_info = {}
        if len(thoughts) > 0:
            cf_info = self.counterfactual_thinking(
                model, initial_state
            )
            
        reward = torch.tensor(-loss_history[-1] if loss_history else 0.0)
        next_state = thought_summary
        
        neuromod = self.compute_neuromodulation(
            initial_state, reward, next_state
        )
        
        decision = self.causal_decision(
            thought_summary, curiosity_info, cf_info, neuromod, training_step
        )
        
        n_mods, record = self.execute_modification(model, decision, training_step)
        
        confidence_raw = decision.confidence
        confidence = self.confidence_ema.item() * self.config.confidence_smoothing + \
                    confidence_raw * (1 - self.config.confidence_smoothing)
        self.confidence_ema.fill_(confidence)
        
        reflection_result = {
            'thought_summary': thought_summary,
            'curiosity_info': curiosity_info,
            'cf_info': cf_info,
            'neuromodulation': {
                'dopamine': neuromod.dopamine.mean().item(),
                'serotonin': neuromod.serotonin.mean().item(),
            },
            'decision': {
                'action': decision.action,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
            },
            'n_modifications': n_mods,
        }
        
        self.reflection_buffer.append(reflection_result)
        
        if record:
            self.modification_history.append(record)
            
        return reflection_result
        
    def should_reflect(self, training_step: int) -> bool:
        """判断是否应该反思"""
        return training_step > 0 and training_step % self.config.reflection_interval == 0
        
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_thinks': self.total_thinks,
            'total_grows': self.total_grows,
            'total_prunes': self.total_prunes,
            'total_interventions': self.total_interventions,
            'current_confidence': self.confidence_ema.item(),
            'dopamine_level': self.dopamine_level.item(),
            'total_reflections': len(self.reflection_buffer),
        }
        
    def reset(self):
        """重置模块状态"""
        self.reflection_buffer.clear()
        self.modification_history.clear()
        self.total_thinks = 0
        self.total_grows = 0
        self.total_prunes = 0
        self.total_interventions = 0
        self.confidence_ema.fill_(0.5)
        self.dopamine_level.fill_(0.0)
