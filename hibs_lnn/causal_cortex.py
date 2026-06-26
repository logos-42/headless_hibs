"""
因果皮层 - Causal Cortex
=========================
整合所有因果推理组件的统一模块：

1. 内在动机 - 好奇心驱动的探索
2. 干预执行 - do-演算语义
3. 情景记忆 - 反事实推理的基础
4. 离线模拟 - 想象和反事实推演
5. 因果学习信号 - 类多巴胺的权重更新

这个模块整合到 SelfAwareTwistorLMT 中，替换原有的固定思考循环
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from .curiosity_module import CuriosityModule, CuriosityConfig
from .intervention_executor import InterventionExecutor, CausalInterventionEngine, DoIntervention, InterventionTarget
from .episodic_memory import EpisodicMemory, HippocampalSystem, MemoryQuery
from .offline_simulator import OfflineSimulator, SimulationConfig, CounterfactualResult, ImaginedTrajectory, DreamCortex
from .causal_learning_signal import CausalLearningSignal, NeuromodulatorySignal, RewardSignal


@dataclass
class CausalCortexConfig:
    """因果皮层配置"""
    hidden_dim: int = 128
    memory_dim: int = 64
    
    enable_curiosity: bool = True
    enable_intervention: bool = True
    enable_episodic_memory: bool = True
    enable_offline_simulation: bool = True
    enable_causal_learning: bool = True
    
    curiosity_weight: float = 1.0
    intervention_cost_weight: float = 0.01
    
    simulation_horizon: int = 15
    simulation_rollouts: int = 5
    
    memory_capacity: int = 1000
    max_episode_length: int = 100


class CausalCortex(nn.Module):
    """
    因果皮层 - 整合所有因果组件
    
    工作流程：
    
    在线学习阶段:
        1. 观察状态
        2. 估计不确定性和新奇性 (curiosity)
        3. 做出干预选择
        4. 执行干预 (intervention)
        5. 观察结果，更新因果知识 (causal learning)
        6. 存储到情景记忆 (episodic memory)
        
    离线推理阶段:
        1. 从情景记忆检索相关经验
        2. 进行反事实推理 (offline simulation)
        3. 评估因果效应
        4. 更新因果结构知识
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[CausalCortexConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or CausalCortexConfig()
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
            
        if self.config.enable_episodic_memory:
            self.hippocampus = HippocampalSystem(
                hidden_dim,
                self.config.memory_dim
            )
            
        if self.config.enable_offline_simulation:
            self.dream_cortex = DreamCortex(hidden_dim)
            self.simulator = OfflineSimulator(
                hidden_dim,
                SimulationConfig(
                    horizon=self.config.simulation_horizon,
                    num_rollouts=self.config.simulation_rollouts,
                )
            )
            
        if self.config.enable_causal_learning:
            self.causal_learning = CausalLearningSignal(hidden_dim)
            
        self.episode_buffer: List[torch.Tensor] = []
        self.current_episode: List[torch.Tensor] = []
        
        self.register_buffer('step_count', torch.tensor(0))
        
    def observe(
        self,
        state: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reward: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        在线观察阶段 - 好奇心驱动的探索
        
        Returns:
            info: 包含好奇心信号、干预选择等的字典
        """
        info = {}
        
        if self.config.enable_curiosity:
            curiosity_result = self.curiosity(state)
            info['curiosity'] = curiosity_result
            info['intrinsic_reward'] = curiosity_result['intrinsic_reward']
            
        if self.config.enable_episodic_memory:
            self.current_episode.append(state.detach())
            
        self.step_count += 1
        
        return info
        
    def intervene(
        self,
        state: torch.Tensor,
        intervention_target: Optional[str] = None,
        intervention_value: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        执行干预
        
        Returns:
            intervened_state: 干预后的状态
            intervention_info: 干预详情
        """
        if not self.config.enable_intervention:
            return state, {}
            
        intervention_info = {}
        
        if intervention_target is None:
            intervention_info = self.intervention_engine.do(
                state,
                'z',
                [np.random.randint(self.hidden_dim)],
                intervention_value or (torch.randn_like(state) * 0.5),
                intervention_type='soft',
            )
        else:
            intervention_info = self.intervention_engine.do(
                state,
                intervention_target,
                [0],
                intervention_value,
                intervention_type='hard',
            )
            
        return intervention_info.intervened_state, intervention_info
        
    def learn_from_experience(
        self,
        state: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        weights: torch.Tensor,
    ) -> Dict:
        """
        从经验中学习 - 因果学习信号更新
        
        Returns:
            learning_info: 学习信号详情
        """
        if not self.config.enable_causal_learning:
            return {}
            
        learning_info = {}
        
        pre_synaptic = state
        post_synaptic = next_state
        
        intrinsic_reward = torch.ones_like(reward) * 0.1
        
        updated_weights, info = self.causal_learning(
            state,
            reward,
            intrinsic_reward,
            next_state,
            weights,
            pre_synaptic,
            post_synaptic,
        )
        
        learning_info['updated_weights'] = updated_weights
        learning_info['dopamine'] = info.get('dopamine')
        learning_info['td_error'] = info.get('td_error')
        
        return learning_info
        
    def store_episode(self):
        """存储当前情节到情景记忆"""
        if not self.config.enable_episodic_memory:
            return
            
        if len(self.current_episode) > 0:
            episode = torch.stack(self.current_episode, dim=0)
            
            if len(self.episode_buffer) < self.config.memory_capacity:
                self.episode_buffer.append(episode)
                
            episode_id = self.hippocampus.encode_and_store(episode)
            
            self.current_episode.clear()
            
            return episode_id
        return None
        
    def retrieve_memories(
        self,
        query_state: torch.Tensor,
        top_k: int = 5,
    ) -> List:
        """检索相关情景记忆"""
        if not self.config.enable_episodic_memory:
            return []
            
        return self.hippocampus.retrieve_similar(query_state, top_k)
        
    def counterfactual_reasoning(
        self,
        model: nn.Module,
        current_state: torch.Tensor,
        intervention_fn,
    ) -> CounterfactualResult:
        """
        反事实推理
        
        Args:
            model: 要模拟的模型
            current_state: 当前状态
            intervention_fn: 干预函数 (state, step) -> intervened_state
            
        Returns:
            CounterfactualResult
        """
        if not self.config.enable_offline_simulation:
            return None
            
        result = self.simulator.counterfactual_rollout(
            model, current_state, intervention_fn,
        )
        
        return result
        
    def imagine_alternatives(
        self,
        model: nn.Module,
        initial_state: torch.Tensor,
        num_alternatives: int = 10,
    ) -> List[ImaginedTrajectory]:
        """
        想象替代方案 - 创意生成
        """
        if not self.config.enable_offline_simulation:
            return []
            
        imagined = self.simulator.imagine_alternatives(
            model, initial_state,
            num_alternatives=num_alternatives,
        )
        
        return imagined
        
    def offline_replay(self) -> List[torch.Tensor]:
        """
        离线重演 - 模拟睡眠期间的记忆重演
        """
        if not self.config.enable_episodic_memory:
            return []
            
        return self.hippocampus.offline_replay()
        
    def do(
        self,
        state: torch.Tensor,
        target_type: str,
        target_indices: List[int],
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        简洁的do-干预接口
        """
        if not self.config.enable_intervention:
            return state
            
        result = self.intervention_engine.do(
            state, target_type, target_indices, value
        )
        
        return result.intervened_state
        
    def what_if(
        self,
        model: nn.Module,
        state: torch.Tensor,
        intervention_fn,
    ) -> CounterfactualResult:
        """
        反事实查询：what if ...?
        """
        return self.counterfactual_reasoning(model, state, intervention_fn)
        
    def reset(self):
        """重置所有子模块"""
        if hasattr(self, 'curiosity'):
            self.curiosity.reset()
        if hasattr(self, 'intervention_engine'):
            self.intervention_engine.executor.reset()
        if hasattr(self, 'hippocampus'):
            self.hippocampus.episodic_memory.reset()
        if hasattr(self, 'simulator'):
            self.simulator.reset()
        if hasattr(self, 'causal_learning'):
            self.causal_learning.reset()
            
        self.episode_buffer.clear()
        self.current_episode.clear()
        self.step_count.fill_(0)


class CausalSelfAwareTwistorLMT(nn.Module):
    """
    因果自感知扭量神经网络
    
    在 SelfAwareTwistorLMT 的基础上集成因果皮层
    支持：
    - 好奇心驱动的内部思考
    - 干预和反事实推理
    - 情景记忆整合
    - 类多巴胺学习信号
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        causal_config: Optional[CausalCortexConfig] = None,
        dt: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_amplitude = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.5)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        self.tau_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Linear(hidden_dim, output_dim)
        
        self.manifold_theta = nn.Parameter(
            torch.randn(hidden_dim, 3) * 0.1
        )
        
        self.sparse_mask = nn.Parameter(torch.ones(hidden_dim, hidden_dim) * -5)
        
        self.causal_cortex = CausalCortex(
            hidden_dim,
            causal_config or CausalCortexConfig(hidden_dim=hidden_dim)
        )
        
        self.register_buffer('z', torch.zeros(hidden_dim, dtype=torch.complex64))
        
    def compute_dzdt(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
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
        
        dz_real = -z_real + W_tanh_real + Ux + self.b
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b
        
        tau = F.sigmoid(self.W_tau(torch.abs(z)))
        tau = tau + self.tau_bias
        tau = torch.clamp(tau, 0.01, 1.0)
        
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)
        
        return dzdt
        
    def think_with_curiosity(
        self,
        x_t: torch.Tensor,
        z: torch.Tensor,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        好奇心驱动的思考循环
        
        替换原有的固定思考循环
        """
        curiosity_info = self.causal_cortex.observe(z.real)
        
        intervened_z, intervention_info = self.causal_cortex.intervene(z.real)
        
        z_complex = torch.complex(intervened_z, torch.zeros_like(intervened_z))
        
        dzdt = self.compute_dzdt(z_complex, x_t)
        z_new = z_complex + self.dt * dzdt
        
        z_new = torch.complex(
            torch.clamp(z_new.real, -100, 100),
            torch.clamp(z_new.imag, -100, 100),
        )
        
        output = self.out(z_new.real)
        
        info = {
            'curiosity': curiosity_info,
            'intervention': intervention_info,
            'z': z_new,
            'output': output,
        }
        
        return z_new, output, info
        
    def forward(
        self,
        x: torch.Tensor,
        enable_causal: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            x: [seq_len, batch, input_dim]
            enable_causal: 是否启用因果皮层
            
        Returns:
            output: [seq_len, batch, output_dim]
            info: 因果处理信息
        """
        T, B, _ = x.shape
        
        outputs = []
        all_info = []
        
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        for t in range(T):
            x_t = x[t]
            
            if enable_causal:
                z, output, info = self.think_with_curiosity(x_t, z, B)
            else:
                dzdt = self.compute_dzdt(z, x_t)
                z = z + self.dt * dzdt
                z = torch.complex(
                    torch.clamp(z.real, -100, 100),
                    torch.clamp(z.imag, -100, 100),
                )
                output = self.out(z.real)
                info = {}
                
            outputs.append(output)
            all_info.append(info)
            
            if enable_causal:
                self.causal_cortex.observe(z.real)
                
        output_seq = torch.stack(outputs, dim=0)
        
        return output_seq, {'causal_info': all_info}
        
    def do_intervention(
        self,
        state: torch.Tensor,
        target_indices: List[int],
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行干预
        """
        return self.causal_cortex.do(state, 'z', target_indices, value)
        
    def what_if(
        self,
        state: torch.Tensor,
        intervention_fn,
    ) -> CounterfactualResult:
        """
        反事实查询
        """
        return self.causal_cortex.what_if(self, state, intervention_fn)
        
    def imagine_alternatives(
        self,
        state: torch.Tensor,
        num_alternatives: int = 10,
    ) -> List[ImaginedTrajectory]:
        """
        想象替代方案
        """
        return self.causal_cortex.imagine_alternatives(self, state, num_alternatives)
        
    def store_memory(self):
        """存储当前情节"""
        return self.causal_cortex.store_episode()
        
    def retrieve_memories(
        self,
        state: torch.Tensor,
        top_k: int = 5,
    ):
        """检索情景记忆"""
        return self.causal_cortex.retrieve_memories(state, top_k)
        
    def offline_replay(self) -> List[torch.Tensor]:
        """离线重演"""
        return self.causal_cortex.offline_replay()
