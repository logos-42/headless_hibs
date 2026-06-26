"""
扭量情绪模块 - Twistor Emotional Module
======================================
在复数扭量空间中实现情绪系统：

1. 幅值情绪 - |Z|反映唤醒度
2. 相位情绪 - 相位同步度反映效价/稳定性
3. 相位速度情绪 - d(phase)/dt反映情绪变化/控制感
4. 情绪调制 - 通过τ(Z)和相位调制扭量状态

核心思想：情绪是相位的集体动力学
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, TYPE_CHECKING, Callable
from dataclasses import dataclass

from .twistor_causal import TwistorCausalModule, TwistorCausalConfig


@dataclass
class TwistorEmotionalConfig:
    """扭量情绪配置"""
    hidden_dim: int = 128
    emotion_dim: int = 64
    decay_rate: float = 0.95
    momentum: float = 0.9


@dataclass
class TwistorEmotionalState:
    """扭量情绪状态"""
    valence: torch.Tensor       # 效价: -1(负面) ~ +1(正面)
    arousal: torch.Tensor      # 唤醒度: 0(平静) ~ 1(兴奋)
    dominance: torch.Tensor    # 控制感: 0(被动) ~ 1(主动)
    coherence: torch.Tensor    # 相位相干度: 0(混乱) ~ 1(一致)
    energy: torch.Tensor      # 能量/活性


class TwistorEmotionExtractor(nn.Module):
    """
    扭量情绪提取器
    
    从复数状态提取情绪：
    - 幅值 |Z| → 唤醒度
    - 相位同步度 → 效价/稳定性
    - 相位速度 → 控制感
    - 幅值变化 → 能量
    """
    
    def __init__(self, hidden_dim: int, emotion_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emotion_dim = emotion_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, emotion_dim),
            nn.LayerNorm(emotion_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emotion_dim, emotion_dim // 2),
            nn.ReLU(),
        )
        
        self.valence_net = nn.Linear(emotion_dim // 2, 1)
        self.arousal_net = nn.Linear(emotion_dim // 2, 1)
        self.dominance_net = nn.Linear(emotion_dim // 2, 1)
        self.coherence_net = nn.Linear(emotion_dim // 2, 1)
        self.energy_net = nn.Linear(emotion_dim // 2, 1)
        
    def extract_phase_coherence(self, z: torch.Tensor) -> torch.Tensor:
        """
        提取相位相干度
        
        计算所有神经元相位的一致性
        相位越一致 → 相干度越高 → 情绪越稳定
        """
        phase = torch.angle(z)
        
        mean_phase = phase.mean(dim=-1, keepdim=True)
        
        phase_diff = phase - mean_phase
        
        coherence = torch.mean(torch.cos(phase_diff), dim=-1)
        
        return coherence
        
    def extract_phase_velocity(self, z: torch.Tensor, z_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        提取相位速度
        
        相位变化越快 → 情绪变化越快 → 控制感越强
        """
        phase = torch.angle(z)
        
        if z_prev is not None:
            prev_phase = torch.angle(z_prev)
            phase_velocity = torch.abs(phase - prev_phase).mean(dim=-1)
        else:
            phase_velocity = torch.zeros(z.size(0), device=z.device)
            
        return phase_velocity
        
    def forward(
        self,
        z: torch.Tensor,
        z_prev: Optional[torch.Tensor] = None,
    ) -> TwistorEmotionalState:
        """
        从扭量状态提取情绪
        """
        amplitude = torch.abs(z)
        phase = torch.angle(z)
        
        amplitude_input = amplitude
        phase_input = phase
        
        combined = torch.cat([amplitude_input, phase_input], dim=-1)
        
        encoded = self.encoder(combined)
        
        valence = torch.tanh(self.valence_net(encoded)).squeeze(-1)
        arousal = torch.sigmoid(self.arousal_net(encoded)).squeeze(-1)
        dominance = torch.sigmoid(self.dominance_net(encoded)).squeeze(-1)
        
        coherence = self.extract_phase_coherence(z)
        energy = torch.mean(amplitude ** 2, dim=-1)
        
        return TwistorEmotionalState(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            coherence=coherence,
            energy=energy,
        )


class TwistorEmotionModulator(nn.Module):
    """
    扭量情绪调制器
    
    用情绪调制扭量状态：
    1. 唤醒度 → 调制τ(Z)时间常数
    2. 效价 → 调制相位演化
    3. 控制感 → 调制连接强度
    4. 相干度 → 调制共振强度
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.tau_modulation = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        self.phase_modulation = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        
        self.connection_modulation = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        dzdt: torch.Tensor,
        emotion: TwistorEmotionalState,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        用情绪调制状态导数
        """
        emotion_vec = torch.stack([
            emotion.valence,
            emotion.arousal,
            emotion.dominance,
        ], dim=-1)
        
        if z is not None:
            amplitude = torch.abs(z).mean(dim=-1, keepdim=True)
            emotion_vec = emotion_vec * amplitude
        
        tau_mod = self.tau_modulation(emotion_vec)
        phase_mod = self.phase_modulation(emotion_vec)
        conn_mod = self.connection_modulation(emotion_vec)
        
        modulated_dzdt = dzdt * tau_mod + phase_mod * dzdt * 0.1
        
        return modulated_dzdt


class TwistorEmotionMemory(nn.Module):
    """
    扭量情绪记忆
    
    存储情境-情绪关联：
    - 复数状态编码 → 检索相似情境
    - 对应情绪状态 → 预测/检索情绪
    """
    
    def __init__(self, hidden_dim: int, memory_size: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        
        self.memory: Optional[torch.Tensor] = None
        self.emotion_memory: Optional[torch.Tensor] = None
        self.memory_ptr = 0
        self.memory_full = False
        
    def init_memory(self, device: torch.device):
        self.memory = torch.zeros(self.memory_size, self.hidden_dim, device=device)
        self.emotion_memory = torch.zeros(self.memory_size, 5, device=device)
        self.memory_ptr = 0
        self.memory_full = False
        
    def store(self, z: torch.Tensor, emotion: TwistorEmotionalState):
        if self.memory is None:
            self.init_memory(z.device)
            
        if z.size(0) > 1:
            z = z[0:1]
            
        encoded = self.state_encoder(
            torch.cat([z.real, z.imag], dim=-1)
        )
        
        idx = self.memory_ptr % self.memory_size
        self.memory[idx] = encoded.detach()
        
        emotion_vec = torch.stack([
            emotion.valence.mean(),
            emotion.arousal.mean(),
            emotion.dominance.mean(),
            emotion.coherence.mean(),
            emotion.energy.mean(),
        ], dim=0)
        self.emotion_memory[idx] = emotion_vec.detach()
        
        self.memory_ptr += 1
        if self.memory_ptr >= self.memory_size:
            self.memory_full = True
            
    def retrieve(
        self,
        z: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.memory is None or self.memory_ptr == 0:
            return torch.zeros(1, 5, device=z.device), torch.zeros(1, device=z.device)
            
        valid_size = self.memory_ptr if not self.memory_full else self.memory_size
        
        if z.size(0) > 1:
            z = z[0:1]
            
        encoded = self.state_encoder(
            torch.cat([z.real, z.imag], dim=-1)
        )
        
        if encoded.dim() == 2 and encoded.size(0) > 1:
            encoded = encoded[0:1]
            
        similarities = F.cosine_similarity(
            encoded.unsqueeze(0).expand(valid_size, -1),
            self.memory[:valid_size],
            dim=-1
        )
        
        top_k = min(top_k, valid_size)
        scores, indices = torch.topk(similarities, top_k)
        
        retrieved_emotions = self.emotion_memory[indices]
        
        return retrieved_emotions, scores


class TwistorEmotionalModule(nn.Module):
    """
    扭量情绪模块
    
    完整整合情绪到扭量空间：
    1. 从复数状态提取情绪
    2. 情绪记忆存储与检索
    3. 情绪调制τ(Z)和相位
    4. 情绪-因果闭环整合
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[TwistorEmotionalConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or TwistorEmotionalConfig()
        self.config.hidden_dim = hidden_dim
        
        self.extractor = TwistorEmotionExtractor(hidden_dim, self.config.emotion_dim)
        self.modulator = TwistorEmotionModulator(hidden_dim)
        self.memory = TwistorEmotionMemory(hidden_dim)
        
        self.previous_z: Optional[torch.Tensor] = None
        
        self.emotion_decay = self.config.decay_rate
        self.emotion_momentum = self.config.momentum
        
    def extract_emotion(
        self,
        z: torch.Tensor,
        store: bool = True,
    ) -> TwistorEmotionalState:
        """
        从扭量状态提取情绪
        """
        emotion = self.extractor(z, self.previous_z)
        
        if store:
            self.memory.store(z, emotion)
            
        self.previous_z = z.detach().clone()
        
        return emotion
        
    def modulate_dzdt(
        self,
        dzdt: torch.Tensor,
        emotion: TwistorEmotionalState,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        用情绪调制状态导数
        """
        return self.modulator(dzdt, emotion, z)
        
    def retrieve_emotion(
        self,
        z: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        检索相似情境的情绪
        """
        return self.memory.retrieve(z, top_k)
        
    def compute_emotion_effect(
        self,
        emotion: TwistorEmotionalState,
    ) -> Dict[str, torch.Tensor]:
        """
        计算情绪对因果的效应
        """
        return {
            'curiosity_drive': emotion.arousal * (1 + emotion.coherence),
            'causal_learning_rate': emotion.valence * 0.5 + 0.5,
            'exploration_bonus': emotion.dominance * emotion.arousal,
            'stability_factor': emotion.coherence,
        }
        
    def forward(
        self,
        z: torch.Tensor,
        dzdt: Optional[torch.Tensor] = None,
    ) -> Tuple[TwistorEmotionalState, Dict]:
        """
        主前向传播
        """
        emotion = self.extract_emotion(z)
        
        modulated_dzdt = None
        if dzdt is not None:
            modulated_dzdt = self.modulate_dzdt(dzdt, emotion, z)
            
        effect = self.compute_emotion_effect(emotion)
        
        info = {
            'emotion': emotion,
            'modulated_dzdt': modulated_dzdt,
            'effect': effect,
        }
        
        return emotion, info


class TwistorCausalEmotionalLoop(nn.Module):
    """
    扭量因果-情绪闭环
    
    在扭量空间中整合因果和情绪：
    
    因果 → 情绪：干预效果 → 情绪反应
    情绪 → 因果：情绪状态 → 调制τ和相位
    相位 → 情绪：相位同步 → 情绪稳定性
    情绪 → 相位：情绪调制 → 相位演化
    """
    
    def __init__(
        self,
        hidden_dim: int,
        causal_config: Optional["TwistorCausalConfig"] = None,
        emotional_config: Optional[TwistorEmotionalConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.causal = TwistorCausalModule(hidden_dim, causal_config)
        self.emotional = TwistorEmotionalModule(hidden_dim, emotional_config)
        
    def forward(
        self,
        z: torch.Tensor,
        intervention_fn: Optional[Callable] = None,
    ) -> Tuple[TwistorEmotionalState, torch.Tensor, Dict]:
        """
        主前向传播
        
        1. 计算相位因果
        2. 提取情绪
        3. 计算情绪效应
        4. 返回调制信号
        """
        phase_causality = self.causal.compute_phase_causality(z)
        
        emotion, emotion_info = self.emotional(z)
        
        effect = self.emotional.compute_emotion_effect(emotion)
        
        curiosity_signal = effect['curiosity_drive']
        causal_learning_rate = effect['causal_learning_rate']
        exploration_bonus = effect['exploration_bonus']
        
        combined_signal = torch.stack([
            curiosity_signal,
            causal_learning_rate,
            exploration_bonus,
            emotion.coherence,
        ], dim=-1)
        
        info = {
            'phase_causality': phase_causality,
            'emotion': emotion,
            'causal_effect': effect,
            'combined_signal': combined_signal,
            'emotion_valence': emotion.valence.mean().item() if emotion.valence.dim() > 0 else emotion.valence.item(),
            'emotion_arousal': emotion.arousal.mean().item() if emotion.arousal.dim() > 0 else emotion.arousal.item(),
            'emotion_coherence': emotion.coherence.mean().item() if emotion.coherence.dim() > 0 else emotion.coherence.item(),
        }
        
        return emotion, phase_causality, info
