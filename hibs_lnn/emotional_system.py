"""
情绪系统 - Emotional System
===========================
完整的情绪-因果闭环系统：

1. 情绪状态 - valence(效价), arousal(唤醒度), dominance(控制感)
2. 情绪反应 - 输出如何产生情绪
3. 情绪影响 - 情绪如何调制处理过程
4. 情绪记忆 - 情境-情绪关联
5. 情绪-因果闭环 - 情绪影响因果推理，因果结果影响情绪

核心思想：情绪是因果推理的"内在尺度"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field


@dataclass
class EmotionalState:
    """情绪状态"""
    valence: torch.Tensor      # 效价: -1(负面) ~ +1(正面)
    arousal: torch.Tensor       # 唤醒度: 0(平静) ~ 1(兴奋)
    dominance: torch.Tensor    # 控制感: 0(被动) ~ 1(主动)
    
    @classmethod
    def create(cls, batch_size: int, device: torch.device):
        return cls(
            valence=torch.zeros(batch_size, device=device),
            arousal=torch.zeros(batch_size, device=device),
            dominance=torch.zeros(batch_size, device=device),
        )
    
    def to_vector(self) -> torch.Tensor:
        v = self.valence.unsqueeze(-1) if self.valence.dim() == 1 else self.valence
        a = self.arousal.unsqueeze(-1) if self.arousal.dim() == 1 else self.arousal
        d = self.dominance.unsqueeze(-1) if self.dominance.dim() == 1 else self.dominance
        return torch.cat([v, a, d], dim=-1)


@dataclass
class EmotionReaction:
    """情绪反应"""
    emotion_type: str          # 情绪类型: joy, sadness, anger, fear, surprise, disgust
    intensity: torch.Tensor     # 强度
    valence_change: torch.Tensor
    arousal_change: torch.Tensor
    novelty: torch.Tensor       # 新奇性


class EmotionalEncoder(nn.Module):
    """情绪编码器 - 将输出编码为情绪反应"""
    
    def __init__(self, output_dim: int, emotional_dim: int = 64):
        super().__init__()
        self.output_dim = output_dim
        self.emotional_dim = emotional_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(output_dim, emotional_dim),
            nn.LayerNorm(emotional_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emotional_dim, emotional_dim // 2),
            nn.ReLU(),
        )
        
        self.valence_net = nn.Linear(emotional_dim // 2, 1)
        self.arousal_net = nn.Linear(emotional_dim // 2, 1)
        self.dominance_net = nn.Linear(emotional_dim // 2, 1)
        self.emotion_type_net = nn.Linear(emotional_dim // 2, 6)
        
    def forward(self, output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(output)
        
        valence = torch.tanh(self.valence_net(encoded))
        arousal = torch.sigmoid(self.arousal_net(encoded))
        dominance = torch.sigmoid(self.dominance_net(encoded))
        emotion_logits = self.emotion_type_net(encoded)
        
        return valence, arousal, dominance, emotion_logits


class EmotionalModulator(nn.Module):
    """情绪调制器 - 情绪如何影响信息处理"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.attention_gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        self.memory_gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
        self.decision_gate = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        state: torch.Tensor,
        emotional_state: EmotionalState,
    ) -> Dict[str, torch.Tensor]:
        emotion_vec = emotional_state.to_vector()
        
        if emotion_vec.dim() > 1:
            emotion_vec = emotion_vec.mean(dim=0)
        
        attention_mod = self.attention_gate(emotion_vec)
        memory_mod = self.memory_gate(emotion_vec)
        decision_mod = self.decision_gate(emotion_vec)
        
        return {
            'attention_gate': attention_mod,
            'memory_gate': memory_mod,
            'decision_gate': decision_mod,
        }


class EmotionalMemory(nn.Module):
    """情绪记忆 - 情境与情绪的关联"""
    
    def __init__(self, state_dim: int, memory_size: int = 1000):
        super().__init__()
        self.state_dim = state_dim
        self.memory_size = memory_size
        
        self.context_encoder = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.LayerNorm(state_dim),
            nn.ReLU(),
        )
        
        self.memory = None
        self.emotion_memory = None
        self.memory_ptr = 0
        self.memory_full = False
        
        self.emotion_predictor = nn.Sequential(
            nn.Linear(state_dim * 2, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, 3),
            nn.Tanh(),
        )
        
    def init_memory(self, device: torch.device):
        self.memory = torch.zeros(self.memory_size, self.state_dim, device=device)
        self.emotion_memory = torch.zeros(self.memory_size, 3, device=device)
        self.memory_ptr = 0
        self.memory_full = False
        
    def store(self, state: torch.Tensor, emotion: EmotionalState):
        if self.memory is None:
            self.init_memory(state.device)
            
        if state.size(0) > 1:
            state = state[0:1]
        if emotion.valence.size(0) > 1:
            emotion = EmotionalState(
                valence=emotion.valence[0:1],
                arousal=emotion.arousal[0:1],
                dominance=emotion.dominance[0:1],
            )
            
        encoded = self.context_encoder(state)
        
        idx = self.memory_ptr % self.memory_size
        self.memory[idx] = encoded.detach()
        
        emotion_vec = torch.stack([
            emotion.valence.mean(),
            emotion.arousal.mean(),
            emotion.dominance.mean(),
        ], dim=0)
        self.emotion_memory[idx] = emotion_vec.detach()
        self.memory_ptr += 1
        if self.memory_ptr >= self.memory_size:
            self.memory_full = True
            
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.memory_full or self.memory_ptr > 0:
            valid_mem = self.memory[:self.memory_ptr] if not self.memory_full else self.memory
            
            if query.dim() > 2:
                query = query.squeeze(0)
            elif query.dim() == 2 and query.size(0) == 1:
                query = query.squeeze(0)
            
            if query.dim() == 1:
                query_expanded = query.unsqueeze(0).expand(valid_mem.size(0), -1)
            else:
                query_expanded = query.unsqueeze(0).expand(valid_mem.size(0), -1)
                
            similarities = F.cosine_similarity(
                query_expanded,
                valid_mem,
                dim=-1
            )
            
            top_k = min(top_k, len(similarities))
            scores, indices = torch.topk(similarities, top_k)
            
            retrieved_emotions = self.emotion_memory[indices]
            
            return retrieved_emotions, scores
        else:
            return torch.zeros(1, 3, device=query.device), torch.zeros(1, device=query.device)
            
    def predict_emotion(self, state: torch.Tensor, retrieved_emotion: torch.Tensor) -> EmotionalState:
        combined = torch.cat([state, retrieved_emotion.mean(dim=0)], dim=-1)
        predicted = self.emotion_predictor(combined)
        
        return EmotionalState(
            valence=predicted[:, 0],
            arousal=predicted[:, 1],
            dominance=predicted[:, 2],
        )


class EmotionalLearning(nn.Module):
    """情绪学习 - 从经验中学习情绪模式"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.reward_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self.baseline = torch.tensor(0.0)
        
    def compute_emotional_reward(
        self,
        current_emotion: EmotionalState,
        expected_emotion: EmotionalState,
        outcome: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        valence_error = (current_emotion.valence - expected_emotion.valence) ** 2
        arousal_error = (current_emotion.arousal - expected_emotion.arousal) ** 2
        dominance_error = (current_emotion.dominance - expected_emotion.dominance) ** 2
        
        reward = -(valence_error + arousal_error + dominance_error)
        
        outcome_reward = self.reward_net(outcome).squeeze(-1)
        
        total_reward = reward + 0.1 * outcome_reward
        
        td_error = total_reward - self.baseline.detach()
        
        self.baseline = self.baseline * 0.99 + total_reward.detach() * 0.01
        
        info = {
            'valence_error': valence_error.mean().item(),
            'arousal_error': arousal_error.mean().item(),
            'dominance_error': dominance_error.mean().item(),
            'td_error': td_error.mean().item(),
            'baseline': self.baseline.item(),
        }
        
        return td_error, info


class EmotionalSystem(nn.Module):
    """
    完整情绪系统
    
    工作流程:
    1. 输出 → 情绪反应 (EmotionalEncoder)
    2. 当前情绪 + 反应 → 新情绪
    3. 情绪 → 调制处理 (EmotionalModulator)
    4. 情境 → 情绪记忆检索
    5. 情绪学习 (EmotionalLearning)
    """
    
    def __init__(self, output_dim: int, hidden_dim: int, memory_size: int = 1000):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = EmotionalEncoder(output_dim, hidden_dim // 2)
        self.modulator = EmotionalModulator(hidden_dim)
        self.memory = EmotionalMemory(hidden_dim, memory_size)
        self.learning = EmotionalLearning(hidden_dim)
        
        self.emotion_decay = 0.95
        self.emotion_momentum = 0.9
        
        self.emotion_history: List[EmotionalState] = []
        
    def compute_reaction(
        self,
        output: torch.Tensor,
        current_emotion: Optional[EmotionalState] = None,
    ) -> EmotionReaction:
        valence, arousal, dominance, emotion_logits = self.encoder(output)
        
        emotion_type_idx = emotion_logits.argmax(dim=-1)
        emotion_types = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        emotion_type = [emotion_types[i] for i in emotion_type_idx.tolist()]
        
        intensity = torch.sigmoid(emotion_logits.max(dim=-1)[0])
        
        novelty = torch.std(output, dim=-1) if output.dim() > 1 else torch.tensor(0.5)
        
        valence_change = valence
        arousal_change = intensity * 0.5
        
        return EmotionReaction(
            emotion_type=emotion_type[0] if isinstance(emotion_type, list) else emotion_type,
            intensity=intensity,
            valence_change=valence_change,
            arousal_change=arousal_change,
            novelty=novelty,
        )
        
    def update_emotion(
        self,
        current: EmotionalState,
        reaction: EmotionReaction,
        retrieved_emotion: Optional[torch.Tensor] = None,
    ) -> EmotionalState:
        valence_delta = reaction.valence_change * 0.3
        arousal_delta = reaction.arousal_change * 0.2
        dominance_delta = reaction.novelty * 0.1
        
        if retrieved_emotion is not None:
            if retrieved_emotion.dim() > 1:
                retrieved_mean = retrieved_emotion.mean(dim=0)
            else:
                retrieved_mean = retrieved_emotion
            valence_delta = valence_delta + retrieved_mean[0] * 0.1
            arousal_delta = arousal_delta + retrieved_mean[1] * 0.1
            dominance_delta = dominance_delta + retrieved_mean[2] * 0.1
            
        new_valence = current.valence * self.emotion_decay + valence_delta * (1 - self.emotion_decay)
        new_arousal = current.arousal * self.emotion_decay + arousal_delta * (1 - self.emotion_decay)
        new_dominance = current.dominance * self.emotion_momentum + dominance_delta * (1 - self.emotion_momentum)
        
        new_valence = torch.clamp(new_valence, -1.0, 1.0)
        new_arousal = torch.clamp(new_arousal, 0.0, 1.0)
        new_dominance = torch.clamp(new_dominance, 0.0, 1.0)
        
        return EmotionalState(
            valence=new_valence,
            arousal=new_arousal,
            dominance=new_dominance,
        )
        
    def modulate(
        self,
        state: torch.Tensor,
        emotion: EmotionalState,
    ) -> Dict[str, torch.Tensor]:
        return self.modulator(state, emotion)
        
    def learn(
        self,
        state: torch.Tensor,
        current_emotion: EmotionalState,
        expected_emotion: EmotionalState,
        outcome: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        return self.learning.compute_emotional_reward(
            current_emotion, expected_emotion, outcome
        )
        
    def forward(
        self,
        output: torch.Tensor,
        state: torch.Tensor,
        current_emotion: Optional[EmotionalState] = None,
    ) -> Tuple[EmotionalState, Dict]:
        reaction = self.compute_reaction(output, current_emotion)
        
        retrieved_emotion, retrieval_scores = self.memory.retrieve(state)
        
        if current_emotion is None:
            current_emotion = EmotionalState.create(output.size(0), output.device)
            
        new_emotion = self.update_emotion(current_emotion, reaction, retrieved_emotion)
        
        self.memory.store(state, new_emotion)
        
        modulations = self.modulate(state, new_emotion)
        
        info = {
            'reaction_type': reaction.emotion_type,
            'reaction_intensity': reaction.intensity.mean().item(),
            'valence': new_emotion.valence.mean().item(),
            'arousal': new_emotion.arousal.mean().item(),
            'dominance': new_emotion.dominance.mean().item(),
            'retrieval_score': retrieval_scores.mean().item() if retrieval_scores.numel() > 0 else 0,
        }
        
        self.emotion_history.append(new_emotion)
        if len(self.emotion_history) > 100:
            self.emotion_history.pop(0)
            
        return new_emotion, info


class EmotionalCausalLoop(nn.Module):
    """
    情绪-因果闭环
    
    核心：情绪和因果互相影响
    - 因果结果 → 情绪反应
    - 情绪状态 → 调制因果推理
    - 情绪记忆 → 影响因果预期
    """
    
    def __init__(self, output_dim: int, hidden_dim: int, memory_size: int = 1000):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.emotional_system = EmotionalSystem(output_dim, hidden_dim, memory_size)
        
        self.curiosity_weight = nn.Parameter(torch.tensor(1.0))
        self.emotion_weight = nn.Parameter(torch.tensor(0.5))
        
    def forward(
        self,
        state: torch.Tensor,
        output: torch.Tensor,
        causal_effect: torch.Tensor,
        current_emotion: Optional[EmotionalState] = None,
    ) -> Tuple[EmotionalState, Dict]:
        new_emotion, emotion_info = self.emotional_system(output, state, current_emotion)
        
        causal_emotion_interaction = torch.cat([
            causal_effect.mean(dim=-1, keepdim=True),
            new_emotion.to_vector(),
        ], dim=-1)
        
        curiosity_mod = torch.sigmoid(self.curiosity_weight * new_emotion.arousal)
        emotion_mod = torch.tanh(self.emotion_weight * new_emotion.valence)
        
        info = {
            'emotion': emotion_info,
            'causal_emotion_interaction': causal_emotion_interaction,
            'curiosity_modulation': curiosity_mod.mean().item(),
            'emotion_modulation': emotion_mod.mean().item(),
        }
        
        return new_emotion, info
