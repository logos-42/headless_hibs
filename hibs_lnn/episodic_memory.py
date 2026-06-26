"""
情景记忆系统 - Episodic Memory System
=====================================
生物启发的记忆系统，用于支持因果推理中的反事实推演：

1. 情景记忆 - 存储完整的状态轨迹片段
2. 情景检索 - 基于当前状态检索相似记忆
3. 记忆整合 - 将新经验整合到长期记忆
4. 离线重演 - 支持在不更新真实权重的情况下重演记忆

核心思想：情景记忆是反事实推理的基础 - "如果当时..."
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import copy


@dataclass
class Episode:
    """情景记忆片段"""
    episode_id: int
    states: torch.Tensor  # (sequence_length, hidden_dim)
    actions: Optional[torch.Tensor] = None  # (sequence_length, action_dim)
    rewards: Optional[torch.Tensor] = None  # (sequence_length,)
    metadata: Dict = field(default_factory=dict)
    importance: float = 1.0
    access_count: int = 0
    last_access: int = 0
    consolidation_level: float = 0.0  # 0=短时, 1=长期
    
    
@dataclass
class MemoryQuery:
    """记忆查询"""
    query_state: torch.Tensor
    query_type: str = 'similarity'  # 'similarity', 'temporal', 'reward'
    top_k: int = 5
    reward_filter: Optional[float] = None
    temporal_range: Optional[Tuple[int, int]] = None


class StateEncoder(nn.Module):
    """状态编码器 - 将原始状态压缩为记忆表征"""
    
    def __init__(self, hidden_dim: int, memory_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, (hidden_dim + memory_dim) // 2),
            nn.LayerNorm((hidden_dim + memory_dim) // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear((hidden_dim + memory_dim) // 2, memory_dim),
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)


class SimilarityRetriever(nn.Module):
    """相似性检索器 - 基于状态相似度检索记忆"""
    
    def __init__(self, memory_dim: int):
        super().__init__()
        self.memory_dim = memory_dim
        
        self.similarity_net = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        query: torch.Tensor,
        memory_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算查询与每个记忆的相似度
        
        Returns:
            similarities: (num_memories,)
            indices: top-k索引
        """
        query = query.unsqueeze(0).expand(memory_embeddings.size(0), -1)
        
        combined = torch.cat([query, memory_embeddings], dim=-1)
        
        similarities = self.similarity_net(combined).squeeze(-1)
        
        return similarities


class TemporalEncoder(nn.Module):
    """时序编码器 - 编码时间关系"""
    
    def __init__(self, memory_dim: int, max_seq_len: int = 1000):
        super().__init__()
        self.memory_dim = memory_dim
        self.max_seq_len = max_seq_len
        
        self.time_encoder = nn.Embedding(max_seq_len, memory_dim)
        
        self.rnn = nn.GRU(
            memory_dim, memory_dim, num_layers=2, 
            batch_first=True, bidirectional=True
        )
        
    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码时序状态
        
        Returns:
            encoded: (batch, memory_dim)
            temporal_features: (batch, seq_len, memory_dim)
        """
        batch_size, seq_len, _ = states.shape
        
        positions = torch.arange(seq_len, device=states.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        time_features = self.time_encoder(positions)
        
        rnn_out, hidden = self.rnn(states + time_features)
        
        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]
        
        encoded = torch.cat([forward_hidden, backward_hidden], dim=-1)
        
        return encoded, rnn_out


class EpisodicMemory(nn.Module):
    """
    情景记忆系统
    
    功能:
    - 存储状态轨迹为情景记忆
    - 基于相似度和时间上下文检索
    - 离线重演用于反事实推理
    - 记忆整合和遗忘
    """
    
    def __init__(
        self,
        hidden_dim: int,
        memory_dim: int = 64,
        max_episodes: int = 10000,
        max_seq_len: int = 100,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        self.max_episodes = max_episodes
        self.max_seq_len = max_seq_len
        
        self.state_encoder = StateEncoder(hidden_dim, memory_dim)
        self.similarity_retriever = SimilarityRetriever(memory_dim)
        self.temporal_encoder = TemporalEncoder(memory_dim, max_seq_len)
        
        self.episodes: List[Episode] = []
        self.episode_embeddings: Optional[torch.Tensor] = None
        
        self.register_buffer('episode_counter', torch.tensor(0))
        self.register_buffer('total_accesses', torch.tensor(0))
        
        self.priority_net = nn.Sequential(
            nn.Linear(memory_dim * 3, memory_dim),
            nn.ReLU(),
            nn.Linear(memory_dim, 1),
            nn.Sigmoid(),
        )
        
    def store(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None,
        importance: float = 1.0,
    ) -> int:
        """
        存储新的情景记忆
        
        Args:
            states: (seq_len, hidden_dim) 状态轨迹
            actions: (seq_len, action_dim) 可选的动作序列
            rewards: (seq_len,) 可选的奖励序列
            metadata: 额外元数据
            importance: 记忆重要性分数
            
        Returns:
            episode_id: 新记忆的ID
        """
        if states.size(0) > self.max_seq_len:
            states = states[:self.max_seq_len]
            
        episode_id = self.episode_counter.item()
        self.episode_counter += 1
        
        episode = Episode(
            episode_id=episode_id,
            states=states,
            actions=actions,
            rewards=rewards,
            metadata=metadata or {},
            importance=importance,
        )
        
        self.episodes.append(episode)
        
        if len(self.episodes) > self.max_episodes:
            self._evict_oldest()
            
        self._update_embeddings()
        
        return episode_id
        
    def retrieve(
        self,
        query: MemoryQuery,
    ) -> List[Tuple[Episode, float]]:
        """
        检索相关记忆
        
        Args:
            query: 记忆查询条件
            
        Returns:
            retrieved: (episode, score)列表，按相关性排序
        """
        if not self.episodes:
            return []
            
        with torch.no_grad():
            query_embedding = self.state_encoder(query.query_state)
            
            if query.query_type == 'similarity':
                similarities = F.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    self.episode_embeddings,
                    dim=-1
                )
            elif query.query_type == 'temporal':
                encoded, _ = self.temporal_encoder(
                    query.query_state.unsqueeze(0)
                )
                similarities = F.cosine_similarity(
                    encoded,
                    self.episode_embeddings,
                    dim=-1
                )
            else:
                similarities = torch.ones(len(self.episodes))
                
            if query.reward_filter is not None:
                reward_scores = torch.tensor([
                    ep.rewards.mean().item() if ep.rewards is not None else 0.0
                    for ep in self.episodes
                ])
                similarities = similarities * (reward_scores > query.reward_filter).float()
                
            top_k = min(query.top_k, len(self.episodes))
            scores, indices = torch.topk(similarities, top_k)
            
            results = []
            for score, idx in zip(scores.tolist(), indices.tolist()):
                episode = self.episodes[idx]
                episode.access_count += 1
                episode.last_access = self.episode_counter.item()
                results.append((episode, score))
                
        return results
        
    def replay(
        self,
        episode_id: int,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        重演指定记忆
        
        用于离线模拟和反事实推理
        
        Returns:
            states: 重演的状态轨迹
            actions: 对应动作（如有）
            rewards: 对应奖励（如有）
        """
        if episode_id < 0 or episode_id >= len(self.episodes):
            raise ValueError(f"Episode {episode_id} not found")
            
        episode = self.episodes[episode_id]
        
        start = start_idx or 0
        end = end_idx or episode.states.size(0)
        
        return (
            episode.states[start:end],
            episode.actions[start:end] if episode.actions is not None else None,
            episode.rewards[start:end] if episode.rewards is not None else None,
        )
        
    def compute_counterfactual(
        self,
        current_state: torch.Tensor,
        retrieved_episode: Episode,
        intervention_fn: callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算反事实：基于当前状态和检索到的记忆
        
        1. 找到记忆中最接近当前状态的时刻
        2. 应用干预
        3. 模拟后续轨迹
        
        Args:
            current_state: (hidden_dim,) 当前状态
            retrieved_episode: 检索到的记忆
            intervention_fn: 干预函数 (state -> intervened_state)
            
        Returns:
            factual_trajectory: 原始轨迹
            counterfactual_trajectory: 反事实轨迹
        """
        similarities = F.cosine_similarity(
            current_state.unsqueeze(0),
            retrieved_episode.states,
            dim=-1
        )
        
        most_similar_idx = similarities.argmax().item()
        
        factual_trajectory = retrieved_episode.states[most_similar_idx:]
        
        cf_state = intervention_fn(retrieved_episode.states[most_similar_idx])
        
        cf_trajectory = [cf_state]
        current_cf = cf_state
        
        for t in range(most_similar_idx + 1, retrieved_episode.states.size(0)):
            next_state = intervention_fn(retrieved_episode.states[t])
            cf_trajectory.append(next_state)
            
        counterfactual_trajectory = torch.stack(cf_trajectory, dim=0)
        
        return factual_trajectory, counterfactual_trajectory
        
    def consolidate(self, episode_id: int):
        """
        整合记忆到长期记忆
        
        增加记忆的consolidation_level
        """
        if 0 <= episode_id < len(self.episodes):
            episode = self.episodes[episode_id]
            episode.consolidation_level = min(1.0, episode.consolidation_level + 0.1)
            
    def _update_embeddings(self):
        """更新所有记忆的嵌入表示"""
        if not self.episodes:
            return None
            
        embeddings = []
        for episode in self.episodes:
            episode_states = episode.states
            
            if episode_states.size(0) == 0:
                embeddings.append(torch.zeros(self.memory_dim, device=episode_states.device))
                continue
                
            encoded, _ = self.temporal_encoder(episode_states.unsqueeze(0))
            embeddings.append(encoded.squeeze(0))
            
        self.episode_embeddings = torch.stack(embeddings, dim=0)
        
    def _evict_oldest(self):
        """驱逐最老的记忆（简单FIFO）"""
        if self.episodes:
            self.episodes.pop(0)
            
    def get_statistics(self) -> Dict:
        """获取记忆系统统计"""
        return {
            'total_episodes': len(self.episodes),
            'total_accesses': self.total_accesses.item(),
            'avg_importance': np.mean([ep.importance for ep in self.episodes]) if self.episodes else 0,
            'avg_consolidation': np.mean([ep.consolidation_level for ep in self.episodes]) if self.episodes else 0,
            'episodes_with_rewards': sum(1 for ep in self.episodes if ep.rewards is not None),
        }
        
    def reset(self):
        """重置记忆系统"""
        self.episodes.clear()
        self.episode_embeddings = None
        self.episode_counter.fill_(0)
        self.total_accesses.fill_(0)


class HippocampalSystem(nn.Module):
    """
    海马体系统 - 整合情景记忆和空间导航
    
    模拟海马体的功能：
    1. 情景记忆编码和提取
    2. 位置细胞和网格细胞表征
    3. 睡眠期间的记忆重演
    """
    
    def __init__(self, hidden_dim: int, memory_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_dim = memory_dim
        
        self.episodic_memory = EpisodicMemory(hidden_dim, memory_dim)
        
        self.place_cell_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, memory_dim),
        )
        
        self.grid_cell_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, memory_dim),
        )
        
        self.memory_consolidation_rnn = nn.GRU(
            memory_dim, memory_dim, num_layers=2, batch_first=True
        )
        
    def encode_and_store(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        rewards: Optional[torch.Tensor] = None,
    ) -> int:
        """
        编码并存储记忆
        """
        episode_id = self.episodic_memory.store(
            states, actions, rewards
        )
        
        return episode_id
        
    def retrieve_similar(
        self,
        state: torch.Tensor,
        top_k: int = 5,
    ) -> List[Tuple[Episode, float]]:
        """
        检索相似记忆
        """
        query = MemoryQuery(
            query_state=state,
            query_type='similarity',
            top_k=top_k,
        )
        
        return self.episodic_memory.retrieve(query)
        
    def offline_replay(
        self,
        num_replays: int = 10,
    ) -> List[torch.Tensor]:
        """
        离线重演 - 模拟睡眠期间的记忆重演
        
        用于：
        1. 记忆整合
        2. 泛化学习
        3. 创造性地组合记忆
        """
        if not self.episodic_memory.episodes:
            return []
            
        replays = []
        
        for _ in range(num_replays):
            episode = np.random.choice(self.episodic_memory.episodes)
            
            start_idx = np.random.randint(0, max(1, episode.states.size(0) - 1))
            
            replay_segment = episode.states[start_idx:]
            
            replays.append(replay_segment)
            
            if episode.consolidation_level < 1.0:
                self.episodic_memory.consolidate(episode.episode_id)
                
        return replays
        
    def forward(
        self,
        state: torch.Tensor,
        store: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Returns:
            place_embedding: 位置细胞表征
            grid_embedding: 网格细胞表征
        """
        place_embedding = self.place_cell_net(state)
        grid_embedding = self.grid_cell_encoder(state)
        
        if store:
            self.episodic_memory.store(state.unsqueeze(0))
            
        return place_embedding, grid_embedding
