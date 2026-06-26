"""
可微分外部记忆系统 - Neural Turing Machine 风格
==============================================
为液态神经网络添加可读写的工作记忆

核心特性:
1. Content-based addressing - 基于内容寻址
2. Location-based addressing - 基于位置寻址
3. Erase and Add mechanism - 擦除和添加机制
4. 专为连续时间动力学设计
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class MemoryConfig:
    """记忆配置"""
    memory_size: int = 128        # 记忆槽位数
    key_dim: int = 64            # 键维度
    value_dim: int = 64           # 值维度
    key_strength_init: float = 1.0 # 注意力强度初始值
    read_strength: float = 0.1    # 读取强度
    write_strength: float = 0.1   # 写入强度
    usage_decay: float = 0.99      # 使用频率衰减
    temporal_link_strength: float = 0.5  # 时间链接强度


class DifferentiableMemory(nn.Module):
    """
    可微分外部记忆

    工作原理:
    1. 读取: query -> content addressing -> 读取相关记忆
    2. 写入: key-value -> erase & add -> 更新记忆

    记忆矩阵 M ∈ ℝ^(N×H):
    - N: 记忆槽位数
    - H: 值维度
    """

    def __init__(
        self,
        memory_size: int = 128,
        key_dim: int = 64,
        value_dim: int = 64,
        config: Optional[MemoryConfig] = None,
    ):
        super().__init__()

        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.config = config or MemoryConfig()

        # 记忆矩阵
        self.M = nn.Parameter(
            torch.randn(memory_size, value_dim) * 0.1
        )

        # 使用频率追踪 (用于LRU替换)
        self.register_buffer('usage', torch.zeros(memory_size))

        # 读写头网络
        self.read_key_net = nn.Linear(key_dim, value_dim)
        self.write_key_net = nn.Linear(key_dim, value_dim)
        self.write_value_net = nn.Linear(key_dim, value_dim)

        # 寻址参数
        self.key_strength = nn.Parameter(
            torch.ones(1) * self.config.key_strength_init
        )

        # 位置移动核 (用于循环移动)
        self.shift_kernel = nn.Parameter(
            torch.tensor([0.5, 0.3, 0.2])
        )

        # 门控网络
        self.erase_gate = nn.Sequential(
            nn.Linear(key_dim, value_dim),
            nn.Sigmoid()
        )
        self.add_gate = nn.Sequential(
            nn.Linear(key_dim, value_dim),
            nn.Tanh()
        )

        # 读写统计
        self.read_history = []
        self.write_history = []

    def content_addressing(
        self,
        query: torch.Tensor,
        strength: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        基于内容的寻址

        Args:
            query: [B, key_dim] 查询向量
            strength: [B, 1] 注意力强度 (可选)

        Returns:
            weights: [B, N] 注意力权重
        """
        # 投影到值空间
        k = self.read_key_net(query)  # [B, value_dim]

        # 计算余弦相似度
        k_norm = F.normalize(k, p=2, dim=-1)  # [B, value_dim]
        m_norm = F.normalize(self.M, p=2, dim=-1)  # [N, value_dim]

        # 相似度分数
        similarity = k_norm @ m_norm.t()  # [B, N]

        # 应用强度
        if strength is None:
            strength = F.softplus(self.key_strength)  # [1]

        strength = strength.unsqueeze(-1) if strength.dim() == 1 else strength
        weights = F.softmax(similarity * strength, dim=-1)

        return weights

    def location_addressing(
        self,
        weights: torch.Tensor,
        shift: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        基于位置的寻址 (循环移动)

        Args:
            weights: [B, N] 当前权重
            shift: [B,] 移动量 (可选)

        Returns:
            shifted_weights: [B, N] 移动后的权重
        """
        if shift is None:
            # 使用可学习的移动核
            shift_weights = F.softmax(self.shift_kernel, dim=0)
        else:
            # 动态shift (简化版)
            shift_weights = F.softmax(
                torch.tensor([1 - shift, 0.5, shift], device=weights.device),
                dim=0
            )

        # 卷积实现循环移动
        shifted = F.conv1d(
            weights.unsqueeze(1),  # [B, 1, N]
            shift_weights.view(1, 1, 3),  # [1, 1, 3]
            padding=1,
            groups=1,
        ).squeeze(1)  # [B, N]

        # 归一化
        shifted_weights = F.softmax(shifted, dim=-1)

        return shifted_weights

    def read(
        self,
        query: torch.Tensor,
        prev_weights: Optional[torch.Tensor] = None,
        use_location: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        读取记忆

        Args:
            query: [B, key_dim] 查询向量
            prev_weights: [B, N] 上次读取权重 (用于链式读取)
            use_location: 是否使用位置寻址

        Returns:
            read_content: [B, value_dim] 读取的内容
            weights: [B, N] 最终注意力权重
        """
        # Content addressing
        weights = self.content_addressing(query)

        # Location addressing (可选)
        if use_location and prev_weights is not None:
            weights = self.location_addressing(weights)

        # 读取内容
        read_content = weights @ self.M  # [B, value_dim]

        # 更新使用频率
        with torch.no_grad():
            self.usage *= self.config.usage_decay
            self.usage += weights.mean(dim=0)

        # 记录
        self.read_history.append({
            'query_norm': query.norm().item(),
            'read_content_norm': read_content.norm().item(),
            'weights_entropy': -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean().item(),
        })

        return read_content, weights

    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        erase_strength: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        写入记忆 (Erase and Add)

        Args:
            key: [B, key_dim] 写入键
            value: [B, value_dim] 写入值
            erase_strength: 擦除强度

        Returns:
            erase_vector: [N, value_dim] 擦除向量
            add_vector: [N, value_dim] 添加向量
        """
        # 计算写入权重
        k = self.write_key_net(key)
        strength = F.softplus(self.key_strength)
        write_weights = F.softmax(
            (k @ F.normalize(self.M, p=2, dim=-1).t()) * strength,
            dim=-1
        )  # [B, N]

        # 擦除门
        erase = self.erase_gate(key) * erase_strength  # [B, value_dim]
        erase_vector = write_weights.unsqueeze(-1) * erase.unsqueeze(1)  # [B, N, value_dim]

        # 添加门
        add = self.add_gate(key)  # [B, value_dim]
        add_vector = write_weights.unsqueeze(-1) * add.unsqueeze(1)  # [B, N, value_dim]

        # 更新记忆
        with torch.no_grad():
            # 擦除: M = M * (1 - erase)
            self.M.data = self.M.data * (1 - erase_vector.mean(dim=0))
            # 添加: M = M + add
            self.M.data = self.M.data + self.config.write_strength * add_vector.mean(dim=0)

        # 记录
        self.write_history.append({
            'key_norm': key.norm().item(),
            'value_norm': value.norm().item(),
            'write_weights_max': write_weights.max(dim=-1)[0].mean().item(),
        })

        return erase_vector.mean(dim=0), add_vector.mean(dim=0)

    def free_gate(self) -> torch.Tensor:
        """
        计算释放门 (Least Recently Used)

        用于选择最不重要的记忆位置进行覆盖
        """
        # 使用频率越低越容易被释放
        free_prob = 1 - F.softmax(self.usage, dim=0)
        return free_prob

    def allocate(self, n_slots: int = 1) -> torch.Tensor:
        """
        分配空闲记忆槽

        Args:
            n_slots: 需要分配的槽数

        Returns:
            indices: 分配的槽索引
        """
        free = self.free_gate()
        _, indices = free.topk(min(n_slots, self.memory_size))
        return indices

    def forward(
        self,
        query: torch.Tensor,
        write_key: Optional[torch.Tensor] = None,
        write_value: Optional[torch.Tensor] = None,
        prev_read: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        完整读写操作

        Args:
            query: [B, key_dim] 查询向量
            write_key: [B, key_dim] 写入键 (可选)
            write_value: [B, value_dim] 写入值 (可选)
            prev_read: [B, value_dim] 上次读取 (可选)

        Returns:
            result: 包含read_content, read_weights, write_weights的字典
        """
        # 读取
        read_content, read_weights = self.read(query, prev_read)

        # 写入
        write_weights = None
        if write_key is not None and write_value is not None:
            _, write_weights = self.write(write_key, write_value)

        return {
            'read_content': read_content,
            'read_weights': read_weights,
            'write_weights': write_weights,
        }

    def get_stats(self) -> Dict:
        """获取记忆统计"""
        return {
            'memory_norm': self.M.norm().item(),
            'memory_std': self.M.std().item(),
            'usage_mean': self.usage.mean().item(),
            'usage_std': self.usage.std().item(),
            'n_reads': len(self.read_history),
            'n_writes': len(self.write_history),
        }

    def reset_stats(self):
        """重置统计"""
        self.read_history.clear()
        self.write_history.clear()


class MemoryAugmentedLMT(nn.Module):
    """
    记忆增强的液态神经网络

    将可微分记忆与ODE动力学整合
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        memory_size: int = 128,
        read_strength: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.read_strength = read_strength

        # 核心液态网络 (使用较简单的版本避免循环导入)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_amplitude = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.5)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        self.out = nn.Linear(hidden_dim, output_dim)

        # 稀疏掩码
        self.sparse_mask = nn.Parameter(torch.ones(hidden_dim, hidden_dim) * -5)

        # 时间常数
        self.dt = 0.1
        self.tau_min = 0.01
        self.tau_max = 1.0

        # 外部记忆
        self.memory = DifferentiableMemory(
            memory_size=memory_size,
            key_dim=hidden_dim,
            value_dim=hidden_dim,
        )

        # 记忆接口
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        # 记忆读写统计
        self.memory_access_history = []

    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """计算状态导数"""
        z_real = z.real
        z_imag = z.imag

        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        # 复数权重
        mask = torch.sigmoid(self.sparse_mask)
        amp = self.W_amplitude * mask
        phase = torch.zeros_like(amp)
        W = amp * torch.exp(1j * phase)

        W_real = W.real
        W_imag = W.imag

        # 矩阵乘法
        W_tanh_real = F.linear(tanh_real, W_real) - F.linear(tanh_imag, W_imag)
        W_tanh_imag = F.linear(tanh_real, W_imag) + F.linear(tanh_imag, W_real)

        Ux = self.U(x)

        dz_real = -z_real + W_tanh_real + Ux + self.b
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b

        # 时间常数
        tau = F.sigmoid(self.W_tau(torch.abs(z)))
        tau = torch.clamp(tau, self.tau_min, self.tau_max)

        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        return dzdt

    def read_memory(self, query: torch.Tensor) -> torch.Tensor:
        """读取记忆并返回"""
        result = self.memory(query)
        return result['read_content']

    def write_memory(self, key: torch.Tensor, value: torch.Tensor):
        """写入记忆"""
        self.memory.write(key, value)

    def forward(
        self,
        x: torch.Tensor,
        write_memory: bool = True,
        return_memory_stats: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播

        Args:
            x: [seq_len, batch, input_dim]
            write_memory: 是否写入记忆
            return_memory_stats: 是否返回记忆统计

        Returns:
            outputs: [seq_len, batch, output_dim]
            memory_stats: 记忆统计 (可选)
        """
        T, B, _ = x.shape

        # 初始化隐藏状态
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        # 初始化prev_read
        prev_read = torch.zeros(B, self.hidden_dim, device=x.device)

        outputs = []
        read_weights_history = []

        for t in range(T):
            x_t = x[t]

            # 1. 生成查询
            query = self.query_proj(z.real)  # [B, hidden]

            # 2. 读取记忆
            read_result = self.memory(query, prev_read=prev_read)
            memory_read = read_result['read_content']
            read_weights_history.append(read_result['read_weights'])

            # 3. ODE更新 + 记忆融合
            dzdt = self.compute_dzdt(z, x_t)

            # 记忆增强
            memory_complex = torch.complex(memory_read, torch.zeros_like(memory_read))
            dzdt = dzdt + self.read_strength * memory_complex

            z = z + self.dt * dzdt

            # 限制状态
            z = torch.complex(
                torch.clamp(z.real, -100, 100),
                torch.clamp(z.imag, -100, 100),
            )

            # 4. 写入记忆 (周期性和基于重要性)
            if write_memory and t % 5 == 0:
                key = self.key_proj(z.real)
                value = self.value_proj(z.real)
                self.write_memory(key, value)

            # 5. 更新prev_read
            prev_read = memory_read

            # 6. 输出
            y_t = self.out(z.real)
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=0)

        if return_memory_stats:
            return outputs, {
                'read_weights': torch.stack(read_weights_history),
                'memory_content': self.memory.M.detach(),
                'memory_stats': self.memory.get_stats(),
            }

        return outputs

    def reset_state(self, batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
        """重置隐藏状态"""
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.complex64, device=device)

    def get_diagnostics(self) -> Dict:
        """获取诊断信息"""
        return {
            'hidden_dim': self.hidden_dim,
            'memory_size': self.memory_size,
            'read_strength': self.read_strength,
            **self.memory.get_stats(),
        }
