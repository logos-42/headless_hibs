"""
Twistor-LMT 100M 参数配置版本
=============================
基于正确的扭量架构设计：
- 复数状态空间：z ∈ ℂⁿ
- 连续时间动力学：dz/dt = (-z + W·tanh(z) + U·x + b) / τ(z)
- 状态依赖时间常数：τ(z) = sigmoid(W_τ·|z|)
- 无注意力机制
- 无 Transformer

参数量配置:
- Small: ~1M
- Medium: ~10M  
- Large: ~100M
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TwistorLMTConfig:
    """Twistor-LMT 配置类"""
    input_dim: int = 1
    output_dim: int = 1
    hidden_dim: int = 2048
    n_layers: int = 16
    dt: float = 0.1
    tau_min: float = 0.01
    tau_max: float = 1.0
    sparsity: float = 0.3
    multi_scale_tau: bool = True
    
    @classmethod
    def small(cls):
        """小规模 ~1M 参数"""
        return cls(
            hidden_dim=256,
            n_layers=4,
        )
    
    @classmethod
    def medium(cls):
        """中等规模 ~10M 参数"""
        return cls(
            hidden_dim=512,
            n_layers=8,
        )
    
    @classmethod
    def large_100m(cls):
        """100M 参数大规模"""
        return cls(
            hidden_dim=2048,
            n_layers=16,
        )
    
    def get_param_count(self) -> int:
        """估算单层的参数量 × 层数"""
        # 每层参数:
        # - W_real: hidden²
        # - W_imag: hidden²
        # - W_tau: hidden²
        # - U: input×hidden ≈ hidden (假设 input=hidden)
        # - out: hidden×output ≈ hidden
        # 每层总计：3×hidden² + 2×hidden
        per_layer = 3 * self.hidden_dim ** 2 + 2 * self.hidden_dim
        return self.n_layers * per_layer


class TwistorLMTBlock(nn.Module):
    """
    Twistor-LMT 块 (单层)
    
    核心动力学:
        dz/dt = (-z + W·tanh(z) + U·x + b) / τ(z)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dt: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        sparsity: float = 0.3,
        multi_scale_tau: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dt = dt
        
        # 动力学权重
        self.W_real = nn.Linear(hidden_dim, hidden_dim)
        self.W_imag = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        
        # 偏置
        self.b_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(hidden_dim))
        
        # 稀疏掩码
        self.sparse_mask_real = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        self.sparse_mask_imag = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        
        # 多尺度 τ
        if multi_scale_tau:
            self.tau_bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.tau_bias = None
        
        # 输出投影
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.W_real.weight, gain=0.5)
        nn.init.orthogonal_(self.W_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau.weight, gain=0.1)
        nn.init.zeros_(self.W_real.bias)
        nn.init.zeros_(self.W_imag.bias)
        nn.init.zeros_(self.U.bias)
        nn.init.zeros_(self.W_tau.bias)
    
    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        z_mod = torch.abs(z)
        tau = torch.sigmoid(self.W_tau(z_mod))
        if self.tau_bias is not None:
            tau = tau + self.tau_bias.unsqueeze(0)
        tau = torch.clamp(tau, 0.01, 1.0) + 1e-6
        return tau
    
    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        z_real = z.real
        z_imag = z.imag
        
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)
        
        # 应用稀疏掩码
        W_tanh_real = F.linear(tanh_real, self.W_real.weight * torch.sigmoid(self.sparse_mask_real), self.W_real.bias)
        W_tanh_imag = F.linear(tanh_imag, self.W_imag.weight * torch.sigmoid(self.sparse_mask_imag), self.W_imag.bias)
        
        Ux = self.U(x)
        
        dz_real = -z_real + W_tanh_real + Ux + self.b_real
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag
        
        tau = self.compute_tau(z)
        
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)
        dzdt = torch.clamp(dzdt.real, -10, 10) + 1j * torch.clamp(dzdt.imag, -10, 10)
        
        return dzdt
    
    def forward(self, x_seq: torch.Tensor, return_states: bool = False):
        """
        Args:
            x_seq: 输入序列 (T, B, input_dim)
            return_states: 是否返回状态
        """
        T, B, _ = x_seq.shape
        
        # 初始状态
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x_seq.device)
        
        outputs = []
        states = []
        
        for t in range(T):
            dzdt = self.compute_dzdt(z, x_seq[t])
            z = z + self.dt * dzdt
            z = torch.complex(
                torch.clamp(z.real, -100, 100),
                torch.clamp(z.imag, -100, 100)
            )
            
            y_t = self.out(z.real)
            outputs.append(y_t)
            if return_states:
                states.append(z)
        
        y_seq = torch.stack(outputs, dim=0)
        
        if return_states:
            states = torch.stack(states, dim=0)
            return y_seq, states
        
        return y_seq


class StackedTwistorLMT(nn.Module):
    """
    堆叠式 Twistor-LMT (多层)
    
    架构:
        Input → [TwistorLMT Block]×N → Output
    """
    
    def __init__(self, config: TwistorLMTConfig):
        super().__init__()
        self.config = config
        
        # 输入投影
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # 堆叠多个 Twistor 层
        self.layers = nn.ModuleList()
        for i in range(config.n_layers):
            layer = TwistorLMTBlock(
                input_dim=config.hidden_dim,  # 层间传递
                hidden_dim=config.hidden_dim,
                dt=config.dt,
                tau_min=config.tau_min,
                tau_max=config.tau_max,
                sparsity=config.sparsity,
                multi_scale_tau=config.multi_scale_tau,
            )
            self.layers.append(layer)
        
        # 输出投影
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
    
    def forward(self, x_seq: torch.Tensor, return_states: bool = False):
        """
        前向传播
        
        Args:
            x_seq: 输入序列 (T, B, input_dim)
            return_states: 是否返回状态
        
        Returns:
            y_seq: 输出序列 (T, B, output_dim)
            states: 各层状态 (可选)
        """
        # 输入投影
        h = self.input_proj(x_seq)  # (T, B, hidden_dim)
        
        all_states = []
        
        # 通过各层
        for layer in self.layers:
            h, states = layer(h, return_states=True)
            if return_states:
                all_states.append(states)
        
        # 输出投影
        y_seq = self.output_proj(h)
        
        if return_states:
            return y_seq, all_states
        
        return y_seq
    
    def get_param_count(self) -> int:
        """获取实际参数量"""
        return sum(p.numel() for p in self.parameters())


def create_twistor_100m() -> StackedTwistorLMT:
    """创建 100M 参数 Twistor-LMT"""
    config = TwistorLMTConfig.large_100m()
    return StackedTwistorLMT(config)


def create_twistor_medium() -> StackedTwistorLMT:
    """创建中等规模 Twistor-LMT"""
    config = TwistorLMTConfig.medium()
    return StackedTwistorLMT(config)


def create_twistor_small() -> StackedTwistorLMT:
    """创建小规模 Twistor-LMT"""
    config = TwistorLMTConfig.small()
    return StackedTwistorLMT(config)


if __name__ == "__main__":
    print("=" * 70)
    print("Twistor-LMT 100M 参数版本测试 (纯扭量架构)")
    print("=" * 70)
    
    configs = [
        ("Small (~1M)", TwistorLMTConfig.small()),
        ("Medium (~10M)", TwistorLMTConfig.medium()),
        ("Large (~100M)", TwistorLMTConfig.large_100m()),
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        print(f"  hidden_dim: {config.hidden_dim}")
        print(f"  n_layers: {config.n_layers}")
        print(f"  预估参数：{config.get_param_count():,}")
        
        model = StackedTwistorLMT(config)
        actual_params = model.get_param_count()
        print(f"  实际参数：{actual_params:,}")
        
        # 测试前向传播
        batch_size = 2
        seq_len = 32
        x = torch.randn(seq_len, batch_size, config.input_dim)
        
        with torch.no_grad():
            y = model(x)
        
        print(f"  输入形状：{x.shape}")
        print(f"  输出形状：{y.shape}")
        print(f"  ✅ 测试通过")
    
    print("\n" + "=" * 70)
    print("所有测试完成！这是纯扭量架构，无注意力机制。")
    print("=" * 70)
