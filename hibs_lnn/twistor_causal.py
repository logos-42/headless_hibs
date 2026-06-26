"""
扭量因果模块 - Twistor Causal Module
====================================
在复数扭量空间中实现因果推理：

1. 复数状态干预 - do(Z = value) 直接修改复数状态
2. 相位因果 - 基于相位关系的因果强度计算
3. 幅值因果 - 基于幅值变化的因果效应
4. 复数ODE反事实推理 - 在复数空间中模拟干预效果

核心思想：相位是因果关系的本质载体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass


@dataclass
class TwistorCausalConfig:
    """扭量因果配置"""
    hidden_dim: int = 128
    resonance_gamma: float = 1.0
    intervention_cost_weight: float = 0.01
    causal_effect_scale: float = 1.0
    sparse_init: float = 0.0  # 0.0≈50%活跃 (增强模式推荐)
    use_learned_sparsity: bool = True
    sparsity_mode: str = "enhanced"  # "multiplicative" | "additive" | "receptor" | "hybrid" | "enhanced"


class PhaseCausalCalculator(nn.Module):
    """
    相位因果计算器
    
    基于相位关系的因果强度：
    相位相近 → 因果关联强（共振）
    相位相反 → 因果关联弱（相消）
    
    支持多种稀疏模式：
    1. multiplicative: 因果 = |cos|^γ × sigmoid(mask) [因果会被削弱]
    2. additive: 因果 = |cos|^γ + 0.1×sigmoid(mask) [保持因果强度]
    3. receptor: 因果 = |cos|^γ × receptor_gate(z) [动态、输入依赖]
    4. hybrid: 因果 = scale × |cos|^γ × mask × gate [平衡方案]
    5. enhanced: 因果 = 0.5×base + 0.5×(base + enhancement) [最佳增强]
    """
    
    def __init__(self, hidden_dim: int, gamma: float = 1.0, 
                 sparse_init: float = 0.0, use_learned: bool = True,
                 sparsity_mode: str = "enhanced"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.use_learned = use_learned
        self.sparsity_mode = sparsity_mode
        
        if use_learned:
            self.sparse_mask = nn.Parameter(torch.ones(hidden_dim, hidden_dim) * sparse_init)
            
            # 受体机制参数
            if sparsity_mode in ["receptor", "hybrid", "enhanced"]:
                self.phase_encoder = nn.Linear(hidden_dim, 4)
                self.gate_proj = nn.Linear(4, hidden_dim * hidden_dim)
            
            # 因果强度缩放
            if sparsity_mode in ["hybrid", "enhanced"]:
                self.causal_scale = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算相位因果矩阵
        
        Args:
            z: (batch, hidden_dim) 复数状态
            
        Returns:
            causal_matrix: (batch, hidden_dim, hidden_dim) 因果强度矩阵
        """
        phase = torch.angle(z)
        phase_diff = phase.unsqueeze(-1) - phase.unsqueeze(-2)
        resonance = torch.abs(torch.cos(phase_diff)) ** self.gamma
        
        if not self.use_learned:
            return resonance
        
        if self.sparsity_mode == "multiplicative":
            # 乘法稀疏：因果被削弱（适用于需要精确控制稀疏度的场景）
            mask = torch.sigmoid(self.sparse_mask)
            return resonance * mask
            
        elif self.sparsity_mode == "additive":
            # 加性稀疏：保持因果强度，但有选择性地增强/抑制
            mask = torch.sigmoid(self.sparse_mask)
            return resonance + 0.1 * mask  # 小幅度调制，不削弱核心因果
            
        elif self.sparsity_mode == "receptor":
            # 受体门控：输入依赖的动态稀疏
            batch_size = z.shape[0]
            phase_abs = phase.abs()
            gate = torch.sigmoid(self.gate_proj(torch.sigmoid(self.phase_encoder(phase_abs))))
            gate_2d = gate.view(batch_size, self.hidden_dim, self.hidden_dim).mean(dim=0)
            return resonance * gate_2d
            
        elif self.sparsity_mode == "hybrid":
            # 混合模式：分离因果强度控制和稀疏度控制
            batch_size = z.shape[0]
            
            # 1. 可学习的因果强度缩放
            scale = self.causal_scale.abs() if hasattr(self, 'causal_scale') else 1.0
            causal = resonance * scale
            
            # 2. 加性稀疏（不削弱因果，只调制）
            mask = torch.sigmoid(self.sparse_mask)
            causal = causal * (0.5 + 0.5 * mask)
            
            # 3. 受体门控（输入依赖的细调）
            phase_abs = phase.abs()
            gate = torch.sigmoid(self.gate_proj(torch.sigmoid(self.phase_encoder(phase_abs))))
            gate_2d = gate.view(batch_size, self.hidden_dim, self.hidden_dim).mean(dim=0)
            causal = causal * (0.5 + 0.5 * gate_2d)
            
            return causal
        
        elif self.sparsity_mode == "enhanced":
            # 增强模式（推荐默认）：硬阈值稀疏 + 可调因果强度
            #
            # 核心思想：
            # 1. 硬阈值：只有 > threshold 的值才保留
            # 2. 阈值由 sigmoid(mask) 控制
            # 3. 被保留的连接强度不被削弱
            
            scale = self.causal_scale.abs() if hasattr(self, 'causal_scale') else 1.0
            
            # 1. 基础因果
            base_causal = resonance * scale
            
            # 2. 计算阈值
            # sigmoid(-5) ≈ 0.007 → threshold ≈ 0.99 (99%稀疏)
            # sigmoid(-3.9) ≈ 0.02 → threshold ≈ 0.95 (98%稀疏)
            # sigmoid(0) = 0.5 → threshold = 0.5 (50%稀疏)
            # sigmoid(1) ≈ 0.73 → threshold ≈ 0.27 (27%稀疏)
            sigmoid_mean = torch.sigmoid(self.sparse_mask).mean()
            threshold = 1.0 - sigmoid_mean  # 当sigmoid=0.02时，threshold=0.98
            
            # 3. 硬阈值稀疏化
            sparse_mask = (base_causal > threshold).float()
            
            # 4. 最终因果
            causal = base_causal * sparse_mask
            
            return causal
        
        return resonance
    
    def get_sparsity(self) -> float:
        """获取当前mask的稀疏度"""
        if not self.use_learned:
            return 0.0
        mask = torch.sigmoid(self.sparse_mask)
        return 1.0 - mask.mean().item()
    
    def get_active_ratio(self) -> float:
        """获取活跃连接比例"""
        if not self.use_learned:
            return 1.0
        mask = torch.sigmoid(self.sparse_mask)
        return mask.mean().item()


class AmplitudeCausalCalculator(nn.Module):
    """
    幅值因果计算器
    
    基于幅值变化的因果效应：
    幅值变化大 → 因果效应强
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.amplitude_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        amplitude = torch.abs(z)
        
        amplitude_effect = self.amplitude_net(amplitude)
        
        return amplitude_effect


class TwistorInterventionExecutor(nn.Module):
    """
    扭量干预执行器
    
    在复数空间中执行do-演算干预：
    1. 硬干预: do(Z_idx = value) - 直接覆盖复数值
    2. 软干预: do(Z := Z * modulation) - 复数乘法调制
    3. 相位干预: do(phase(Z_idx) = new_phase) - 只改相位
    4. 幅值干预: do(|Z_idx| = new_amplitude) - 只改幅值
    """
    
    def __init__(self, hidden_dim: int, config: Optional[TwistorCausalConfig] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or TwistorCausalConfig()
        self.config.hidden_dim = hidden_dim
        
        self.phase_causal = PhaseCausalCalculator(hidden_dim, self.config.resonance_gamma)
        self.amplitude_causal = AmplitudeCausalCalculator(hidden_dim)
        
        self.intervention_history: List[Dict] = []
        
    def do_hard(self, z: torch.Tensor, idx: int, value) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        硬干预: do(Z_idx = value)
        
        直接覆盖目标维度的复数值
        """
        z_intervened = z.clone()
        
        if isinstance(value, complex):
            z_intervened[:, idx] = torch.complex(
                torch.tensor(value.real, device=z.device, dtype=z.real.dtype),
                torch.tensor(value.imag, device=z.device, dtype=z.imag.dtype)
            )
        elif torch.is_complex(value):
            z_intervened[:, idx] = value
        else:
            z_intervened[:, idx] = torch.complex(
                torch.tensor(value, device=z.device, dtype=z.real.dtype),
                torch.zeros_like(z_intervened.real[:, idx])
            )
        
        causal_effect = z_intervened - z
        
        self.intervention_history.append({
            'type': 'hard',
            'idx': idx,
            'value': value,
        })
        
        return z_intervened, causal_effect
        
    def do_soft(self, z: torch.Tensor, idx: int, modulation: complex) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        软干预: do(Z := Z * modulation)
        
        通过复数乘法调制（同时改变幅值和相位）
        """
        z_intervened = z.clone()
        
        if torch.is_complex(modulation):
            z_intervened[:, idx] = z[:, idx] * modulation
        else:
            modulation_tensor = torch.tensor(complex(modulation), device=z.device, dtype=torch.complex64)
            z_intervened[:, idx] = z[:, idx] * modulation_tensor
            
        causal_effect = z_intervened - z
        
        return z_intervened, causal_effect
        
    def do_phase(self, z: torch.Tensor, idx: int, new_phase: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        相位干预: do(phase(Z_idx) = new_phase)
        
        只改变相位，保持幅值不变
        """
        z_intervened = z.clone()
        
        old_amplitude = torch.abs(z[:, idx])
        
        new_phase_tensor = torch.tensor(new_phase, device=z.device, dtype=z.real.dtype)
        new_complex = old_amplitude * torch.exp(1j * new_phase_tensor)
        z_intervened[:, idx] = new_complex
        
        causal_effect = z_intervened - z
        
        return z_intervened, causal_effect
        
    def do_amplitude(self, z: torch.Tensor, idx: int, new_amplitude: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        幅值干预: do(|Z_idx| = new_amplitude)
        
        只改变幅值，保持相位不变
        """
        z_intervened = z.clone()
        
        old_phase = torch.angle(z[:, idx])
        new_amp_tensor = torch.tensor(new_amplitude, device=z.device, dtype=z.real.dtype)
        new_complex = new_amp_tensor * torch.exp(1j * old_phase.real)
        z_intervened[:, idx] = new_complex
        
        causal_effect = z_intervened - z
        
        return z_intervened, causal_effect
        
    def do_z_dynamics(self, z: torch.Tensor, idx: int, dzdt: torch.Tensor, dt: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        动力学干预: do(dZ/dt = dzdt)
        
        通过修改状态导数来影响演化
        """
        z_intervened = z + dt * dzdt
        
        causal_effect = z_intervened - z
        
        return z_intervened, causal_effect


class TwistorOfflineSimulator(nn.Module):
    """
    扭量离线模拟器
    
    在复数空间中执行反事实推理：
    1. 冻结模型参数
    2. 应用干预
    3. 运行复数ODE
    4. 比较事实与反事实轨迹
    """
    
    def __init__(self, hidden_dim: int, config: Optional[TwistorCausalConfig] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or TwistorCausalConfig()
        
    def simulate_factual(
        self,
        model: nn.Module,
        initial_z: torch.Tensor,
        num_steps: int,
        dt: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模拟事实轨迹（无干预）
        """
        z = initial_z.clone()
        trajectory = [z]
        
        for step in range(num_steps):
            with torch.no_grad():
                if hasattr(model, 'compute_dzdt'):
                    dzdt = model.compute_dzdt(z, torch.zeros_like(z.real))
                else:
                    dzdt = -z
                    
                z = z + dt * dzdt
                z = torch.complex(
                    torch.clamp(z.real, -100, 100),
                    torch.clamp(z.imag, -100, 100),
                )
                trajectory.append(z)
                
        trajectory = torch.stack(trajectory, dim=0)
        final_z = trajectory[-1]
        
        return trajectory, final_z
        
    def simulate_counterfactual(
        self,
        model: nn.Module,
        initial_z: torch.Tensor,
        intervention_fn: Callable,
        num_steps: int,
        dt: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模拟反事实轨迹（有干预）
        """
        z = initial_z.clone()
        trajectory = [z]
        
        for step in range(num_steps):
            with torch.no_grad():
                intervention = intervention_fn(z, step)
                if intervention is not None:
                    z = intervention
                    
                if hasattr(model, 'compute_dzdt'):
                    dzdt = model.compute_dzdt(z, torch.zeros_like(z.real))
                else:
                    dzdt = -z
                    
                z = z + dt * dzdt
                z = torch.complex(
                    torch.clamp(z.real, -100, 100),
                    torch.clamp(z.imag, -100, 100),
                )
                trajectory.append(z)
                
        trajectory = torch.stack(trajectory, dim=0)
        final_z = trajectory[-1]
        
        return trajectory, final_z
        
    def compute_causal_effect(
        self,
        factual_traj: torch.Tensor,
        cf_traj: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        计算因果效应
        """
        effect = cf_traj - factual_traj
        
        effect_magnitude = torch.norm(effect[-1]).item()
        
        trajectory_divergence = torch.mean(torch.abs(effect)).item()
        
        return effect, effect_magnitude


class TwistorCausalModule(nn.Module):
    """
    扭量因果模块
    
    完整整合因果推理到扭量空间：
    1. 相位因果计算
    2. 干预执行（硬/软/相位/幅值）
    3. 反事实推理
    4. 因果效应量化
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[TwistorCausalConfig] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.config = config or TwistorCausalConfig()
        self.config.hidden_dim = hidden_dim
        
        self.intervention = TwistorInterventionExecutor(hidden_dim, self.config)
        self.simulator = TwistorOfflineSimulator(hidden_dim, self.config)
        
        self.phase_causal = PhaseCausalCalculator(
            hidden_dim, 
            self.config.resonance_gamma,
            self.config.sparse_init,
            self.config.use_learned_sparsity,
            self.config.sparsity_mode
        )
        
    def compute_phase_causality(self, z: torch.Tensor) -> torch.Tensor:
        """
        计算相位因果矩阵
        """
        return self.phase_causal(z)
    
    def get_sparsity(self) -> float:
        """获取当前因果矩阵稀疏度"""
        return self.phase_causal.get_sparsity()
    
    def get_active_ratio(self) -> float:
        """获取活跃连接比例"""
        return self.phase_causal.get_active_ratio()
        
    def do(
        self,
        z: torch.Tensor,
        intervention_type: str,
        idx: int,
        value,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行干预
        """
        if intervention_type == 'hard':
            return self.intervention.do_hard(z, idx, value)
        elif intervention_type == 'soft':
            return self.intervention.do_soft(z, idx, value)
        elif intervention_type == 'phase':
            return self.intervention.do_phase(z, idx, value)
        elif intervention_type == 'amplitude':
            return self.intervention.do_amplitude(z, idx, value)
        else:
            return z, torch.zeros_like(z)
            
    def counterfactual(
        self,
        model: nn.Module,
        initial_z: torch.Tensor,
        intervention_fn: Callable,
        num_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        执行反事实推理
        """
        factual_traj, final_factual = self.simulator.simulate_factual(
            model, initial_z, num_steps
        )
        
        cf_traj, final_cf = self.simulator.simulate_counterfactual(
            model, initial_z, intervention_fn, num_steps
        )
        
        effect, magnitude = self.simulator.compute_causal_effect(
            factual_traj, cf_traj
        )
        
        return factual_traj, cf_traj, magnitude
        
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        主前向传播：计算相位因果结构
        """
        phase_causality = self.compute_phase_causality(z)
        
        return {
            'phase_causality': phase_causality,
            'hidden_dim': self.hidden_dim,
            'sparsity': self.get_sparsity(),
        }
