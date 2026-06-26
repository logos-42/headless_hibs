"""
长时间训练可增长受体门控扭量液态神经网络
==========================================
支持持续数小时的训练，具备：
1. 参数预分配 + 动态扩展（无需重建模型）
2. 自动生长与剪枝机制
3. 动态受体门控
4. 检查点保存与恢复
5. 学习率调度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import math
import os
import time
import json
from datetime import datetime


@dataclass
class LongTrainingConfig:
    """长时间训练配置"""
    
    # 架构参数
    input_dim: int = 100
    output_dim: int = 100
    max_hidden_dim: int = 512      # 最大神经元数
    start_hidden_dim: int = 4       # 起始神经元数
    
    # 训练参数
    max_steps: int = 50000           # 最大训练步数
    batch_size: int = 32
    base_lr: float = 0.002
    min_lr: float = 1e-6
    
    # 生长参数
    growth_interval: int = 200       # 每200步检查一次生长
    prune_interval: int = 500        # 每500步检查一次剪枝
    neurons_per_growth: int = 2      # 每次生长增加的神经元数
    max_growth_per_check: int = 5    # 每次最多增加5个神经元
    
    # 生长触发条件
    loss_stagnation_threshold: float = 0.01  # 损失变化小于此值视为停滞
    loss_stagnation_window: int = 50         # 窗口大小
    
    # 检查点
    checkpoint_interval: int = 3000  # 每3000步保存一次
    checkpoint_dir: str = "checkpoints"
    max_checkpoints: int = 10        # 保留最近N个检查点
    
    # 受体参数
    n_channels: int = 3
    n_receptor_types: int = 4
    
    # 动力学参数
    dt: float = 0.1
    tau_min: float = 0.01
    tau_max: float = 1.0
    dzdt_max: float = 10.0
    z_max: float = 100.0
    sparsity: float = 0.3
    
    # 莫比乌斯约束
    enable_mobius: bool = True
    mobius_strength: float = 0.1
    
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)


class LongTrainingGrowableReceptorLMT(nn.Module):
    """
    长时间训练的可增长受体门控扭量网络
    
    核心特性：
    - 参数预分配到 max_hidden_dim，避免重建模型
    - 通过 active_hidden_dim 追踪当前使用的神经元
    - 支持生长、剪枝、受体门控、检查点
    """
    
    def __init__(
        self,
        config: LongTrainingConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.config = config
        self.device = device
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.max_hidden_dim = config.max_hidden_dim
        self.hidden_dim = config.start_hidden_dim  # 当前活跃维度
        
        # 训练状态
        self.training_step = 0
        self.best_loss = float('inf')
        self.loss_history = []
        self.growth_history = []
        self.learning_rates = []
        
        # 预分配所有参数到 max_hidden_dim
        self._preallocate_parameters()
        
        # 初始化莫比乌斯约束
        if config.enable_mobius:
            self._init_mobius_constraint()
        
        # 记录神经元激活统计
        self._activation_stats = torch.zeros(self.max_hidden_dim)
        self._activation_buffer = []
        self._max_buffer_size = 100
        
        # 初始化流形坐标
        self._init_manifold_coords()
        
        print(f"🚀 初始化完成: {config.start_hidden_dim} → {config.max_hidden_dim} 神经元")
        print(f"   设备: {device}, 检查点目录: {config.checkpoint_dir}")
    
    def _preallocate_parameters(self):
        """预分配所有参数到最大维度"""
        max_h = self.max_hidden_dim
        d = self.config
        
        # 1. 流形坐标 (每个神经元3维)
        self.manifold_theta = nn.Parameter(torch.randn(max_h, 3) * 0.1)
        
        # 2. 振幅矩阵
        self.W_amplitude = nn.Parameter(torch.randn(max_h, max_h) * 0.3)
        
        # 3. 输入层
        self.U = nn.Linear(self.input_dim, max_h)
        
        # 4. 时间尺度网络
        self.W_tau = nn.Linear(max_h, max_h)
        self.tau_bias = nn.Parameter(torch.zeros(max_h))
        
        # 5. 偏置
        self.b_real = nn.Parameter(torch.zeros(max_h))
        self.b_imag = nn.Parameter(torch.zeros(max_h))
        
        # 6. 输出层
        self.out = nn.Linear(max_h, self.output_dim)
        
        # 7. 稀疏掩码
        self.sparse_mask = nn.Parameter(torch.ones(max_h, max_h) * -5)
        
        # 8. 受体系统参数
        # 通道生成网络
        self.channel_net = nn.Sequential(
            nn.Linear(self.input_dim, d.n_channels * 2),
            nn.ReLU(),
            nn.Linear(d.n_channels * 2, d.n_channels),
        )
        
        # 受体权重
        self.receptor_weights = nn.Parameter(
            torch.randn(max_h, d.n_receptor_types) * 0.3
        )
        
        # 通道到受体映射
        self.channel_to_receptor = nn.Linear(d.n_channels, d.n_receptor_types)
        
        # 低秩门控投影
        self.gate_row_proj = nn.Linear(d.n_receptor_types, max_h)
        self.gate_col_proj = nn.Linear(d.n_receptor_types, max_h)
        
        # 相位调制网络
        self.phase_net = nn.Linear(d.n_receptor_types, max_h * max_h)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau.weight, gain=0.1)
        nn.init.orthogonal_(self.W_amplitude, gain=0.3)
        
        # 稀疏初始化
        if self.config.sparsity > 0:
            with torch.no_grad():
                mask = (
                    torch.rand(self.max_hidden_dim, self.max_hidden_dim) 
                    > self.config.sparsity
                ).float()
                self.sparse_mask.data.copy_(mask * 4 - 5)
    
    def _init_manifold_coords(self):
        """初始化流形坐标"""
        n = self.hidden_dim
        for i in range(n):
            theta = 2 * math.pi * i / max(1, n)
            self.manifold_theta.data[i, 0] = theta
            self.manifold_theta.data[i, 1] = 0.0
            self.manifold_theta.data[i, 2] = 0.0
    
    def _init_mobius_constraint(self):
        """初始化莫比乌斯约束"""
        from .mobius import MobiusConstraint
        
        max_dim = max(self.max_hidden_dim * 4, 512)
        self.mobius = MobiusConstraint(
            max_dim=max_dim,
            constraint_strength=self.config.mobius_strength,
            enable_learning=True,
            device=self.device,
        )
    
    # ==================== 受体系统 ====================
    
    def compute_channels(self, x: torch.Tensor) -> torch.Tensor:
        """计算动态递质浓度"""
        c = torch.sigmoid(self.channel_net(x))
        return c
    
    def compute_receptor_activation(self, c: torch.Tensor) -> torch.Tensor:
        """计算受体激活"""
        return torch.tanh(self.channel_to_receptor(c))
    
    def compute_gate(self, receptor_act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算门控信号"""
        g_row = torch.sigmoid(self.gate_row_proj(receptor_act))
        g_col = torch.sigmoid(self.gate_col_proj(receptor_act))
        return g_row, g_col
    
    def compute_twist_phase(self) -> torch.Tensor:
        """计算莫比乌斯扭转相位"""
        n = self.hidden_dim
        theta = self.manifold_theta[:n, 0]
        
        i = theta.unsqueeze(1)
        j = theta.unsqueeze(0)
        
        twist_mobius = math.pi * (i + j) / (2 * max(n, 1))
        twist_klein = 2 * math.pi * (i * j) / (max(n, 1) ** 2)
        
        phase = 0.5 * twist_mobius + 0.5 * twist_klein
        return phase
    
    def compute_phase_modulation(self, receptor_act: torch.Tensor) -> torch.Tensor:
        """计算相位调制"""
        n = self.hidden_dim
        batch_size = receptor_act.shape[0]
        
        phase_shift_flat = self.phase_net(receptor_act)
        phase_shift = phase_shift_flat.view(-1, n, n)
        
        # 限制相位偏移范围
        phase_shift = torch.tanh(phase_shift) * math.pi
        return phase_shift
    
    def get_complex_weight(self, c: torch.Tensor) -> torch.Tensor:
        """获取复数权重矩阵"""
        n = self.hidden_dim
        batch_size = c.shape[0]
        
        # 1. 受体激活
        receptor_act = self.compute_receptor_activation(c)
        
        # 2. 门控信号
        g_row, g_col = self.compute_gate(receptor_act)
        
        # 3. 振幅 + 门控
        A = self.W_amplitude[:n, :n].unsqueeze(0)
        g_row_active = g_row[:, :n].unsqueeze(2)
        g_col_active = g_col[:, :n].unsqueeze(1)
        A_gated = A * g_row_active * g_col_active
        
        # 4. 相位
        phase_base = self.compute_twist_phase().unsqueeze(0)
        phase_shift = self.compute_phase_modulation(receptor_act)
        phase = phase_base + phase_shift[:, :n, :n]
        
        # 5. 稀疏掩码
        mask = torch.sigmoid(self.sparse_mask[:n, :n]).unsqueeze(0)
        
        # 6. 复数权重
        W = A_gated * mask * torch.exp(1j * phase)
        return W
    
    # ==================== 动力学 ====================
    
    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """计算自适应时间尺度"""
        z_mod = torch.abs(z)[:, :self.hidden_dim]
        
        tau = F.sigmoid(
            F.linear(z_mod, self.W_tau.weight[:self.hidden_dim, :self.hidden_dim],
                     self.W_tau.bias[:self.hidden_dim])
        )
        
        tau = tau + self.tau_bias[:self.hidden_dim].unsqueeze(0)
        tau = torch.clamp(tau, self.config.tau_min, self.config.tau_max)
        return tau + 1e-6
    
    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """计算状态导数"""
        n = self.hidden_dim
        
        z_real = z.real[:, :n]
        z_imag = z.imag[:, :n]
        
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)
        
        # 复数权重
        W = self.get_complex_weight(c)
        
        # 复数矩阵乘法
        W_real = W.real
        W_imag = W.imag
        
        W_tanh_real = torch.bmm(W_real, tanh_real.unsqueeze(2)).squeeze(2) - \
                      torch.bmm(W_imag, tanh_imag.unsqueeze(2)).squeeze(2)
        W_tanh_imag = torch.bmm(W_real, tanh_imag.unsqueeze(2)).squeeze(2) + \
                      torch.bmm(W_imag, tanh_real.unsqueeze(2)).squeeze(2)
        
        Ux = self.U(x)[:, :n]
        
        dz_real = -z_real + W_tanh_real + Ux + self.b_real[:n].unsqueeze(0)
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag[:n].unsqueeze(0)
        
        tau = self.compute_tau(z)
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)
        
        # 限制导数范围
        dzdt_real = torch.clamp(dzdt.real, -self.config.dzdt_max, self.config.dzdt_max)
        dzdt_imag = torch.clamp(dzdt.imag, -self.config.dzdt_max, self.config.dzdt_max)
        dzdt = torch.complex(dzdt_real, dzdt_imag)
        
        return dzdt
    
    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """前向传播"""
        T, B, _ = x.shape
        
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        outputs = []
        states = []
        
        for t in range(T):
            x_t = x[t]
            c = self.compute_channels(x_t)
            
            dzdt = self.compute_dzdt(z, x_t, c)
            z = z + self.config.dt * dzdt
            
            # 莫比乌斯投影
            if hasattr(self, 'mobius') and self.mobius is not None:
                z = self.mobius.project_state(z)
            
            # 限制状态范围
            z = torch.complex(
                torch.clamp(z.real, -self.config.z_max, self.config.z_max),
                torch.clamp(z.imag, -self.config.z_max, self.config.z_max),
            )
            
            y_t = F.linear(z.real, self.out.weight[:, :self.hidden_dim], self.out.bias)
            outputs.append(y_t)
            
            if return_states:
                states.append(z.detach().cpu())
            
            # 记录激活统计
            if self.training:
                act = torch.abs(z).mean(dim=0).detach().cpu()
                if len(self._activation_buffer) >= self._max_buffer_size:
                    self._activation_buffer.pop(0)
                self._activation_buffer.append(act[:self.hidden_dim])
        
        y = torch.stack(outputs, dim=0)
        
        if return_states:
            return y, torch.stack(states, dim=0)
        return y
    
    # ==================== 生长系统 ====================
    
    def _update_activation_stats(self):
        """更新激活统计"""
        if len(self._activation_buffer) < 10:
            return
        
        buffer = torch.stack(self._activation_buffer[-50:])
        mean_act = buffer.mean(dim=0)
        self._activation_stats[:len(mean_act)] = mean_act
    
    def should_grow(self) -> bool:
        """判断是否应该生长"""
        if self.hidden_dim >= self.config.max_hidden_dim:
            return False
        
        if len(self.loss_history) < self.config.loss_stagnation_window:
            return False
        
        # 检查损失是否停滞
        recent_losses = self.loss_history[-self.config.loss_stagnation_window:]
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        
        if loss_change < self.config.loss_stagnation_threshold:
            return True
        
        # 每隔固定间隔也检查一次
        if self.training_step % self.config.growth_interval == 0:
            return True
        
        return False
    
    def expand_neurons(self, n_new: int = 2):
        """扩展神经元 - O(1)操作"""
        if self.hidden_dim + n_new > self.config.max_hidden_dim:
            n_new = self.config.max_hidden_dim - self.hidden_dim
        
        if n_new <= 0:
            return 0
        
        old_dim = self.hidden_dim
        new_dim = old_dim + n_new
        
        print(f"  🌱 生长: {old_dim} → {new_dim} 神经元")
        
        # 初始化新神经元的参数
        with torch.no_grad():
            # 流形坐标: 在父神经元附近
            for i in range(n_new):
                idx = old_dim + i
                parent_idx = np.random.randint(0, old_dim) if old_dim > 0 else 0
                self.manifold_theta.data[idx] = (
                    self.manifold_theta.data[parent_idx] + torch.randn(3) * 0.1
                )
            
            # 振幅: 复制相邻神经元的初始值
            for i in range(n_new):
                idx = old_dim + i
                parent_idx = np.random.randint(0, old_dim) if old_dim > 0 else 0
                self.W_amplitude.data[idx, :old_dim] = self.W_amplitude.data[parent_idx, :old_dim]
                self.W_amplitude.data[:old_dim, idx] = self.W_amplitude.data[:old_dim, parent_idx]
                self.sparse_mask.data[idx, :old_dim] = self.sparse_mask.data[parent_idx, :old_dim]
                self.sparse_mask.data[:old_dim, idx] = self.sparse_mask.data[:old_dim, parent_idx]
            
            # 受体权重: 小随机初始化
            nn.init.xavier_uniform_(self.receptor_weights.data[old_dim:new_dim])
            
            # 偏置: 继承父神经元
            for i in range(n_new):
                idx = old_dim + i
                parent_idx = np.random.randint(0, old_dim) if old_dim > 0 else 0
                self.b_real.data[idx] = self.b_real.data[parent_idx] * 0.5
                self.b_imag.data[idx] = self.b_imag.data[parent_idx] * 0.5
                self.tau_bias.data[idx] = self.tau_bias.data[parent_idx] * 0.5
        
        self.hidden_dim = new_dim
        self.growth_history.append((self.training_step, new_dim))
        
        return n_new
    
    def prune_weak_connections(self):
        """剪枝弱连接"""
        n = self.hidden_dim
        if n == 0:
            return
        
        amp = self.W_amplitude[:n, :n].abs()
        mask = torch.sigmoid(self.sparse_mask[:n, :n])
        
        # 找出弱的连接
        weak_mask = (amp * mask < self.config.prune_threshold).float()
        weak_ratio = weak_mask.mean().item()
        
        if weak_ratio > 0.1:  # 如果弱连接超过10%
            with torch.no_grad():
                self.sparse_mask.data[:n, :n] += weak_mask * 0.5
    
    # ==================== 检查点 ====================
    
    def save_checkpoint(self, optimizer: torch.optim.Optimizer, 
                        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                        extra_info: Dict = None) -> str:
        """保存检查点"""
        step = self.training_step
        
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_step_{step}_{timestamp}.pt"
        filepath = os.path.join(self.config.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'hidden_dim': self.hidden_dim,
            'training_step': self.training_step,
            'best_loss': self.best_loss,
            'loss_history': self.loss_history,
            'growth_history': self.growth_history,
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(self.config),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_info:
            checkpoint['extra_info'] = extra_info
        
        torch.save(checkpoint, filepath)
        print(f"  💾 检查点已保存: {filepath}")
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        return filepath
    
    def load_checkpoint(self, filepath: str, optimizer: torch.optim.Optimizer = None,
                        scheduler: torch.optim.lr_scheduler._LRScheduler = None) -> Dict:
        """加载检查点"""
        print(f"  📂 加载检查点: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.hidden_dim = checkpoint['hidden_dim']
        self.training_step = checkpoint['training_step']
        self.best_loss = checkpoint['best_loss']
        self.loss_history = checkpoint['loss_history']
        self.growth_history = checkpoint['growth_history']
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"  ✅ 已恢复: step={self.training_step}, hidden_dim={self.hidden_dim}")
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点，只保留最近的N个"""
        checkpoint_dir = self.config.checkpoint_dir
        
        if not os.path.exists(checkpoint_dir):
            return
        
        files = [
            os.path.join(checkpoint_dir, f) 
            for f in os.listdir(checkpoint_dir) 
            if f.startswith("checkpoint_") and f.endswith(".pt")
        ]
        
        if len(files) <= self.config.max_checkpoints:
            return
        
        # 按修改时间排序
        files.sort(key=lambda x: os.path.getmtime(x))
        
        # 删除最旧的
        for f in files[:-self.config.max_checkpoints]:
            os.remove(f)
            print(f"  🗑️ 删除旧检查点: {os.path.basename(f)}")
    
    # ==================== 诊断 ====================
    
    def get_diagnostics(self) -> Dict:
        """获取诊断信息"""
        diag = {
            'hidden_dim': self.hidden_dim,
            'max_hidden_dim': self.config.max_hidden_dim,
            'training_step': self.training_step,
            'best_loss': self.best_loss,
            'current_lr': self.learning_rates[-1] if self.learning_rates else self.config.base_lr,
        }
        
        if len(self.loss_history) > 0:
            recent = self.loss_history[-min(100, len(self.loss_history)):]
            diag['avg_loss_100'] = sum(recent) / len(recent)
        
        if self.hidden_dim > 0 and len(self._activation_buffer) > 0:
            diag['avg_activation'] = self._activation_stats[:self.hidden_dim].mean().item()
        
        diag['growth_count'] = len(self.growth_history)
        
        return diag
    
    def print_status(self):
        """打印状态"""
        diag = self.get_diagnostics()
        print(f"  状态: step={diag['training_step']}, neurons={diag['hidden_dim']}, "
              f"loss={diag.get('avg_loss_100', 0):.4f}, lr={diag['current_lr']:.6f}")


class LongTrainingLoop:
    """
    长时间训练循环管理器
    """
    
    def __init__(
        self,
        model: LongTrainingGrowableReceptorLMT,
        config: LongTrainingConfig,
        train_data: torch.Tensor = None,
        train_targets: torch.Tensor = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 数据
        self.train_data = train_data
        self.train_targets = train_targets
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.base_lr,
            weight_decay=1e-5
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5000, T_mult=2, eta_min=config.min_lr
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 统计
        self.start_time = time.time()
        self.last_print_time = time.time()
        
        # 梯度监控
        self.gradient_history = []
    
    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> float:
        """单步训练"""
        self.model.train()
        
        # 前向传播
        pred = self.model(batch_x)
        
        # 计算损失 (seq_len, batch, vocab) -> (batch, vocab)
        if pred.shape[0] == batch_y.shape[0]:
            pred_last = pred[-1]
        else:
            pred_last = pred.squeeze(1)
        
        loss = self.criterion(pred_last, batch_y)
        
        # 额外正则化
        if hasattr(self.model, 'compute_amplitude_regularization'):
            reg = self.model.compute_amplitude_regularization(l1_weight=0.01, l2_weight=0.001)
            loss = loss + reg
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )
        self.gradient_history.append(grad_norm)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def run(self):
        """运行训练循环"""
        print("=" * 70)
        print("🚀 开始长时间训练")
        print("=" * 70)
        
        print(f"模型配置:")
        print(f"  - 最大神经元数: {self.config.max_hidden_dim}")
        print(f"  - 起始神经元数: {self.config.start_hidden_dim}")
        print(f"  - 最大训练步数: {self.config.max_steps}")
        print(f"  - 批次大小: {self.config.batch_size}")
        print(f"  - 检查点间隔: {self.config.checkpoint_interval}")
        print()
        
        # 主循环
        step = self.model.training_step
        
        while step < self.config.max_steps:
            # 生成批次
            if self.train_data is not None and self.train_targets is not None:
                batch_idx = torch.randint(
                    0, len(self.train_data), 
                    (self.config.batch_size,)
                )
                batch_x = self.train_data[batch_idx].to(self.device)
                batch_y = self.train_targets[batch_idx].to(self.device)
            else:
                # 随机生成数据
                batch_x = torch.randn(
                    self.config.batch_size, self.model.input_dim
                ).to(self.device)
                batch_y = torch.randint(
                    0, self.model.output_dim, (self.config.batch_size,)
                ).to(self.device)
            
            # 训练步骤
            loss = self.train_step(batch_x, batch_y)
            
            # 更新统计
            step += 1
            self.model.training_step = step
            self.model.loss_history.append(loss)
            self.model.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            
            # 更新最佳损失
            if loss < self.model.best_loss:
                self.model.best_loss = loss
            
            # 更新激活统计
            if step % 10 == 0:
                self.model._update_activation_stats()
            
            # 生长检查
            if step % self.config.growth_interval == 0:
                if self.model.should_grow():
                    n_new = min(
                        self.config.neurons_per_growth,
                        self.config.max_growth_per_check,
                        self.config.max_hidden_dim - self.model.hidden_dim
                    )
                    if n_new > 0:
                        self.model.expand_neurons(n_new)
            
            # 剪枝检查
            if step % self.config.prune_interval == 0:
                self.model.prune_weak_connections()
            
            # 打印进度
            if step % 100 == 0:
                elapsed = time.time() - self.start_time
                avg_time = elapsed / step
                eta = avg_time * (self.config.max_steps - step)
                
                loss_100 = sum(self.model.loss_history[-100:]) / min(100, len(self.model.loss_history))
                
                print(f"Step {step:6d} | Loss: {loss_100:.4f} | "
                      f"Neurons: {self.model.hidden_dim:4d} | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                      f"Time: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m")
            
            # 保存检查点
            if step % self.config.checkpoint_interval == 0:
                extra = {
                    'current_loss': loss,
                    'avg_loss_100': sum(self.model.loss_history[-100:]) / min(100, len(self.model.loss_history)),
                }
                self.model.save_checkpoint(
                    self.optimizer, self.scheduler, extra_info=extra
                )
            
            # 检查梯度爆炸
            if len(self.gradient_history) > 0 and self.gradient_history[-1] > 10:
                print(f"  ⚠️ 梯度爆炸警告: {self.gradient_history[-1]:.2f}")
            
            # 检查 NaN
            if math.isnan(loss) or math.isinf(loss):
                print(f"  ❌ 损失异常: {loss}, 停止训练")
                break
        
        # 最终检查点
        print("\n保存最终检查点...")
        self.model.save_checkpoint(self.optimizer, self.scheduler)
        
        # 总结
        total_time = time.time() - self.start_time
        print("\n" + "=" * 70)
        print("🏁 训练完成!")
        print("=" * 70)
        print(f"  总步数: {step}")
        print(f"  最终神经元数: {self.model.hidden_dim}")
        print(f"  最佳损失: {self.model.best_loss:.6f}")
        print(f"  总时间: {total_time/3600:.2f} 小时")
        print(f"  生长次数: {len(self.model.growth_history)}")
        
        return self.model


# ==================== 便捷函数 ====================

def create_vocabulary_dataset(vocab_size: int, n_samples: int, seq_len: int):
    """创建词汇表数据集"""
    data = torch.randint(0, vocab_size, (n_samples, seq_len))
    x = data[:, :-1].long()
    y = data[:, 1:].long()
    return x, y


def run_long_training(
    input_dim: int = 100,
    output_dim: int = 100,
    max_hidden_dim: int = 512,
    max_steps: int = 50000,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """运行长时间训练"""
    
    # 配置
    config = LongTrainingConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        max_hidden_dim=max_hidden_dim,
        start_hidden_dim=4,
        max_steps=max_steps,
        batch_size=batch_size,
        checkpoint_interval=3000,
        growth_interval=200,
        neurons_per_growth=2,
    )
    
    # 模型
    model = LongTrainingGrowableReceptorLMT(config, device=device)
    
    # 训练循环
    trainer = LongTrainingLoop(model, config, device=device)
    trained_model = trainer.run()
    
    return trained_model


if __name__ == "__main__":
    print("🧪 测试长时间训练系统...")
    
    # 测试运行
    model = run_long_training(
        input_dim=50,
        output_dim=50,
        max_hidden_dim=64,
        max_steps=1000,
        batch_size=16,
    )
    
    print("\n模型诊断信息:")
    diag = model.get_diagnostics()
    for k, v in diag.items():
        print(f"  {k}: {v}")