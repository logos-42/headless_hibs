"""
扭量旋开生长 (TwistGrow) — 复数隐藏维度的渐进增加
====================================================

灵感: C-route (POC 线) 的"增量生长"是实空间加 1 个神经元 (POC-7 报告 §3 关键).
本模块把它**扭量化**: 不是在实空间加维度, 而是在复数空间"旋开"新相位维度.

核心思想:
- 实空间生长:  h_new ∈ ℝ,  h ∈ ℝ^n → ℝ^(n+1)  ← 离散加挂
- 复数空间生长: z_new ∈ ℂ, z ∈ ℂ^n → ℂ^(n+step)
                 新维度的初相位 = π/(active+step) 的等分  ← 螺旋延展

复数空间生长的扭量优势:
1. 新维度天然带"相位信息", 几何上螺旋延展, 不是垂直插入
2. 配合 Möbius 流形投影, 新维度从零相位学到独特动力学的过程被几何约束
3. 配合 τ(z) 状态依赖时间常数, 新维度的"学习速度"自动差异化

触发逻辑 (仿 C-route):
- 每 N epoch 检查一次 recent_avg_loss
- 阈值 < 3.0 才允许生长 (loss 太差不生长)
- 每次生长 step 个维度
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TwistGrowCell(nn.Module):
    """
    扭量旋开生长单元: 复数 ODE 主干的"自适应容量"扩展.

    与标准 LMT/CNN 的区别:
    - 标准: hidden_dim 固定, 训练时不变
    - C-route (实空间): hidden_dim 从 4 增到 max, 通过 mask 截断
    - TwistGrow (复数空间): active_phase_dim 从 init 增到 max,
                            新维度的初相位按 π 等分, 通过 active_mask 控制

    Args:
        max_dim: 预分配的最大复数维数 (init 之后能旋开到的上限)
        init_dim: 初始激活的复数维数
        grow_step: 每次旋开几个维度
        grow_threshold: 允许生长的 loss 阈值 (仿 C-route 3.0)
        grow_interval: 两次生长的最小 epoch 间隔
    """

    def __init__(
        self,
        max_dim: int = 128,
        init_dim: int = 16,
        grow_step: int = 8,
        grow_threshold: float = 3.0,
        grow_interval: int = 10,
    ):
        super().__init__()
        assert init_dim <= max_dim
        self.max_dim = max_dim
        self.init_dim = init_dim
        self.grow_step = grow_step
        self.grow_threshold = grow_threshold
        self.grow_interval = grow_interval

        # 当前激活的复数维数
        self.register_buffer("active_dim", torch.tensor(init_dim, dtype=torch.long))

        # 激活 mask: 1=激活, 0=未旋开
        self.register_buffer("active_mask", torch.zeros(max_dim))
        self.active_mask[:init_dim] = 1.0

        # ★ 扭量特异性: 新维度的初相位 (π 等分, 螺旋延展)
        # 第 k 个旋开批次的新维度起始相位 = (2k-1) · π / (2 · batch_size)
        #   batch_size=1: π/2 (垂直于 0 相位)
        #   batch_size=2: π/4, 3π/4
        #   batch_size=n: 均匀分布
        self.register_buffer("init_phases", torch.zeros(max_dim))
        for i in range(max_dim):
            # 螺旋延展: i 个总维度, 新加入的维度相位 = π · (i+1) / (max_dim+1)
            self.init_phases[i] = math.pi * (i + 1) / (max_dim + 1)

        # 生长事件历史
        self.grow_events = []  # list of {epoch, from_dim, to_dim, loss}
        self._last_grow_epoch = -grow_interval  # 立即允许第一次生长

    @property
    def current_dim(self) -> int:
        return int(self.active_dim.item())

    def can_grow(self, current_epoch: int, recent_loss: float) -> bool:
        """判断是否应该旋开新维度"""
        if int(self.active_dim.item()) >= self.max_dim:
            return False
        if current_epoch - self._last_grow_epoch < self.grow_interval:
            return False
        if recent_loss > self.grow_threshold:
            return False
        return True

    def twist_grow(self, current_epoch: int, recent_loss: float) -> bool:
        """
        旋开新维度 (复数生长).

        Returns:
            True = 成功生长, False = 不满足条件
        """
        if not self.can_grow(current_epoch, recent_loss):
            return False

        old_dim = int(self.active_dim.item())
        new_dim = min(old_dim + self.grow_step, self.max_dim)
        self.active_mask[old_dim:new_dim] = 1.0
        self.active_dim.fill_(new_dim)
        self._last_grow_epoch = current_epoch

        event = {
            "epoch": current_epoch,
            "from": old_dim,
            "to": new_dim,
            "loss": float(recent_loss),
            "init_phases": [float(self.init_phases[i].item()) for i in range(old_dim, new_dim)],
        }
        self.grow_events.append(event)
        return True

    def apply_mask(self, z: torch.Tensor) -> torch.Tensor:
        """
        把 z 限制到 active_dim 范围内 (未旋开的维度置零).

        Args:
            z: 复数张量, 最后一维 = max_dim
        Returns:
            掩码后的复数张量, 同形
        """
        mask = self.active_mask.to(z.device).to(z.dtype if not z.is_complex() else torch.float32)
        # 复数 z 不能直接乘 float mask, 拆成 real/imag
        z_real = z.real * mask
        z_imag = z.imag * mask
        return torch.complex(z_real, z_imag)

    def get_init_phase(self, dim_idx: int) -> float:
        """获取第 dim_idx 个维度的初相位 (用于可视化)"""
        if dim_idx < 0 or dim_idx >= self.max_dim:
            raise IndexError(f"dim_idx {dim_idx} out of range [0, {self.max_dim})")
        return float(self.init_phases[dim_idx].item())

    def get_state(self) -> dict:
        """获取当前状态 (用于日志/序列化)"""
        return {
            "active_dim": self.current_dim,
            "max_dim": self.max_dim,
            "init_dim": self.init_dim,
            "grow_step": self.grow_step,
            "n_grow_events": len(self.grow_events),
            "grow_events": self.grow_events,
        }


class TwistGrowMultimodalTwistorLMT(nn.Module):
    """
    扭量旋开生长的多模态 Twistor-LMT.

    在 MultimodalTwistorLMT 基础上, 把 backbone 替换为"可旋开生长"的版本.

    工作流:
        raw x (modality)
            ↓ MultimodalNormalizer.encode
        z_entry (复数, 维度 = max_dim, 初始只有 init_dim 激活)
            ↓ 预分配 backbone (TwistorLMT with max_dim)
            ↓ 每次 forward 通过 TwistGrowCell.apply_mask 限制
        z (复数, active_dim 维)
            ↓
        输出

    训练时, 每 grow_interval epoch 检查 loss, 满足条件就旋开新维度.
    """

    def __init__(
        self,
        hidden_dim: int = 64,        # 兼容 MultimodalTwistorLMT 的接口
        max_dim: int = 64,           # 预分配 (主干的 hidden_dim 必须 = max_dim)
        init_dim: int = 16,          # 初始激活维数 (必须 ≤ hidden_dim)
        n_layers: int = 4,
        grow_step: int = 8,          # 每次旋开几个维度
        grow_threshold: float = 3.0,
        grow_interval: int = 10,
        modality_dims: Optional[dict] = None,
        head_specs: Optional[dict] = None,
        enable_mobius: bool = True,
        enable_injector: bool = True,
        enable_isometry: bool = True,
        isometry_weight: float = 0.1,
        tau_scales: Optional[dict] = None,
        dt: float = 0.1,
        sparsity: float = 0.3,
    ):
        super().__init__()
        from .multimodal_model import MultimodalTwistorLMT
        from .multimodal_normalizer import (
            MODALITY_TEXT, MODALITY_VISION, MODALITY_AUDIO,
        )

        # 兼容性检查
        if hidden_dim != max_dim:
            raise ValueError(
                f"TwistGrow 模式要求 hidden_dim == max_dim, "
                f"got hidden_dim={hidden_dim} max_dim={max_dim}"
            )

        self.hidden_dim = hidden_dim
        self.max_dim = max_dim
        self.init_dim = init_dim
        self.grow_step = grow_step

        # 1) 旋开生长单元
        self.twist_grow = TwistGrowCell(
            max_dim=max_dim,
            init_dim=init_dim,
            grow_step=grow_step,
            grow_threshold=grow_threshold,
            grow_interval=grow_interval,
        )

        # 2) 完整 MultimodalTwistorLMT 作为底层 (复用)
        self._mm = MultimodalTwistorLMT(
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            modality_dims=modality_dims,
            head_specs=head_specs,
            enable_mobius=enable_mobius,
            enable_injector=enable_injector,
            enable_isometry=enable_isometry,
            isometry_weight=isometry_weight,
            tau_scales=tau_scales,
            dt=dt,
            sparsity=sparsity,
        )
        # 强制 backbone 的 z_max 留出余量 (避免旋开后 z 被 clamp)
        # (TwistorLMT 默认 z_max=100, max_dim=64 足够)

    def twist_grow_step(self, epoch: int, recent_loss: float) -> bool:
        """训练循环调用: 检查并执行旋开生长"""
        return self.twist_grow.twist_grow(epoch, recent_loss)

    def forward(self, x: torch.Tensor, modality: str, return_z: bool = False, seq_len: int = 32):
        """
        前向传播, 在 backbone 内部把 z 限制到 active_dim.

        Args:
            seq_len: ODE 时序步数 (生成时用 1 提速)
        """
        # 1) 编码 (复用 normalizer)
        z_entry = self._mm.encode(x, modality, seq_len=seq_len)

        # 2) 把 z_entry 限制到 active_dim 范围
        z_entry = self.twist_grow.apply_mask(z_entry)

        # 3) 调底层 _backbone_with_inject (复用 MultimodalTwistorLMT)
        # 关键: 在每个 ODE 步之后再次掩码 (防止 backbone 内部把 z 推到 mask 范围外)
        T, B, D = z_entry.shape
        z = torch.zeros(B, self.max_dim, dtype=torch.complex64, device=z_entry.device)
        rho = self._mm.normalizer.tau_scale(modality)
        mod_embed = self._mm.normalizer.modality_embed(modality)
        original_dt = self._mm.backbone.dt
        self._mm.backbone.dt = float(original_dt * rho.item())
        try:
            outputs = []
            for t in range(T):
                x_t = z_entry[t].real
                dzdt = self._mm.backbone.compute_dzdt(z, x_t)
                dzdt = self._mm.bridge.inject(dzdt, z, mod_embed)
                z = z + self._mm.backbone.dt * dzdt
                if self._mm.backbone.mobius is not None:
                    z = self._mm.backbone.mobius.project_state(z)
                z = torch.complex(
                    torch.clamp(z.real, -self._mm.backbone.z_max, self._mm.backbone.z_max),
                    torch.clamp(z.imag, -self._mm.backbone.z_max, self._mm.backbone.z_max),
                )
                # ★ 关键: 每步都掩码, 保持 z 在 active_dim 内
                z = self.twist_grow.apply_mask(z)
                outputs.append(z)
            z_out = torch.stack(outputs, dim=0)
        finally:
            self._mm.backbone.dt = original_dt

        # 4) 解码 (用底层 head)
        out = self._mm.heads.decode(z_out[-1] if z_out.dim() == 3 else z_out, modality)

        result = {
            "output": out,
            "modality_embed": mod_embed,
            "tau_scale": float(rho.item()),
        }
        if return_z:
            result["z"] = z_out
        return result

    def compute_loss(self, *args, **kwargs):
        return self._mm.compute_loss(*args, **kwargs)

    def get_grow_state(self) -> dict:
        return self.twist_grow.get_state()

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
