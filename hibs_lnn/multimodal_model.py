"""
多模态 Twistor-LMT 顶层模型 (Multimodal Twistor-LMT)
====================================================

设计原则:
- 复用现有 TwistorLMT 做主干, 不破坏 core.py
- Möbius 隐空间一致化在 normalizer + bridge 层完成
- 解码延后到 heads
- 完全可拔插: 关闭 multimodal 时行为等价于 baseline

工作流:
    raw_x (modality-specific)
        ↓ MultimodalNormalizer.encode
    z_entry (复数)
        ↓ 注入到 TwistorLMT 主干
    z (复数, 沿流形演化)
        ↓ TwistorLMT.forward (含 mobius/resonance)
    output (复数隐藏状态)
        ↓ MultimodalHeads.decode
    per-modality prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple

from .core import TwistorLMT
from .multimodal_normalizer import (
    MultimodalNormalizer,
    create_multimodal_normalizer,
    MODALITY_TEXT,
    MODALITY_VISION,
    MODALITY_AUDIO,
    SUPPORTED_MODALITIES,
)
from .multimodal_bridge import MultimodalBridge
from .multimodal_heads import MultimodalHeads


class MultimodalTwistorLMT(nn.Module):
    """
    多模态 Twistor-LMT: 在 Möbius 隐空间一致化的统一模型.

    Args:
        hidden_dim: 主干隐藏维度 (与 TwistorLMT.hidden_dim 一致)
        n_layers: TwistorLMT 层数
        modality_dims: 每模态原始信号维度
        head_specs: 每模态解码头规格
        enable_mobius: 是否启用 Möbius 流形约束
        enable_resonance: 是否启用共振
        enable_injector: 是否启用模态注入
        enable_isometry: 是否启用等距正则
        isometry_weight: 等距损失权重
        tau_scales: 每模态 τ 缩放先验
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 4,
        modality_dims: Optional[Dict[str, int]] = None,
        head_specs: Optional[Dict[str, Dict]] = None,
        enable_mobius: bool = True,
        enable_resonance: bool = True,
        enable_injector: bool = True,
        enable_isometry: bool = True,
        isometry_weight: float = 0.1,
        tau_scales: Optional[Dict[str, float]] = None,
        dt: float = 0.1,
        sparsity: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.isometry_weight = isometry_weight

        # 1) 模态前端
        if modality_dims is None:
            modality_dims = {
                MODALITY_TEXT: hidden_dim,
                MODALITY_VISION: hidden_dim // 4,
                MODALITY_AUDIO: hidden_dim // 4,
            }
        self.normalizer = create_multimodal_normalizer(
            hidden_dim=hidden_dim,
            modality_dims=modality_dims,
            tau_scales=tau_scales,
            max_norm=5.0,
        )

        # 2) 主干 (复用现有 TwistorLMT)
        self.backbone = TwistorLMT(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # 中间维度, 由 head 决定最终输出
            dt=dt,
            sparsity=sparsity,
        )
        if enable_mobius or enable_resonance:
            self.backbone.enable_mobius_resonance(
                enable_mobius=enable_mobius,
                enable_resonance=enable_resonance,
                mobius_strength=0.1,
                resonance_strength=0.1,
                sparse_resonance=True,
                learn_manifold_dim=True,
                resonance_mode="additive",
            )

        # 3) 模态桥接 (注入 + 等距)
        self.bridge = MultimodalBridge(
            hidden_dim=hidden_dim,
            rank=min(8, hidden_dim // 4),
            strength=0.1,
            enable_injector=enable_injector,
            enable_isometry=enable_isometry,
        )

        # 4) 解码头
        if head_specs is None:
            head_specs = {
                MODALITY_TEXT: {"type": "text", "out_dim": 100},
                MODALITY_VISION: {"type": "continuous", "out_dim": modality_dims.get(MODALITY_VISION, hidden_dim // 4)},
                MODALITY_AUDIO: {"type": "continuous", "out_dim": modality_dims.get(MODALITY_AUDIO, hidden_dim // 4)},
            }
        self.heads = MultimodalHeads(hidden_dim=hidden_dim, head_specs=head_specs)

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def encode(
        self,
        x: torch.Tensor,
        modality: str,
        seq_len: int = 32,
    ) -> torch.Tensor:
        """
        把原始信号编码成主干入口 z_entry (复数).

        Args:
            x: (B, raw_dim) 或 (B, T, raw_dim)
            modality: 模态 ID
            seq_len: 主干时序长度
        Returns:
            z_entry: (T, B, hidden_dim) 复数
        """
        z_entry = self.normalizer.encode(x, modality)  # 复数 (B, [T,] D)
        # 统一成 (T, B, D) 形式
        if z_entry.dim() == 2:
            z_entry = z_entry.unsqueeze(0).expand(seq_len, -1, -1)
        elif z_entry.dim() == 3 and z_entry.shape[1] == seq_len:
            z_entry = z_entry.transpose(0, 1)
        return z_entry.contiguous()

    def forward(
        self,
        x: torch.Tensor,
        modality: str,
        return_z: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播: raw_x → normalizer → backbone(注入模态) → head.

        Args:
            x: (B, raw_dim) 或 (B, T, raw_dim)
            modality: 模态 ID
            return_z: 是否返回主干隐藏状态
        Returns:
            dict: {
                "output": head 输出,
                "z": (可选) 复数隐藏状态 (T, B, D),
                "modality_embed": 模态 embedding,
            }
        """
        # 1) 编码
        z_entry = self.encode(x, modality)  # (T, B, D) 复数
        T, B, D = z_entry.shape

        # 2) 模态 τ 缩放 (异步采样)
        rho = self.normalizer.tau_scale(modality)
        # 临时调整主干 dt
        original_dt = self.backbone.dt
        self.backbone.dt = float(original_dt * rho.item())

        try:
            # 3) 主干 (注入模态)
            mod_embed = self.normalizer.modality_embed(modality)
            # 通过自定义的"注入式"主干, 这里包装一下
            z = self._backbone_with_inject(z_entry, mod_embed)
        finally:
            # 恢复 dt
            self.backbone.dt = original_dt

        # 4) 解码
        out = self.heads.decode(z[-1] if z.dim() == 3 else z, modality)

        result = {
            "output": out,
            "modality_embed": mod_embed,
            "tau_scale": float(rho.item()),
        }
        if return_z:
            result["z"] = z
        return result

    def _backbone_with_inject(
        self,
        z_entry: torch.Tensor,
        modality_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        主干演化 + 模态注入.

        我们手动跑一个 ODE step 循环, 这样可以在 compute_dzdt 后注入模态项.
        """
        T, B, D = z_entry.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=z_entry.device)

        outputs = []
        for t in range(T):
            x_t = z_entry[t].real  # TwistorLMT 内部用的是实数 Linear, 我们传实部
            dzdt = self.backbone.compute_dzdt(z, x_t)
            # 模态注入 (附加项)
            dzdt = self.bridge.inject(dzdt, z, modality_embed)
            z = z + self.backbone.dt * dzdt
            # 流形约束
            if self.backbone.mobius is not None:
                z = self.backbone.mobius.project_state(z)
            z = torch.complex(
                torch.clamp(z.real, -self.backbone.z_max, self.backbone.z_max),
                torch.clamp(z.imag, -self.backbone.z_max, self.backbone.z_max),
            )
            outputs.append(z)
        return torch.stack(outputs, dim=0)  # (T, B, D)

    # ------------------------------------------------------------------
    # Warmup 接口: 渐进打开 Möbius 约束
    # ------------------------------------------------------------------
    def set_mobius_warmup(
        self,
        epoch: int,
        warmup_epochs: int = 30,
        total_epochs: int = 200,
        max_strength: float = 0.1,
    ) -> float:
        """
        线性 ramp Möbius 约束强度.

        阶段:
          - epoch < warmup_epochs: 强度 = 0 (完全关)
          - warmup <= epoch < total: 线性 ramp 0 → max_strength
          - epoch >= total: 强度 = max_strength

        Returns:
            当前生效的强度值.
        """
        if self.backbone.mobius is None:
            return 0.0
        if epoch < warmup_epochs:
            s = 0.0
        elif epoch < total_epochs:
            t = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            s = float(t) * float(max_strength)
        else:
            s = float(max_strength)
        # 兼容 float / Parameter 两种存储方式
        self.backbone.mobius.set_constraint_strength(s)
        return s

    def set_isometry_warmup(
        self,
        epoch: int,
        warmup_epochs: int = 30,
        max_weight: float = 0.1,
    ) -> float:
        """
        渐进打开等距正则权重.

        阶段:
          - epoch < warmup_epochs: weight = 0
          - epoch >= warmup_epochs: 线性 ramp 0 → max_weight (再 30 epoch)
        """
        if epoch < warmup_epochs:
            w = 0.0
        else:
            # 30 epoch ramp from 0 to max_weight
            ramp = (epoch - warmup_epochs) / 30.0
            w = float(min(ramp, 1.0)) * float(max_weight)
        self.isometry_weight = w
        return w

    # ------------------------------------------------------------------
    # 损失
    # ------------------------------------------------------------------
    def compute_loss(
        self,
        z_dict: Dict[str, torch.Tensor],
        target_dict: Dict[str, torch.Tensor],
        pred_dict: Dict[str, torch.Tensor],
        modality: str,
        task_loss_fn=None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        总损失 = 任务损失 + 等距正则.

        Args:
            z_dict: {modality: (B, D) 复数最终状态}
            target_dict: {modality: target tensor}
            pred_dict: {modality: prediction tensor}
            modality: 当前主任务模态
            task_loss_fn: 任务损失函数 (pred, target) → loss
        Returns:
            (total_loss, metrics_dict)
        """
        metrics = {}

        # 任务损失
        if task_loss_fn is None:
            task_loss_fn = F.mse_loss
        target = target_dict[modality]
        pred = pred_dict[modality]
        task_loss = task_loss_fn(pred, target)
        metrics["task_loss"] = float(task_loss.item())

        # 等距正则
        iso_loss = self.bridge.isometry(z_dict)
        metrics["isometry_loss"] = float(iso_loss.item())

        total = task_loss + self.isometry_weight * iso_loss
        metrics["total_loss"] = float(total.item())
        return total, metrics

    def get_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_param_breakdown(self) -> Dict[str, int]:
        return {
            "normalizer": sum(p.numel() for p in self.normalizer.parameters() if p.requires_grad),
            "backbone": sum(p.numel() for p in self.backbone.parameters() if p.requires_grad),
            "bridge": sum(p.numel() for p in self.bridge.parameters() if p.requires_grad),
            "heads": sum(p.numel() for p in self.heads.parameters() if p.requires_grad),
        }
