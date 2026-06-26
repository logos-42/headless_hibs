"""
V14: 预训练 TwistorLMT + 纤维丛自适应微调
========================================

核心思路:
  1. 加载预训练 TwistorLMT (WikiText-2, 5000步)
  2. 冻结 backbone (不计算梯度, torch.no_grad)
  3. 纤维丛作为"可训练舵" — 在不改动原有动力学的前提下重定向状态
  4. 只训练纤维丛 ~1K 参数 → 极快, 极轻量

速度优势:
  - ODE 步数不计梯度 (torch.no_grad 跳过 backward)
  - 纤维丛 1K 参数 backward 极快
  - 预训练输出的质量 + 纤维丛的适应性
"""

import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from .fiber_bundle import TwistFiberBundle


class PretrainedFiberTwistor(nn.Module):
    """
    预训练 TwistorLMT + 纤维丛自适应.

    Args:
        backbone: 预训练 TwistorLMT
        fiber_bundle: TwistFiberBundle (训练)
        n_internal: ODE 步数/时间步
        lr_fiber: 纤维丛学习率
        freeze_backbone: True=冻结 backbone (仅训练纤维丛), False=全部训练
    """

    def __init__(
        self,
        backbone: nn.Module,
        fiber_bundle: TwistFiberBundle,
        n_internal: int = 1,
        lr_fiber: float = 1e-3,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.fiber_bundle = fiber_bundle
        self.n_internal = n_internal
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # 纤维丛初始化: 零振幅 (初始 = 无变化)
        with torch.no_grad():
            self.fiber_bundle.fiber_amps.zero_()

        # 纤维丛参数统计
        self.fiber_params = sum(p.numel() for p in self.fiber_bundle.parameters())
        self.backbone_params = sum(p.numel() for p in self.backbone.parameters())

        self.loss_history = []
        self.adapt_events = 0
        self._step_count = 0

    def reset_state(self, batch: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.zeros(batch, self.backbone.hidden_dim,
                          dtype=torch.complex64, device=device)

    # ============================================================
    # 核心: 冻结 ODE + 可训练纤维投影
    # ============================================================
    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, V) one-hot input
            z: 可选初始状态 (None = 零)
        Returns:
            logits: (B, V) 或 (logits, z)
        """
        B = x.shape[0]
        if z is None:
            z = self.reset_state(B, x.device)

        # === 阶段 1: ODE 演化 ===
        ctx = torch.no_grad if self.freeze_backbone else torch.enable_grad
        with ctx():
            for t in range(self.n_internal):
                dzdt = self.backbone.compute_dzdt(z, x)
                z = z + self.backbone.dt * dzdt
                if self.backbone.mobius is not None:
                    z = self.backbone.mobius.project_state(z)
                z = torch.complex(
                    torch.clamp(z.real, -self.backbone.z_max, self.backbone.z_max),
                    torch.clamp(z.imag, -self.backbone.z_max, self.backbone.z_max),
                )

        # === 阶段 2: 纤维丛自适应投影 (可训练) ===
        # 残差式: 纤维丛学的是"微调偏移", 不截断 z
        D = z.shape[-1]; k = min(D, self.fiber_bundle.max_fiber_dim)
        dirs = self.fiber_bundle.fiber_dirs[:k]  # (k, base_dim)
        amp = self.fiber_bundle.fiber_amps[:k].unsqueeze(0)  # (1, k)
        fidx = self.fiber_bundle.active_mask[:k].unsqueeze(0) > 0  # (1, k)
        # 实部/虚部分别投影 → 重建
        steer_real = (z.real @ dirs) @ dirs.T  # (B, k)
        steer_imag = (z.imag @ dirs) @ dirs.T
        steer = torch.complex(steer_real, steer_imag)
        # 残差: z += 振幅 * 方向偏移 (仅激活纤维)
        z = z + (amp * fidx.to(amp.dtype)).to(z.dtype) * steer

        # === 解码 (冻结的 out) ===
        logits = self.backbone.out(z.real)

        self._step_count += 1
        if return_state:
            return logits, (z.detach() if self.freeze_backbone else z)
        return logits

    # ============================================================
    # 在线学习 (只训练纤维丛)
    # ============================================================
    def learn(
        self,
        ids: torch.Tensor,
        n_epochs: int = 1,
        lr: float = 1e-3,
        report_every: int = 1000,
    ) -> Dict:
        """
        在线学习: 只训练纤维丛, backbone 冻结.

        Args:
            ids: (N,) token indices
            n_epochs: 遍历次数
            lr: 学习率
        Returns:
            stats
        """
        self.fiber_bundle.train()
        V = self.backbone.vocab_size if hasattr(self.backbone, 'vocab_size') else \
            self.backbone.out.weight.shape[0]
        trainable = list(self.fiber_bundle.parameters())
        if not self.freeze_backbone:
            trainable += list(self.backbone.parameters())
        opt = torch.optim.Adam(trainable, lr=lr)
        n = len(ids)
        losses = []
        t0 = time.time()

        for epoch in range(n_epochs):
            z = self.reset_state(1, ids.device)
            total_loss = 0.0

            for i in range(n - 1):
                x = F.one_hot(ids[i:i+1], V).float().to(ids.device)
                target = ids[i+1:i+2]

                logits, z = self(x, z=z.detach(), return_state=True)
                loss = F.cross_entropy(logits, target)

                opt.zero_grad()
                loss.backward()
                trainable_params = list(self.fiber_bundle.parameters())
                if not self.freeze_backbone:
                    trainable_params += list(self.backbone.parameters())
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                opt.step()

                losses.append(float(loss.item()))
                total_loss += float(loss.item())

                if (i + 1) % report_every == 0:
                    print(f"  epoch {epoch+1}/{n_epochs}  step {i+1:6d}/{n-1}  "
                          f"loss={total_loss/report_every:.4f}  fiber_dim={self.fiber_bundle.current_fiber_dim}")
                    total_loss = 0.0

        elapsed = time.time() - t0
        return {
            "n_tokens": (n - 1) * n_epochs,
            "final_loss": float(np.mean(losses[-200:])),
            "losses": losses,
            "elapsed_s": elapsed,
            "fiber_dim": self.fiber_bundle.current_fiber_dim,
        }

    # ============================================================
    # 生成 (不计梯度, 始终)
    # ============================================================
    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        prompt: str,
        max_len: int = 100,
        temperature: float = 0.8,
        top_k: int = 20,
    ) -> str:
        self.eval()
        encoded = tokenizer.encode(prompt)
        generated = list(encoded)

        # 预热
        z = self.reset_state(1, device=next(self.parameters()).device)
        V = self.backbone.out.weight.shape[0]
        for t in range(len(encoded)):
            x = F.one_hot(torch.tensor([encoded[t]], device=z.device), V).float()
            logits, z = self(x, z=z, return_state=True)

        # 自回归
        for _ in range(max_len):
            x = F.one_hot(torch.tensor([generated[-1]], device=z.device), V).float()
            logits, z = self(x, z=z, return_state=True)

            logits = logits / temperature
            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            generated.append(next_idx)

        return tokenizer.decode(torch.tensor(generated))
