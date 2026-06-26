"""
V13: 有状态纤维丛文本生成器
============================

核心改进 vs V12:
  1. 有状态 (stateful z) — 跨 token 保持隐藏状态, 实现上下文记忆
  2. 真正自我评估 — 复用 InternalCritic + MicroModifier
  3. 更快的 token 速度 — 直接 embedding lookup, 不创建 one-hot
  4. 输出才 token 化 — 内部始终在连续 embedding 空间

架构:
  token → embedding (normalizer 权重查找) → ODE × n_internal 步 (stateful z)
      → TextHead → logits → softmax → sample → next token
                        ↑
                   InternalCritic (自我评估)
                        ↓
              MicroModifier (微量调参)
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from .self_feedback import InternalCritic, MicroModifier, SelfFeedbackConfig


class StatefulFiberGenerator(nn.Module):
    """
    有状态纤维丛文本生成器.

    维护复数隐藏状态 z 跨 token, 引入上下文记忆.
    内部始终在连续 embedding 空间运算, 只在最终输出时 token 化.

    Args:
        model: TwistGrowMultimodalTwistorLMT (已含纤维丛包装)
        fiber_bundle: TwistFiberBundle 实例
        vocab_size: tokenizer 词表大小
        hidden_dim: 隐藏维度 (默认 64)
        n_internal: 每个 token 内部 ODE 步数 (默认 8)
    """

    def __init__(
        self,
        model,
        fiber_bundle,
        vocab_size: int,
        hidden_dim: int = 64,
        n_internal: int = 1,
        z_leak: float = 1e-4,
        z_max_norm: float = 50.0,
        reset_interval: int = 200,
    ):
        super().__init__()
        self.model = model
        self.fiber_bundle = fiber_bundle
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_internal = n_internal
        self.z_leak = z_leak
        self.z_max_norm = z_max_norm
        self.reset_interval = reset_interval

        # 共享 normalizer 的投影权重作为 embedding
        self.embed = model._mm.normalizer.normalizers["text"].proj
        self.mod_embed = model._mm.normalizer.modality_embed("text")

        # 隐藏状态 (跨 token 持久)
        self.register_buffer("_z", torch.zeros(1, hidden_dim, dtype=torch.complex64))
        self._step_count = 0

        # 自我评估器 (从 self_feedback 复用)
        self.config = SelfFeedbackConfig(
            min_think_steps=1, max_think_steps=1,
            tau_adjust_rate=0.005, weight_adjust_rate=0.0005,
        )
        self.critic = InternalCritic(hidden_dim, vocab_size)
        self.modifier = MicroModifier(hidden_dim, self.config)

        # 历史追踪
        self.loss_history = []
        self.eval_history = []
        self.adapt_events = 0

    # ============================================================
    # State management
    # ============================================================
    @property
    def z(self) -> torch.Tensor:
        return self._z

    def reset(self, batch: int = 1):
        """重置隐藏状态 (新序列开始)"""
        self._z = torch.zeros(batch, self.hidden_dim, dtype=torch.complex64, device=self._z.device)
        self._step_count = 0

    def _get_embed(self, token: torch.Tensor) -> torch.Tensor:
        """
        embedding lookup — 比 one-hot + linear 快 ~vocab_size 倍.
        Linear weight: (hidden_dim, vocab_size), 所以 weight[:, token] 是嵌入向量.

        Args:
            token: (B,) 或 (B, 1) token indices
        Returns:
            (B, hidden_dim) complex (实部=虚部=embed, 带 max-norm 约束)
        """
        if token.dim() > 1:
            token = token.squeeze(-1)
        e = self.embed.weight[:, token].T  # (B, hidden_dim)
        # max-norm 约束 (与 ModalityNormalizer.forward 一致)
        norm = e.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (self.model._mm.normalizer.normalizers["text"].max_norm / norm).clamp(max=1.0)
        e = e * scale
        return torch.complex(e, e)

    # ============================================================
    # 核心前向: 单 token 处理 (有状态)
    # ============================================================
    def forward(
        self,
        x: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
        """
        处理一个 token, 更新隐藏状态.

        Args:
            x: (B,) token indices
            return_diagnostics: 是否返回评估信息
        Returns:
            logits: (B, vocab_size)
        """
        B = x.shape[0] if x.dim() > 0 else 1
        z = self._z
        embed = self._get_embed(x)  # (B, D) complex
        dt = self.model._mm.backbone.dt

        # --- ODE × n_internal 步 (带泄露防止漂移) ---
        for t in range(self.n_internal):
            dzdt = self.model._mm.backbone.compute_dzdt(z, embed.real)
            dzdt = self.model._mm.bridge.inject(dzdt, z, self.mod_embed)
            z = z * (1.0 - self.z_leak) + dt * dzdt  # leaky integration
            z = self.fiber_bundle.project(z)
            z = torch.complex(
                torch.clamp(z.real, -self.model._mm.backbone.z_max, self.model._mm.backbone.z_max),
                torch.clamp(z.imag, -self.model._mm.backbone.z_max, self.model._mm.backbone.z_max),
            )

        # --- 数值安全防护 ---
        if torch.isnan(z).any() or torch.isinf(z).any():
            z = torch.where(torch.isnan(z) | torch.isinf(z), torch.zeros_like(z), z)
        z_norm = z.abs()
        if z_norm.max() > self.z_max_norm:
            scale = (self.z_max_norm / z_norm).clamp(max=1.0)
            z = z * scale.to(z.dtype)

        # --- 更新持久状态 (截断梯度) ---
        self._z = z.detach()
        self._step_count += 1

        # --- 周期性重置 (防止跨段落漂移) ---
        if self.reset_interval > 0 and self._step_count % self.reset_interval == 0:
            self._z = self._z * 0.1  # 衰减而不是清零, 保留弱上下文

        # --- 解码 ---
        logits = self.model._mm.heads.decode(z, "text")

        if return_diagnostics:
            # 自我评估
            prev_z = None
            prev_out = None
            if self._step_count > 1:
                pass  # 简化: 只用当前状态评估
            eval_scores = self.critic.evaluate(
                logits.detach(), z.real.detach(), prev_out, prev_z
            )
            return logits, eval_scores

        return logits

    # ============================================================
    # 自回归生成
    # ============================================================
    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        prompt: str,
        max_len: int = 100,
        temperature: float = 0.8,
        top_k: int = 20,
        show_progress: bool = False,
    ) -> str:
        """
        自回归文本生成 (有状态).

        Args:
            tokenizer: CharTokenizer
            prompt: 提示文本
            max_len: 生成最大长度
            temperature: 采样温度
            top_k: top-k 采样
        Returns:
            generated: 生成的文本
        """
        self.eval()
        self.reset()

        # 编码 prompt (先跑 prompt 不生成)
        encoded = tokenizer.encode(prompt)
        generated = list(encoded)

        # 预热: 跑 prompt token 但不采样
        for t in range(len(encoded)):
            inp = torch.tensor([encoded[t]], device=self._z.device)
            self(inp)

        # 自回归生成
        for _ in range(max_len):
            inp = torch.tensor([generated[-1]], device=self._z.device)
            logits = self(inp)

            if temperature > 0:
                logits = logits / temperature
            if top_k > 0:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).item()
            generated.append(next_idx)

            if show_progress and len(generated) % 20 == 0:
                print(f"  gen {len(generated)} tokens...")

        return tokenizer.decode(torch.tensor(generated))

    # ============================================================
    # 在线学习 (streaming next-token prediction + 自我评估)
    # ============================================================
    def learn(
        self,
        tokenizer,
        text: str,
        n_epochs: int = 5,
        lr: float = 1e-3,
        report_every: int = 500,
        eval_interval: int = 50,
    ) -> Dict:
        """
        在线学习: 有状态, 流式 next-token prediction.

        每步:
          1. 前向 (stateful) → logits
          2. CE loss vs 真实 next token
          3. backward + Adam step
          4. 自我评估 (每 eval_interval 步)
          5. 微量调参 (评估分数低时)

        Args:
            tokenizer: CharTokenizer
            text: 训练文本
            n_epochs: 遍历文本次数
            lr: 学习率
        Returns:
            stats: 学习统计
        """
        self.train()
        ids = tokenizer.encode(text).to(self._z.device)
        n = len(ids)

        params = list(self.model.parameters()) + list(self.fiber_bundle.parameters())
        opt = torch.optim.Adam(params, lr=lr)

        losses = []
        t0 = time.time()

        for epoch in range(n_epochs):
            self.reset()
            total_loss = 0.0
            # 在 learn 中 reset_interval 控制分段
            self.reset_interval = min(self.reset_interval, n // 4) if n > 100 else 0

            for i in range(n - 1):
                inp = ids[i:i+1]  # (1,)
                target = ids[i+1:i+2]  # (1,)

                logits = self(inp)
                if torch.isnan(logits).any():
                    # logits 已崩 → 跳过此步, 重置状态
                    self.reset()
                    continue
                loss = F.cross_entropy(logits, target)

                if torch.isnan(loss):
                    self.reset()
                    continue

                opt.zero_grad()
                loss.backward()
                # 裁剪梯度防止爆炸
                torch.nn.utils.clip_grad_norm_(
                    [p for p in params if p.grad is not None], 1.0
                )
                # 跳过 nan 梯度
                if any(p.grad is not None and torch.isnan(p.grad).any() for p in params):
                    opt.zero_grad()
                    self.reset()
                    continue

                opt.step()

                total_loss += float(loss.item())
                losses.append(float(loss.item()))

                # 自我评估 + 微量调参
                if i % eval_interval == 0 and i > 0:
                    self._self_evaluate(logits, self._z, i)

                # 报告
                if (i + 1) % report_every == 0:
                    avg = total_loss / report_every
                    total_loss = 0.0
                    print(f"  epoch {epoch+1}/{n_epochs}  step {i+1:6d}/{n-1}  "
                          f"loss={avg:.4f}  adapt={self.adapt_events}  "
                          f"fiber_dim={self.fiber_bundle.current_fiber_dim}")

            # 每 epoch 结束: 尝试生长纤维
            if self.fiber_bundle.current_fiber_dim < self.fiber_bundle.max_fiber_dim:
                n_prev = self.fiber_bundle.current_fiber_dim
                self.fiber_bundle.grow_fiber(n_new=8)
                print(f"  [生长] fiber_dim: {n_prev} → {self.fiber_bundle.current_fiber_dim}")

        elapsed = time.time() - t0
        return {
            "n_tokens": (n - 1) * n_epochs,
            "final_loss": float(np.mean(losses[-100:])),
            "losses": losses,
            "adapt_events": self.adapt_events,
            "elapsed_s": elapsed,
            "fiber_dim": self.fiber_bundle.current_fiber_dim,
        }

    def _self_evaluate(self, logits: torch.Tensor, z: torch.Tensor, step: int):
        """内部自我评估 + 微量参数调整"""
        with torch.no_grad():
            # 评估
            scores = self.critic.evaluate(logits, z.real)
            self.eval_history.append(scores)

            # 如果整体分数低 → 微量扰动纤维方向探索
            if scores["overall"] < 0.3:
                cur = self.fiber_bundle.current_fiber_dim
                if cur > 0:
                    noise = torch.randn_like(self.fiber_bundle.fiber_dirs[:cur]) * 0.01
                    self.fiber_bundle.fiber_dirs[:cur] += noise
                    self.adapt_events += 1
