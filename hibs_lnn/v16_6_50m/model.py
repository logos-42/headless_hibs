"""
hibs 0.16 (基于 hibs 0.16): 完整语言模型
=====================

架构总览:
    Embedding(vocab, d_model=768)
      ↓
    [SSM_Layer_V16_6_Scaled × 12]   <- 每层含复值 κ 调制
      ↓
    LayerNorm
      ↓
    Linear(vocab, weight=embed.T)   <- 权重共享

目标参数量: ~50M
  - Embedding: 8192 × 768 = 6.3M (12.6%)
  - 12 × SSM Layer: ~3.6M/层 = 43.2M (86.4%)
  - 其他: ~0.5M
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import SSM_Layer_V16_6_Scaled
from .fiber import HibsFiberBundleV2_Scaled


@dataclass
class Hibs_0_16_50M_Config:
    """hibs 0.16 (基于 hibs 0.16) 配置"""
    vocab_size: int = 8192
    d_model: int = 768
    n_layers: int = 12
    d_state: int = 16
    max_seq_len: int = 1024
    conv_kernel: int = 4

    # 纤维丛
    fiber_base_dim: int = 64
    fiber_max_dim: int = 16
    fiber_init_dim: int = 8

    # 训练
    dropout: float = 0.0
    tie_weights: bool = True


class Hibs_0_16_50M(nn.Module):
    """
    hibs 0.16 (基于 hibs 0.16) 参数语言模型.

    使用示例:
        >>> cfg = Hibs_0_16_50M_Config()
        >>> model = Hibs_0_16_50M(cfg)
        >>> ids = torch.randint(0, cfg.vocab_size, (1, 64))
        >>> logits = model(ids)  # (1, 64, vocab_size)
    """

    def __init__(self, config: Hibs_0_16_50M_Config):
        super().__init__()
        self.config = config

        # 共享纤维丛
        self.fiber = HibsFiberBundleV2_Scaled(
            base_dim=config.fiber_base_dim,
            max_fiber_dim=config.fiber_max_dim,
            init_fiber_dim=config.fiber_init_dim,
        )

        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # SSM 层栈
        self.layers = nn.ModuleList([
            SSM_Layer_V16_6_Scaled(
                d_model=config.d_model,
                d_state=config.d_state,
                fiber=self.fiber,
                layer_idx=i,
                conv_kernel=config.conv_kernel,
            )
            for i in range(config.n_layers)
        ])

        # 最终 norm
        self.norm_f = nn.LayerNorm(config.d_model)

        # 输出 (权重共享)
        if config.tie_weights:
            self.out = nn.Linear(config.d_model, config.vocab_size, bias=False)
            self.out.weight = self.embed.weight
        else:
            self.out = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """正交初始化 (与 hibs 0.16 一致)"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ids: (B, L) token ids
        Returns:
            logits: (B, L, vocab_size)
        """
        x = self.embed(ids)  # (B, L, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.out(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        自回归生成.

        Args:
            ids: (B, L) 初始 prompt
            max_new_tokens: 生成 token 数
            temperature: 采样温度
            top_k: top-k 采样
            top_p: nucleus 采样
        Returns:
            generated: (B, L + max_new_tokens)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 截断到 max_seq_len
            ids_cond = ids if ids.size(1) <= self.config.max_seq_len else ids[:, -self.config.max_seq_len:]
            logits = self(ids_cond)[:, -1, :] / max(temperature, 1e-5)

            # Top-k + top-p 过滤
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_mask = cumprobs > top_p
                sorted_mask[:, 0] = False
                mask = sorted_mask.scatter(1, sorted_idx, sorted_mask)
                logits[mask] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, next_id], dim=1)
        return ids

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_v16_6_50m(vocab_size: int = 8192) -> Hibs_0_16_50M:
    """便捷创建函数"""
    cfg = Hibs_0_16_50M_Config(vocab_size=vocab_size)
    return Hibs_0_16_50M(cfg)