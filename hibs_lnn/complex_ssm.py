"""
V16: 复数对角 SSM + 纤维丛
==============================

架构:
  x → in_proj → Conv1d → SiLU → SSM(复数A) → fiber → gate → out_proj → y
                               ↑
                          选择性 B, C

SSM 核心 (复数对角):
  h' = A·h + B·x     A = diag(-σ_k + i·θ_k)  复数频率
  y  = C·h + D·x

离散化 (ZOH):
  h_{k+1} = exp(Δ·A) · h_k + (exp(Δ·A) - I) / A · B · x_k
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class ComplexSSM(nn.Module):
    """
    复数对角状态空间模型 (S4D 风格 + 选择性 B/C).

    A = diag(-exp(log_σ) + i·θ)  复数对角, 可学习频率.
    Δ, B, C 依赖输入 (Mamba 选择性机制).

    Args:
        d_inner: 内部维度 (通常 = expand * d_model)
        d_state: SSM 状态维度 (每个隐藏维度的状态大小, 典型 16-64)
        dt_rank: Δ 投影的秩 (0 = d_inner 直接)
    """

    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int = 16):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        # 复数对角 A: λ_k = -exp(log_σ_k) + i·θ_k  (d_inner, d_state)
        self.log_sigma = nn.Parameter(torch.randn(d_inner, d_state) * 0.1)
        self.theta = nn.Parameter(torch.randn(d_inner, d_state) * 0.5)

        # Δ 投影: 输入 → Δ (选择性步长)
        if dt_rank > 0:
            self.dt_rank = dt_rank
            self.dt_proj = nn.Sequential(
                nn.Linear(d_inner, dt_rank),
                nn.Linear(dt_rank, d_inner),
            )
        else:
            self.dt_rank = d_inner
            self.dt_proj = nn.Linear(d_inner, d_inner)

        # B, C 投影 (选择性)
        self.B_proj = nn.Linear(d_inner, d_state)
        self.C_proj = nn.Linear(d_inner, d_state)

        # D (跳跃连接, 可学习)
        self.D = nn.Parameter(torch.ones(d_inner))

    @property
    def A_complex(self) -> torch.Tensor:
        """复数对角矩阵 A  (d_inner, d_state) complex"""
        return -self.log_sigma.exp() + 1j * self.theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_inner) 输入序列
        Returns:
            y: (B, L, d_inner) 输出序列
        """
        B, L, D = x.shape
        device = x.device

        # 复对角 A
        A = self.A_complex  # (D, N) complex

        # 选择性 Δ, B, C
        dt = F.softplus(self.dt_proj(x)) + 1e-4  # (B, L, D)
        B_k = self.B_proj(x)  # (B, L, N)
        C_k = self.C_proj(x)  # (B, L, N)

        # ZOH 离散化
        # Ā = exp(Δ·A) → 对角矩阵指数 = 逐元素 exp
        # h_{k+1} = Ā·h_k + (Ā - I)/A · B · x
        dt_A = dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)  # (B, L, D, N) complex
        A_bar = torch.exp(dt_A)  # (B, L, D, N) complex

        # Ā·h: 逐元素乘, 不是矩阵乘! (因为 A, h 都是对角)
        # h: (B, D, N) complex, A_bar: (B, L, D, N) → h * A_bar: (B, D, N)
        # 扫描: h_{k+1} = A_bar_k * h_k + B_eff_k * x_k

        # B_eff = (Ā - I) / A · B_k  (对角简化, 因为 A 可逐元素求逆)
        epsilon = 1e-8
        B_eff = (A_bar - 1.0) / (A.unsqueeze(0).unsqueeze(0) + epsilon)  # (B, L, D, N) complex
        B_eff = B_eff * B_k.unsqueeze(2)  # (B, L, D, N) complex

        # 序列扫描 (先 sequential, 后续可以换关联扫描)
        h = torch.zeros(B, D, self.d_state, dtype=torch.complex64, device=device)
        outputs = []
        for k in range(L):
            # h = Ā·h + B_eff·x_k
            h = A_bar[:, k] * h + B_eff[:, k] * x[:, k].unsqueeze(-1)
            # y = C·h + D·x  取实部
            y_k = (C_k[:, k].unsqueeze(1) * h).sum(dim=-1).real  # (B, D)
            y_k = y_k + self.D * x[:, k]
            outputs.append(y_k)

        return torch.stack(outputs, dim=1)


class ComplexSSMBlock(nn.Module):
    """
    完整 SSM 块: expansion → Conv1d → SiLU → SSM → 纤维丛 → gate → out_proj.

    类似 Mamba 的 macro block.

    Args:
        d_model: 输入/输出维度
        expand: 内部扩展因子 (默认 2)
        d_state: SSM 状态维度
        dt_rank: Δ 投影秩
    """

    def __init__(
        self,
        d_model: int,
        expand: int = 2,
        d_state: int = 16,
        dt_rank: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state

        # 输入投影: d_model → d_inner × 2 (SSM 分支 + 门分支)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 卷积 (因果, 深度可分离)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            groups=self.d_inner,
            padding=3,  # 因果: left padding
        )
        self.act = nn.SiLU()

        # SSM 核心
        self.ssm = ComplexSSM(
            d_inner=self.d_inner,
            d_state=d_state,
            dt_rank=dt_rank,
        )

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            y: (B, L, d_model)
        """
        B, L, D = x.shape

        # 输入投影 → 分割 SSM/门分支
        x_proj = self.in_proj(x)  # (B, L, 2*d_inner)
        ssm_branch, gate_branch = x_proj.chunk(2, dim=-1)  # 各 (B, L, d_inner)

        # 卷积 (因果) + SiLU
        ssm_branch = ssm_branch.transpose(-1, -2)  # (B, d_inner, L) for Conv1d
        ssm_branch = self.conv1d(ssm_branch)[:, :, :L]  # 截断到 L
        ssm_branch = self.act(ssm_branch)
        ssm_branch = ssm_branch.transpose(-1, -2)  # (B, L, d_inner)

        # SSM
        ssm_out = self.ssm(ssm_branch)  # (B, L, d_inner)

        # 门 (SwiGLU 风格)
        y = ssm_out * F.silu(gate_branch)

        # 输出投影
        y = self.out_proj(y)  # (B, L, d_model)

        return y


class SSMFiberTwistor(nn.Module):
    """
    SSM + 纤维丛: Mamba 风格 backbone + 可生长子空间 steering.

    Args:
        vocab_size: 词表大小
        d_model: 模型维度
        d_state: SSM 状态维度
        fiber: TwistFiberBundle 实例 (可选)
        n_layers: SSM 块层数
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        d_state: int = 16,
        fiber=None,
        n_layers: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # SSM 块
        self.blocks = nn.ModuleList([
            ComplexSSMBlock(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])

        # 纤维丛 (可选, 放在最后一个 SSM 块之后)
        self.fiber = fiber

        # 输出投影
        self.out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        ids: torch.Tensor,
        return_logits: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            ids: (B, L) token indices
        Returns:
            logits: (B, L, vocab_size)
        """
        x = self.embed(ids)  # (B, L, d_model)

        for block in self.blocks:
            x = block(x)

        if self.fiber is not None:
            x = self.fiber.project(x)  # (B, L, D) — 现在直接支持 3D 输入

        logits = self.out(x)  # (B, L, vocab_size)
        return logits

    def generate(self, tokenizer, prompt: str, max_len: int = 100,
                 temperature: float = 0.8, top_k: int = 20) -> str:
        self.eval()
        with torch.no_grad():
            ids = tokenizer.encode(prompt).unsqueeze(0)
            # 使用设备
            device = next(self.parameters()).device
            ids = ids.to(device)
            generated = ids[0].tolist()

            for _ in range(max_len):
                logits = self(ids, return_logits=True)  # (1, L, V)
                next_logits = logits[0, -1, :] / temperature

                if top_k > 0:
                    vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < vals[-1]] = float('-inf')

                probs = F.softmax(next_logits, dim=-1)
                next_idx = torch.multinomial(probs, 1).item()
                generated.append(next_idx)
                ids = torch.tensor([[next_idx]], device=device)

        return tokenizer.decode(torch.tensor(generated))
