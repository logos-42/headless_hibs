"""
hibs 0.16 (基于 hibs 0.16): 复值 κ 调制的 SSM 层
=============================

继承自 hibs 0.16 验证的 SSM_Layer_V16_6:
  - 振幅调制: a *= 1 + tanh(Σ |κ_f| · w_amp + b_amp)
  - 相位调制: Im(A) += 0.5 · Σ arg(κ_f) · w_phase + b_phase (频率偏移)
  - ZOH 解析解: h_{k+1} = exp(Δ·A)·h_k + (exp(Δ·A)-I)/A·B·x_k

50M 扩展:
  - d_model: 128 -> 768
  - 支持可选的 FlashAttention-style 加速 (torch.compile)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fiber import HibsFiberBundleV2_Scaled


class SSM_Layer_V16_6_Scaled(nn.Module):
    """
    hibs 0.16 SSM 层 - 复值 κ 调制 (50M 规模).

    架构:
        x -> LayerNorm -> in_proj (4d) -> split -> Conv1d -> SiLU
        -> SSM(complex A modulated by complex κ)
        -> gate -> out_proj -> + residual
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        fiber: HibsFiberBundleV2_Scaled = None,
        layer_idx: int = 0,
        conv_kernel: int = 4,
        use_torch_compile: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.layer_idx = layer_idx

        # Norm + Projection
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 4 * d_model, bias=False)
        self.conv = nn.Conv1d(
            2 * d_model, 2 * d_model,
            kernel_size=conv_kernel,
            groups=2 * d_model,
            padding=conv_kernel - 1,
        )
        self.out_proj = nn.Linear(2 * d_model, d_model, bias=False)

        # SSM 参数 (复数对角 A = -|σ| + i·θ)
        self.ssm_logsigma = nn.Parameter(torch.randn(2 * d_model, d_state) * 0.1)
        self.ssm_theta = nn.Parameter(torch.randn(2 * d_model, d_state) * 0.5)
        self.ssm_D = nn.Parameter(torch.ones(2 * d_model))

        # Δ / B / C 投影 (Mamba 选择性)
        self.dt_proj = nn.Sequential(
            nn.Linear(2 * d_model, 16),
            nn.Linear(16, 2 * d_model),
        )
        self.B_proj = nn.Linear(2 * d_model, d_state)
        self.C_proj = nn.Linear(2 * d_model, d_state)

        # 复值 κ 调制
        self.fiber = fiber
        if fiber is not None:
            nf = fiber.current_fiber_dim
            self.kappa_amp_weight = nn.Parameter(torch.randn(nf) * 0.1)
            self.kappa_phase_weight = nn.Parameter(torch.randn(nf) * 0.1)
            self.kappa_amp_bias = nn.Parameter(torch.zeros(1))
            self.kappa_phase_bias = nn.Parameter(torch.zeros(1))

        self.use_torch_compile = use_torch_compile

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            y: (B, L, d_model)
        """
        residual = x
        x = self.norm(x)

        # 输入投影 + 分割 SSM/门分支
        a, b = self.in_proj(x).chunk(2, dim=-1)  # 各 (B, L, 2*d_model)

        # 因果卷积 + SiLU
        a = a.transpose(-1, -2)  # (B, 2*d_model, L)
        a = self.conv(a)[:, :, :x.shape[1]]  # 截断到 L (因果)
        a = F.silu(a.transpose(-1, -2))  # (B, L, 2*d_model)

        # 复值 κ 调制
        if self.fiber is not None:
            kappa = self.fiber.compute_complex_kappa(x)  # (B, L, nf) complex
            amp_k = kappa.abs()
            phase_k = kappa.angle()

            amp_mod = (amp_k * self.kappa_amp_weight).sum(dim=-1, keepdim=True) + self.kappa_amp_bias
            phase_mod = (phase_k * self.kappa_phase_weight).sum(dim=-1, keepdim=True) + self.kappa_phase_bias

            # 振幅调制: 门控输入
            a = a * (1 + torch.tanh(amp_mod))

            # 相位调制: 频率偏移 (加法, 保持 |σ| 稳定)
            sigma = -self.ssm_logsigma.exp()  # (2*d_model, d_state)
            theta0 = self.ssm_theta
            theta_shift = (phase_mod * 0.5).unsqueeze(-1)
            A_eff = sigma.unsqueeze(0).unsqueeze(0) + 1j * (
                theta0.unsqueeze(0).unsqueeze(0) + theta_shift
            )
        else:
            A_eff = (-self.ssm_logsigma.exp() + 1j * self.ssm_theta).unsqueeze(0).unsqueeze(0)

        # SSM 扫描
        B_, L_, D_ = a.shape
        S_ = self.ssm_logsigma.shape[-1]

        dt = F.softplus(self.dt_proj(a)) + 1e-4
        Bk = self.B_proj(a)
        Ck = self.C_proj(a)

        # ZOH 离散化
        Ab = torch.exp(dt.unsqueeze(-1) * A_eff)  # (B, L, 2*d_model, S)
        bk = (Ab - 1.0) / (A_eff + 1e-8) * Bk.unsqueeze(2) * a.unsqueeze(-1)

        # 顺序扫描 (Python loop, 占 85% 时间)
        # TODO: 用关联扫描替换
        h = torch.zeros(B_, D_, S_, dtype=torch.complex64, device=Ab.device)
        ys = torch.empty(B_, L_, D_, device=a.device, dtype=a.dtype)
        for k in range(L_):
            h = Ab[:, k] * h + bk[:, k]
            y = (Ck[:, k].unsqueeze(1) * h).sum(dim=-1).real + self.ssm_D * a[:, k]
            ys[:, k] = y

        # 输出 + 残差
        return self.out_proj(ys * F.silu(b)) + residual