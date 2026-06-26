"""
Hibs hibs 0.16: 扭量纤维丛 (复值 κ 扩展版)
=====================================

继承自 V16.6 验证的 TwistFiberBundle:
  - κ_f = z_base @ dirs_f · exp(i · phase_f)  (complex per fiber)
  - |κ|: amplitude modulation
  - arg(κ): phase modulation of eigenvalues
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..fiber_bundle import TwistFiberBundle as HibsFiberBundle


class HibsFiberBundleV2_Scaled(HibsFiberBundle):
    """
    hibs 0.16 复值 κ 纤维丛 (50M 扩展版).

    与 247K 版本关键差异:
      - base_dim 默认 64 (而非 16), 以匹配 768 维 d_model
      - max_fiber_dim 默认 16
      - 每个 fiber 独立可学习 phase
    """

    def __init__(
        self,
        base_dim: int = 64,
        max_fiber_dim: int = 16,
        init_fiber_dim: int = 8,
    ):
        super().__init__(base_dim, max_fiber_dim, init_fiber_dim)
        # 可学习相位 (per fiber, radians)
        self.fiber_phases = nn.Parameter(
            torch.randn(max_fiber_dim) * 0.5
        )

    def compute_complex_kappa(self, z: torch.Tensor) -> torch.Tensor:
        """
        返回 (..., n_fibers) 复值 κ per fiber.

        Args:
            z: (B, L, D) 或 (B, D) 实数张量
        Returns:
            kappa: (B, L, n_fibers) 或 (B, n_fibers) 复数张量
        """
        orig_dim = z.dim()
        if orig_dim == 3:
            B, L, D = z.shape
            z_2d = z.reshape(-1, D)
        else:
            B_2d, D = z.shape
            z_2d = z

        z_base = z_2d[:, :self.base_dim]
        dirs = self.fiber_dirs  # (n_fibers, base_dim)
        amp = self.fiber_amps[:self.current_fiber_dim]  # (n_fibers,)

        # 实数投影 per fiber
        proj = z_base @ dirs.T  # (N, n_fibers)
        proj = proj * amp.unsqueeze(0)

        # 复数旋转
        phase = self.fiber_phases[:self.current_fiber_dim]  # (n_fibers,)
        kappa = proj * torch.exp(1j * phase.unsqueeze(0))

        if orig_dim == 3:
            kappa = kappa.view(B, L, -1)
        return kappa

    def get_complex_stats(self):
        """调试用: 返回振幅与相位统计"""
        amps = self.fiber_amps[:self.current_fiber_dim].detach().cpu()
        phases = self.fiber_phases[:self.current_fiber_dim].detach().cpu()
        return {
            "amplitudes": amps.tolist(),
            "phases_rad": phases.tolist(),
            "amp_mean": float(amps.mean()),
            "phase_mean": float(phases.mean()),
            "phase_std": float(phases.std()),
        }