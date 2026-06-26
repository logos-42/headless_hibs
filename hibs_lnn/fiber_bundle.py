"""
扭量纤维丛 (Twist Fiber Bundle) — V11+ 旋转联络版
=================================================

纤维之间通过固定 Möbius 旋转连接, 不是独立参数.

核心结构:
  - 基空间: Möbius 环 (固定相位 θ_i = π·(i+1)/(n+1))
  - 初始方向: dirs_0 (可学, base_dim 维向量)
  - 联络: R(θ) — 沿基空间通过 2D 块旋转
  - 纤维方向: dirs_i = R(θ_i · freq) · dirs_0

不再有独立 fiber_dirs. 所有纤维方向是 dirs_0 沿 Möbius 环旋转的像.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwistFiberBundle(nn.Module):
    def __init__(self, base_dim=16, max_fiber_dim=16, init_fiber_dim=8):
        super().__init__()
        self.base_dim = base_dim
        self.max_fiber_dim = min(max_fiber_dim, base_dim)
        self.init_fiber_dim = min(init_fiber_dim, self.max_fiber_dim)

        self.register_buffer("base_phases", torch.zeros(self.max_fiber_dim))
        for i in range(self.max_fiber_dim):
            self.base_phases[i] = math.pi * (i + 1) / (self.max_fiber_dim + 1)

        n_blocks = base_dim // 2
        self.register_buffer("freq_mult", torch.arange(1, n_blocks + 1).float())

        self.dirs_0 = nn.Parameter(torch.randn(base_dim) * 0.1)
        self.fiber_amps = nn.Parameter(torch.ones(self.max_fiber_dim) * 0.5)
        self.register_buffer("active_mask", torch.zeros(self.max_fiber_dim))
        self.active_mask[:self.init_fiber_dim] = 1.0
        self.dir_history = []

    @property
    def current_fiber_dim(self):
        return int(self.active_mask.sum().item())

    @property
    def fiber_dirs(self):
        n = self.current_fiber_dim
        if n == 0:
            return torch.zeros(0, self.base_dim, device=self.dirs_0.device)
        d = self.base_dim
        nb = d // 2
        phases = self.base_phases[:n]
        angles = phases[:, None] * self.freq_mult[None, :nb]  # (n, nb): use ALL frequencies
        c, s = torch.cos(angles), torch.sin(angles)
        xy = self.dirs_0.view(1, nb, 2)
        rx = c * xy[:, :, 0] - s * xy[:, :, 1]
        ry = s * xy[:, :, 0] + c * xy[:, :, 1]
        return torch.stack([rx, ry], dim=-1).reshape(n, d)

    def compute_kappa(self, z):
        orig_dim = z.dim()
        if orig_dim == 3:
            B, L, D = z.shape
            z_2d = z.reshape(-1, D)
        else:
            B_, D = z.shape
            z_2d = z
        z_base = z_2d[:, :self.base_dim]
        dirs = self.fiber_dirs
        proj = (z_base @ dirs.T) @ dirs
        kappa = proj.norm(dim=-1, keepdim=True)
        if orig_dim == 3:
            kappa = kappa.view(B, L, 1)
        return kappa

    def project(self, z):
        if z.is_complex():
            return self._project_complex(z)
        return self._project_real(z)

    def _project_real(self, z):
        orig_shape = z.shape
        if z.dim() == 3:
            B, L, D = z.shape
            z = z.reshape(-1, D)
        else:
            B_, D = z.shape
        z_base = z[:, :self.base_dim]
        dirs = self.fiber_dirs
        amp = self.fiber_amps[:self.current_fiber_dim].unsqueeze(0)
        mask = self.active_mask[:self.current_fiber_dim].unsqueeze(0)
        proj = (z_base @ dirs.T) @ dirs
        z = z.clone()
        z[:, :self.base_dim] = z[:, :self.base_dim] + mask * amp * proj
        if len(orig_shape) == 3:
            z = z.reshape(orig_shape)
        return z

    def _project_complex(self, z):
        B, D = z.shape
        dirs = self.fiber_dirs[:, :D]
        z_real = z.real @ dirs.T
        z_imag = z.imag @ dirs.T
        twist_base = torch.exp(
            1j * ((z_real.abs() + z_imag.abs()) * self.base_phases[:self.current_fiber_dim].unsqueeze(0)).sum(-1, keepdim=True)
        )
        amp = self.fiber_amps[:self.current_fiber_dim].unsqueeze(0)
        z_twisted = z * (1 + amp * (twist_base - 1).real.to(z.dtype))
        mask = self.active_mask[:self.current_fiber_dim].to(z.device)
        return z_twisted * mask.unsqueeze(0)

    def compute_complex_kappa(self, z):
        """Compute complex κ = (z@dirs) · A · exp(i·Φ) for each fiber.
        
        Returns (B, L, nf) complex tensor where each fiber's κ value
        encodes both amplitude (scattering strength) and phase (wave curvature).
        """
        orig_shape = z.shape
        z_flat = z.reshape(-1, self.base_dim)
        dirs = self.fiber_dirs[:, :self.base_dim]
        amps = self.fiber_amps[:self.current_fiber_dim].unsqueeze(0)
        phases = self.base_phases[:self.current_fiber_dim].unsqueeze(0)
        # κ_f = (z · dir_f) · A_f · exp(i · Φ_f)
        kappa = (z_flat @ dirs.T) * amps * torch.exp(1j * phases)
        return kappa.reshape(*orig_shape[:-1], -1)

    def grow_fiber(self, n_new=4):
        cur = self.current_fiber_dim
        if cur >= self.max_fiber_dim:
            return False
        end = min(cur + n_new, self.max_fiber_dim)
        with torch.no_grad():
            for i in range(cur, end):
                self.fiber_amps[i] = 0.5
        self.active_mask[cur:end] = 1.0
        return True

    def adapt_direction(self, loss_signal, lr=0.01, threshold=2.3):
        changed = False
        with torch.no_grad():
            if loss_signal > threshold:
                theta = math.pi / 4
                c, s = math.cos(theta), math.sin(theta)
                d = self.base_dim
                nb = d // 2
                xy = self.dirs_0.view(nb, 2)
                rx = c * xy[:, 0] - s * xy[:, 1]
                ry = s * xy[:, 0] + c * xy[:, 1]
                self.dirs_0.data = torch.stack([rx, ry]).t().reshape(-1).contiguous()
                self.dir_history.append({"event": "adapt", "loss_signal": loss_signal})
                changed = True
        return changed

    def get_state(self):
        return {
            "base_dim": self.base_dim,
            "max_fiber_dim": self.max_fiber_dim,
            "current_fiber_dim": self.current_fiber_dim,
            "n_adapt_events": sum(1 for h in self.dir_history if h.get("event") == "adapt"),
            "dir_history": self.dir_history[-10:],
        }


class FiberDirectionAdaptor(nn.Module):
    def __init__(self, bundle, ic_threshold=2.3, cooldown=30):
        super().__init__()
        self.bundle = bundle
        self.ic_threshold = ic_threshold
        self.cooldown = cooldown
        self.last_adapt_epoch = -cooldown
        self.n_adaptations = 0

    def step(self, epoch, infonce_value):
        if epoch - self.last_adapt_epoch < self.cooldown:
            return False
        if infonce_value > self.ic_threshold:
            changed = self.bundle.adapt_direction(infonce_value)
            if changed:
                self.last_adapt_epoch = epoch
                self.n_adaptations += 1
                return True
        return False
