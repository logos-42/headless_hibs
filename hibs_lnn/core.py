"""
Twistor-LMT Core Module
=======================
Core Twistor-inspired Liquid Neural Network implementation.

Key features:
- Complex-valued hidden state z ∈ ℂ^n
- State-dependent time constant τ(z)
- Sparse connectivity
- Multi-scale tau
- Stability monitoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np


class TwistorLMT(nn.Module):
    """
    Twistor-inspired Liquid Neural Network with complex-valued states.
    Stability-optimized version.

    Dynamics: dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        output_dim: int = 1,
        sparsity: float = 0.3,
        multi_scale_tau: bool = True,
        dt: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        dzdt_max: float = 10.0,
        z_max: float = 100.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.multi_scale_tau = multi_scale_tau

        # Stability parameters
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dzdt_max = dzdt_max
        self.z_max = z_max

        # Weight matrices
        self.W_real = nn.Linear(hidden_dim, hidden_dim)
        self.W_imag = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)

        # Sparse masks
        self.sparse_mask_real = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        self.sparse_mask_imag = nn.Parameter(torch.ones(hidden_dim, hidden_dim))

        # Multi-scale tau bias
        if multi_scale_tau:
            self.tau_bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.tau_bias = None

        # Bias terms
        self.b_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(hidden_dim))

        # Output projection
        self.out = nn.Linear(hidden_dim, output_dim)

        # Mobius constraint and resonance attention (optional, disabled by default)
        self.mobius = None
        self.resonance = None
        self._resonance_mode = "additive"

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.W_real.weight, gain=0.5)
        nn.init.orthogonal_(self.W_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau.weight, gain=0.1)
        nn.init.zeros_(self.W_real.bias)
        nn.init.zeros_(self.W_imag.bias)
        nn.init.zeros_(self.U.bias)
        nn.init.zeros_(self.W_tau.bias)
        nn.init.zeros_(self.b_real)
        nn.init.zeros_(self.b_imag)

        if self.sparsity > 0:
            with torch.no_grad():
                mask_real = (
                    torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity
                ).float()
                mask_imag = (
                    torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity
                ).float()
                self.sparse_mask_real.copy_(mask_real)
                self.sparse_mask_imag.copy_(mask_imag)

        if self.multi_scale_tau and self.tau_bias is not None:
            nn.init.zeros_(self.tau_bias)

    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """Compute state-dependent time constant with clamping."""
        z_mod = torch.abs(z)
        tau = F.sigmoid(self.W_tau(z_mod))

        if self.multi_scale_tau and self.tau_bias is not None:
            tau = tau + self.tau_bias.unsqueeze(0)

        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau + 1e-6

    def check_numerical_stability(
        self, z: torch.Tensor, dzdt: torch.Tensor
    ) -> Dict[str, bool]:
        """Check for NaN/Inf."""
        return {
            "z_nan": torch.isnan(z).any().item(),
            "z_inf": torch.isinf(z).any().item(),
            "dzdt_nan": torch.isnan(dzdt).any().item(),
            "dzdt_inf": torch.isinf(dzdt).any().item(),
        }

    # === Tensor Decoder ===
    def decode_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """Decode to second-order tensor: v ⊗ v"""
        v = z.real
        return torch.einsum("bi,bj->bij", v, v)

    def decode_tensor_flat(self, z: torch.Tensor) -> torch.Tensor:
        """Decode to flattened tensor."""
        tensor = self.decode_tensor(z)
        return tensor.view(tensor.size(0), -1)

    # === RK4 Integrator ===
    def rk4_step(
        self, z: torch.Tensor, x: torch.Tensor, dt: float = None
    ) -> torch.Tensor:
        """Runge-Kutta 4th order integration."""
        if dt is None:
            dt = self.dt

        k1 = self.compute_dzdt(z, x)
        k2 = self.compute_dzdt(z + 0.5 * dt * k1, x)
        k3 = self.compute_dzdt(z + 0.5 * dt * k2, x)
        k4 = self.compute_dzdt(z + dt * k3, x)

        return z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def forward_rk4(
        self, x: torch.Tensor, return_states: bool = False, dt: float = None
    ):
        """Forward with RK4 integration."""
        if dt is None:
            dt = self.dt

        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []

        for t in range(T):
            x_t = x[t]
            z = self.rk4_step(z, x_t, dt)

            if self.mobius is not None:
                z = self.mobius.project_state(z)

            z = torch.complex(
                torch.clamp(z.real, -self.z_max, self.z_max),
                torch.clamp(z.imag, -self.z_max, self.z_max),
            )

            y_t = self.out(z.real)
            outputs.append(y_t)
            if return_states:
                states.append(z)

        y = torch.stack(outputs, dim=0)

        if return_states:
            states = torch.stack(states, dim=0)
            return y, states
        return y

    def step_rk4(self, z: torch.Tensor, x: torch.Tensor, dt: float = None):
        """Single step with RK4."""
        if dt is None:
            dt = self.dt

        z_new = self.rk4_step(z, x, dt)

        if self.mobius is not None:
            z_new = self.mobius.project_state(z_new)

        z_new = torch.complex(
            torch.clamp(z_new.real, -self.z_max, self.z_max),
            torch.clamp(z_new.imag, -self.z_max, self.z_max),
        )

        output = self.out(z_new.real)
        return z_new, output

    def get_tau_statistics(self, z: torch.Tensor) -> Dict[str, float]:
        """Get tau statistics."""
        tau = self.compute_tau(z)
        return {
            "tau_mean": tau.mean().item(),
            "tau_std": tau.std().item(),
            "tau_min": tau.min().item(),
            "tau_max": tau.max().item(),
        }

    def reset_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """Reset hidden state."""
        return torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.complex64, device=device
        )

    # === Mobius Manifold Constraint & Resonance Attention ===
    def enable_mobius_resonance(
        self,
        enable_mobius: bool = True,
        enable_resonance: bool = True,
        mobius_strength: float = 0.1,
        resonance_strength: float = 0.1,
        sparse_resonance: bool = True,
        learn_manifold_dim: bool = True,
        resonance_mode: str = "additive",
        **mobius_kwargs,
    ):
        """
        启用莫比乌斯约束和共振注意力 (插件式集成)

        Args:
            enable_mobius: 启用莫比乌斯流形约束
            enable_resonance: 启用扭量共振注意力
            mobius_strength: 莫比乌斯约束强度
            resonance_strength: 共振注意力强度
            sparse_resonance: 使用稀疏共振 (由莫比乌斯拓扑控制)
            learn_manifold_dim: 可学习流形维度决策
            resonance_mode: 共振应用模式 ('additive', 'multiplicative', 'gating')
            **mobius_kwargs: 传递给 MobiusConstraint 的额外参数
        """
        if enable_mobius:
            from .mobius import MobiusConstraint

            self.mobius = MobiusConstraint(
                max_dim=max(self.hidden_dim * 4, 512),
                constraint_strength=mobius_strength,
                enable_learning=learn_manifold_dim,
                device=str(self.W_real.weight.device),
                **mobius_kwargs,
            )

        if enable_resonance:
            from .resonance import TwistorResonance

            self.resonance = TwistorResonance(
                hidden_dim=self.hidden_dim,
                resonance_strength=resonance_strength,
                sparse_mode=sparse_resonance,
                device=str(self.W_real.weight.device),
            )

        self._resonance_mode = resonance_mode

    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute time derivative with stability normalization."""
        z_real = z.real
        z_imag = z.imag

        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        # Sparse connectivity
        if self.sparsity > 0:
            W_real_sparse = self.W_real.weight * torch.sigmoid(self.sparse_mask_real)
            W_imag_sparse = self.W_imag.weight * torch.sigmoid(self.sparse_mask_imag)
            W_tanh_real = F.linear(tanh_real, W_real_sparse, self.W_real.bias)
            W_tanh_imag = F.linear(tanh_imag, W_imag_sparse, self.W_imag.bias)
        else:
            W_tanh_real = self.W_real(tanh_real)
            W_tanh_imag = self.W_imag(tanh_imag)

        Ux = self.U(x)

        dz_real = -z_real + W_tanh_real + Ux + self.b_real
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag

        tau = self.compute_tau(z)
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        # === 扭量共振注意力 (附加项，不改变原有动力学) ===
        if self.resonance is not None:
            topo_weights = None
            if self.mobius is not None:
                topo_weights = self.mobius.topology_weight_matrix(self.hidden_dim)

            dzdt_resonance = self.resonance(
                z, topology_weights=topo_weights, mode=self._resonance_mode
            )
            dzdt = dzdt + dzdt_resonance

        # Clamp dz/dt
        dzdt_real = torch.clamp(dzdt.real, -self.dzdt_max, self.dzdt_max)
        dzdt_imag = torch.clamp(dzdt.imag, -self.dzdt_max, self.dzdt_max)
        dzdt_clipped = torch.complex(dzdt_real, dzdt_imag)

        # Scale if too large
        dzdt_norm = torch.abs(dzdt_clipped)
        mean_norm = dzdt_norm.mean()
        if mean_norm > self.dzdt_max / 2:
            scale = (self.dzdt_max / 2) / (mean_norm + 1e-6)
            dzdt_clipped = dzdt_clipped * scale

        return dzdt_clipped

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Forward pass with Euler integration."""
        T, B, _ = x.shape

        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []
        diagnostics = {
            "z_norm": [],
            "dzdt_norm": [],
            "tau_mean": [],
            "tau_std": [],
            "has_nan": False,
            "has_inf": False,
        }

        for t in range(T):
            x_t = x[t]
            dzdt = self.compute_dzdt(z, x_t)

            stability_check = self.check_numerical_stability(z, dzdt)
            if stability_check["z_nan"] or stability_check["z_inf"]:
                diagnostics["has_nan"] = stability_check["z_nan"]
                diagnostics["has_inf"] = stability_check["z_inf"]

            if return_diagnostics:
                diagnostics["z_norm"].append(torch.abs(z).mean().item())
                diagnostics["dzdt_norm"].append(torch.abs(dzdt).mean().item())

            z = z + self.dt * dzdt

            # === 莫比乌斯状态投影约束 (Euler 步进后施加) ===
            if self.mobius is not None:
                z = self.mobius.project_state(z)

            z = torch.complex(
                torch.clamp(z.real, -self.z_max, self.z_max),
                torch.clamp(z.imag, -self.z_max, self.z_max),
            )

            if return_diagnostics:
                tau_t = self.compute_tau(z)
                diagnostics["tau_mean"].append(tau_t.mean().item())
                diagnostics["tau_std"].append(tau_t.std().item())

            y_t = self.out(z.real)
            outputs.append(y_t)
            if return_states:
                states.append(z)

        y = torch.stack(outputs, dim=0)
        result = [y]

        if return_states:
            states = torch.stack(states, dim=0)
            result.append(states)

        if return_diagnostics:
            diagnostics["z_norm"] = np.array(diagnostics["z_norm"])
            diagnostics["dzdt_norm"] = np.array(diagnostics["dzdt_norm"])
            diagnostics["tau_mean"] = np.array(diagnostics["tau_mean"])
            diagnostics["tau_std"] = np.array(diagnostics["tau_std"])
            result.append(diagnostics)

        return tuple(result) if len(result) > 1 else result[0]

    def get_mobius_info(self) -> Optional[Dict]:
        """获取莫比乌斯流形当前状态"""
        if self.mobius is None:
            return None
        return self.mobius.get_manifold_info(self.hidden_dim)

    def step(self, z: torch.Tensor, x: torch.Tensor, dt: float = None):
        """Single step for agent."""
        if dt is None:
            dt = self.dt

        dzdt = self.compute_dzdt(z, x)
        z_new = z + dt * dzdt

        # 莫比乌斯约束
        if self.mobius is not None:
            z_new = self.mobius.project_state(z_new)

        z_new = torch.complex(
            torch.clamp(z_new.real, -self.z_max, self.z_max),
            torch.clamp(z_new.imag, -self.z_max, self.z_max),
        )

        output = self.out(z_new.real)
        return z_new, output
