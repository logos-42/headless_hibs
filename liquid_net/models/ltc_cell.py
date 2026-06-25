"""
LTC Cell - Core dynamics for Twistor-inspired Liquid Neural Network.
Stability-optimized version.

Implements the continuous-time dynamics:
    dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)

Stability features:
- Clamped tau (tau_min, tau_max)
- Normalized dz/dt
- NaN/Inf detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class LTCCell(nn.Module):
    """
    Liquid Time-Constant Cell with complex-valued states.
    Stability-optimized version.

    Mathematical formulation:
        dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)

    where:
        - z is complex hidden state
        - W is recurrent weight matrix (separate for real/imag)
        - U is input weight matrix
        - b is bias term
        - tau(z) = clamp(sigmoid(W_tau * |z|), tau_min, tau_max)
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 16,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        dzdt_max: float = 10.0,
    ):
        """
        Initialize LTC Cell.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state (default: 16)
            tau_min: Minimum time constant (default: 0.01)
            tau_max: Maximum time constant (default: 1.0)
            dzdt_max: Maximum |dz/dt| for normalization (default: 10.0)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dzdt_max = dzdt_max

        # Weight matrices - SEPARATE for real and imag parts
        self.W_real = nn.Linear(hidden_dim, hidden_dim)
        self.W_imag = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        
        # Bias terms
        self.b_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(hidden_dim))

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization for stability."""
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

    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent time constant with clamping.

        tau(z) = clamp(sigmoid(W_tau(|z|)), tau_min, tau_max) + epsilon

        Args:
            z: Complex state (B, hidden_dim), dtype=complex

        Returns:
            tau: Clamped time constant (B, hidden_dim), in [tau_min, tau_max]
        """
        z_mod = torch.abs(z)
        tau = F.sigmoid(self.W_tau(z_mod))
        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau + 1e-6

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative dz/dt with stability normalization.

        Args:
            z: Complex hidden state (B, hidden_dim), dtype=complex
            x: Input (B, input_dim)

        Returns:
            dzdt: Normalized time derivative (B, hidden_dim), dtype=complex
        """
        z_real = z.real
        z_imag = z.imag

        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        W_tanh_real = self.W_real(tanh_real)
        W_tanh_imag = self.W_imag(tanh_imag)
        Ux = self.U(x)

        dz_real = -z_real + W_tanh_real + Ux + self.b_real
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag

        tau = self.compute_tau(z)

        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        # Normalize dz/dt to prevent explosion
        # Clip real and imag parts separately (clamp doesn't support complex)
        dzdt_real = torch.clamp(dzdt.real, -self.dzdt_max, self.dzdt_max)
        dzdt_imag = torch.clamp(dzdt.imag, -self.dzdt_max, self.dzdt_max)
        dzdt_clipped = torch.complex(dzdt_real, dzdt_imag)
        
        # Additional scaling if mean norm is too high
        dzdt_norm = torch.abs(dzdt_clipped)
        mean_norm = dzdt_norm.mean()
        if mean_norm > self.dzdt_max / 2:
            scale = (self.dzdt_max / 2) / (mean_norm + 1e-6)
            dzdt_clipped = dzdt_clipped * scale

        return dzdt_clipped

    def check_stability(self, z: torch.Tensor, dzdt: torch.Tensor) -> Dict[str, bool]:
        """Check for numerical instability (NaN/Inf)."""
        return {
            'z_nan': torch.isnan(z).any().item(),
            'z_inf': torch.isinf(z).any().item(),
            'dzdt_nan': torch.isnan(dzdt).any().item(),
            'dzdt_inf': torch.isinf(dzdt).any().item(),
        }
