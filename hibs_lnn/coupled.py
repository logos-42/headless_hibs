"""
Twistor-LMT Coupled Module
==========================
Multi-space coupled Twistor-LMT.

Two state spaces:
- h: Behavior space (real, standard LMT)
- z: Structure space (complex, twistor-inspired)

Coupled dynamics:
- dh/dt = f(h, z, x)
- dz/dt = g(z, h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .integrators import rk4_step, euler_step


class CoupledTwistorLMT(nn.Module):
    """
    Multi-space coupled Twistor-LMT.

    Two state spaces with bidirectional coupling:
    - h: Behavior space (real LMT) - handles input/output
    - z: Structure space (complex Twistor) - provides geometric structure
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        output_dim: int = 1,
        coupling_strength: float = 0.1,
        use_rk4: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.coupling_strength = coupling_strength
        self.use_rk4 = use_rk4
        self.dt = 0.1

        # Behavior space (real LMT)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.U_h = nn.Linear(input_dim, hidden_dim)
        self.W_tau_h = nn.Linear(hidden_dim, hidden_dim)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        # Structure space (complex Twistor)
        self.W_z_real = nn.Linear(hidden_dim, hidden_dim)
        self.W_z_imag = nn.Linear(hidden_dim, hidden_dim)
        self.U_z = nn.Linear(input_dim, hidden_dim)
        self.W_tau_z = nn.Linear(hidden_dim, hidden_dim)
        self.b_z_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_z_imag = nn.Parameter(torch.zeros(hidden_dim))

        # Coupling: h ↔ z
        self.h_to_z_coupling = nn.Linear(hidden_dim, hidden_dim)
        self.z_to_h_coupling = nn.Linear(hidden_dim, hidden_dim)

        # Output decoder (uses both h and z)
        self.out_h = nn.Linear(hidden_dim, output_dim)
        self.out_z = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.orthogonal_(self.W_h.weight, gain=0.5)
        nn.init.orthogonal_(self.W_z_real.weight, gain=0.5)
        nn.init.orthogonal_(self.W_z_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U_h.weight, gain=0.5)
        nn.init.orthogonal_(self.U_z.weight, gain=0.5)
        nn.init.orthogonal_(self.W_tau_h.weight, gain=0.1)
        nn.init.orthogonal_(self.W_tau_z.weight, gain=0.1)
        nn.init.zeros_(self.W_h.bias)
        nn.init.zeros_(self.W_z_real.bias)
        nn.init.zeros_(self.W_z_imag.bias)
        nn.init.zeros_(self.U_h.bias)
        nn.init.zeros_(self.U_z.bias)
        nn.init.zeros_(self.W_tau_h.bias)
        nn.init.zeros_(self.W_tau_z.bias)

    def compute_dhdt(
        self, h: torch.Tensor, z: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute behavior space derivative.

        dh/dt = (-h + W_h·tanh(h) + U_h·x + b_h + coupling_from_z) / τ(h)
        """
        tau_h = torch.sigmoid(self.W_tau_h(h)).clamp(0.01, 1.0) + 1e-6
        coupling_from_z = self.z_to_h_coupling(z.real) * self.coupling_strength

        dh = (
            -h + torch.tanh(self.W_h(h)) + self.U_h(x) + self.b_h + coupling_from_z
        ) / tau_h
        return dh

    def compute_dzdt(
        self, z: torch.Tensor, h: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute structure space derivative.

        dz/dt = (-z + W_z·tanh(z) + U_z·x + b_z + coupling_from_h) / τ(z)
        """
        z_real = z.real
        z_imag = z.imag

        tau_z = torch.sigmoid(self.W_tau_z(torch.abs(z))).clamp(0.01, 1.0) + 1e-6

        coupling_from_h = self.h_to_z_coupling(h) * self.coupling_strength

        dz_real = (
            -z_real
            + torch.tanh(self.W_z_real(z_real))
            + self.U_z(x)
            + self.b_z_real
            + coupling_from_h
        )
        dz_imag = (
            -z_imag
            + torch.tanh(self.W_z_imag(z_imag))
            + self.U_z(x)
            + self.b_z_imag
            + coupling_from_h
        )

        dzdt = torch.complex(dz_real / tau_z, dz_imag / tau_z)
        return dzdt

    def step(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        dt: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step for agent.

        Args:
            h: Current behavior state (B, hidden_dim)
            z: Current structure state (B, hidden_dim), complex
            x: Input/observation (B, input_dim)
            dt: Time step

        Returns:
            h_new: Next behavior state
            z_new: Next structure state
            output: Action/prediction
        """
        if dt is None:
            dt = self.dt

        if self.use_rk4:
            h_new = rk4_step(self.compute_dhdt, h, z, dt, z, x)
            z_new = rk4_step(self.compute_dzdt, z, h, dt, h, x)
        else:
            dhdt = self.compute_dhdt(h, z, x)
            dzdt = self.compute_dzdt(z, h, x)
            h_new = h + dt * dhdt
            z_new = z + dt * dzdt

        output = self.out_h(h_new) + self.out_z(z_new.real)

        return h_new, z_new, output

    def forward(
        self, x: torch.Tensor, return_states: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with coupled dynamics.

        Args:
            x: Input sequence (T, B, input_dim)
            return_states: If True, return both h and z states

        Returns:
            y: Output sequence (T, B, output_dim)
            h_states: Behavior states if return_states=True
            z_states: Structure states if return_states=True
        """
        T, B, _ = x.shape

        h = torch.zeros(B, self.hidden_dim, device=x.device)
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        h_states = []
        z_states = []

        for t in range(T):
            x_t = x[t]

            if self.use_rk4:
                h = rk4_step(self.compute_dhdt, h, z, self.dt, z, x_t)
                z = rk4_step(self.compute_dzdt, z, h, self.dt, h, x_t)
            else:
                dhdt = self.compute_dhdt(h, z, x_t)
                dzdt = self.compute_dzdt(z, h, x_t)
                h = h + self.dt * dhdt
                z = z + self.dt * dzdt

            y_t = self.out_h(h) + self.out_z(z.real)
            outputs.append(y_t)

            if return_states:
                h_states.append(h)
                z_states.append(z)

        y = torch.stack(outputs, dim=0)

        if return_states:
            h_states = torch.stack(h_states, dim=0)
            z_states = torch.stack(z_states, dim=0)
            return y, h_states, z_states

        return y

    def reset_state(
        self, batch_size: int = 1, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset both states.

        Args:
            batch_size: Number of parallel environments
            device: Device

        Returns:
            h: Zero behavior state
            z: Zero structure state
        """
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.complex64, device=device
        )
        return h, z


class StackedCoupledLMT(nn.Module):
    """
    Stacked coupled LMT layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        coupling_strength: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                CoupledTwistorLMT(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    hidden_dim if i < num_layers - 1 else output_dim,
                    coupling_strength,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, return_states: bool = False):
        """Forward through all coupled layers."""
        for layer in self.layers:
            x = layer(x, return_states=False)
        return x


def create_coupled_LMT(
    input_dim: int, hidden_dim: int, output_dim: int, num_spaces: int = 2, **kwargs
) -> CoupledTwistorLMT:
    """
    Factory to create coupled LMT.

    Args:
        input_dim: Input dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        num_spaces: Number of coupled spaces
        **kwargs: Additional arguments

    Returns:
        model: CoupledTwistorLMT instance
    """
    return CoupledTwistorLMT(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, **kwargs
    )
