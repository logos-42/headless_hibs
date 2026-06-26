"""
Twistor-inspired Liquid Neural Network (Complex-valued LMT) - Stability Optimized
==================================================================================
Implements continuous-time dynamics: dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)

Stability Features:
- Complex-valued hidden state z (torch.complex)
- State-dependent time constant tau(z) with clamping
- dz/dt normalization to prevent explosion
- Gradient clipping during training
- L2 regularization on z
- Tunable dt parameter
- NaN/Inf detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional

from twistor_LMT.datasets import generate_sine_dataset


class TwistorLMT(nn.Module):
    """
    Twistor-inspired Liquid Neural Network with complex-valued states.
    Stability-optimized version.

    The dynamics follow: dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)
    where:
        - z ∈ ℂⁿ is complex hidden state
        - W is recurrent weight matrix (separate for real/imag)
        - U is input weight matrix
        - b is bias term (separate for real/imag)
        - tau(z) = clamp(sigmoid(W_tau * |z|), tau_min, tau_max) is state-dependent time constant
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
        self.dt = dt  # Time step (tunable)
        self.tau_min = tau_min  # Minimum time constant
        self.tau_max = tau_max  # Maximum time constant
        self.dzdt_max = dzdt_max  # Maximum |dz/dt|
        self.z_max = z_max  # Maximum |z|

        # Weight matrices - SEPARATE for real and imag parts
        self.W_real = nn.Linear(hidden_dim, hidden_dim)
        self.W_imag = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)

        # Sparse connectivity masks
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

        # Initialize sparse masks
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

        # Initialize multi-scale tau bias
        if self.multi_scale_tau and self.tau_bias is not None:
            nn.init.zeros_(self.tau_bias)

    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent time constant with clamping.

        tau_i(z) = clamp(sigmoid(W_tau(|z|)_i + tau_bias_i), tau_min, tau_max)

        Args:
            z: Complex state (B, hidden_dim), dtype=complex

        Returns:
            tau: Clamped time constant (B, hidden_dim), in [tau_min, tau_max]
        """
        z_mod = torch.abs(z)  # (B, hidden_dim)
        tau = F.sigmoid(self.W_tau(z_mod))

        # Add per-neuron bias for multi-scale time constants
        if self.multi_scale_tau and self.tau_bias is not None:
            tau = tau + self.tau_bias.unsqueeze(0)

        # Clamp tau to [tau_min, tau_max] for stability
        tau = torch.clamp(tau, self.tau_min, self.tau_max)

        return tau + 1e-6  # epsilon for numerical stability

    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative dz/dt with stability normalization.

        Dynamics: dz/dt = normalize((-z + W*tanh(z) + Ux + b) / tau(z))

        Args:
            z: Complex hidden state (B, hidden_dim), dtype=complex
            x: Input (B, input_dim)

        Returns:
            dzdt: Normalized time derivative (B, hidden_dim), dtype=complex
        """
        z_real = z.real
        z_imag = z.imag

        # Apply tanh to real and imag parts separately
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        # Compute numerator with sparse masks
        if self.sparsity > 0:
            W_real_sparse = self.W_real.weight * torch.sigmoid(self.sparse_mask_real)
            W_imag_sparse = self.W_imag.weight * torch.sigmoid(self.sparse_mask_imag)
            W_tanh_real = F.linear(tanh_real, W_real_sparse, self.W_real.bias)
            W_tanh_imag = F.linear(tanh_imag, W_imag_sparse, self.W_imag.bias)
        else:
            W_tanh_real = self.W_real(tanh_real)
            W_tanh_imag = self.W_imag(tanh_imag)

        # Input affects both real and imag parts
        Ux = self.U(x)

        # Compute derivatives
        dz_real = -z_real + W_tanh_real + Ux + self.b_real
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag

        # Compute clamped time constant
        tau = self.compute_tau(z)

        # Divide by tau
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)

        # Normalize dz/dt to prevent explosion
        # Clip real and imag parts separately (clamp doesn't support complex)
        dzdt_real = torch.clamp(dzdt.real, -self.dzdt_max, self.dzdt_max)
        dzdt_imag = torch.clamp(dzdt.imag, -self.dzdt_max, self.dzdt_max)
        dzdt_clipped = torch.complex(dzdt_real, dzdt_imag)

        # Additional normalization: if mean |dz/dt| > threshold, scale down
        dzdt_norm = torch.abs(dzdt_clipped)
        mean_norm = dzdt_norm.mean()
        if mean_norm > self.dzdt_max / 2:
            scale = (self.dzdt_max / 2) / (mean_norm + 1e-6)
            dzdt_clipped = dzdt_clipped * scale

        return dzdt_clipped

    def check_numerical_stability(
        self, z: torch.Tensor, dzdt: torch.Tensor
    ) -> Dict[str, bool]:
        """
        Check for numerical instability (NaN/Inf).

        Args:
            z: Current state
            dzdt: Time derivative

        Returns:
            Dictionary with stability flags
        """
        return {
            "z_nan": torch.isnan(z).any().item(),
            "z_inf": torch.isinf(z).any().item(),
            "dzdt_nan": torch.isnan(dzdt).any().item(),
            "dzdt_inf": torch.isinf(dzdt).any().item(),
        }

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with Euler integration and stability monitoring.

        Args:
            x: Input sequence (T, B, input_dim)
            return_states: If True, return all hidden states
            return_diagnostics: If True, return stability diagnostics

        Returns:
            y: Output sequence (T, B, output_dim)
            states: All hidden states (T, B, hidden_dim) if return_states=True
            diagnostics: Stability info if return_diagnostics=True
        """
        T, B, _ = x.shape

        # Initialize complex hidden state to zero
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []
        dzdts = []
        taus = []
        diagnostics = {
            "z_norm": [],
            "dzdt_norm": [],
            "tau_mean": [],
            "tau_std": [],
            "has_nan": False,
            "has_inf": False,
        }

        # Time loop: Euler integration
        for t in range(T):
            x_t = x[t]  # (B, input_dim)

            # Compute time derivative
            dzdt = self.compute_dzdt(z, x_t)

            # Check for numerical issues
            stability_check = self.check_numerical_stability(z, dzdt)
            if stability_check["z_nan"] or stability_check["z_inf"]:
                diagnostics["has_nan"] = stability_check["z_nan"]
                diagnostics["has_inf"] = stability_check["z_inf"]
                print(f"  Warning: Numerical instability at t={t}: {stability_check}")

            # Record diagnostics
            if return_diagnostics:
                diagnostics["z_norm"].append(torch.abs(z).mean().item())
                diagnostics["dzdt_norm"].append(torch.abs(dzdt).mean().item())

            # Euler step: z(t+dt) = z(t) + dt * dz/dt
            z = z + self.dt * dzdt

            # Clamp z to prevent explosion (separate for real/imag)
            z_real_clamped = torch.clamp(z.real, -self.z_max, self.z_max)
            z_imag_clamped = torch.clamp(z.imag, -self.z_max, self.z_max)
            z = torch.complex(z_real_clamped, z_imag_clamped)

            # Compute tau for diagnostics
            if return_diagnostics:
                tau_t = self.compute_tau(z)
                diagnostics["tau_mean"].append(tau_t.mean().item())
                diagnostics["tau_std"].append(tau_t.std().item())
                dzdts.append(dzdt)
                taus.append(tau_t)

            # Output from real part only
            y_t = self.out(z.real)

            outputs.append(y_t)
            if return_states:
                states.append(z)

        # Stack outputs
        y = torch.stack(outputs, dim=0)  # (T, B, output_dim)

        result = [y]

        if return_states:
            states = torch.stack(states, dim=0)  # (T, B, hidden_dim)
            result.append(states)

        if return_diagnostics:
            diagnostics["z_norm"] = np.array(diagnostics["z_norm"])
            diagnostics["dzdt_norm"] = np.array(diagnostics["dzdt_norm"])
            diagnostics["tau_mean"] = np.array(diagnostics["tau_mean"])
            diagnostics["tau_std"] = np.array(diagnostics["tau_std"])
            if dzdts:
                diagnostics["dzdts"] = torch.stack(dzdts, dim=0)
            if taus:
                diagnostics["taus"] = torch.stack(taus, dim=0)
            result.append(diagnostics)

        return tuple(result) if len(result) > 1 else result[0]

    def get_tau_statistics(self, z: torch.Tensor) -> Dict[str, float]:
        """
        Get statistics of time constant tau for a given state.

        Args:
            z: Complex state

        Returns:
            Dictionary with tau statistics
        """
        tau = self.compute_tau(z)
        return {
            "tau_mean": tau.mean().item(),
            "tau_std": tau.std().item(),
            "tau_min": tau.min().item(),
            "tau_max": tau.max().item(),
        }

    # ============================================================
    # Tensor Decoder: z → v ⊗ v (外积生成二阶张量)
    # ============================================================
    def decode_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode complex state to second-order tensor via outer product.

        z → v (real part) → v ⊗ v

        Args:
            z: Complex state (B, hidden_dim)

        Returns:
            tensor: Second-order tensor (B, hidden_dim, hidden_dim)
        """
        v = z.real  # (B, hidden_dim)
        tensor = torch.einsum("bi,bj->bij", v, v)  # (B, hidden_dim, hidden_dim)
        return tensor

    def decode_tensor_flat(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode to flattened tensor for output projection.

        Args:
            z: Complex state (B, hidden_dim)

        Returns:
            tensor_flat: Flattened tensor (B, hidden_dim * hidden_dim)
        """
        tensor = self.decode_tensor(z)
        return tensor.view(tensor.size(0), -1)

    # ============================================================
    # RK4 Integrator: 更精确的数值积分
    # ============================================================
    def rk4_step(
        self, z: torch.Tensor, x: torch.Tensor, dt: float = None
    ) -> torch.Tensor:
        """
        Runge-Kutta 4th order integration.

        More accurate than Euler, better for complex dynamics.

        Args:
            z: Current complex state (B, hidden_dim)
            x: Input (B, input_dim)
            dt: Time step (defaults to self.dt)

        Returns:
            z_new: Next state (B, hidden_dim)
        """
        if dt is None:
            dt = self.dt

        k1 = self.compute_dzdt(z, x)
        k2 = self.compute_dzdt(z + 0.5 * dt * k1, x)
        k3 = self.compute_dzdt(z + 0.5 * dt * k2, x)
        k4 = self.compute_dzdt(z + dt * k3, x)

        z_new = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return z_new

    def forward_rk4(
        self, x: torch.Tensor, return_states: bool = False, dt: float = None
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with RK4 integration instead of Euler.

        Args:
            x: Input sequence (T, B, input_dim)
            return_states: If True, return all hidden states
            dt: Time step (defaults to self.dt)

        Returns:
            y: Output sequence (T, B, output_dim)
            states: All hidden states if return_states=True
        """
        if dt is None:
            dt = self.dt

        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)

        outputs = []
        states = []

        for t in range(T):
            x_t = x[t]
            z = self.rk4_step(z, x_t, dt)
            z = torch.clamp(z, -self.z_max, self.z_max)

            y_t = self.out(z.real)
            outputs.append(y_t)
            if return_states:
                states.append(z)

        y = torch.stack(outputs, dim=0)

        if return_states:
            states = torch.stack(states, dim=0)
            return y, states

        return y

    # ============================================================
    # Agent Interface: 单步演化用于强化学习
    # ============================================================
    def step(
        self, z: torch.Tensor, x: torch.Tensor, dt: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step evolution for agent/RL use.

        Args:
            z: Current complex state (B, hidden_dim), complex
            x: Input/observation (B, input_dim), real
            dt: Time step (defaults to self.dt)

        Returns:
            z_new: Next state (B, hidden_dim), complex
            output: Action/prediction (B, output_dim), real
        """
        if dt is None:
            dt = self.dt

        dzdt = self.compute_dzdt(z, x)
        z_new = z + dt * dzdt
        z_new = torch.clamp(z_new, -self.z_max, self.z_max)

        output = self.out(z_new.real)
        return z_new, output

    def step_rk4(
        self, z: torch.Tensor, x: torch.Tensor, dt: float = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single step with RK4 integration for agent.

        Args:
            z: Current complex state (B, hidden_dim), complex
            x: Input/observation (B, input_dim), real
            dt: Time step (defaults to self.dt)

        Returns:
            z_new: Next state (B, hidden_dim), complex
            output: Action/prediction (B, output_dim), real
        """
        if dt is None:
            dt = self.dt

        z_new = self.rk4_step(z, x, dt)
        z_new = torch.clamp(z_new, -self.z_max, self.z_max)

        output = self.out(z_new.real)
        return z_new, output

    def reset_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """
        Reset hidden state to zero.

        Args:
            batch_size: Number of parallel environments
            device: Device to create state on

        Returns:
            z: Zero state (batch_size, hidden_dim), complex
        """
        return torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.complex64, device=device
        )


# ============================================================
# Coupled Twistor-LMT: 多空间耦合 (h + z)
# ============================================================
class CoupledTwistorLMT(nn.Module):
    """
    Multi-space coupled Twistor-LMT.

    Two state spaces:
    - h: Behavior space (real, standard LMT)
    - z: Structure space (complex, twistor-inspired)

    Coupled dynamics:
    - dh/dt = f(h, z, x)
    - dz/dt = g(z, h)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        output_dim: int = 1,
        coupling_strength: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.coupling_strength = coupling_strength
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

        # Coupling: h → z and z → h
        self.h_to_z_coupling = nn.Linear(hidden_dim, hidden_dim)
        self.z_to_h_coupling = nn.Linear(hidden_dim, hidden_dim)

        # Output decoder (uses both h and z)
        self.out_h = nn.Linear(hidden_dim, output_dim)
        self.out_z = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
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
        """Compute behavior space derivative."""
        tau_h = torch.sigmoid(self.W_tau_h(h)) + 1e-6
        coupling_from_z = self.z_to_h_coupling(z.real) * self.coupling_strength
        dh = (
            -h + torch.tanh(self.W_h(h)) + self.U_h(x) + self.b_h + coupling_from_z
        ) / tau_h
        return dh

    def compute_dzdt(
        self, z: torch.Tensor, h: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Compute structure space derivative."""
        z_real = z.real
        z_imag = z.imag

        tau_z = torch.sigmoid(self.W_tau_z(torch.abs(z))) + 1e-6

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

    def step(
        self, h: torch.Tensor, z: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single step for agent.

        Args:
            h: Current behavior state (B, hidden_dim)
            z: Current structure state (B, hidden_dim), complex
            x: Input/observation (B, input_dim)

        Returns:
            h_new: Next behavior state
            z_new: Next structure state
            output: Action/prediction
        """
        dhdt = self.compute_dhdt(h, z, x)
        dzdt = self.compute_dzdt(z, h, x)

        h_new = h + self.dt * dhdt
        z_new = z + self.dt * dzdt

        output = self.out_h(h_new) + self.out_z(z_new.real)

        return h_new, z_new, output

    def reset_state(
        self, batch_size: int = 1, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset both states.

        Args:
            batch_size: Number of parallel environments
            device: Device to create states on

        Returns:
            h: Zero behavior state
            z: Zero structure state
        """
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(
            batch_size, self.hidden_dim, dtype=torch.complex64, device=device
        )
        return h, z


# ============================================================
# Twistor Agent: 智能体封装
# ============================================================
class TwistorAgent:
    """
    Agent wrapper for Twistor-LMT.

    Usage:
        agent = TwistorAgent(obs_dim=4, action_dim=2, hidden_dim=32)
        obs = env.reset()
        agent.reset()

        for step in range(max_steps):
            action = agent.act(obs)
            obs, reward, done, _ = env.step(action)
            if done:
                agent.reset()
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 32,
        use_rk4: bool = False,
        dt: float = 0.1,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.use_rk4 = use_rk4
        self.dt = dt

        self.model = TwistorLMT(
            input_dim=obs_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )
        self.z = None

    def reset(self, batch_size: int = 1, device: str = "cpu"):
        """Reset agent state."""
        self.z = self.model.reset_state(batch_size, device)
        return self.z

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from observation.

        Args:
            obs: Observation (obs_dim,) or (batch, obs_dim)
            deterministic: If True, return argmax (for discrete)

        Returns:
            action: Action (action_dim,) or (batch, action_dim)
        """
        if self.z is None:
            self.reset()

        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            batch_mode = False
        else:
            batch_mode = True

        with torch.no_grad():
            if self.use_rk4:
                self.z, action = self.model.step_rk4(self.z, obs, self.dt)
            else:
                self.z, action = self.model.step(self.z, obs, self.dt)

        if not batch_mode:
            action = action.squeeze(0)

        return action.cpu().numpy()

    def update(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Update state without getting action (for inference).

        Args:
            obs: Observation

        Returns:
            output: Model output
        """
        if self.z is None:
            self.reset(obs.size(0), str(obs.device))

        with torch.no_grad():
            self.z, output = self.model.step(self.z, obs, self.dt)

        return output


# ============================================================================
# Phase 1: GQA Attention + Twistor-LMT 融合 (LFM2 风格)
# ============================================================================

class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (Grouped Query Attention)
    
    类似 LFM2 的 GQA 实现：
    - n_heads 个查询头
    - n_kv_heads 个 KV 头 (更少，共享)
    - 每 n_heads/n_kv_heads 个查询头共享一个 KV 头
    
    优势：减少 KV 内存，提高效率
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        qk_layer_norm: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        # Q 投影：每个头独立
        self.W_q = nn.Linear(dim, dim)
        # KV 投影：更少的头，共享
        self.W_k = nn.Linear(dim, n_kv_heads * self.head_dim)
        self.W_v = nn.Linear(dim, n_kv_heads * self.head_dim)
        # 输出投影
        self.W_o = nn.Linear(dim, dim)
        
        # QK LayerNorm (类似 LFM2，增加稳定性)
        if qk_layer_norm:
            self.q_layer_norm = nn.LayerNorm(self.head_dim)
            self.k_layer_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_layer_norm = None
            self.k_layer_norm = None
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入 (B, T, D)
            mask: 注意力掩码 (B, n_heads, T, T)
        
        Returns:
            输出 (B, T, D)
        """
        B, T, D = x.shape
        
        # Q: (B, T, n_heads, head_dim) → 转置为 (B, n_heads, T, head_dim)
        q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # KV: (B, T, n_kv_heads, head_dim) -> 转置为 (B, n_kv_heads, T, head_dim)
        k = self.W_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # 先进行 repeat_interleave，再应用 LayerNorm
        # 分组广播：每 n_heads/n_kv_heads 个 Q 头共享一个 KV 头
        n_repeat = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(n_repeat, dim=1)  # (B, n_heads, T, head_dim)
        v = v.repeat_interleave(n_repeat, dim=1)  # (B, n_heads, T, head_dim)
        
        # 注意：QK LayerNorm 在 4D tensor 上应用较复杂，这里暂时跳过
        # 如需启用，需先 reshape 到 3D，应用 LayerNorm，再 reshape 回 4D
        # 保持 LayerNorm 参数但暂不使用，待后续优化
        
        # Attention 计算: (B, n_heads, T, T)
        scale = self.head_dim ** -0.5
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)  # (B, n_heads, T, head_dim)
        
        # 合并头并输出
        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.W_o(out)
        
        return out


class TwistorLMTwithGQA(nn.Module):
    """
    Twistor-LMT + GQA 融合版本 (Twistor-LMT-Edge)
    
    核心设计：
    - 大部分层：ODE 动力学（局部建模，类似 LFM2 的短卷积）
    - 关键节点：GQA Attention（全局建模，稀疏触发）
    - τ(z) 阈值决定是否触发 Attention
    
    公式：
        dz/dt = (-z + W·tanh(z) + U·x) / τ(z)  # 局部 ODE
        + [可选] GQA_Attention(z, z; τ)         # 全局交互（τ < 阈值时）
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
        # GQA 参数
        use_gqa: bool = True,
        n_heads: int = 4,
        n_kv_heads: int = 1,
        attention_interval: int = 5,
        tau_attention_threshold: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        self.multi_scale_tau = multi_scale_tau
        
        # 稳定性参数
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dzdt_max = dzdt_max
        self.z_max = z_max
        
        # GQA 参数
        self.use_gqa = use_gqa
        self.attention_interval = attention_interval
        self.tau_attention_threshold = tau_attention_threshold
        
        # Twistor-LMT 权重
        self.W_real = nn.Linear(hidden_dim, hidden_dim)
        self.W_imag = nn.Linear(hidden_dim, hidden_dim)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        
        self.sparse_mask_real = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        self.sparse_mask_imag = nn.Parameter(torch.ones(hidden_dim, hidden_dim))
        
        if multi_scale_tau:
            self.tau_bias = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.tau_bias = None
        
        self.b_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(hidden_dim))
        
        self.out = nn.Linear(hidden_dim, output_dim)
        
        # GQA 模块
        if use_gqa:
            self.gqa = GroupedQueryAttention(
                dim=hidden_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                qk_layer_norm=True,
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
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
        
        # 稀疏掩码
        if self.sparsity > 0:
            with torch.no_grad():
                mask_real = (torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity).float()
                mask_imag = (torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity).float()
                self.sparse_mask_real.copy_(mask_real)
                self.sparse_mask_imag.copy_(mask_imag)
        
        if self.multi_scale_tau and self.tau_bias is not None:
            nn.init.zeros_(self.tau_bias)
    
    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """计算时间常数"""
        z_mod = torch.abs(z)
        tau = F.sigmoid(self.W_tau(z_mod))
        
        if self.multi_scale_tau and self.tau_bias is not None:
            tau = tau + self.tau_bias.unsqueeze(0)
        
        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau + 1e-6
    
    def should_trigger_attention(self, tau: torch.Tensor) -> torch.Tensor:
        """
        基于 τ 决定是否触发 Attention
        当 τ 较小时（状态变化快），需要更多全局信息
        
        Returns:
            trigger: (B,) bool tensor
        """
        tau_mean = tau.mean(dim=-1)  # (B,)
        return tau_mean < self.tau_attention_threshold
    
    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """计算 ODE 动力学项"""
        z_real = z.real
        z_imag = z.imag
        
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)
        
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
        
        # 限幅
        dzdt_real = torch.clamp(dzdt.real, -self.dzdt_max, self.dzdt_max)
        dzdt_imag = torch.clamp(dzdt.imag, -self.dzdt_max, self.dzdt_max)
        dzdt = torch.complex(dzdt_real, dzdt_imag)
        
        return dzdt
    
    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播：ODE 动力学 + 稀疏 GQA Attention
        
        Args:
            x: 输入序列 (T, B, input_dim)
            
        Returns:
            y: 输出序列 (T, B, output_dim)
        """
        T, B, _ = x.shape
        
        # 初始化复数状态
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        outputs = []
        states = []
        attention_count = 0
        
        for t in range(T):
            x_t = x[t]
            
            # ===== 1. ODE 动力学（局部建模）=====
            dzdt = self.compute_dzdt(z, x_t)
            z = z + self.dt * dzdt
            z = torch.complex(
                torch.clamp(z.real, -self.z_max, self.z_max),
                torch.clamp(z.imag, -self.z_max, self.z_max)
            )
            
            # ===== 2. 可选：GQA Attention（全局建模）=====
            if self.use_gqa:
                tau = self.compute_tau(z)
                
                # 触发条件：τ < 阈值 或 每隔固定步数
                trigger_condition = (
                    (t % self.attention_interval == 0) or 
                    self.should_trigger_attention(tau)
                )
                
                # trigger_condition 可能是 (B,) 的 tensor，需要用 any() 判断
                should_attend = (t % self.attention_interval == 0) or self.should_trigger_attention(tau).any()
                
                if should_attend:
                    # 将复数状态转为实数序列用于 Attention
                    z_real = z.real  # (B, hidden_dim)
                    
                    # 简化：用当前状态做 self-attention
                    # 在实际应用中可以用过去所有状态
                    z_seq = z_real.unsqueeze(1)  # (B, 1, D)
                    attn_out = self.gqa(z_seq)  # (B, 1, D)
                    
                    # 将 Attention 结果加回状态
                    z = torch.complex(
                        z.real + attn_out.squeeze(1) * 0.1,  # 残差连接
                        z.imag
                    )
                    attention_count += 1
            
            # 输出
            y_t = self.out(z.real)
            outputs.append(y_t)
            
            if return_states:
                states.append(z)
        
        y = torch.stack(outputs, dim=0)
        
        if return_states:
            states = torch.stack(states, dim=0)
            return y, states
        
        return y
    
    def step(self, z: torch.Tensor, x: torch.Tensor, dt: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步演化（用于 Agent）"""
        if dt is None:
            dt = self.dt
        
        # ODE 动力学
        dzdt = self.compute_dzdt(z, x)
        z_new = z + dt * dzdt
        z_new = torch.complex(
            torch.clamp(z_new.real, -self.z_max, self.z_max),
            torch.clamp(z_new.imag, -self.z_max, self.z_max)
        )
        
        # 可选 GQA
        if self.use_gqa:
            tau = self.compute_tau(z_new)
            if self.should_trigger_attention(tau).any():
                z_real = z_new.real
                z_seq = z_real.unsqueeze(1)
                attn_out = self.gqa(z_seq)
                z_new = torch.complex(
                    z_new.real + attn_out.squeeze(1) * 0.1,
                    z_new.imag
                )
        
        output = self.out(z_new.real)
        return z_new, output
    
    def reset_state(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        """重置状态"""
        return torch.zeros(batch_size, self.hidden_dim, dtype=torch.complex64, device=device)


# 保留原有类名作为别名，方便迁移
TwistorLMTEdge = TwistorLMTwithGQA


if __name__ == "__main__":
    # 简单测试
    print("=" * 60)
    print("Twistor-LMT-Edge (GQA 融合版本) 测试")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 创建模型
    model = TwistorLMTwithGQA(
        input_dim=2,
        hidden_dim=32,
        output_dim=1,
        use_gqa=True,
        n_heads=4,
        n_kv_heads=1,
        attention_interval=3,
        tau_attention_threshold=0.3,
    ).to(device)
    
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 生成测试数据
    X = torch.randn(20, 4, 2).to(device)  # (T, B, input_dim)
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        y = model(X)
        print(f"输入形状: {X.shape}")
        print(f"输出形状: {y.shape}")
    
    # 测试单步
    z = model.reset_state(1, device)
    x = torch.randn(1, 2).to(device)
    z, output = model.step(z, x)
    print(f"单步输出: {output.shape}")
    
    print("\n测试完成！")



    """Plot training curves."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_mse"], label="Train MSE")
    plt.plot(history["val_mse"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Mean Squared Error")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved to 'training_curves.png'")
    plt.close()


def plot_predictions(
    model: TwistorLMT,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str,
    n_samples: int = 5,
):
    """Plot sample predictions."""
    model.eval()

    with torch.no_grad():
        x_test = X_test[:n_samples].transpose(0, 1)
        y_pred = model(x_test).transpose(0, 1)
        y_true = y_test[:n_samples]

    plt.figure(figsize=(14, 8))

    for i in range(n_samples):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(
            y_true[i].cpu().numpy().flatten(),
            "o-",
            label="True",
            alpha=0.7,
            markersize=4,
        )
        plt.plot(
            y_pred[i].cpu().numpy().flatten(),
            "s-",
            label="Predicted",
            alpha=0.7,
            markersize=4,
        )
        plt.ylabel("Amplitude")
        plt.title(f"Sample {i + 1}")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)

    plt.xlabel("Time Step")
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    print("Sample predictions saved to 'predictions.png'")
    plt.close()


def plot_z_trajectory(diagnostics: Dict, save_path: str = "z_trajectory.png"):
    """
    Plot z trajectory and tau distribution from diagnostics.

    Args:
        diagnostics: Dictionary from forward pass with return_diagnostics=True
        save_path: Path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: |z| over time
    ax1 = axes[0, 0]
    ax1.plot(diagnostics["z_norm"], "b-", linewidth=2)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("|z| (mean)")
    ax1.set_title("State Norm Over Time")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=diagnostics["z_norm"].mean(), color="r", linestyle="--", label="Mean")
    ax1.legend()

    # Plot 2: |dz/dt| over time
    ax2 = axes[0, 1]
    ax2.plot(diagnostics["dzdt_norm"], "g-", linewidth=2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("|dz/dt| (mean)")
    ax2.set_title("Time Derivative Norm Over Time")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(
        y=diagnostics["dzdt_norm"].mean(), color="r", linestyle="--", label="Mean"
    )
    ax2.legend()

    # Plot 3: tau mean over time
    ax3 = axes[1, 0]
    ax3.plot(diagnostics["tau_mean"], "m-", linewidth=2)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("τ (mean)")
    ax3.set_title("Time Constant Mean Over Time")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(
        y=diagnostics["tau_mean"].mean(), color="r", linestyle="--", label="Mean"
    )
    ax3.legend()

    # Plot 4: tau distribution (histogram)
    ax4 = axes[1, 1]
    if "taus" in diagnostics:
        all_taus = diagnostics["taus"].flatten().cpu().numpy()
        ax4.hist(all_taus, bins=50, edgecolor="black", alpha=0.7)
        ax4.set_xlabel("τ value")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Time Constant Distribution")
        ax4.grid(True, alpha=0.3)
        ax4.axvline(
            x=np.mean(all_taus),
            color="r",
            linestyle="--",
            label=f"Mean: {np.mean(all_taus):.4f}",
        )
        ax4.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Z trajectory saved to '{save_path}'")
    plt.close()


def print_tau_diagnostics(model: TwistorLMT, device: str = "cpu"):
    """
    Print tau distribution statistics for debugging.

    Args:
        model: Trained TwistorLMT model
        device: Device to run on
    """
    model.eval()

    # Generate random state
    z = torch.randn(1, model.hidden_dim, dtype=torch.complex64, device=device)

    stats = model.get_tau_statistics(z)

    print("\n" + "=" * 50)
    print("Tau Distribution Statistics:")
    print("=" * 50)
    print(f"  τ mean:  {stats['tau_mean']:.6f}")
    print(f"  τ std:   {stats['tau_std']:.6f}")
    print(f"  τ min:   {stats['tau_min']:.6f}")
    print(f"  τ max:   {stats['tau_max']:.6f}")
    print(f"  τ range: [{model.tau_min:.4f}, {model.tau_max:.4f}] (configured)")
    print("=" * 50)


def train_twistor_LMT(
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-2,
    hidden_dim: int = 16,
    stability_weight: float = 0.01,
    l2_weight: float = 0.001,
    sparsity: float = 0.3,
    multi_scale_tau: bool = True,
    dt: float = 0.1,
    tau_min: float = 0.01,
    tau_max: float = 1.0,
    dzdt_max: float = 10.0,
    z_max: float = 100.0,
    grad_clip: float = 1.0,
    device: str = "cpu",
    plot_diagnostics: bool = True,
):
    """
    Train the Twistor LMT on sine wave prediction with stability optimizations.

    Args:
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        hidden_dim: Hidden dimension
        stability_weight: Weight for ||dz/dt||^2 regularization
        l2_weight: Weight for L2 regularization on z
        sparsity: Sparsity level for recurrent weights
        multi_scale_tau: Use per-neuron tau bias
        dt: Time step for Euler integration
        tau_min: Minimum time constant
        tau_max: Maximum time constant
        dzdt_max: Maximum |dz/dt|
        z_max: Maximum |z|
        grad_clip: Gradient clipping threshold
        device: Device to train on
        plot_diagnostics: If True, plot z trajectory and tau distribution

    Returns:
        model: Trained model
        history: Training history
    """
    print("=" * 60)
    print("Twistor-inspired Liquid Neural Network Training")
    print("(Stability Optimized Version)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Time step dt: {dt}")
    print(f"Tau range: [{tau_min}, {tau_max}]")
    print(f"Max |dz/dt|: {dzdt_max}")
    print(f"Max |z|: {z_max}")
    print(f"Gradient clip: {grad_clip}")
    print(f"Stability weight: {stability_weight}")
    print(f"L2 weight: {l2_weight}")
    print()

    # Generate dataset
    print("Generating synthetic sine wave dataset...")
    X_train, y_train = generate_sine_dataset(n_samples=1000, seq_len=50, device=device)
    X_val, y_val = generate_sine_dataset(n_samples=200, seq_len=50, device=device)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Sequence length: {X_train.shape[1]}, Input dim: {X_train.shape[2]}")
    print()

    # Initialize model with stability parameters
    model = TwistorLMT(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_dim,
        output_dim=1,
        sparsity=sparsity,
        multi_scale_tau=multi_scale_tau,
        dt=dt,
        tau_min=tau_min,
        tau_max=tau_max,
        dzdt_max=dzdt_max,
        z_max=z_max,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Print initial tau statistics
    print_tau_diagnostics(model, device)
    print()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.5
    )

    # Training loop
    n_batches = len(X_train) // batch_size
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
        "stability_loss": [],
        "l2_loss": [],
    }

    print("Starting training...")
    print("-" * 60)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        epoch_stability = 0.0
        epoch_l2 = 0.0

        # Shuffle data
        perm = torch.randperm(len(X_train), device=device)
        X_train = X_train[perm]
        y_train = y_train[perm]

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size

            x_batch = X_train[start_idx:end_idx].transpose(0, 1)
            y_batch = y_train[start_idx:end_idx].transpose(0, 1)

            optimizer.zero_grad()

            # Forward pass with states and diagnostics
            y_pred, states, diagnostics = model(
                x_batch, return_states=True, return_diagnostics=True
            )

            # Check for numerical issues
            if diagnostics["has_nan"] or diagnostics["has_inf"]:
                print(f"  Warning: Numerical instability detected at epoch {epoch + 1}")

            # MSE loss
            mse_loss = F.mse_loss(y_pred, y_batch)

            # Stability regularization: ||dz/dt||^2
            dzdt_norm_sq = 0.0
            for t in range(len(states) - 1):
                dzdt = states[t + 1] - states[t]
                dzdt_norm_sq += (dzdt.abs() ** 2).mean()
            stability_loss = dzdt_norm_sq / (len(states) - 1)

            # L2 regularization on z
            l2_loss = (states.abs() ** 2).mean()

            # Total loss
            loss = mse_loss + stability_weight * stability_loss + l2_weight * l2_loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_stability += stability_loss.item()
            epoch_l2 += l2_loss.item()

        # Average losses
        avg_train_loss = epoch_loss / n_batches
        avg_train_mse = epoch_mse / n_batches
        avg_stability = epoch_stability / n_batches
        avg_l2 = epoch_l2 / n_batches

        history["train_loss"].append(avg_train_loss)
        history["train_mse"].append(avg_train_mse)
        history["stability_loss"].append(avg_stability)
        history["l2_loss"].append(avg_l2)

        # Validation
        model.eval()
        with torch.no_grad():
            x_val = X_val.transpose(0, 1)
            y_val_t = y_val.transpose(0, 1)
            y_val_pred = model(x_val)
            val_mse = F.mse_loss(y_val_pred, y_val_t).item()
            history["val_loss"].append(val_mse)
            history["val_mse"].append(val_mse)

        # Update learning rate
        scheduler.step(avg_train_loss)

        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:4d}/{n_epochs}: "
                f"Train Loss = {avg_train_loss:.6f}, "
                f"Train MSE = {avg_train_mse:.6f}, "
                f"Val MSE = {val_mse:.6f}, "
                f"Stab = {avg_stability:.6f}, "
                f"L2 = {avg_l2:.6f}, "
                f"LR = {optimizer.param_groups[0]['lr']:.6f}"
            )

    print("-" * 60)
    print(f"Training complete! Final Val MSE: {history['val_mse'][-1]:.6f}")
    print()

    # Print final tau statistics
    print_tau_diagnostics(model, device)

    # Plot results
    plot_training_results(history)
    plot_predictions(model, X_val, y_val, device)

    # Plot z trajectory and tau distribution
    if plot_diagnostics:
        print("\nGenerating diagnostics plots...")
        model.eval()
        with torch.no_grad():
            x_test = X_val[:1].transpose(0, 1)
            _, _, diagnostics = model(
                x_test, return_states=True, return_diagnostics=True
            )
        plot_z_trajectory(diagnostics)

    return model, history


if __name__ == "__main__":
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Train model with stability optimizations (reduced epochs for testing)
    model, history = train_twistor_LMT(
        n_epochs=50,  # Reduced for faster testing
        batch_size=32,
        lr=1e-2,
        hidden_dim=16,
        stability_weight=0.01,
        l2_weight=0.001,
        sparsity=0.3,
        multi_scale_tau=True,
        dt=0.1,
        tau_min=0.01,
        tau_max=1.0,
        dzdt_max=10.0,
        z_max=100.0,
        grad_clip=1.0,
        device=device,
        plot_diagnostics=True,
    )

    # Save model
    torch.save(model.state_dict(), "twistor_LMT_stable.pth")
    print("Model saved to 'twistor_LMT_stable.pth'")

    print()
    print("=" * 60)
    print("Training Summary:")
    print(f"  Initial Train Loss: {history['train_loss'][0]:.6f}")
    print(f"  Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"  Initial Val MSE: {history['val_mse'][0]:.6f}")
    print(f"  Final Val MSE: {history['val_mse'][-1]:.6f}")
    print(
        f"  Convergence: {'Yes' if history['train_loss'][-1] < history['train_loss'][0] * 0.5 else 'Partial'}"
    )
    print("=" * 60)


# ============================================================================
# v2.0: 多任务和零样本学习扩展
# ============================================================================

from dataclasses import dataclass
from typing import List, Optional as TypingOptional


@dataclass
class TaskConfig:
    """任务配置"""
    task_id: int
    task_name: str
    input_dim: int
    output_dim: int


class MultiTaskTwistorLMT(nn.Module):
    """
    支持多任务学习的 Twistor-LMT v2.0
    
    核心思想:
    1. 共享的动力学核心 (Shared Dynamics Core)
    2. 任务特定的嵌入 (Task-specific Embeddings)
    3. 任务特定的输入/输出投影 (Task-specific Projections)
    """
    
    def __init__(
        self,
        task_configs: List[TaskConfig],
        hidden_dim: int = 32,
        task_embedding_dim: int = 8,
        dt: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
    ):
        super().__init__()
        self.task_configs = task_configs
        self.hidden_dim = hidden_dim
        self.task_embedding_dim = task_embedding_dim
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_tasks = len(task_configs)
        
        # 1. 任务嵌入 (共享)
        self.task_embeddings = nn.ParameterDict({
            cfg.task_name: nn.Parameter(torch.randn(task_embedding_dim))
            for cfg in task_configs
        })
        
        # 2. 共享动力学核心
        self.dynamics_core = nn.ModuleDict({
            'W_z': nn.Linear(hidden_dim, hidden_dim),
            'W_x': nn.Linear(hidden_dim + task_embedding_dim, hidden_dim),
            'W_tau': nn.Linear(hidden_dim, hidden_dim),
        })
        self.tau_bias = nn.Parameter(torch.zeros(hidden_dim))
        
        # 3. 任务特定的输入投影
        self.input_projections = nn.ModuleDict({
            cfg.task_name: nn.Linear(cfg.input_dim, hidden_dim)
            for cfg in task_configs
        })
        
        # 4. 任务特定的输出投影
        self.output_projections = nn.ModuleDict({
            cfg.task_name: nn.Linear(hidden_dim, cfg.output_dim)
            for cfg in task_configs
        })
        
        # 5. 任务门控网络
        self.task_gates = nn.ModuleDict({
            cfg.task_name: nn.Sequential(
                nn.Linear(task_embedding_dim, hidden_dim),
                nn.Sigmoid()
            )
            for cfg in task_configs
        })
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.dynamics_core.items():
            if isinstance(param, nn.Linear):
                nn.init.orthogonal_(param.weight, gain=0.5)
                nn.init.zeros_(param.bias)
        
        for proj in self.input_projections.values():
            nn.init.orthogonal_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)
        
        for proj in self.output_projections.values():
            nn.init.orthogonal_(proj.weight, gain=0.5)
            nn.init.zeros_(proj.bias)
    
    def compute_dzdt(
        self, 
        z: torch.Tensor, 
        x: torch.Tensor, 
        task_emb: torch.Tensor,
        task_name: str
    ) -> torch.Tensor:
        """计算动力学方程 (任务条件化)"""
        z_real = z.real
        z_imag = z.imag
        
        # 任务门控
        gate = self.task_gates[task_name](task_emb)
        
        # 非线性项
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)
        
        W_tanh_real = self.dynamics_core['W_z'](tanh_real)
        W_tanh_imag = self.dynamics_core['W_z'](tanh_imag)
        
        # 输入项 (拼接任务嵌入)
        x_task = torch.cat([x, task_emb.unsqueeze(0).expand(x.size(0), -1)], dim=-1)
        Ux = self.dynamics_core['W_x'](x_task)
        
        # 应用门控
        W_tanh_real = gate * W_tanh_real
        W_tanh_imag = gate * W_tanh_imag
        Ux = gate * Ux
        
        # 动力学方程
        dz_real = -z_real + W_tanh_real + Ux
        dz_imag = -z_imag + W_tanh_imag + Ux
        
        # 时间常数
        z_mod = torch.abs(z)
        tau = torch.sigmoid(self.dynamics_core['W_tau'](z_mod))
        tau = tau + self.tau_bias.unsqueeze(0)
        tau = torch.clamp(tau, self.tau_min, self.tau_max) + 1e-6
        
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)
        dzdt = torch.clamp(dzdt.real, -10, 10) + 1j * torch.clamp(dzdt.imag, -10, 10)
        
        return dzdt
    
    def forward(
        self, 
        x: torch.Tensor, 
        task_name: str,
        return_states: bool = False
    ) -> torch.Tensor:
        """前向传播 (指定任务)"""
        T, B, _ = x.shape
        
        # 获取任务嵌入
        task_emb = self.task_embeddings[task_name]
        
        # 初始化状态
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        # 获取任务特定的投影
        input_proj = self.input_projections[task_name]
        output_proj = self.output_projections[task_name]
        
        outputs = []
        states = []
        
        for t in range(T):
            x_t = x[t]
            
            # 输入投影
            x_encoded = input_proj(x_t)
            
            # 动力学演化
            dzdt = self.compute_dzdt(z, x_encoded, task_emb, task_name)
            z = z + self.dt * dzdt
            
            # 状态限幅
            z = torch.complex(
                torch.clamp(z.real, -100, 100),
                torch.clamp(z.imag, -100, 100)
            )
            
            # 输出投影
            y_t = output_proj(z.real)
            outputs.append(y_t)
            
            if return_states:
                states.append(z)
        
        y = torch.stack(outputs, dim=0)
        
        if return_states:
            states = torch.stack(states, dim=0)
            return y, states
        
        return y
    
    def zero_shot_transfer(
        self, 
        x: torch.Tensor, 
        source_task: str, 
        target_task: str
    ) -> torch.Tensor:
        """零样本迁移：使用源任务的输入投影 + 目标任务的输出投影"""
        T, B, _ = x.shape
        task_emb = self.task_embeddings[target_task]
        
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        # 使用源任务的输入投影
        input_proj = self.input_projections[source_task]
        # 使用目标任务的输出投影
        output_proj = self.output_projections[target_task]
        
        outputs = []
        
        for t in range(T):
            x_t = x[t]
            x_encoded = input_proj(x_t)
            
            dzdt = self.compute_dzdt(z, x_encoded, task_emb, target_task)
            z = z + self.dt * dzdt
            
            y_t = output_proj(z.real)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=0)


class MetaTwistorLMT(nn.Module):
    """
    基于元学习的 Twistor-LMT v2.0，支持零样本适应新任务
    
    使用 MAML (Model-Agnostic Meta-Learning) 算法
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dt: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        
        # 共享参数 (元学习初始化)
        self.meta_params = nn.ParameterDict({
            'W_z': nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1),
            'W_x': nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1),
            'W_out': nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1),
            'W_tau': nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01),
            'b_z': nn.Parameter(torch.zeros(hidden_dim)),
            'b_x': nn.Parameter(torch.zeros(hidden_dim)),
            'b_out': nn.Parameter(torch.zeros(output_dim)),
        })
    
    def compute_dzdt_fast(
        self, 
        z: torch.Tensor, 
        x: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """快速动力学计算 (使用给定参数)"""
        z_real = z.real
        z_imag = z.imag
        
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)
        
        W_tanh_real = F.linear(tanh_real, params['W_z'], params['b_z'])
        W_tanh_imag = F.linear(tanh_imag, params['W_z'], params['b_z'])
        Ux = F.linear(x, params['W_x'], params['b_x'])
        
        dz_real = -z_real + W_tanh_real + Ux
        dz_imag = -z_imag + W_tanh_imag
        
        z_mod = torch.abs(z)
        tau = torch.sigmoid(F.linear(z_mod, params['W_tau'])) + 1e-6
        
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)
        dzdt = torch.clamp(dzdt.real, -10, 10) + 1j * torch.clamp(dzdt.imag, -10, 10)
        
        return dzdt
    
    def forward_step(
        self, 
        z: torch.Tensor, 
        x: torch.Tensor, 
        params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步前向传播"""
        dzdt = self.compute_dzdt_fast(z, x, params)
        z_new = z + self.dt * dzdt
        y = F.linear(z_new.real, params['W_out'], params['b_out'])
        return z_new, y
    
    def forward(
        self, 
        x: torch.Tensor, 
        params: TypingOptional[Dict] = None
    ) -> torch.Tensor:
        """前向传播"""
        if params is None:
            params = self.meta_params
        
        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        outputs = []
        for t in range(T):
            z, y_t = self.forward_step(z, x[t], params)
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=0)
    
    def meta_update(
        self, 
        x_support: torch.Tensor, 
        y_support: torch.Tensor,
        x_query: torch.Tensor,
        y_query: torch.Tensor,
        inner_lr: float = 0.1,
        inner_steps: int = 5
    ) -> Tuple[torch.Tensor, Dict]:
        """MAML 元更新"""
        # 1. 复制参数进行适应
        adapted_params = {k: v.clone() for k, v in self.meta_params.items()}
        
        # 2. 在支持集上进行梯度下降适应
        for _ in range(inner_steps):
            y_pred = self.forward(x_support, adapted_params)
            support_loss = F.mse_loss(y_pred, y_support)
            
            # 计算梯度并更新
            grads = torch.autograd.grad(
                support_loss, 
                adapted_params.values(),
                create_graph=True
            )
            
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - inner_lr * grad
        
        # 3. 在查询集上评估
        y_query_pred = self.forward(x_query, adapted_params)
        query_loss = F.mse_loss(y_query_pred, y_query)
        
        return query_loss, adapted_params
    
    def zero_shot_adapt(
        self, 
        x_few_shot: torch.Tensor, 
        y_few_shot: torch.Tensor,
        x_test: torch.Tensor,
        adapt_steps: int = 10,
        adapt_lr: float = 0.1
    ) -> torch.Tensor:
        """零样本适应：使用少量样本快速适应新任务"""
        # 复制参数 (需要梯度)
        adapted_params = {}
        for name, param in self.meta_params.items():
            adapted_params[name] = param.clone().detach().requires_grad_(True)
        
        # 快速适应
        for _ in range(adapt_steps):
            y_pred = self.forward(x_few_shot, adapted_params)
            loss = F.mse_loss(y_pred, y_few_shot)
            
            grads = torch.autograd.grad(
                loss, 
                adapted_params.values(),
                retain_graph=True,
                create_graph=False
            )
            
            for (name, param), grad in zip(adapted_params.items(), grads):
                adapted_params[name] = param - adapt_lr * grad
        
        # 使用适应后的参数进行预测
        y_test_pred = self.forward(x_test, adapted_params)
        
        return y_test_pred


class PromptTwistorLMT(nn.Module):
    """
    基于提示学习的 Twistor-LMT v2.0
    
    核心思想：
    1. 学习一组"提示"向量 (Prompt Vectors)
    2. 不同任务使用不同的提示组合
    3. 新任务通过提示组合实现零样本迁移
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        output_dim: int = 1,
        n_prompts: int = 10,
        prompt_dim: int = 8,
        dt: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.n_prompts = n_prompts
        
        # 提示库
        self.prompt_bank = nn.Parameter(torch.randn(n_prompts, prompt_dim))
        
        # 提示选择网络
        self.prompt_selector = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_prompts),
            nn.Softmax(dim=-1)
        )
        
        # 提示投影
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)
        
        # 核心动力学
        self.core = nn.ModuleDict({
            'W_z': nn.Linear(hidden_dim, hidden_dim),
            'W_x': nn.Linear(input_dim, hidden_dim),
            'W_tau': nn.Linear(hidden_dim, hidden_dim),
        })
        
        # 输出层
        self.out = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.core['W_z'].weight, gain=0.5)
        nn.init.orthogonal_(self.core['W_x'].weight, gain=0.5)
        nn.init.orthogonal_(self.core['W_tau'].weight, gain=0.1)
        nn.init.zeros_(self.core['W_z'].bias)
        nn.init.zeros_(self.core['W_x'].bias)
        nn.init.zeros_(self.core['W_tau'].bias)
    
    def get_prompt(self, x: torch.Tensor) -> torch.Tensor:
        """获取输入相关的提示"""
        weights = self.prompt_selector(x)
        prompt = torch.einsum('bn,np->bp', weights, self.prompt_bank)
        return self.prompt_proj(prompt)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        outputs = []
        
        for t in range(T):
            x_t = x[t]
            
            # 获取提示
            prompt = self.get_prompt(x_t)
            
            # 动力学
            z_real = z.real
            z_imag = z.imag
            
            tanh_real = torch.tanh(z_real)
            tanh_imag = torch.tanh(z_imag)
            
            W_tanh_real = self.core['W_z'](tanh_real)
            W_tanh_imag = self.core['W_z'](tanh_imag)
            Ux = self.core['W_x'](x_t)
            
            # 添加提示影响
            W_tanh_real = W_tanh_real + prompt
            W_tanh_imag = W_tanh_imag + prompt
            
            dz_real = -z_real + W_tanh_real + Ux
            dz_imag = -z_imag + W_tanh_imag
            
            z_mod = torch.abs(z)
            tau = torch.sigmoid(self.core['W_tau'](z_mod)) + 1e-6
            
            dzdt = torch.complex(dz_real / tau, dz_imag / tau)
            dzdt = torch.clamp(dzdt.real, -10, 10) + 1j * torch.clamp(dzdt.imag, -10, 10)

            z = z + self.dt * dzdt

            y_t = self.out(z.real)
            outputs.append(y_t)

        return torch.stack(outputs, dim=0)


# ============================================================================
# 0.2.1: 自生产数据和自训练循环扩展
# ============================================================================

class SelfTrainingTwistorLMT(nn.Module):
    """
    支持自生产数据和自训练循环的 Twistor-LMT 0.2.1
    
    核心能力:
    1. 自动生成训练数据
    2. 自动训练循环
    3. 性能评估
    4. 数据生成策略调整
    5. 持续迭代改进
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        output_dim: int = 1,
        dt: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        
        # 核心模型
        self.model = TwistorLMT(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dt=dt,
            **kwargs
        )
        
        # 数据生成参数
        self.data_params = {
            'freq_range': (0.5, 2.0),
            'noise_std': 0.1,
            'seq_len': 50,
            'n_samples': 100,
        }
        
        # 训练历史
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'data_quality': [],
        }
        
        # 性能指标
        self.performance_metrics = {
            'target_loss': 0.01,
            'min_samples': 50,
            'max_samples': 500,
            'patience': 10,
        }
    
    def generate_data(
        self,
        task_type: str = 'sine',
        n_samples: Optional[int] = None,
        seq_len: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """自动生成训练数据"""
        n_samples = n_samples or self.data_params['n_samples']
        seq_len = seq_len or self.data_params['seq_len']
        
        if task_type == 'sine':
            return self._generate_sine_data(n_samples, seq_len, **kwargs)
        elif task_type == 'lorenz':
            return self._generate_lorenz_data(n_samples, seq_len, **kwargs)
        elif task_type == 'custom':
            return self._generate_custom_data(n_samples, seq_len, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _generate_sine_data(
        self, n_samples: int, seq_len: int,
        freq_range: Optional[Tuple] = None,
        noise_std: Optional[float] = None,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成正弦波数据"""
        freq_range = freq_range or self.data_params['freq_range']
        noise_std = noise_std or self.data_params['noise_std']
        
        X, y = [], []
        for _ in range(n_samples):
            freq = np.random.uniform(*freq_range)
            phase = np.random.uniform(0, 2 * np.pi)
            t = np.linspace(0, 4 * np.pi, seq_len + 1)
            
            signal = np.sin(freq * t + phase) + np.random.randn(len(t)) * noise_std
            
            x_seq = np.stack([
                signal[:-1],
                np.cos(freq * t[:-1] + phase) + np.random.randn(seq_len) * noise_std
            ], axis=-1)
            y_seq = signal[1:].reshape(-1, 1)
            
            X.append(x_seq)
            y.append(y_seq)
        
        X = torch.FloatTensor(np.stack(X)).to(device)
        y = torch.FloatTensor(np.stack(y)).to(device)
        return X, y
    
    def _generate_lorenz_data(
        self, n_samples: int, seq_len: int,
        sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0/3.0,
        dt: float = 0.01, device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成 Lorenz 吸引子数据"""
        X, y = [], []
        for _ in range(n_samples):
            x0 = np.random.uniform(-10, 10, 3)
            trajectory = [x0]
            for _ in range(seq_len):
                x, y_, z = trajectory[-1]
                dx = sigma * (y_ - x) * dt
                dy = (x * (rho - z) - y_) * dt
                dz = (x * y_ - beta * z) * dt
                trajectory.append([x + dx, y_ + dy, z + dz])
            
            trajectory = np.array(trajectory) + np.random.randn(seq_len + 1, 3) * 0.01
            X.append(trajectory[:-1])
            y.append(trajectory[1:])
        
        X = torch.FloatTensor(np.stack(X)).to(device)
        y = torch.FloatTensor(np.stack(y)).to(device)
        return X, y
    
    def _generate_custom_data(
        self, n_samples: int, seq_len: int,
        func: Optional[callable] = None, device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成自定义数据"""
        if func is None:
            def func(t, params):
                a, b, c = params
                return a * t + b * np.sin(c * t)
        
        X, y = [], []
        for _ in range(n_samples):
            params = np.random.uniform(-1, 1, 3)
            t = np.linspace(0, 10, seq_len + 1)
            signal = func(t, params) + np.random.randn(len(t)) * 0.05
            X.append(signal[:-1].reshape(-1, 1))
            y.append(signal[1:].reshape(-1, 1))
        
        X = torch.FloatTensor(np.stack(X)).to(device)
        y = torch.FloatTensor(np.stack(y)).to(device)
        return X, y
    
    def evaluate_performance(self, X_val: torch.Tensor, y_val: torch.Tensor) -> Dict[str, float]:
        """评估模型性能"""
        self.model.eval()
        with torch.no_grad():
            x_val = X_val.transpose(0, 1)
            y_val_t = y_val.transpose(0, 1)
            y_pred = self.model(x_val)
            
            mse = F.mse_loss(y_pred, y_val_t).item()
            mae = F.l1_loss(y_pred, y_val_t).item()
            signal_power = torch.var(y_val_t)
            noise_power = torch.var(y_val_t - y_pred)
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()
        
        return {'mse': mse, 'mae': mae, 'snr': snr, 'loss': mse}
    
    def adjust_data_strategy(self, current_metrics: Dict[str, float]) -> Dict:
        """根据性能调整数据生成策略"""
        current_loss = current_metrics.get('loss', 1.0)
        target_loss = self.performance_metrics['target_loss']
        
        if current_loss > target_loss * 10:
            self.data_params['n_samples'] = min(
                self.data_params['n_samples'] * 2, self.performance_metrics['max_samples']
            )
            self.data_params['noise_std'] = max(self.data_params['noise_std'] * 0.8, 0.01)
        elif current_loss > target_loss:
            self.data_params['n_samples'] = min(
                int(self.data_params['n_samples'] * 1.2), self.performance_metrics['max_samples']
            )
        else:
            self.data_params['n_samples'] = max(
                int(self.data_params['n_samples'] * 0.9), self.performance_metrics['min_samples']
            )
        
        return self.data_params
    
    def self_training_loop(
        self, n_iterations: int = 10, epochs_per_iteration: int = 50,
        task_type: str = 'sine', val_split: float = 0.2,
        device: str = 'cpu', verbose: bool = True,
    ) -> Dict:
        """自训练循环：生成数据 → 训练 → 评估 → 调整策略"""
        if verbose:
            print("=" * 60)
            print("自训练循环开始")
            print("=" * 60)
        
        for iteration in range(n_iterations):
            if verbose:
                print(f"\n迭代 {iteration + 1}/{n_iterations}")
                print("-" * 40)
            
            # 1. 生成数据
            X, y = self.generate_data(task_type=task_type)
            
            # 2. 划分训练/验证集
            n_val = int(len(X) * val_split)
            indices = torch.randperm(len(X))
            train_idx, val_idx = indices[n_val:], indices[:n_val]
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            if verbose:
                print(f"   生成数据：{len(X_train)} 训练，{len(X_val)} 验证")
                print(f"   数据参数：n_samples={self.data_params['n_samples']}, "
                      f"noise_std={self.data_params['noise_std']}")
            
            # 3. 训练
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            train_losses = []
            for epoch in range(epochs_per_iteration):
                optimizer.zero_grad()
                perm = torch.randperm(len(X_train), device=device)
                X_batch = X_train[perm].transpose(0, 1)
                y_batch = y_train[perm].transpose(0, 1)
                
                y_pred = self.model(X_batch)
                loss = F.mse_loss(y_pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())
                scheduler.step(loss)
            
            avg_train_loss = np.mean(train_losses)
            
            # 4. 评估
            metrics = self.evaluate_performance(X_val, y_val)
            
            if verbose:
                print(f"   训练损失：{avg_train_loss:.6f}")
                print(f"   验证 MSE: {metrics['mse']:.6f}")
                print(f"   验证 SNR: {metrics['snr']:.2f} dB")
            
            # 5. 记录历史
            self.training_history['epoch'].append(iteration)
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(metrics['mse'])
            self.training_history['data_quality'].append(metrics['snr'])
            
            # 6. 调整策略
            old_params = self.data_params.copy()
            self.adjust_data_strategy(metrics)
            
            if verbose and old_params != self.data_params:
                print(f"   策略调整：n_samples {old_params['n_samples']} → {self.data_params['n_samples']}")
            
            # 7. 检查收敛
            if metrics['mse'] < self.performance_metrics['target_loss']:
                if verbose:
                    print(f"   ✅ 达到目标损失！提前终止")
                break
        
        if verbose:
            print("\n" + "=" * 60)
            print("自训练循环完成")
            print("=" * 60)
        
        return {
            'history': self.training_history,
            'final_metrics': self.evaluate_performance(X_val, y_val),
            'data_params': self.data_params,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(x)


class AutoCurriculumTrainer:
    """自动课程学习训练器"""
    
    def __init__(self, model: SelfTrainingTwistorLMT):
        self.model = model
        self.curriculum_stage = 0
    
    def define_curriculum(self, stages: Optional[List[Dict]] = None):
        """定义课程阶段"""
        if stages is None:
            self.curriculum_stages = [
                {'name': '简单正弦波', 'task_type': 'sine', 'freq_range': (0.5, 1.0), 'noise_std': 0.05, 'seq_len': 30, 'target_loss': 0.1},
                {'name': '中等正弦波', 'task_type': 'sine', 'freq_range': (0.5, 2.0), 'noise_std': 0.1, 'seq_len': 50, 'target_loss': 0.05},
                {'name': '复杂正弦波', 'task_type': 'sine', 'freq_range': (1.0, 3.0), 'noise_std': 0.15, 'seq_len': 70, 'target_loss': 0.02},
                {'name': 'Lorenz 吸引子', 'task_type': 'lorenz', 'seq_len': 50, 'target_loss': 0.05},
            ]
        else:
            self.curriculum_stages = stages
        self.n_stages = len(self.curriculum_stages)
    
    def train_with_curriculum(self, epochs_per_stage: int = 100, device: str = 'cpu', verbose: bool = True) -> Dict:
        """使用课程学习进行训练"""
        if verbose:
            print("=" * 60)
            print("课程学习训练开始")
            print(f"课程阶段数：{self.n_stages}")
            print("=" * 60)
        
        self.curriculum_history = {'stage': [], 'stage_name': [], 'loss': [], 'completed': []}
        
        for stage_idx, stage in enumerate(self.curriculum_stages):
            if verbose:
                print(f"\n阶段 {stage_idx + 1}/{self.n_stages}: {stage['name']}")
            
            self.model.data_params.update({k: v for k, v in stage.items() if k in ['freq_range', 'noise_std', 'seq_len', 'n_samples']})
            
            result = self.model.self_training_loop(
                n_iterations=3, epochs_per_iteration=epochs_per_stage // 3,
                task_type=stage['task_type'], device=device, verbose=verbose,
            )
            
            final_loss = result['final_metrics']['mse']
            target_loss = stage.get('target_loss', 0.01)
            completed = final_loss < target_loss
            
            self.curriculum_history['stage'].append(stage_idx)
            self.curriculum_history['stage_name'].append(stage['name'])
            self.curriculum_history['loss'].append(final_loss)
            self.curriculum_history['completed'].append(completed)
            
            if verbose:
                print(f"   完成：{'✅ 是' if completed else '❌ 否'}")
                print(f"   最终损失：{final_loss:.6f} (目标：{target_loss})")
        
        if verbose:
            print("\n" + "=" * 60)
            print("课程学习训练完成")
            n_completed = sum(self.curriculum_history['completed'])
            print(f"完成阶段：{n_completed}/{self.n_stages}")
        
        return self.curriculum_history
