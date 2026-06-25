"""
Twistor-inspired Liquid Neural Network - Main network module.
Supports multiple integration methods: Euler, RK4, and torchdiffeq ODE solvers.

Core dynamics: dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional
from .ltc_cell import LTCCell
from ..solvers.rk4 import RK4Integrator


class TwistorLMT(nn.Module):
    """
    Twistor-inspired Liquid Neural Network with complex-valued states.
    Supports Euler and RK4 integration methods.

    The network evolves the complex hidden state over time following:
        dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)

    Key features:
        - Complex-valued hidden state z (torch.complex)
        - State-dependent time constant tau(z) = clamp(sigmoid(W_tau * |z|))
        - Bias terms b for both real and imag parts
        - Input U*x affects both real and imag parts
        - Output from real part only
        - Euler or RK4 integration
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 16, 
        output_dim: int = 1, 
        dt: float = 0.1,
        tau_min: float = 0.01,
        tau_max: float = 1.0,
        dzdt_max: float = 10.0,
        z_max: float = 100.0,
        integration_method: str = 'euler',  # 'euler' or 'rk4'
    ):
        """
        Initialize Twistor LMT.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state (default: 16)
            output_dim: Dimension of output (default: 1)
            dt: Time step for integration (default: 0.1)
            tau_min: Minimum time constant (default: 0.01)
            tau_max: Maximum time constant (default: 1.0)
            dzdt_max: Maximum |dz/dt| (default: 10.0)
            z_max: Maximum |z| state (default: 100.0)
            integration_method: 'euler' or 'rk4' (default: 'euler')
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dt = dt
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.dzdt_max = dzdt_max
        self.z_max = z_max
        self.integration_method = integration_method

        # Core dynamics cell
        self.cell = LTCCell(
            input_dim, 
            hidden_dim, 
            tau_min=tau_min, 
            tau_max=tau_max,
            dzdt_max=dzdt_max,
        )

        # RK4 integrator
        self.rk4 = RK4Integrator(dt=dt)

        # Output projection (real part only)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(
        self, 
        x: torch.Tensor, 
        return_states: bool = False,
        return_diagnostics: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with selectable integration method.

        Args:
            x: Input sequence (T, B, input_dim)
            return_states: If True, return all hidden states
            return_diagnostics: If True, return stability diagnostics

        Returns:
            y: Output sequence (T, B, output_dim)
            states: All hidden states if return_states=True
            diagnostics: Stability info if return_diagnostics=True
        """
        if self.integration_method == 'rk4':
            y, states = self._rk4_forward(x)
        else:
            y, states = self._euler_forward(x)

        if not return_states and not return_diagnostics:
            return y

        result = [y]
        
        if return_states:
            result.append(states)
        
        if return_diagnostics:
            diagnostics = self._compute_diagnostics(states)
            result.append(diagnostics)

        return tuple(result) if len(result) > 1 else result[0]

    def _euler_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with Euler integration."""
        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        outputs = []
        states = [z]

        for t in range(T):
            dzdt = self.cell(z, x[t])
            z = z + self.dt * dzdt
            
            # Clamp z
            z = torch.complex(
                torch.clamp(z.real, -self.z_max, self.z_max),
                torch.clamp(z.imag, -self.z_max, self.z_max)
            )
            
            y_t = self.out(z.real)
            outputs.append(y_t)
            states.append(z)

        return torch.stack(outputs, dim=0), states

    def _rk4_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with RK4 integration."""
        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        outputs = []
        states = [z]

        def dzdt_func(z_state, x_input):
            return self.cell(z_state, x_input)

        for t in range(T):
            z = self.rk4.step(dzdt_func, z, x[t], t)
            
            # Clamp z
            z = torch.complex(
                torch.clamp(z.real, -self.z_max, self.z_max),
                torch.clamp(z.imag, -self.z_max, self.z_max)
            )
            
            y_t = self.out(z.real)
            outputs.append(y_t)
            states.append(z)

        return torch.stack(outputs, dim=0), states

    def _compute_diagnostics(self, states: List[torch.Tensor]) -> Dict:
        """Compute diagnostic information from states."""
        diagnostics = {
            'z_norm': [],
            'dzdt_norm': [],
            'tau_mean': [],
            'tau_std': [],
            'has_nan': False,
            'has_inf': False,
        }

        for i, z in enumerate(states):
            diagnostics['z_norm'].append(torch.abs(z).mean().item())
            
            if i < len(states) - 1:
                dzdt = (states[i + 1] - z) / self.dt
                diagnostics['dzdt_norm'].append(torch.abs(dzdt).mean().item())
            
            tau = self.cell.compute_tau(z)
            diagnostics['tau_mean'].append(tau.mean().item())
            diagnostics['tau_std'].append(tau.std().item())
            
            if torch.isnan(z).any() or torch.isinf(z).any():
                diagnostics['has_nan'] = True
                diagnostics['has_inf'] = True

        diagnostics['z_norm'] = torch.tensor(diagnostics['z_norm'])
        diagnostics['dzdt_norm'] = torch.tensor(diagnostics['dzdt_norm'])
        diagnostics['tau_mean'] = torch.tensor(diagnostics['tau_mean'])
        diagnostics['tau_std'] = torch.tensor(diagnostics['tau_std'])

        return diagnostics

    def get_tau_statistics(self, z: torch.Tensor) -> Dict[str, float]:
        """Get statistics of time constant tau."""
        tau = self.cell.compute_tau(z)
        return {
            'tau_mean': tau.mean().item(),
            'tau_std': tau.std().item(),
            'tau_min': tau.min().item(),
            'tau_max': tau.max().item(),
        }
