"""
Twistor-LMT ODE Solver Module (Twistor-specific)
================================================
High-level ODE solver interface specifically designed for TwistorLMT models.

Note: Base implementations are in liquid_net.solvers.
This module provides Twistor-specific convenience classes.

Usage:
    from twistor_LMT import TwistorODE, create_ode_solver
    solver = create_ode_solver(model)
    output = solver.forward(sequence)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Import from liquid_net.solvers
try:
    from liquid_net.solvers import ODESolver, AdjointODESolver, TORCHDIFFEQ_AVAILABLE
except ImportError:
    from .integrators import TORCHDIFFEQ_AVAILABLE

    ODESolver = None
    AdjointODESolver = None


class ODEDynamics(nn.Module):
    """
    Wrapper to make TwistorLMT compatible with torchdiffeq ODE solvers.

    Converts model dynamics to the format expected by torchdiffeq.
    """

    def __init__(self, model: nn.Module, input_sequence: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = model
        self.input_sequence = input_sequence

    def set_input(self, input_sequence: torch.Tensor):
        """Set input sequence for ODE integration."""
        self.input_sequence = input_sequence

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute dz/dt at time t.

        Args:
            t: Current time (scalar or tensor)
            z: Current state

        Returns:
            dzdt: Time derivative
        """
        if self.input_sequence is None:
            raise ValueError("input_sequence not set")

        # Convert time to index
        t_val = t.item() if isinstance(t, torch.Tensor) else t
        t_idx = int(t_val * (len(self.input_sequence) - 1))
        t_idx = max(0, min(t_idx, len(self.input_sequence) - 1))

        x = self.input_sequence[t_idx]

        # If z has time dimension, take the last one
        if z.dim() > 2:
            z = z[-1]

        return self.model.compute_dzdt(z, x)


class TwistorODE:
    """
    High-level ODE solver interface for TwistorLMT models.

    Provides convenient forward() method for sequence processing.
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "dopri5",
        adjoint: bool = False,
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ):
        self.model = model
        self.method = method
        self.adjoint = adjoint
        self.rtol = rtol
        self.atol = atol

    def forward(
        self,
        x: torch.Tensor,
        return_states: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with ODE solver.

        Args:
            x: Input sequence (T, B, input_dim)
            return_states: Return full trajectory

        Returns:
            y: Output sequence (T, B, output_dim)
            states: State trajectory if return_states=True
        """
        if ODESolver is None:
            raise ImportError("Use liquid_net.solvers.ODESolver")

        T, B, _ = x.shape

        # Initial state
        z0 = torch.zeros(
            B, self.model.hidden_dim, dtype=torch.complex64, device=x.device
        )

        # Create dynamics wrapper
        dynamics = ODEDynamics(self.model)
        dynamics.set_input(x)
        dynamics.to(x.device)

        # Time points
        t_span = torch.linspace(0, 1, T, device=x.device)

        # Use appropriate solver
        if self.adjoint and TORCHDIFFEQ_AVAILABLE:
            from torchdiffeq import odeint_adjoint

            solver = odeint_adjoint
        elif TORCHDIFFEQ_AVAILABLE:
            from torchdiffeq import odeint

            solver = odeint
        else:
            # Fallback to simple Euler
            solver = None

        if solver is None:
            # Simple Euler fallback
            z = z0
            z_trajectory = [z]
            for t_idx in range(T):
                dzdt = self.model.compute_dzdt(z, x[t_idx])
                z = z + self.model.dt * dzdt
                z_trajectory.append(z)
            z_trajectory = torch.stack(z_trajectory[:-1], dim=0)
        else:
            options = {"rtol": self.rtol, "atol": self.atol}
            z_trajectory = solver(
                dynamics,
                z0,
                t_span,
                method=self.method,
                options=options,
            )

        # Compute outputs
        outputs = []
        for t in range(T):
            y_t = self.model.out(z_trajectory[t].real)
            outputs.append(y_t)

        y = torch.stack(outputs, dim=0)

        if return_states:
            return y, z_trajectory
        return y

    def compute_trajectory(
        self,
        x: torch.Tensor,
        z0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute state trajectory only (no output).

        Args:
            x: Input sequence
            z0: Initial state (if None, zeros)

        Returns:
            z_trajectory: (T, B, hidden_dim)
        """
        T, B, _ = x.shape

        if z0 is None:
            z0 = torch.zeros(
                B, self.model.hidden_dim, dtype=torch.complex64, device=x.device
            )

        dynamics = ODEDynamics(self.model)
        dynamics.set_input(x)
        dynamics.to(x.device)

        t_span = torch.linspace(0, 1, T, device=x.device)

        if TORCHDIFFEQ_AVAILABLE:
            from torchdiffeq import odeint

            options = {"rtol": self.rtol, "atol": self.atol}
            return odeint(dynamics, z0, t_span, method=self.method, options=options)
        else:
            # Fallback
            z = z0
            trajectory = [z]
            for t_idx in range(T):
                dzdt = self.model.compute_dzdt(z, x[t_idx])
                z = z + self.model.dt * dzdt
                trajectory.append(z)
            return torch.stack(trajectory[:-1], dim=0)


def odeint_wrapper(
    model: nn.Module,
    z0: torch.Tensor,
    input_sequence: torch.Tensor,
    method: str = "dopri5",
    adjoint: bool = False,
    options: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate ODE using torchdiffeq (convenience wrapper).

    Args:
        model: TwistorLMT model
        z0: Initial state
        input_sequence: Input sequence (T, B, input_dim)
        method: Integration method
        adjoint: Use adjoint method
        options: Solver options

    Returns:
        z_trajectory: State trajectory
        t_eval: Time points
    """
    solver = TwistorODE(model, method=method, adjoint=adjoint)
    dynamics = ODEDynamics(model)
    dynamics.set_input(input_sequence)

    T = len(input_sequence)
    t_span = torch.linspace(0, 1, T, device=z0.device)

    options = options or {}

    if adjoint and TORCHDIFFEQ_AVAILABLE:
        from torchdiffeq import odeint_adjoint

        z_trajectory = odeint_adjoint(dynamics, z0, t_span, method=method, **options)
    elif TORCHDIFFEQ_AVAILABLE:
        from torchdiffeq import odeint

        z_trajectory = odeint(dynamics, z0, t_span, method=method, **options)
    else:
        # Fallback
        z = z0
        trajectory = [z]
        for t_idx in range(T):
            dzdt = model.compute_dzdt(z, input_sequence[t_idx])
            z = z + model.dt * dzdt
            trajectory.append(z)
        z_trajectory = torch.stack(trajectory[:-1], dim=0)

    return z_trajectory, t_span


def create_ode_solver(model: nn.Module, method: str = "dopri5", **kwargs) -> TwistorODE:
    """
    Factory function to create TwistorODE solver.

    Args:
        model: TwistorLMT model
        method: Integration method
        **kwargs: Additional arguments

    Returns:
        solver: TwistorODE instance
    """
    return TwistorODE(model, method=method, **kwargs)
