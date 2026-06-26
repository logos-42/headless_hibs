"""
Twistor-LMT Integrators Module
==============================
Numerical integration methods for ODE dynamics.

Methods:
- Euler: First-order, simple
- RK4: Fourth-order Runge-Kutta
- ODESolver: torchdiffeq wrapper with multiple methods

This module consolidates all solver implementations.
"""

import torch
from typing import Callable, Optional, List
from functools import wraps

# Try to import torchdiffeq
try:
    from torchdiffeq import odeint, odeint_adjoint

    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    odeint = None
    odeint_adjoint = None


# ============================================================
# Basic Euler Integration
# ============================================================


def euler_step(z: torch.Tensor, dzdt: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """
    Perform a single Euler integration step.

    Euler: z(t+dt) = z(t) + dt * dz/dt

    Args:
        z: Current state (can be complex)
        dzdt: Time derivative at current state
        dt: Time step size

    Returns:
        z_new: Updated state
    """
    return z + dt * dzdt


# ============================================================
# Runge-Kutta 4th Order
# ============================================================


class RK4Integrator:
    """
    Runge-Kutta 4th order integrator for complex-valued ODEs.

    RK4 formula:
        k1 = f(t, z)
        k2 = f(t + dt/2, z + dt*k1/2)
        k3 = f(t + dt/2, z + dt*k2/2)
        k4 = f(t + dt, z + dt*k3)
        z(t+dt) = z(t) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    """

    def __init__(self, dt: float = 0.1):
        self.dt = dt

    def step(
        self, dzdt_func: Callable, z: torch.Tensor, x: torch.Tensor, t: int = 0
    ) -> torch.Tensor:
        """Single RK4 step."""
        dt = self.dt

        k1 = dzdt_func(z, x)
        k2 = dzdt_func(z + dt * k1 / 2, x)
        k3 = dzdt_func(z + dt * k2 / 2, x)
        k4 = dzdt_func(z + dt * k3, x)

        return z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def integrate(
        self, dzdt_func: Callable, z0: torch.Tensor, x_seq: torch.Tensor
    ) -> List[torch.Tensor]:
        """Integrate over sequence."""
        T, B, _ = x_seq.shape
        z = z0
        states = [z]

        for t in range(T):
            z = self.step(dzdt_func, z, x_seq[t], t)
            states.append(z)

        return states


def rk4_step(
    dzdt_fn: Callable, z: torch.Tensor, x: torch.Tensor, dt: float, *args, **kwargs
) -> torch.Tensor:
    """Simple functional RK4 step."""
    k1 = dzdt_fn(z, x, *args, **kwargs)
    k2 = dzdt_fn(z + 0.5 * dt * k1, x, *args, **kwargs)
    k3 = dzdt_fn(z + 0.5 * dt * k2, x, *args, **kwargs)
    k4 = dzdt_fn(z + dt * k3, x, *args, **kwargs)
    return z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


# ============================================================
# ODESolver with torchdiffeq
# ============================================================


class ODESolver:
    """
    ODE Solver wrapper supporting multiple integration methods.

    Supports:
    - 'euler': Forward Euler
    - 'rk4': Runge-Kutta 4th order
    - 'dopri5': Dormand-Prince (with torchdiffeq)
    - 'rk45': RK-Fehlberg (with torchdiffeq)
    """

    AVAILABLE_METHODS = ["euler", "rk4", "dopri5", "rk45", "adjoint"]

    def __init__(
        self,
        method: str = "dopri5",
        dt: float = 0.1,
        rtol: float = 1e-5,
        atol: float = 1e-5,
    ):
        """
        Initialize ODE Solver.

        Args:
            method: Integration method
            dt: Time step for discrete methods
            rtol: Relative tolerance (adaptive solvers)
            atol: Absolute tolerance (adaptive solvers)
        """
        self.method = method
        self.dt = dt
        self.rtol = rtol
        self.atol = atol

        if method not in self.AVAILABLE_METHODS:
            if not TORCHDIFFEQ_AVAILABLE:
                print(
                    f"Warning: Unknown method '{method}' and torchdiffeq not available. Using Euler."
                )
                self.method = "euler"

    def solve(
        self,
        func: Callable,
        y0: torch.Tensor,
        t: torch.Tensor,
        x_interp: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Solve ODE using specified method."""
        if self.method == "euler":
            return self._euler_solve(func, y0, t, x_interp)
        elif self.method == "rk4":
            return self._rk4_solve(func, y0, t, x_interp)
        elif self.method in ["dopri5", "rk45"]:
            if TORCHDIFFEQ_AVAILABLE:
                return odeint(
                    func,
                    y0,
                    t,
                    method=self.method,
                    rtol=self.rtol,
                    atol=self.atol,
                    **kwargs,
                )
            else:
                print(f"Warning: torchdiffeq not available. Falling back to Euler.")
                return self._euler_solve(func, y0, t, x_interp)
        elif self.method == "adjoint":
            if TORCHDIFFEQ_AVAILABLE:
                return odeint_adjoint(
                    func, y0, t, rtol=self.rtol, atol=self.atol, **kwargs
                )
            else:
                return self._euler_solve(func, y0, t, x_interp)
        else:
            return self._euler_solve(func, y0, t, x_interp)

    def _euler_solve(
        self,
        func: Callable,
        y0: torch.Tensor,
        t: torch.Tensor,
        x_interp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve using Euler."""
        solutions = [y0]
        y = y0

        for i in range(len(t) - 1):
            dt = t[i + 1] - t[i]
            x_t = x_interp[i] if x_interp is not None else None
            if x_t is not None:
                dydt = func(t[i], y, x_t)
            else:
                dydt = func(t[i], y)
            y = y + dt * dydt
            solutions.append(y)

        return torch.stack(solutions, dim=0)

    def _rk4_solve(
        self,
        func: Callable,
        y0: torch.Tensor,
        t: torch.Tensor,
        x_interp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Solve using RK4."""
        dt = self.dt
        solutions = [y0]
        y = y0

        for i in range(len(t) - 1):
            x_t = x_interp[i] if x_interp is not None else None

            if x_t is not None:
                k1 = func(t[i], y, x_t)
                k2 = func(t[i] + dt / 2, y + dt * k1 / 2, x_t)
                k3 = func(t[i] + dt / 2, y + dt * k2 / 2, x_t)
                k4 = func(t[i] + dt, y + dt * k3, x_t)
            else:
                k1 = func(t[i], y)
                k2 = func(t[i] + dt / 2, y + dt * k1 / 2)
                k3 = func(t[i] + dt / 2, y + dt * k2 / 2)
                k4 = func(t[i] + dt, y + dt * k3)

            y = y + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            solutions.append(y)

        return torch.stack(solutions, dim=0)


class AdjointODESolver:
    """
    Adjoint ODE solver for memory-efficient backpropagation.
    Uses torchdiffeq's odeint_adjoint.
    """

    def __init__(self, method: str = "dopri5", rtol: float = 1e-5, atol: float = 1e-5):
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def solve(
        self, func: Callable, y0: torch.Tensor, t: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Solve using adjoint method."""
        if not TORCHDIFFEQ_AVAILABLE:
            raise ImportError("torchdiffeq required for adjoint ODE solving")
        return odeint_adjoint(
            func, y0, t, method=self.method, rtol=self.rtol, atol=self.atol, **kwargs
        )


# ============================================================
# Additional Methods
# ============================================================


def heun_step(
    dzdt_fn: Callable, z: torch.Tensor, x: torch.Tensor, dt: float, *args, **kwargs
) -> torch.Tensor:
    """Heun's method (second-order)."""
    k1 = dzdt_fn(z, x, *args, **kwargs)
    k2 = dzdt_fn(z + dt * k1, x, *args, **kwargs)
    return z + (dt / 2) * (k1 + k2)


def dopri5_step(
    dzdt_fn: Callable, z: torch.Tensor, x: torch.Tensor, dt: float, *args, **kwargs
) -> torch.Tensor:
    """Dormand-Prince 5th order (simplified)."""
    c = torch.tensor(
        [0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1], device=z.device, dtype=z.dtype
    )
    a = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ],
        device=z.device,
        dtype=z.dtype,
    )
    b = torch.tensor(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        device=z.device,
        dtype=z.dtype,
    )

    k = [dzdt_fn(z, x, *args, **kwargs)]
    for i in range(1, 7):
        z_temp = z + dt * sum(a[i, j] * k[j] for j in range(i))
        k.append(dzdt_fn(z_temp, x, *args, **kwargs))

    return z + dt * sum(b[i] * k[i] for i in range(7))


def create_integrator(method: str = "dopri5", **kwargs):
    """Factory to create integrator."""
    if method in ["dopri5", "rk45", "adjoint"]:
        if method == "adjoint":
            return AdjointODESolver(**kwargs)
        return ODESolver(method=method, **kwargs)
    elif method == "rk4":
        return RK4Integrator(**kwargs)
    else:
        return ODESolver(method="euler", **kwargs)
