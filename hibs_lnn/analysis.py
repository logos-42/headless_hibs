"""
Twistor-LMT Analysis Module
===========================
Fixed point analysis and stability analysis tools.

Features:
- Fixed point finding via gradient descent
- Eigenvalue analysis for stability
- Jacobian computation
- Phase space visualization helpers
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, List
from scipy import linalg
from scipy.optimize import fsolve


class FixedPointFinder:
    """
    Find fixed points of the dynamics: dz/dt = 0

    Fixed points satisfy: -z + W*tanh(z) + U*x + b = 0
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    def find_fixed_point(
        self,
        x: torch.Tensor,
        z_init: Optional[torch.Tensor] = None,
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Find fixed point using gradient descent.

        Solve: dz/dt = 0

        Args:
            x: Input (B, input_dim)
            z_init: Initial guess (B, hidden_dim), complex
            lr: Learning rate
            max_iter: Maximum iterations
            tol: Tolerance

        Returns:
            z_fixed: Fixed point
            info: Dictionary with convergence info
        """
        B = x.size(0)
        hidden_dim = self.model.hidden_dim

        # Initialize
        if z_init is None:
            z = torch.zeros(B, hidden_dim, dtype=torch.complex64, device=self.device)
        else:
            z = z_init.clone().detach().requires_grad_(True)

        x = x.to(self.device).detach()

        # Optimizer
        optimizer = torch.optim.Adam([z], lr=lr)

        history = []

        for i in range(max_iter):
            optimizer.zero_grad()

            # Compute dz/dt
            dzdt = self.model.compute_dzdt(z, x)

            # Loss: minimize ||dz/dt||^2
            loss = (torch.abs(dzdt) ** 2).mean()

            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            history.append(loss_val)

            if loss_val < tol:
                break

        info = {
            "converged": loss_val < tol,
            "final_loss": loss_val,
            "iterations": i + 1,
            "history": history,
        }

        return z.detach(), info

    def find_multiple_fixed_points(
        self, x: torch.Tensor, num_points: int = 10, **kwargs
    ) -> List[Tuple[torch.Tensor, Dict]]:
        """
        Find multiple fixed points with different initializations.

        Args:
            x: Input
            num_points: Number of different initializations
            **kwargs: Arguments for find_fixed_point

        Returns:
            List of (z_fixed, info) tuples
        """
        B = x.size(0)
        hidden_dim = self.model.hidden_dim
        results = []

        for i in range(num_points):
            # Random initialization
            z_init = (
                torch.randn(B, hidden_dim, dtype=torch.complex64, device=self.device)
                * (i + 1)
                * 0.5
            )

            z_fixed, info = self.find_fixed_point(x, z_init=z_init, **kwargs)
            results.append((z_fixed, info))

        return results


class StabilityAnalyzer:
    """
    Analyze stability of fixed points via eigenvalue analysis.

    For complex systems, we analyze the real Jacobian:
    J = d(dz/dt)/dz
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    def compute_jacobian(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jacobian J = d(dz/dt)/dz at given point.

        Args:
            z: State (B, hidden_dim), complex
            x: Input (B, input_dim)

        Returns:
            J: Jacobian (B, 2*hidden_dim, 2*hidden_dim)
               First hidden_dim rows: d(dz_real)/dz
               Last hidden_dim rows: d(dz_imag)/dz
        """
        z = z.detach().requires_grad_(True)
        x = x.detach()

        # Compute dz/dt
        dzdt = self.model.compute_dzdt(z, x)

        # Jacobian for real part
        J_real = []
        for i in range(z.size(1)):
            grad = torch.autograd.grad(dzdt.real[:, i].sum(), z, retain_graph=True)[0]
            J_real.append(grad.real)
        J_real = torch.stack(J_real, dim=0).transpose(0, 1)  # (B, hidden, hidden)

        # Jacobian for imag part
        J_imag = []
        for i in range(z.size(1)):
            grad = torch.autograd.grad(dzdt.imag[:, i].sum(), z, retain_graph=True)[0]
            J_imag.append(grad.imag)
        J_imag = torch.stack(J_imag, dim=0).transpose(0, 1)

        # Combined Jacobian (real system of size 2*hidden_dim)
        # J = [d(dz_real)/dz_real  d(dz_real)/dz_imag]
        #     [d(dz_imag)/dz_real  d(dz_imag)/dz_imag]

        # d(dz_real)/dz_imag and d(dz_imag)/dz_real need gradients through abs
        J_cross_real_imag = []
        for i in range(z.size(1)):
            grad = torch.autograd.grad(dzdt.real[:, i].sum(), z, retain_graph=True)[0]
            J_cross_real_imag.append(grad.imag)
        J_cross_real_imag = torch.stack(J_cross_real_imag, dim=0).transpose(0, 1)

        J_cross_imag_real = []
        for i in range(z.size(1)):
            grad = torch.autograd.grad(dzdt.imag[:, i].sum(), z, retain_graph=True)[0]
            J_cross_imag_real.append(grad.real)
        J_cross_imag_real = torch.stack(J_cross_imag_real, dim=0).transpose(0, 1)

        # Full Jacobian
        J = torch.zeros(z.size(0), 2 * hidden_dim, 2 * hidden_dim, device=self.device)

        J[:, :hidden_dim, :hidden_dim] = J_real
        J[:, :hidden_dim, hidden_dim:] = J_cross_real_imag
        J[:, hidden_dim:, :hidden_dim] = J_cross_imag_real
        J[:, hidden_dim:, hidden_dim:] = J_imag

        return J

    def analyze_stability(
        self,
        z: torch.Tensor,
        x: torch.Tensor,
    ) -> Dict:
        """
        Analyze stability of a fixed point.

        Args:
            z: Fixed point (B, hidden_dim), complex
            x: Input (B, input_dim)

        Returns:
            Dictionary with stability info
        """
        J = self.compute_jacobian(z, x)

        # For a single point
        if z.size(0) == 1:
            J_np = J[0].cpu().numpy()
        else:
            J_np = J.mean(0).cpu().numpy()

        # Eigenvalues
        eigenvalues = np.linalg.eigvals(J_np)

        # Stability criteria
        real_parts = eigenvalues.real
        max_real = real_parts.max()

        # Determine stability
        if max_real < 0:
            stability = "stable"  # All eigenvalues have negative real part
        elif max_real > 0:
            stability = "unstable"  # At least one eigenvalue has positive real part
        else:
            stability = "marginal"  # On the boundary

        # Oscillation detection (complex eigenvalues)
        imag_parts = np.abs(eigenvalues.imag)
        has_oscillation = np.any(imag_parts > 1e-6)

        # Oscillation frequency
        if has_oscillation:
            osc_indices = np.where(imag_parts > 1e-6)[0]
            frequencies = np.abs(eigenvalues[osc_indices].imag) / (2 * np.pi)
            dominant_freq = frequencies.max()
        else:
            dominant_freq = 0.0

        return {
            "eigenvalues": eigenvalues,
            "max_real_part": max_real,
            "stability": stability,
            "has_oscillation": has_oscillation,
            "oscillation_frequency": dominant_freq,
            "jacobian": J_np,
        }

    def compute_lyapunov_exponents(
        self,
        x: torch.Tensor,
        z0: Optional[torch.Tensor] = None,
        num_steps: int = 100,
        dt: float = 0.1,
    ) -> np.ndarray:
        """
        Estimate Lyapunov exponents via trajectory linearization.

        Args:
            x: Input sequence
            z0: Initial state
            num_steps: Number of steps
            dt: Time step

        Returns:
            lyapunov_exponents: Estimated exponents
        """
        B = x.size(1) if x.dim() > 1 else 1
        hidden_dim = self.model.hidden_dim

        if z0 is None:
            z = torch.zeros(B, hidden_dim, dtype=torch.complex64, device=self.device)
        else:
            z = z0.clone()

        # Accumulator for Jacobian products
        Q = np.eye(2 * hidden_dim)

        for t in range(num_steps):
            x_t = x[t] if t < len(x) else x[-1]

            # Compute Jacobian
            J = self.compute_jacobian(z, x_t)

            # For single batch
            J_np = J[0].cpu().numpy() if B > 1 else J.cpu().numpy()

            # QR decomposition
            Q, R = np.linalg.qr(Q @ (np.eye(2 * hidden_dim) + dt * J_np))

        # Lyapunov exponents from diagonal of R
        lyapunov = np.log(np.abs(np.diag(R))) / (num_steps * dt)

        return lyapunov


class BifurcationAnalyzer:
    """
    Analyze bifurcations as parameters vary.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.finder = FixedPointFinder(model)
        self.analyzer = StabilityAnalyzer(model)

    def sweep_parameter(
        self,
        param_name: str,
        param_values: List[float],
        x: torch.Tensor,
    ) -> List[Dict]:
        """
        Sweep a parameter and analyze fixed points.

        Args:
            param_name: Name of parameter to sweep
            param_values: List of values to sweep
            x: Input

        Returns:
            List of analysis results
        """
        results = []

        original_value = getattr(self.model, param_name, None)

        for val in param_values:
            setattr(self.model, param_name, val)

            # Find fixed point
            z_fixed, info = self.finder.find_fixed_point(x)

            # Analyze stability
            stability = self.analyzer.analyze_stability(z_fixed, x)

            results.append(
                {
                    "param_value": val,
                    "fixed_point": z_fixed,
                    "stability": stability,
                    "convergence": info,
                }
            )

        # Restore original value
        if original_value is not None:
            setattr(self.model, param_name, original_value)

        return results


def analyze_model(model: nn.Module, x: torch.Tensor) -> Dict:
    """
    Complete analysis of a TwistorLMT model.

    Args:
        model: TwistorLMT model
        x: Sample input

    Returns:
        Dictionary with all analysis results
    """
    finder = FixedPointFinder(model)
    analyzer = StabilityAnalyzer(model)

    # Find fixed point
    z_fixed, info = finder.find_fixed_point(x)

    # Analyze stability
    stability = analyzer.analyze_stability(z_fixed, x)

    # Compute tau statistics
    tau_stats = model.get_tau_statistics(z_fixed)

    return {
        "fixed_point": z_fixed,
        "fixed_point_info": info,
        "stability": stability,
        "tau_statistics": tau_stats,
    }
