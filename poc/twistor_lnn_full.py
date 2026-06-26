"""
Twistor-inspired Liquid Neural Network (Complex-valued LMT) - RK4 & ODE Integrated
===================================================================================
Implements continuous-time dynamics: dz/dt = (-z + W*tanh(z) + U*x + b) / tau(z)

Integration Methods:
- Euler (default, fast)
- RK4 (more accurate)
- torchdiffeq ODE solvers (most accurate, if available)

Features:
- Complex-valued hidden state z (torch.complex)
- State-dependent time constant tau(z) with clamping
- dz/dt normalization to prevent explosion
- Gradient clipping during training
- L2 regularization on z
- Tunable dt parameter
- NaN/Inf detection
- Fixed-point analysis tools
- Phase space visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional, List, Union
from dataclasses import dataclass

# Try to import torchdiffeq for advanced ODE solving
try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("Warning: torchdiffeq not available. Install with: pip install torchdiffeq")


@dataclass
class IntegrationConfig:
    """Configuration for ODE integration."""
    method: str = 'euler'  # 'euler', 'rk4', 'dopri5', 'rk45'
    dt: float = 0.1
    rtol: float = 1e-5  # Relative tolerance (for ODE solvers)
    atol: float = 1e-5  # Absolute tolerance (for ODE solvers)


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
    
    def step(self, dzdt_func, z: torch.Tensor, x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """
        Perform a single RK4 integration step.
        
        Args:
            dzdt_func: Function computing dz/dt given (z, x)
            z: Current state (B, hidden_dim), dtype=complex
            x: Input (B, input_dim)
            t: Time step index
            
        Returns:
            z_new: Updated state after one RK4 step
        """
        dt = self.dt
        
        # k1 = f(t, z)
        k1 = dzdt_func(z, x)
        
        # k2 = f(t + dt/2, z + dt*k1/2)
        z_mid = z + dt * k1 / 2
        k2 = dzdt_func(z_mid, x)
        
        # k3 = f(t + dt/2, z + dt*k2/2)
        z_mid = z + dt * k2 / 2
        k3 = dzdt_func(z_mid, x)
        
        # k4 = f(t + dt, z + dt*k3)
        z_end = z + dt * k3
        k4 = dzdt_func(z_end, x)
        
        # z(t+dt) = z(t) + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        z_new = z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        return z_new
    
    def integrate(self, dzdt_func, z0: torch.Tensor, x_seq: torch.Tensor) -> List[torch.Tensor]:
        """
        Integrate over a sequence of inputs.
        
        Args:
            dzdt_func: Function computing dz/dt given (z, x)
            z0: Initial state (B, hidden_dim)
            x_seq: Input sequence (T, B, input_dim)
            
        Returns:
            List of states at each time step
        """
        T, B, _ = x_seq.shape
        z = z0
        states = [z]
        
        for t in range(T):
            z = self.step(dzdt_func, z, x_seq[t], t)
            states.append(z)
        
        return states


class ODESolverWrapper:
    """
    Wrapper for torchdiffeq ODE solvers with complex state support.
    """
    
    def __init__(self, config: IntegrationConfig = None):
        self.config = config or IntegrationConfig()
        
        if self.config.method != 'euler' and not TORCHDIFFEQ_AVAILABLE:
            print(f"Warning: torchdiffeq not available. Falling back to Euler method.")
            self.config.method = 'euler'
    
    def solve(self, dzdt_func, z0: torch.Tensor, t_span: torch.Tensor, 
              x_interp: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
        Solve ODE using specified method.
        
        Args:
            dzdt_func: Function computing dz/dt given (t, z)
            z0: Initial state (B, hidden_dim)
            t_span: Time points to evaluate (T,)
            x_interp: Interpolated inputs at time points (T, B, input_dim)
            
        Returns:
            Solution at time points (T, B, hidden_dim)
        """
        if self.config.method == 'euler':
            return self._euler_solve(dzdt_func, z0, t_span, x_interp)
        elif self.config.method == 'rk4':
            return self._rk4_solve(dzdt_func, z0, t_span, x_interp)
        else:
            if TORCHDIFFEQ_AVAILABLE:
                return odeint(dzdt_func, z0, t_span, 
                             method=self.config.method,
                             rtol=self.config.rtol,
                             atol=self.config.atol,
                             **kwargs)
            else:
                return self._euler_solve(dzdt_func, z0, t_span, x_interp)
    
    def _euler_solve(self, dzdt_func, z0: torch.Tensor, t_span: torch.Tensor,
                     x_interp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Euler integration."""
        solutions = [z0]
        z = z0
        
        for i in range(len(t_span) - 1):
            dt = t_span[i + 1] - t_span[i]
            x_t = x_interp[i] if x_interp is not None else None
            if x_t is not None:
                dydt = dzdt_func(t_span[i], z, x_t)
            else:
                dydt = dzdt_func(t_span[i], z)
            z = z + dt * dydt
            solutions.append(z)
        
        return torch.stack(solutions, dim=0)
    
    def _rk4_solve(self, dzdt_func, z0: torch.Tensor, t_span: torch.Tensor,
                   x_interp: Optional[torch.Tensor] = None) -> torch.Tensor:
        """RK4 integration."""
        rk4 = RK4Integrator()
        
        def step_func(z, x):
            return dzdt_func(torch.tensor(0.0), z, x)
        
        states = rk4.integrate(step_func, z0, x_interp if x_interp is not None else torch.zeros(len(t_span), 1, 1))
        return torch.stack(states[:-1], dim=0)


class TwistorLMT(nn.Module):
    """
    Twistor-inspired Liquid Neural Network with complex-valued states.
    Stability-optimized version with multiple integration methods.

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
        integration_method: str = 'euler',  # 'euler', 'rk4', 'dopri5'
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
        
        # Integration method
        self.integration_method = integration_method
        self.rk4_integrator = RK4Integrator(dt=dt)
        self.ode_solver = ODESolverWrapper(IntegrationConfig(method=integration_method, dt=dt))

        # Weight matrices
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
                mask_real = (torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity).float()
                mask_imag = (torch.rand(self.hidden_dim, self.hidden_dim) > self.sparsity).float()
                self.sparse_mask_real.copy_(mask_real)
                self.sparse_mask_imag.copy_(mask_imag)

        # Initialize multi-scale tau bias
        if self.multi_scale_tau and self.tau_bias is not None:
            nn.init.zeros_(self.tau_bias)

    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent time constant with clamping.
        """
        z_mod = torch.abs(z)
        tau = F.sigmoid(self.W_tau(z_mod))

        if self.multi_scale_tau and self.tau_bias is not None:
            tau = tau + self.tau_bias.unsqueeze(0)

        tau = torch.clamp(tau, self.tau_min, self.tau_max)
        return tau + 1e-6

    def compute_dzdt(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative dz/dt with stability normalization.
        """
        z_real = z.real
        z_imag = z.imag

        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)

        # Apply sparse masks
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

        # Normalize dz/dt
        dzdt_real = torch.clamp(dzdt.real, -self.dzdt_max, self.dzdt_max)
        dzdt_imag = torch.clamp(dzdt.imag, -self.dzdt_max, self.dzdt_max)
        dzdt_clipped = torch.complex(dzdt_real, dzdt_imag)
        
        dzdt_norm = torch.abs(dzdt_clipped)
        mean_norm = dzdt_norm.mean()
        if mean_norm > self.dzdt_max / 2:
            scale = (self.dzdt_max / 2) / (mean_norm + 1e-6)
            dzdt_clipped = dzdt_clipped * scale

        return dzdt_clipped

    def _euler_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with Euler integration."""
        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        outputs = []
        states = [z]

        for t in range(T):
            dzdt = self.compute_dzdt(z, x[t])
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
            return self.compute_dzdt(z_state, x_input)

        for t in range(T):
            z = self.rk4_integrator.step(dzdt_func, z, x[t], t)
            
            # Clamp z
            z = torch.complex(
                torch.clamp(z.real, -self.z_max, self.z_max),
                torch.clamp(z.imag, -self.z_max, self.z_max)
            )
            
            y_t = self.out(z.real)
            outputs.append(y_t)
            states.append(z)

        return torch.stack(outputs, dim=0), states

    def _ode_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with ODE solver (torchdiffeq)."""
        T, B, _ = x.shape
        z = torch.zeros(B, self.hidden_dim, dtype=torch.complex64, device=x.device)
        
        t_span = torch.linspace(0, T * self.dt, T + 1, device=x.device)
        
        # Create interpolated inputs
        x_interp = F.interpolate(x.transpose(0, 1).unsqueeze(0).float(), 
                                  size=T, mode='linear', align_corners=False)
        x_interp = x_interp.squeeze(0).transpose(0, 1)
        
        def ode_func(t, z_state, x_input=None):
            # For ODE solver, we need to handle time-varying inputs
            if x_input is not None:
                t_idx = int(t / self.dt)
                t_idx = min(t_idx, T - 1)
                return self.compute_dzdt(z_state, x_input[t_idx])
            return self.compute_dzdt(z_state, torch.zeros(B, self.input_dim, device=z_state.device))
        
        # Use ODE solver
        if TORCHDIFFEQ_AVAILABLE:
            z_traj = odeint(lambda t, z: ode_func(t, z, x_interp), z, t_span, 
                           method=self.integration_method if self.integration_method != 'euler' else 'dopri5')
        else:
            z_traj = self.ode_solver.solve(lambda t, z: ode_func(t, z, x_interp), z, t_span)
        
        outputs = []
        for t in range(T):
            y_t = self.out(z_traj[t].real)
            outputs.append(y_t)

        return torch.stack(outputs, dim=0), z_traj.unbind(0)

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
        # Select integration method
        if self.integration_method == 'rk4':
            y, states = self._rk4_forward(x)
        elif self.integration_method in ['dopri5', 'rk45', 'dopri8']:
            y, states = self._ode_forward(x)
        else:  # euler
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
            
            tau = self.compute_tau(z)
            diagnostics['tau_mean'].append(tau.mean().item())
            diagnostics['tau_std'].append(tau.std().item())
            
            if torch.isnan(z).any() or torch.isinf(z).any():
                diagnostics['has_nan'] = True
                diagnostics['has_inf'] = True

        diagnostics['z_norm'] = np.array(diagnostics['z_norm'])
        diagnostics['dzdt_norm'] = np.array(diagnostics['dzdt_norm'])
        diagnostics['tau_mean'] = np.array(diagnostics['tau_mean'])
        diagnostics['tau_std'] = np.array(diagnostics['tau_std'])

        return diagnostics

    def analyze_fixed_points(self, x_const: torch.Tensor, max_iter: int = 1000, 
                            tol: float = 1e-6) -> Dict:
        """
        Analyze fixed points of the dynamics for constant input.
        
        A fixed point z* satisfies: dz/dt(z*, x) = 0
        
        Args:
            x_const: Constant input (B, input_dim)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            
        Returns:
            Dictionary with fixed point analysis results
        """
        B = x_const.shape[0]
        z = torch.randn(B, self.hidden_dim, dtype=torch.complex64, device=x_const.device) * 0.1
        
        z_history = []
        dzdt_norms = []
        
        for i in range(max_iter):
            dzdt = self.compute_dzdt(z, x_const)
            dzdt_norm = torch.abs(dzdt).mean().item()
            dzdt_norms.append(dzdt_norm)
            
            # Euler step to find fixed point
            z_new = z + self.dt * dzdt
            
            # Check convergence
            delta_z = torch.abs(z_new - z).mean().item()
            z_history.append(z.clone())
            
            if delta_z < tol and dzdt_norm < tol:
                break
            
            z = z_new
        
        # Compute Jacobian eigenvalues at fixed point (linear stability analysis)
        eigenvalues = self._compute_jacobian_eigenvalues(z, x_const)
        
        return {
            'fixed_point': z,
            'converged': dzdt_norms[-1] < tol,
            'iterations': len(z_history),
            'dzdt_final': dzdt_norms[-1],
            'dzdt_history': dzdt_norms,
            'eigenvalues': eigenvalues,
            'is_stable': eigenvalues is not None and all(e.real < 0 for e in eigenvalues),
        }

    def _compute_jacobian_eigenvalues(self, z: torch.Tensor, x: torch.Tensor, 
                                      eps: float = 1e-5) -> Optional[List[complex]]:
        """
        Compute eigenvalues of the Jacobian at a given state.
        
        J_ij = ∂(dz_i/dt) / ∂z_j
        
        This is a numerical approximation using finite differences.
        """
        B, N = z.shape
        if B > 1:
            z = z[0:1]  # Use first batch element
            x = x[0:1]
        
        dzdt_base = self.compute_dzdt(z, x)
        
        J = torch.zeros(N, N, dtype=torch.complex64, device=z.device)
        
        for j in range(N):
            # Perturb real part
            z_pert = z.clone()
            z_pert[:, j] = z_pert[:, j] + eps
            dzdt_pert = self.compute_dzdt(z_pert, x)
            J[:, j] = (dzdt_pert - dzdt_base).squeeze(0) / eps
        
        # Compute eigenvalues
        try:
            eigenvalues = torch.linalg.eigvals(J)
            return eigenvalues.tolist()
        except:
            return None

    def get_tau_statistics(self, z: torch.Tensor) -> Dict[str, float]:
        """Get statistics of time constant tau."""
        tau = self.compute_tau(z)
        return {
            'tau_mean': tau.mean().item(),
            'tau_std': tau.std().item(),
            'tau_min': tau.min().item(),
            'tau_max': tau.max().item(),
        }


def plot_phase_space(model: TwistorLMT, x: torch.Tensor, n_steps: int = 100,
                     save_path: str = 'phase_space.png'):
    """
    Plot phase space trajectory.
    
    Args:
        model: Trained TwistorLMT model
        x: Input sequence
        n_steps: Number of steps to plot
        save_path: Path to save figure
    """
    model.eval()
    
    with torch.no_grad():
        y, states, diag = model(x[:n_steps], return_states=True, return_diagnostics=True)
    
    if isinstance(states, list):
        states = torch.stack(states[1:], dim=0)  # Skip initial zero state
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Real vs Imag phase portrait (first 2 neurons)
    ax1 = axes[0, 0]
    z_sample = states[:, 0, :2]  # First batch, first 2 neurons
    ax1.plot(z_sample.real.cpu(), z_sample.imag.cpu(), 'b-', alpha=0.5)
    ax1.scatter(z_sample.real[0].cpu(), z_sample.imag[0].cpu(), c='g', s=50, label='Start')
    ax1.scatter(z_sample.real[-1].cpu(), z_sample.imag[-1].cpu(), c='r', s=50, label='End')
    ax1.set_xlabel('Re(z)')
    ax1.set_ylabel('Im(z)')
    ax1.set_title('Phase Portrait (Re vs Im)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: |z| over time
    ax2 = axes[0, 1]
    ax2.plot(diag['z_norm'], 'b-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('|z| (mean)')
    ax2.set_title('State Norm Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: |dz/dt| over time
    ax3 = axes[1, 0]
    ax3.plot(diag['dzdt_norm'], 'g-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('|dz/dt| (mean)')
    ax3.set_title('Time Derivative Norm Over Time')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: τ distribution
    ax4 = axes[1, 1]
    all_taus = []
    for z in states:
        tau = model.compute_tau(z)
        all_taus.extend(tau.flatten().cpu().numpy())
    ax4.hist(all_taus, bins=50, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('τ value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Time Constant Distribution')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Phase space plot saved to '{save_path}'")
    plt.close()


def plot_z_trajectory(diagnostics: Dict, save_path: str = 'z_trajectory.png'):
    """
    Plot z trajectory and tau distribution from diagnostics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.plot(diagnostics['z_norm'], 'b-', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('|z| (mean)')
    ax1.set_title('State Norm Over Time')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(diagnostics['dzdt_norm'], 'g-', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('|dz/dt| (mean)')
    ax2.set_title('Time Derivative Norm Over Time')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(diagnostics['tau_mean'], 'm-', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('τ (mean)')
    ax3.set_title('Time Constant Mean Over Time')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    if 'taus' in diagnostics:
        all_taus = diagnostics['taus'].flatten().cpu().numpy()
        ax4.hist(all_taus, bins=50, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('τ value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Time Constant Distribution')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Z trajectory saved to '{save_path}'")
    plt.close()
