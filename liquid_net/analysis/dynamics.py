"""
Dynamics Analysis Tools for Twistor-inspired Liquid Neural Network.

Provides tools for analyzing the dynamical system:
- Fixed point analysis
- Jacobian eigenvalue computation
- Phase space visualization
- Stability analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


class DynamicsAnalyzer:
    """
    Analyzer for Twistor LMT dynamics.
    """
    
    def __init__(self, model: torch.nn.Module):
        """
        Initialize dynamics analyzer.
        
        Args:
            model: TwistorLMT model to analyze
        """
        self.model = model
        self.dt = model.dt
    
    def find_fixed_point(self, x_const: torch.Tensor, z0: Optional[torch.Tensor] = None,
                         max_iter: int = 1000, tol: float = 1e-8,
                         method: str = 'euler') -> Dict:
        """
        Find fixed point for constant input.
        
        A fixed point z* satisfies: dz/dt(z*, x) = 0
        
        Args:
            x_const: Constant input (B, input_dim)
            z0: Initial state guess (default: random)
            max_iter: Maximum iterations
            tol: Convergence tolerance
            method: 'euler' or 'newton'
            
        Returns:
            Dictionary with analysis results
        """
        B = x_const.shape[0]
        device = x_const.device
        
        if z0 is None:
            z = torch.randn(B, self.model.hidden_dim, dtype=torch.complex64, device=device) * 0.1
        else:
            z = z0
        
        dzdt_history = []
        z_history = []
        
        for i in range(max_iter):
            # Compute dz/dt using the cell
            dzdt = self.model.cell(z, x_const)
            dzdt_norm = torch.abs(dzdt).mean().item()
            dzdt_history.append(dzdt_norm)
            z_history.append(z.clone())
            
            # Check convergence
            if dzdt_norm < tol:
                break
            
            # Euler step
            z = z + self.model.dt * dzdt
            
            # Clamp for stability
            z = torch.complex(
                torch.clamp(z.real, -self.model.z_max, self.model.z_max),
                torch.clamp(z.imag, -self.model.z_max, self.model.z_max)
            )
        
        # Compute Jacobian eigenvalues at fixed point
        eigenvalues = self.compute_jacobian_eigenvalues(z, x_const)
        
        return {
            'fixed_point': z,
            'converged': dzdt_history[-1] < tol,
            'iterations': len(z_history),
            'dzdt_final': dzdt_history[-1],
            'dzdt_history': dzdt_history,
            'eigenvalues': eigenvalues,
            'is_stable': eigenvalues is not None and all(e.real < 0 for e in eigenvalues),
        }
    
    def compute_jacobian_eigenvalues(self, z: torch.Tensor, x: torch.Tensor,
                                     eps: float = 1e-5) -> Optional[List[complex]]:
        """
        Compute eigenvalues of the Jacobian at a given state.
        
        J_ij = ∂(dz_i/dt) / ∂z_j
        
        Uses numerical differentiation (finite differences).
        
        Args:
            z: State (B, hidden_dim)
            x: Input (B, input_dim)
            eps: Perturbation for finite differences
            
        Returns:
            List of eigenvalues (complex numbers), or None if computation fails
        """
        B, N = z.shape
        if B > 1:
            z = z[0:1]
            x = x[0:1]
        
        dzdt_base = self.model.cell(z, x)
        
        # Build Jacobian matrix using finite differences
        J = torch.zeros(N, N, dtype=torch.complex64, device=z.device)
        
        for j in range(N):
            # Perturb real part
            z_pert = z.clone()
            z_pert[:, j] = z_pert[:, j] + eps
            dzdt_pert = self.model.cell(z_pert, x)
            J[:, j] = (dzdt_pert - dzdt_base).squeeze(0) / eps
        
        # Compute eigenvalues
        try:
            eigenvalues = torch.linalg.eigvals(J)
            return eigenvalues.tolist()
        except Exception as e:
            print(f"Warning: Could not compute eigenvalues: {e}")
            return None
    
    def analyze_stability(self, x_const: torch.Tensor, n_trajectories: int = 10,
                         n_steps: int = 100) -> Dict:
        """
        Analyze stability by perturbing fixed point.
        
        Args:
            x_const: Constant input
            n_trajectories: Number of perturbed trajectories
            n_steps: Number of steps to simulate
            
        Returns:
            Dictionary with stability analysis results
        """
        # Find fixed point
        fp_result = self.find_fixed_point(x_const)
        
        if not fp_result['converged']:
            return {'error': 'Fixed point not found', **fp_result}
        
        z_star = fp_result['fixed_point']
        
        # Perturb and simulate
        perturbations = np.logspace(-6, -1, n_trajectories)
        divergence_rates = []
        
        for pert in perturbations:
            # Random perturbation
            z_pert = z_star + (torch.randn_like(z_star) * pert)

            # Simulate
            distances = []
            z = z_pert
            for _ in range(n_steps):
                dzdt = self.model.cell(z, x_const)
                z = z + self.model.dt * dzdt
                distance = torch.abs(z - z_star).mean().item()
                distances.append(distance)
            
            # Estimate divergence rate (linear fit in log space)
            log_distances = np.log(distances + 1e-10)
            try:
                slope = np.polyfit(np.arange(len(distances)), log_distances, 1)[0]
                divergence_rates.append(slope)
            except:
                divergence_rates.append(0)
        
        return {
            'fixed_point': z_star,
            'perturbations': perturbations,
            'divergence_rates': divergence_rates,
            'is_stable': all(r < 0 for r in divergence_rates),
            'max_divergence_rate': max(divergence_rates),
        }
    
    def plot_phase_portrait(self, x_const: torch.Tensor, 
                           n_trajectories: int = 20, n_steps: int = 100,
                           save_path: str = 'phase_portrait.png'):
        """
        Plot phase portrait showing multiple trajectories.
        
        Args:
            x_const: Constant input
            n_trajectories: Number of trajectories to plot
            n_steps: Number of steps per trajectory
            save_path: Path to save figure
        """
        # Find fixed point
        fp_result = self.find_fixed_point(x_const)
        z_star = fp_result['fixed_point']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Generate random initial conditions
        B = n_trajectories
        z0_list = torch.randn(B, self.model.hidden_dim, dtype=torch.complex64, 
                              device=x_const.device) * 2
        
        all_trajectories = []
        
        for i in range(B):
            z = z0_list[i:i+1]
            trajectory = [z.clone()]

            for _ in range(n_steps):
                dzdt = self.model.cell(z, x_const)
                z = z + self.model.dt * dzdt
                trajectory.append(z.clone())
            
            trajectory = torch.cat(trajectory, dim=0)  # (n_steps+1, hidden_dim)
            all_trajectories.append(trajectory)
        
        # Plot first 2 neurons (Re vs Im)
        ax = axes[0]
        for i, traj in enumerate(all_trajectories[:min(10, n_trajectories)]):
            traj_np = traj[:, :2].cpu().numpy()
            ax.plot(traj_np.real, traj_np.imag, alpha=0.5, linewidth=1)
            ax.scatter(traj_np.real[0], traj_np.imag[0], c='C0', s=10, alpha=0.5)
        
        # Mark fixed point
        ax.scatter(z_star.real[0, :2].cpu(), z_star.imag[0, :2].cpu(), 
                  c='red', s=100, marker='x', label='Fixed Point')
        
        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Im(z)')
        ax.set_title('Phase Portrait (First 2 Neurons)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot distance from fixed point over time
        ax2 = axes[1]
        for i, traj in enumerate(all_trajectories[:min(10, n_trajectories)]):
            distances = torch.abs(traj - z_star).mean(dim=1).cpu().numpy()
            ax2.semilogy(distances, alpha=0.5, linewidth=1)
        
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Distance from Fixed Point')
        ax2.set_title('Convergence to Fixed Point')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Phase portrait saved to '{save_path}'")
        plt.close()
    
    def compute_lyapunov_exponent(self, x_const: torch.Tensor, 
                                  n_steps: int = 500) -> float:
        """
        Compute the largest Lyapunov exponent (measure of chaos).
        
        Positive exponent indicates chaos.
        
        Args:
            x_const: Constant input
            n_steps: Number of steps
            
        Returns:
            Largest Lyapunov exponent estimate
        """
        # Find fixed point
        fp_result = self.find_fixed_point(x_const)
        z_star = fp_result['fixed_point']
        
        # Start with small perturbation
        z1 = z_star.clone()
        z2 = z_star + 1e-8
        
        lyapunov_sum = 0
        n_lyapunov = 0
        
        for _ in range(n_steps):
            dzdt1 = self.model.cell(z1, x_const)
            dzdt2 = self.model.cell(z2, x_const)

            z1 = z1 + self.model.dt * dzdt1
            z2 = z2 + self.model.dt * dzdt2
            
            # Distance between trajectories
            d = torch.abs(z2 - z1).mean().item()
            
            if d > 1e-4:  # Renormalize if too large
                lyapunov_sum += np.log(d / 1e-8)
                n_lyapunov += 1
                z2 = z1 + (z2 - z1) * (1e-8 / d)
        
        if n_lyapunov > 0:
            return lyapunov_sum / n_lyapunov / self.dt
        return 0.0


def plot_bifurcation_diagram(model: torch.nn.Module, param_name: str,
                            param_range: Tuple[float, float], n_values: int = 50,
                            x_const: torch.Tensor = None, n_steps: int = 200,
                            save_path: str = 'bifurcation_diagram.png'):
    """
    Plot bifurcation diagram by varying a parameter.
    
    Args:
        model: TwistorLMT model
        param_name: Name of parameter to vary
        param_range: (min, max) range for parameter
        n_values: Number of parameter values to test
        x_const: Constant input (default: zeros)
        n_steps: Number of steps per simulation
        save_path: Path to save figure
    """
    if x_const is None:
        x_const = torch.zeros(1, model.input_dim, device=next(model.parameters()).device)
    
    param_values = np.linspace(param_range[0], param_range[1], n_values)
    
    # Store final |z| values for each parameter
    final_z_norms = []
    
    original_param = getattr(model, param_name).item() if hasattr(getattr(model, param_name), 'item') else None
    
    for p in param_values:
        # Set parameter
        if hasattr(model, param_name):
            setattr(model, param_name, torch.tensor(p) if original_param is None else p)
        
        # Simulate
        z = torch.zeros(1, model.hidden_dim, dtype=torch.complex64, device=x_const.device)
        for _ in range(n_steps):
            dzdt = model.cell(z, x_const)
            z = z + model.dt * dzdt
        
        final_z_norms.append(torch.abs(z).mean().item())
    
    # Restore original parameter
    if original_param is not None:
        setattr(model, param_name, original_param)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, final_z_norms, 'b-', linewidth=2)
    plt.xlabel(param_name)
    plt.ylabel('|z| (final)')
    plt.title(f'Bifurcation Diagram: {param_name}')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150)
    print(f"Bifurcation diagram saved to '{save_path}'")
    plt.close()
