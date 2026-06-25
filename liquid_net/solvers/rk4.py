"""
RK4 Integrator for Twistor-inspired Liquid Neural Network.

Runge-Kutta 4th order integrator for complex-valued ODEs.
"""

import torch
from typing import Callable, List


class RK4Integrator:
    """Runge-Kutta 4th order integrator."""
    
    def __init__(self, dt: float = 0.1):
        self.dt = dt
    
    def step(self, dzdt_func: Callable, z: torch.Tensor, 
             x: torch.Tensor, t: int = 0) -> torch.Tensor:
        """Perform a single RK4 integration step."""
        dt = self.dt
        
        k1 = dzdt_func(z, x)
        z_mid = z + dt * k1 / 2
        k2 = dzdt_func(z_mid, x)
        z_mid = z + dt * k2 / 2
        k3 = dzdt_func(z_mid, x)
        z_end = z + dt * k3
        k4 = dzdt_func(z_end, x)
        
        z_new = z + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return z_new
    
    def integrate(self, dzdt_func: Callable, z0: torch.Tensor, 
                  x_seq: torch.Tensor) -> List[torch.Tensor]:
        """Integrate over a sequence of inputs."""
        T, B, _ = x_seq.shape
        z = z0
        states = [z]
        
        for t in range(T):
            z = self.step(dzdt_func, z, x_seq[t], t)
            states.append(z)
        
        return states
