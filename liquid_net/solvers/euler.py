"""Euler solver."""
import torch

def euler_step(z: torch.Tensor, dzdt: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """Perform a single Euler integration step."""
    return z + dt * dzdt
