"""
Loss functions for Twistor-inspired Liquid Neural Network.

Implements MSE loss with stability regularization.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


def twistor_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    states: Optional[List[torch.Tensor]] = None,
    stability_weight: float = 0.01
) -> torch.Tensor:
    """
    Compute loss for Twistor LMT with stability regularization.
    
    Loss = MSE + stability_weight * ||dz/dt||^2
    
    Args:
        y_pred: Predicted output (T, B, output_dim)
        y_true: True output (T, B, output_dim)
        states: Hidden states for stability regularization (optional)
        stability_weight: Weight for stability term (default: 0.01)
    
    Returns:
        loss: Total loss
    """
    # MSE loss
    mse_loss = F.mse_loss(y_pred, y_true)
    
    # Stability regularization: ||dz/dt||^2
    if states is not None and len(states) > 1:
        dzdt_norm_sq = 0.0
        for t in range(len(states) - 1):
            # Approximate dz/dt from consecutive states
            dzdt = states[t + 1] - states[t]
            dzdt_norm_sq += (dzdt.abs() ** 2).mean()
        stability_loss = dzdt_norm_sq / (len(states) - 1)
    else:
        stability_loss = torch.tensor(0.0, device=y_pred.device)
    
    # Total loss
    loss = mse_loss + stability_weight * stability_loss
    
    return loss
