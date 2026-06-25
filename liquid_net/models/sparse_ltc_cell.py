"""
LTC Cell with Sparse Connectivity and Multi-Scale Time Constants.

Features:
1. Sparse connectivity via learnable masks (L1 regularization for sparsity)
2. Multi-scale time constants τᵢ for each neuron

Mathematical formulation:
    τᵢ(z) = sigmoid(w_τᵢ · |z| + b_τᵢ) + ε
    
    dz/dt = (-z + W_sparse·tanh(z) + U·x + b) / τ(z)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseLTCCell(nn.Module):
    """
    Liquid Time-Constant Cell with:
    1. Sparse connectivity via learnable masks
    2. Multi-scale time constants (one τᵢ per neuron)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,
        sparsity: float = 0.3,
        use_multi_scale_tau: bool = True
    ):
        """
        Initialize Sparse LTC Cell.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden state
            sparsity: Target sparsity ratio (0.0-1.0). 0.3 = 30% connections active
            use_multi_scale_tau: If True, each neuron has independent τᵢ
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity = sparsity
        self.use_multi_scale_tau = use_multi_scale_tau
        
        # ============ Weight matrices ============
        # W_real: recurrent weight for real part
        self.W_real = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # W_imag: recurrent weight for imag part
        self.W_imag = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # U: input weight (shared)
        self.U = nn.Linear(input_dim, hidden_dim)
        
        # ============ Sparse masks (learnable) ============
        # Binary masks for sparse connectivity
        # Using straight-through estimator for discrete masks
        self.mask_real = nn.Parameter(self._init_sparse_mask(hidden_dim, hidden_dim))
        self.mask_imag = nn.Parameter(self._init_sparse_mask(hidden_dim, hidden_dim))
        
        # ============ Multi-scale time constants ============
        if use_multi_scale_tau:
            # Each neuron has its own τᵢ parameter
            # tau_i = sigmoid(gain_i * |z_i| + bias_i) + epsilon
            self.tau_gain = nn.Parameter(torch.ones(hidden_dim))  # gain per neuron
            self.tau_bias = nn.Parameter(torch.zeros(hidden_dim))  # bias per neuron
        else:
            # Shared τ for all neurons
            self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        
        # ============ Bias terms ============
        self.b_real = nn.Parameter(torch.zeros(hidden_dim))
        self.b_imag = nn.Parameter(torch.zeros(hidden_dim))
        
        # Initialize weights
        self._init_weights()
    
    def _init_sparse_mask(self, rows: int, cols: int) -> torch.Tensor:
        """
        Initialize sparse mask.
        
        Strategy: Start with target sparsity, use relaxed伯努利 distribution
        """
        # Start with target sparsity
        mask = torch.zeros(rows, cols)
        n_keep = int(rows * cols * (1 - self.sparsity))
        
        # Random selection
        indices = torch.randperm(rows * cols)[:n_keep]
        mask.view(-1)[indices] = 1.0
        
        return mask
    
    def _apply_sparse_mask(self, weight: nn.Parameter) -> torch.Tensor:
        """
        Apply sparse mask to weight matrix.
        
        Uses hard thresholding during forward pass:
        W_sparse = W * (mask > 0.5)
        """
        # Hard threshold during forward
        mask_binary = (self.mask_real > 0.5).float()
        return weight * mask_binary
    
    def _init_weights(self):
        """Initialize weights with stability considerations."""
        # Recurrent weights
        nn.init.orthogonal_(self.W_real.weight, gain=0.5)
        nn.init.orthogonal_(self.W_imag.weight, gain=0.5)
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        
        # Initialize masks with target sparsity
        with torch.no_grad():
            self.mask_real.data = self._init_sparse_mask(
                self.hidden_dim, self.hidden_dim
            ).to(self.mask_real.device)
            self.mask_imag.data = self._init_sparse_mask(
                self.hidden_dim, self.hidden_dim
            ).to(self.mask_imag.device)
        
        # Tau network
        if not self.use_multi_scale_tau:
            nn.init.orthogonal_(self.W_tau.weight, gain=0.1)
            nn.init.zeros_(self.W_tau.bias)
    
    def compute_tau(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute state-dependent time constants.
        
        Multi-scale version: τᵢ = sigmoid(gainᵢ * |zᵢ| + biasᵢ) + ε
        
        This gives each neuron its own adaptive time constant.
        
        Args:
            z: Complex state (B, hidden_dim)
        
        Returns:
            tau: Time constants (B, hidden_dim), one per neuron
        """
        # Get modulus |z| for each neuron
        z_mod = torch.abs(z)  # (B, hidden_dim)
        
        if self.use_multi_scale_tau:
            # Per-neuron time constant
            # τᵢ = sigmoid(gainᵢ * |zᵢ| + biasᵢ) + ε
            tau = torch.sigmoid(
                self.tau_gain.unsqueeze(0) * z_mod + self.tau_bias.unsqueeze(0)
            ) + 1e-6
            
            # Clamp to reasonable range [0.01, 10]
            tau = torch.clamp(tau, min=0.01, max=10.0)
        else:
            # Shared τ (original version)
            tau = torch.sigmoid(self.W_tau(z_mod)) + 1e-6
        
        return tau
    
    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative dz/dt.
        
        Dynamics:
            dz/dt = (-z + W_sparse·tanh(z) + U·x + b) / τ(z)
        
        Where:
            - W_sparse = W * mask (sparse connectivity)
            - τᵢ is per-neuron (multi-scale)
        
        Args:
            z: Complex hidden state (B, hidden_dim)
            x: Input (B, input_dim)
        
        Returns:
            dzdt: Time derivative (B, hidden_dim), complex
        """
        # Extract parts
        z_real = z.real
        z_imag = z.imag
        
        # Apply tanh
        tanh_real = torch.tanh(z_real)
        tanh_imag = torch.tanh(z_imag)
        
        # ============ Apply sparse weights ============
        # Apply mask to weights (multiplication, gradients flow through mask)
        # This creates learnable sparsity
        W_real_sparse = self.W_real.weight * self.mask_real
        W_imag_sparse = self.W_imag.weight * self.mask_imag
        
        # Compute recurrent term with sparse weights
        W_tanh_real = F.linear(tanh_real, W_real_sparse)
        W_tanh_imag = F.linear(tanh_imag, W_imag_sparse)
        
        # Input term
        Ux = self.U(x)
        
        # ============ Compute derivatives ============
        dz_real = -z_real + W_tanh_real + Ux + self.b_real
        dz_imag = -z_imag + W_tanh_imag + Ux + self.b_imag
        
        # ============ Multi-scale τ ============
        tau = self.compute_tau(z)
        
        # Divide by tau
        dzdt = torch.complex(dz_real / tau, dz_imag / tau)
        
        return dzdt
    
    def get_sparsity(self) -> dict:
        """
        Get current sparsity statistics.
        
        Returns:
            dict with sparsity info
        """
        mask_real_binary = (self.mask_real > 0.5).float()
        mask_imag_binary = (self.mask_imag > 0.5).float()
        
        return {
            'W_real_active': mask_real_binary.sum().item(),
            'W_real_total': mask_real_binary.numel(),
            'W_real_sparsity': 1.0 - mask_real_binary.mean().item(),
            'W_imag_active': mask_imag_binary.sum().item(),
            'W_imag_total': mask_imag_binary.numel(),
            'W_imag_sparsity': 1.0 - mask_imag_binary.mean().item(),
        }


class SparseLTTLNCell(nn.Module):
    """
    Sparse Liquid Time-Constant LTC Cell.
    
    Simplified version with sparse connectivity only.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 16, sparsity: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Weights
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.U = nn.Linear(input_dim, hidden_dim)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        
        # Sparse mask
        self.mask = nn.Parameter(self._create_mask(hidden_dim, hidden_dim, sparsity))
        
        # Time constant (single shared τ for simplicity)
        self.W_tau = nn.Linear(hidden_dim, hidden_dim)
        
        self._init_weights()
    
    def _create_mask(self, rows: int, cols: int, sparsity: float) -> torch.Tensor:
        mask = torch.zeros(rows, cols)
        n_keep = int(rows * cols * (1 - sparsity))
        indices = torch.randperm(rows * cols)[:n_keep]
        mask.view(-1)[indices] = 1.0
        return mask
    
    def _init_weights(self):
        nn.init.orthogonal_(self.W.weight, gain=0.5)
        nn.init.orthogonal_(self.U.weight, gain=0.5)
        nn.init.zeros_(self.W_tau.bias)
    
    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward for real-valued states (standard LTC).
        
        Args:
            h: Hidden state (B, hidden_dim)
            x: Input (B, input_dim)
        
        Returns:
            dhdt: Time derivative (B, hidden_dim)
        """
        # Apply sparse weight
        W_sparse = self.W.weight * self.mask
        
        # Compute derivative
        dh = -h + F.linear(torch.tanh(h), W_sparse) + self.U(x) + self.b
        
        # Time constant
        tau = torch.sigmoid(self.W_tau(h)) + 1e-6
        
        return dh / tau


# Alias for backward compatibility
LTCCell = SparseLTCCell
