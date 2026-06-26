"""
Twistor-LMT Decoder Module
==========================
Multi-space decoding from complex twistor state z.

Decoding modes:
- vector: z.real → ℝ^n
- tensor: z → v ⊗ v (outer product)
- scalar: |z| (norm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TwistorDecoder(nn.Module):
    """
    Multi-space decoder from twistor state.

    Decodes complex state z ∈ ℂ^n to:
    - vector: v = Re(z) ∈ ℝ^n
    - tensor: T = v ⊗ v ∈ ℝ^(n×n)
    - scalar: |z|
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        use_tensor: bool = False,
        tensor_hidden: int = 32,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_tensor = use_tensor

        self.vector_decoder = nn.Linear(hidden_dim, output_dim)

        if use_tensor:
            self.tensor_decoder = nn.Linear(hidden_dim * hidden_dim, output_dim)

    def decode_vector(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode to vector: v = Re(z)

        Args:
            z: Complex state (B, hidden_dim)

        Returns:
            v: Vector (B, hidden_dim)
        """
        return z.real

    def decode_tensor(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode to second-order tensor via outer product.

        z → v (real part) → v ⊗ v

        Args:
            z: Complex state (B, hidden_dim)

        Returns:
            tensor: Second-order tensor (B, hidden_dim, hidden_dim)
        """
        v = z.real
        tensor = torch.einsum("bi,bj->bij", v, v)
        return tensor

    def decode_tensor_flat(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode to flattened tensor.

        Args:
            z: Complex state (B, hidden_dim)

        Returns:
            tensor_flat: Flattened tensor (B, hidden_dim * hidden_dim)
        """
        tensor = self.decode_tensor(z)
        return tensor.view(tensor.size(0), -1)

    def decode_scalar(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode to scalar: |z| (modulus/energy)

        Args:
            z: Complex state (B, hidden_dim)

        Returns:
            scalar: Modulus (B, hidden_dim)
        """
        return torch.abs(z)

    def forward(self, z: torch.Tensor, mode: str = "vector") -> torch.Tensor:
        """
        Decode with specified mode.

        Args:
            z: Complex state
            mode: 'vector', 'tensor', 'scalar', or 'both'

        Returns:
            output: Decoded tensor
        """
        if mode == "vector":
            v = self.decode_vector(z)
            return self.vector_decoder(v)

        elif mode == "tensor":
            tensor_flat = self.decode_tensor_flat(z)
            return self.tensor_decoder(tensor_flat)

        elif mode == "scalar":
            s = self.decode_scalar(z)
            return self.vector_decoder(s)

        elif mode == "both":
            v = self.decode_vector(z)
            tensor_flat = self.decode_tensor_flat(z)
            vec_out = self.vector_decoder(v)
            tens_out = self.tensor_decoder(tensor_flat)
            return vec_out + 0.1 * tens_out

        else:
            raise ValueError(f"Unknown mode: {mode}")


class TensorTwistorDecoder(nn.Module):
    """
    Advanced decoder with learnable tensor transformations.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        tensor_rank: int = 2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tensor_rank = tensor_rank

        self.vector_proj = nn.Linear(hidden_dim, hidden_dim)

        tensor_dim = hidden_dim**tensor_rank
        self.tensor_proj = nn.Sequential(
            nn.Linear(tensor_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(self.vector_proj(z.real))

        if self.tensor_rank == 2:
            tensor = torch.einsum("bi,bj->bij", v, v)
        elif self.tensor_rank == 3:
            tensor = torch.einsum("bi,bj,bk->bijk", v, v, v)
        else:
            raise ValueError(f"Unsupported rank: {self.tensor_rank}")

        tensor_flat = tensor.view(tensor.size(0), -1)
        return self.tensor_proj(tensor_flat)


def create_decoder(
    hidden_dim: int, output_dim: int, decoder_type: str = "simple", **kwargs
) -> nn.Module:
    """
    Factory function to create decoder.

    Args:
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        decoder_type: 'simple', 'tensor', or 'advanced'

    Returns:
        decoder: Decoder module
    """
    if decoder_type == "simple":
        return TwistorDecoder(hidden_dim, output_dim, use_tensor=False)

    elif decoder_type == "tensor":
        return TwistorDecoder(hidden_dim, output_dim, use_tensor=True)

    elif decoder_type == "advanced":
        return TensorTwistorDecoder(
            hidden_dim, output_dim, tensor_rank=kwargs.get("tensor_rank", 2)
        )

    else:
        raise ValueError(f"Unknown decoder type: {decoder_type}")
