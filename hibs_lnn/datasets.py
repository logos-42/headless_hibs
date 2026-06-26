"""
Twistor-LMT Dataset Generators
=============================
Synthetic datasets for testing TwistorLMT models.

Datasets:
- Sine Wave: Simple baseline
- Lorenz System: Chaotic dynamics
- Mackey-Glass: Time-delay system
- Van der Pol: Oscillator
"""

import torch
import numpy as np
from typing import Tuple, Optional


def generate_lorenz_dataset(
    n_samples: int = 1000,
    seq_len: int = 50,
    dt: float = 0.01,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    noise_std: float = 0.1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Lorenz system dataset.

    Lorenz equations:
        dx/dt = sigma * (y - x)
        dy/dt = x * (rho - z) - y
        dz/dt = x * y - beta * z

    Chaotic behavior for rho > 24.74

    Args:
        n_samples: Number of sequences
        seq_len: Length of each sequence
        dt: Time step
        sigma, rho, beta: Lorenz parameters
        noise_std: Noise standard deviation
        device: Device

    Returns:
        X: Input sequences (n_samples, seq_len, 3)
        y: Target sequences (n_samples, seq_len, 3)
    """
    X = []
    y = []

    for _ in range(n_samples):
        # Random initial condition
        x0 = np.random.uniform(-20, 20)
        y0 = np.random.uniform(-30, 30)
        z0 = np.random.uniform(0, 50)

        # Generate trajectory
        states = []
        x, y_val, z = x0, y0, z0

        for i in range(seq_len + 1):
            states.append([x, y_val, z])

            # Lorenz dynamics
            dx = sigma * (y_val - x) * dt
            dy = (x * (rho - z) - y_val) * dt
            dz = (x * y_val - beta * z) * dt

            x += dx
            y_val += dy
            z += dz

        states = np.array(states)

        # Add noise
        states += np.random.randn(*states.shape) * noise_std

        # Input: first 3 dims, target: next step
        X.append(states[:-1])  # (seq_len, 3)
        y.append(states[1:])  # (seq_len, 3)

    X = torch.FloatTensor(np.stack(X)).to(device)
    y = torch.FloatTensor(np.stack(y)).to(device)

    return X, y


def generate_mackey_glass_dataset(
    n_samples: int = 1000,
    seq_len: int = 50,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n0: float = 1.0,
    dt: float = 0.1,
    noise_std: float = 0.01,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Mackey-Glass time series.

    Mackey-Glass equation:
        dx/dt = beta * x(t - tau) / (1 + x(t - tau)^n0) - gamma * x(t)

    Chaotic for tau > 25

    Args:
        n_samples: Number of sequences
        seq_len: Length of each sequence
        tau: Delay parameter
        beta, gamma, n0: Equation parameters
        dt: Time step
        noise_std: Noise standard deviation
        device: Device

    Returns:
        X: Input sequences (n_samples, seq_len, 1)
        y: Target sequences (n_samples, seq_len, 1)
    """
    X = []
    y = []

    # Total steps needed: warmup + seq_len + 1 (for target)
    total_steps = tau + 100 + seq_len + 1

    for _ in range(n_samples):
        # Initialize history
        x = np.zeros(total_steps)
        x[: tau + 100] = n0 + np.random.randn(tau + 100) * 0.01

        # Generate trajectory
        for i in range(tau + 100, total_steps - 1):
            x[i + 1] = x[i] + dt * (
                beta * x[i - tau] / (1 + x[i - tau] ** n0) - gamma * x[i]
            )

        # Add noise and extract sequences
        x_noisy = x[tau + 100 :] + np.random.randn(seq_len + 1) * noise_std

        X.append(x_noisy[:-1].reshape(-1, 1))
        y.append(x_noisy[1:].reshape(-1, 1))

    X = torch.FloatTensor(np.stack(X)).to(device)
    y = torch.FloatTensor(np.stack(y)).to(device)

    return X, y


def generate_van_der_pol_dataset(
    n_samples: int = 1000,
    seq_len: int = 50,
    mu: float = 1.0,
    dt: float = 0.05,
    noise_std: float = 0.1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate Van der Pol oscillator dataset.

    Van der Pol equation:
        d²x/dt² = mu * (1 - x²) * dx/dt - x

    Args:
        n_samples: Number of sequences
        seq_len: Length of each sequence
        mu: Damping parameter
        dt: Time step
        noise_std: Noise standard deviation
        device: Device

    Returns:
        X: Input sequences (n_samples, seq_len, 2)
        y: Target sequences (n_samples, seq_len, 2)
    """
    X = []
    y = []

    for _ in range(n_samples):
        # Random initial condition
        x0 = np.random.uniform(-3, 3)
        v0 = np.random.uniform(-3, 3)

        # Generate trajectory
        states = []
        x, v = x0, v0

        for _ in range(seq_len + 1):
            states.append([x, v])

            # Van der Pol (convert to first order)
            dx = v * dt
            dv = (mu * (1 - x**2) * v - x) * dt

            x += dx
            v += dv

        states = np.array(states)
        states += np.random.randn(*states.shape) * noise_std

        X.append(states[:-1])
        y.append(states[1:])

    X = torch.FloatTensor(np.stack(X)).to(device)
    y = torch.FloatTensor(np.stack(y)).to(device)

    return X, y


def generate_sine_dataset(
    n_samples: int = 1000,
    seq_len: int = 50,
    input_dim: int = 2,
    noise_std: float = 0.1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate sine wave prediction dataset.

    Input: [sin(t), cos(t)]
    Target: sin(t + dt)

    Args:
        n_samples: Number of sequences
        seq_len: Length of each sequence
        input_dim: Input dimension
        noise_std: Noise standard deviation
        device: Device

    Returns:
        X: Input sequences
        y: Target sequences
    """
    X = []
    y = []

    for _ in range(n_samples):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)

        t = np.linspace(0, 4 * np.pi, seq_len + 1)
        signal = np.sin(freq * t + phase)
        signal += np.random.randn(len(t)) * noise_std

        sin_component = signal[:-1]
        cos_component = (
            np.cos(freq * t[:-1] + phase) + np.random.randn(seq_len) * noise_std
        )

        x_seq = np.stack([sin_component, cos_component], axis=-1)
        y_seq = signal[1:].reshape(-1, 1)

        X.append(x_seq)
        y.append(y_seq)

    X = torch.FloatTensor(np.stack(X)).to(device)
    y = torch.FloatTensor(np.stack(y)).to(device)

    return X, y


def generate_multi_step_dataset(
    n_samples: int = 1000,
    seq_len: int = 50,
    pred_steps: int = 5,
    noise_std: float = 0.1,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate multi-step prediction dataset.

    Input: sequence
    Target: sequence shifted by pred_steps

    Args:
        n_samples: Number of sequences
        seq_len: Input sequence length
        pred_steps: Number of steps to predict ahead
        noise_std: Noise standard deviation
        device: Device

    Returns:
        X: Input sequences (n_samples, seq_len, 1)
        y: Target sequences (n_samples, pred_steps, 1)
    """
    X = []
    y = []

    for _ in range(n_samples):
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)

        t = np.linspace(0, 4 * np.pi, seq_len + pred_steps + 1)
        signal = np.sin(freq * t + phase)
        signal += np.random.randn(len(t)) * noise_std

        X.append(signal[:seq_len].reshape(-1, 1))
        y.append(signal[pred_steps : seq_len + pred_steps].reshape(-1, 1))

    X = torch.FloatTensor(np.stack(X)).to(device)
    y = torch.FloatTensor(np.stack(y)).to(device)

    return X, y


def create_dataset(
    name: str = "lorenz", n_samples: int = 1000, seq_len: int = 50, **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Factory function to create datasets.

    Args:
        name: Dataset name ('lorenz', 'mackey_glass', 'van_der_pol', 'sine')
        n_samples: Number of samples
        seq_len: Sequence length
        **kwargs: Additional arguments

    Returns:
        X, y: Input and target tensors
    """
    generators = {
        "lorenz": generate_lorenz_dataset,
        "mackey_glass": generate_mackey_glass_dataset,
        "van_der_pol": generate_van_der_pol_dataset,
        "sine": generate_sine_dataset,
    }

    if name not in generators:
        raise ValueError(
            f"Unknown dataset: {name}. Available: {list(generators.keys())}"
        )

    return generators[name](n_samples=n_samples, seq_len=seq_len, **kwargs)
