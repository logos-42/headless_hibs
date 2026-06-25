"""
Training utilities for Twistor-inspired Liquid Neural Network.

Includes dataset generation and training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict

from ..models.liquid_net import TwistorLMT
from .loss import twistor_loss


def generate_sine_dataset(
    n_samples: int = 1000,
    seq_len: int = 50,
    input_dim: int = 2,
    noise_std: float = 0.1,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic sine wave prediction dataset.
    
    Input: sine waves with random frequencies and phases
    Target: next values in the sequence
    
    Args:
        n_samples: Number of sequences
        seq_len: Length of each sequence
        input_dim: Input dimension (default 2: sin + cos)
        noise_std: Standard deviation of noise
        device: Device to place tensors on
    
    Returns:
        X: Input sequences (n_samples, seq_len, input_dim)
        y: Target sequences (n_samples, seq_len, 1)
    """
    X = []
    y = []
    
    for _ in range(n_samples):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Time steps
        t = np.linspace(0, 4 * np.pi, seq_len + 1)  # +1 for target
        
        # Generate sine wave
        signal = np.sin(freq * t + phase)
        
        # Add noise
        signal += np.random.randn(len(t)) * noise_std
        
        # Input: sin and cos components
        sin_component = signal[:-1]
        cos_component = np.cos(freq * t[:-1] + phase) + np.random.randn(seq_len) * noise_std
        
        x_seq = np.stack([sin_component, cos_component], axis=-1)  # (seq_len, 2)
        
        # Target: next value (prediction task)
        y_seq = signal[1:].reshape(-1, 1)  # (seq_len, 1)
        
        X.append(x_seq)
        y.append(y_seq)
    
    X = torch.FloatTensor(np.stack(X)).to(device)  # (n_samples, seq_len, input_dim)
    y = torch.FloatTensor(np.stack(y)).to(device)  # (n_samples, seq_len, 1)
    
    return X, y


def plot_training_results(history: Dict[str, list]):
    """Plot training curves."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mse'], label='Train MSE')
    plt.plot(history['val_mse'], label='Val MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Mean Squared Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("Training curves saved to 'training_curves.png'")
    plt.close()


def plot_predictions(
    model: TwistorLMT,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    device: str,
    n_samples: int = 5
):
    """Plot sample predictions."""
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        x_test = X_test[:n_samples].transpose(0, 1)
        y_pred = model(x_test).transpose(0, 1)  # (n_samples, seq_len, 1)
        y_true = y_test[:n_samples]
    
    plt.figure(figsize=(14, 8))
    
    for i in range(n_samples):
        plt.subplot(n_samples, 1, i + 1)
        plt.plot(y_true[i].cpu().numpy().flatten(), 'o-', label='True', alpha=0.7, markersize=4)
        plt.plot(y_pred[i].cpu().numpy().flatten(), 's-', label='Predicted', alpha=0.7, markersize=4)
        plt.ylabel('Amplitude')
        plt.title(f'Sample {i + 1}')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    
    plt.xlabel('Time Step')
    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150)
    print("Sample predictions saved to 'predictions.png'")
    plt.close()


def train_twistor_LMT(
    n_epochs: int = 200,
    batch_size: int = 32,
    lr: float = 1e-2,
    hidden_dim: int = 16,
    stability_weight: float = 0.01,
    device: str = 'cpu'
) -> Tuple[TwistorLMT, Dict[str, list]]:
    """
    Train the Twistor LMT on sine wave prediction.
    
    Args:
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        hidden_dim: Hidden dimension
        stability_weight: Weight for ||dz/dt||^2 regularization
        device: Device to train on
    
    Returns:
        model: Trained model
        history: Training history
    """
    print("=" * 60)
    print("Twistor-inspired Liquid Neural Network Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Stability weight: {stability_weight}")
    print()
    
    # Generate dataset
    print("Generating synthetic sine wave dataset...")
    X_train, y_train = generate_sine_dataset(n_samples=1000, seq_len=50, device=device)
    X_val, y_val = generate_sine_dataset(n_samples=200, seq_len=50, device=device)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    print(f"Sequence length: {X_train.shape[1]}, Input dim: {X_train.shape[2]}")
    print()
    
    # Initialize model
    model = TwistorLMT(
        input_dim=X_train.shape[2],
        hidden_dim=hidden_dim,
        output_dim=1
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    
    # Training loop
    n_batches = len(X_train) // batch_size
    history = {'train_loss': [], 'val_loss': [], 'train_mse': [], 'val_mse': []}
    
    print("Starting training...")
    print("-" * 60)
    
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_mse = 0.0
        
        # Shuffle data
        perm = torch.randperm(len(X_train), device=device)
        X_train = X_train[perm]
        y_train = y_train[perm]
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            # Get batch (T, B, input_dim) format
            x_batch = X_train[start_idx:end_idx].transpose(0, 1)  # (seq_len, batch, input_dim)
            y_batch = y_train[start_idx:end_idx].transpose(0, 1)  # (seq_len, batch, 1)
            
            optimizer.zero_grad()
            
            # Forward pass with states for stability loss
            y_pred, states = model(x_batch, return_states=True)
            
            # Compute loss
            loss = twistor_loss(y_pred, y_batch, states, stability_weight)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_mse += F.mse_loss(y_pred, y_batch).item()
        
        # Average losses
        avg_train_loss = epoch_loss / n_batches
        avg_train_mse = epoch_mse / n_batches
        history['train_loss'].append(avg_train_loss)
        history['train_mse'].append(avg_train_mse)
        
        # Validation
        model.eval()
        with torch.no_grad():
            x_val = X_val.transpose(0, 1)
            y_val_t = y_val.transpose(0, 1)
            y_val_pred = model(x_val)
            val_mse = F.mse_loss(y_val_pred, y_val_t).item()
            history['val_loss'].append(val_mse)  # Use MSE as val loss
            history['val_mse'].append(val_mse)
        
        # Update learning rate
        scheduler.step(avg_train_loss)
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:4d}/{n_epochs}: "
                  f"Train Loss = {avg_train_loss:.6f}, "
                  f"Train MSE = {avg_train_mse:.6f}, "
                  f"Val MSE = {val_mse:.6f}, "
                  f"LR = {optimizer.param_groups[0]['lr']:.6f}")
    
    print("-" * 60)
    print(f"Training complete! Final Val MSE: {history['val_mse'][-1]:.6f}")
    print()
    
    # Plot results
    plot_training_results(history)
    plot_predictions(model, X_val, y_val, device)
    
    return model, history
