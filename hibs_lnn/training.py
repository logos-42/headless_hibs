"""
Twistor-LMT Training Module
==========================
Training utilities for TwistorLMT models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import time

from .datasets import create_dataset


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.01,
    val_split: float = 0.2,
    device: str = "cpu",
    print_every: int = 10,
) -> Tuple[nn.Module, Dict]:
    """
    Train TwistorLMT model.

    Args:
        model: TwistorLMT model
        X: Input data (N, T, D_in)
        y: Target data (N, T, D_out)
        n_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        val_split: Validation split ratio
        device: Device
        print_every: Print frequency

    Returns:
        model: Trained model
        history: Training history
    """
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

    # Split data
    n = len(X)
    n_val = int(n * val_split)
    indices = torch.randperm(n)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mse": [],
        "val_mse": [],
    }

    n_batches = len(X_train) // batch_size

    print(f"Training: {len(X_train)} samples, Val: {len(X_val)} samples")
    print(f"Batches: {n_batches}, Epochs: {n_epochs}")
    print("-" * 50)

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_mse = 0.0

        # Shuffle
        perm = torch.randperm(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_start = time.time()

        for i in range(n_batches):
            start = i * batch_size
            end = start + batch_size

            x_batch = X_train[start:end].transpose(0, 1)  # (T, B, D)
            y_batch = y_train[start:end].transpose(0, 1)

            optimizer.zero_grad()

            pred = model(x_batch)
            mse_loss = F.mse_loss(pred, y_batch)

            # Add stability regularization
            if hasattr(model, "compute_dzdt"):
                stability_loss = 0.0
                z = torch.zeros(
                    batch_size, model.hidden_dim, dtype=torch.complex64, device=device
                )
                for t in range(min(x_batch.size(0), 10)):
                    dzdt = model.compute_dzdt(z, x_batch[t])
                    stability_loss += (torch.abs(dzdt) ** 2).mean()
                    z = z + model.dt * dzdt
                stability_loss = stability_loss / 10 * 0.01
                loss = mse_loss + stability_loss
            else:
                loss = mse_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_mse += mse_loss.item()

        train_loss /= n_batches
        train_mse /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            x_val = X_val.transpose(0, 1)
            y_val_t = y_val.transpose(0, 1)
            pred_val = model(x_val)
            val_mse = F.mse_loss(pred_val, y_val_t).item()
            val_loss = val_mse  # No stability on val

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)

        scheduler.step(val_mse)

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch + 1:3d}/{n_epochs} | "
                f"Train: {train_loss:.4f} ({train_mse:.4f}) | "
                f"Val: {val_loss:.4f} | "
                f"Time: {time.time() - epoch_start:.1f}s"
            )

    return model, history


def train_on_task(
    model_class,
    task: str = "lorenz",
    hidden_dim: int = 32,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.01,
    device: str = "cpu",
    **model_kwargs,
) -> Tuple[nn.Module, Dict]:
    """
    Train model on a specific task.

    Args:
        model_class: Model class (TwistorLMT, CoupledTwistorLMT, etc.)
        task: Task name ('lorenz', 'mackey_glass', 'van_der_pol', 'sine')
        hidden_dim: Hidden dimension
        n_epochs: Number of epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device
        **model_kwargs: Additional model arguments

    Returns:
        model: Trained model
        history: Training history
    """
    print(f"=" * 60)
    print(f"Training on {task.upper()} dataset")
    print(f"=" * 60)

    # Create dataset
    n_samples = 500 if task != "sine" else 1000
    seq_len = 40 if task == "sine" else 50

    X, y = create_dataset(task, n_samples=n_samples, seq_len=seq_len, device=device)

    input_dim = X.shape[2]
    output_dim = y.shape[2]

    print(f"Dataset: {X.shape} -> {y.shape}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")

    # Create model
    model = model_class(input_dim, hidden_dim, output_dim, **model_kwargs)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    model, history = train_model(
        model,
        X,
        y,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )

    # Final evaluation
    model.eval()
    with torch.no_grad():
        X_test = X[:50].transpose(0, 1)
        y_test = y[:50].transpose(0, 1)
        pred = model(X_test)
        final_mse = F.mse_loss(pred, y_test).item()

    print("-" * 60)
    print(f"Final Test MSE: {final_mse:.6f}")
    print(f"Final Test RMSE: {np.sqrt(final_mse):.6f}")

    return model, history


def plot_training_results(history: Dict, save_path: str = None):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # MSE
    axes[1].plot(history["train_mse"], label="Train")
    axes[1].plot(history["val_mse"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Mean Squared Error")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")

    plt.close()


def plot_predictions(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    n_samples: int = 5,
    save_path: str = None,
):
    """Plot prediction vs ground truth."""
    model.eval()

    with torch.no_grad():
        x = X[:n_samples].transpose(0, 1)
        pred = model(x).transpose(0, 1).cpu().numpy()
        true = y[:n_samples].cpu().numpy()

    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        axes[i].plot(true[i], "b-", label="True", linewidth=2)
        axes[i].plot(pred[i], "r--", label="Pred", linewidth=2)
        axes[i].set_ylabel(f"Sample {i + 1}")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    plt.suptitle("Predictions vs Ground Truth")

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")

    plt.close()
