"""
Twistor-LMT Visualization Module
==================================
Phase space visualization and analysis tools.

Features:
- Phase space trajectory plotting
- Vector field visualization
- Tau evolution tracking
- Stability analysis plots
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple, Dict
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend


def plot_phase_space_2d(
    z_trajectory: torch.Tensor,
    fixed_points: Optional[List[torch.Tensor]] = None,
    title: str = "Phase Space Trajectory",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot 2D phase space (first two dimensions).

    Args:
        z_trajectory: State trajectory (T, B, hidden) or (T, hidden)
        fixed_points: List of fixed points to mark
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    z_np = z_trajectory.detach().cpu().numpy()

    if z_np.ndim == 3:
        # Multiple trajectories
        z_np = z_np[:, 0, :2]  # First batch, first 2 dims
    elif z_np.ndim == 4:
        z_np = z_np[:, 0, 0, :2]
    else:
        z_np = z_np[:, :2]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot real part
    ax.plot(
        z_np[:, 0].real, z_np[:, 1].real, "b-", linewidth=1, alpha=0.7, label="Real"
    )
    ax.plot(z_np[:, 0].real[-1], z_np[:, 1].real[-1], "bo", markersize=10, label="End")

    # Plot imaginary part
    ax.plot(
        z_np[:, 0].imag, z_np[:, 1].imag, "r--", linewidth=1, alpha=0.5, label="Imag"
    )

    # Mark fixed points
    if fixed_points is not None:
        for i, fp in enumerate(fixed_points):
            fp_np = fp.detach().cpu().numpy()
            if fp_np.ndim > 1:
                fp_np = fp_np[0]
            ax.plot(
                fp_np[0].real,
                fp_np[1].real,
                "g*",
                markersize=15,
                label=f"Fixed Point {i}",
            )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_phase_space_3d(
    z_trajectory: torch.Tensor,
    title: str = "3D Phase Space",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
):
    """
    Plot 3D phase space.
    """
    z_np = z_trajectory.detach().cpu().numpy()

    if z_np.dtype == np.complex64 or z_np.dtype == np.complex128:
        # Complex - use real parts
        if z_np.ndim == 3:
            z_np = z_np[:, 0, :3].real
        else:
            z_np = z_np[:, :3].real
    else:
        if z_np.ndim == 3:
            z_np = z_np[:, 0, :3]
        else:
            z_np = z_np[:, :3]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(z_np[:, 0], z_np[:, 1], z_np[:, 2], "b-", linewidth=0.5, alpha=0.7)
    ax.scatter(z_np[0, 0], z_np[0, 1], z_np[0, 2], c="green", s=100, label="Start")
    ax.scatter(z_np[-1, 0], z_np[-1, 1], z_np[-1, 2], c="red", s=100, label="End")

    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_vector_field(
    model,
    x_input: torch.Tensor,
    grid_size: int = 20,
    title: str = "Vector Field",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot vector field of the dynamics.

    Args:
        model: TwistorLMT model
        x_input: Input to the model
        grid_size: Grid resolution
        title: Plot title
        save_path: Path to save
        figsize: Figure size
    """
    hidden_dim = model.hidden_dim

    # Create grid (first 2 dimensions)
    x_range = np.linspace(-2, 2, grid_size)
    y_range = np.linspace(-2, 2, grid_size)

    dx = np.zeros((grid_size, grid_size))
    dy = np.zeros((grid_size, grid_size))

    x_input = x_input.to(model.device)

    for i, x_val in enumerate(x_range):
        for j, y_val in enumerate(y_range):
            # Create state
            z = torch.zeros(1, hidden_dim, dtype=torch.complex64, device=model.device)
            z[0, 0] = complex(x_val, y_val)
            z[0, 1] = complex(x_val, y_val)

            # Compute derivative
            with torch.no_grad():
                dzdt = model.compute_dzdt(z, x_input[:1])

            dx[i, j] = dzdt[0, 0].real.item()
            dy[i, j] = dzdt[0, 1].real.item()

    fig, ax = plt.subplots(figsize=figsize)

    # Stream plot
    ax.streamplot(
        x_range,
        y_range,
        dx,
        dy,
        color=np.hypot(dx, dy),
        cmap="viridis",
        density=1.5,
        arrowsize=1,
    )

    ax.set_xlabel("Re(Z₁)")
    ax.set_ylabel("Re(Z₂)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_tau_evolution(
    z_trajectory: torch.Tensor,
    model,
    title: str = "Time Constant τ Evolution",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot τ(z) evolution over time.
    """
    T, B, hidden = z_trajectory.shape

    z_np = z_trajectory.detach().cpu()

    # Compute tau for each time step
    taus = []
    for t in range(min(T, 100)):
        z_t = z_np[t, 0]  # First batch
        tau_t = model.compute_tau(z_t.unsqueeze(0))
        taus.append(tau_t[0].cpu().numpy())

    taus = np.array(taus)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Mean tau over neurons
    ax1 = axes[0]
    ax1.plot(taus.mean(axis=1), "b-", linewidth=2)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Mean τ")
    ax1.set_title("Mean Time Constant")
    ax1.grid(True, alpha=0.3)

    # Tau for first few neurons
    ax2 = axes[1]
    for i in range(min(5, hidden)):
        ax2.plot(taus[:, i], label=f"Neuron {i}", linewidth=1.5)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("τ")
    ax2.set_title("Per-Neuron Time Constants")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_complex_plane(
    z_trajectory: torch.Tensor,
    title: str = "Complex Plane Trajectory",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
):
    """
    Plot trajectory in complex plane (Re vs Im for a single dimension).
    """
    z_np = z_trajectory.detach().cpu().numpy()

    if z_np.ndim == 3:
        z_np = z_np[:, 0]  # First batch

    fig, ax = plt.subplots(figsize=figsize)

    # Plot trajectory
    ax.plot(
        z_np[:, 0].real,
        z_np[:, 0].imag,
        "b-",
        linewidth=1,
        alpha=0.7,
        label="Trajectory",
    )
    ax.scatter(
        z_np[0, 0].real, z_np[0, 0].imag, c="green", s=100, zorder=5, label="Start"
    )
    ax.scatter(
        z_np[-1, 0].real, z_np[-1, 0].imag, c="red", s=100, zorder=5, label="End"
    )

    # Unit circle for reference
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), "k--", alpha=0.3, label="Unit Circle")

    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(title)
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_stability_analysis(
    stability_result: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """
    Plot stability analysis results.
    """
    eigenvalues = stability_result["eigenvalues"]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Eigenvalue distribution in complex plane
    ax1 = axes[0]
    ax1.scatter(eigenvalues.real, eigenvalues.imag, s=100, c="blue", alpha=0.7)
    ax1.axvline(x=0, color="red", linestyle="--", alpha=0.5)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    # Mark stable region
    x_fill = np.linspace(-5, 0, 50)
    y_fill = np.sqrt(25 - x_fill**2)
    ax1.fill_between(
        x_fill, -y_fill, y_fill, alpha=0.1, color="green", label="Stable Region"
    )

    ax1.set_xlabel("Real Part")
    ax1.set_ylabel("Imaginary Part")
    ax1.set_title("Eigenvalue Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_aspect("equal")

    # Summary text
    ax2 = axes[1]
    ax2.axis("off")

    summary = f"""
    Stability Analysis Summary
    ==========================
    
    Stability: {stability_result["stability"]}
    Max Real Part: {stability_result["max_real_part"]:.4f}
    
    Oscillation: {"Yes" if stability_result["has_oscillation"] else "No"}
    Dominant Frequency: {stability_result["oscillation_frequency"]:.4f}
    
    Number of Eigenvalues: {len(eigenvalues)}
    """

    ax2.text(
        0.1,
        0.9,
        summary,
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.suptitle("Fixed Point Stability Analysis")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_diagnostics(
    history: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot comprehensive training diagnostics.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Loss curves
    ax1 = axes[0, 0]
    if "train_loss" in history:
        ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    if "val_loss" in history:
        ax1.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MSE
    ax2 = axes[0, 1]
    if "train_mse" in history:
        ax2.plot(history["train_mse"], label="Train MSE", linewidth=2)
    if "val_mse" in history:
        ax2.plot(history["val_mse"], label="Val MSE", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE")
    ax2.set_title("Mean Squared Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # LR schedule
    ax3 = axes[1, 0]
    if "lr" in history:
        ax3.plot(history["lr"], linewidth=2, color="orange")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Learning Rate")
        ax3.set_title("Learning Rate Schedule")
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale("log")

    # Convergence info
    ax4 = axes[1, 1]
    ax4.axis("off")

    info_text = "Training Complete\n" + "=" * 20 + "\n\n"
    if "train_loss" in history:
        info_text += f"Initial Loss: {history['train_loss'][0]:.6f}\n"
        info_text += f"Final Loss: {history['train_loss'][-1]:.6f}\n"
        info_text += f"Improvement: {(history['train_loss'][0] - history['train_loss'][-1]) / history['train_loss'][0] * 100:.1f}%\n"

    ax4.text(
        0.1,
        0.9,
        info_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def create_animation(
    z_trajectory: torch.Tensor,
    save_path: str,
    fps: int = 10,
):
    """
    Create animation of phase space trajectory.

    Note: Requires ffmpeg installed.
    """
    try:
        import matplotlib.animation as animation
        from matplotlib.animation import FuncAnimation

        z_np = z_trajectory.detach().cpu().numpy()

        if z_np.ndim == 3:
            z_np = z_np[:, 0, :2]  # First batch, first 2 dims
        else:
            z_np = z_np[:, :2]

        fig, ax = plt.subplots()

        (line,) = ax.plot([], [], "b-", linewidth=1)
        (point,) = ax.plot([], [], "ro", markersize=10)

        ax.set_xlim(z_np[:, 0].real.min() - 0.5, z_np[:, 0].real.max() + 0.5)
        ax.set_ylim(z_np[:, 1].real.min() - 0.5, z_np[:, 1].real.max() + 0.5)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(True, alpha=0.3)

        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def animate(i):
            line.set_data(z_np[: i + 1, 0].real, z_np[: i + 1, 1].real)
            point.set_data([z_np[i, 0].real], [z_np[i, 1].real])
            return line, point

        anim = FuncAnimation(
            fig, animate, init_func=init, frames=len(z_np), interval=1000 / fps
        )

        anim.save(save_path, writer="ffmpeg", dpi=150)
        plt.close()

    except Exception as e:
        print(f"Animation failed: {e}")
