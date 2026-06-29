import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(time: np.ndarray, p_traj: np.ndarray, p_ref: np.ndarray):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    labels = ["x", "y", "z"]
    for i, ax in enumerate(axes):
        ax.plot(time, p_traj[:, i], label=f"{labels[i]} actual")
        ax.plot(time, p_ref[:, i], "--", label=f"{labels[i]} reference")
        ax.set_ylabel(labels[i])
        ax.grid(True)
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend()
    fig.tight_layout()
    return fig
