import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def animate_xy_traj(time: np.ndarray, p_traj: np.ndarray, filename: str | None = None):
    """
    Animate x-y trajectory as a moving point with path trace.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    x = p_traj[:, 0]
    y = p_traj[:, 1]

    line, = ax.plot([], [], "-", color="C0", lw=1.5)
    point, = ax.plot([], [], "o", color="C1")

    ax.set_xlim(float(np.min(x) - 0.5), float(np.max(x) + 0.5))
    ax.set_ylim(float(np.min(y) - 0.5), float(np.max(y) + 0.5))

    def init():
        line.set_data([], [])
        point.set_data([], [])
        return line, point

    def update(frame):
        line.set_data(x[:frame + 1], y[:frame + 1])
        point.set_data(x[frame], y[frame])
        ax.set_title(f"t = {time[frame]:.2f} s")
        return line, point

    ani = FuncAnimation(fig, update, frames=len(time), init_func=init,
                        interval=20, blit=True)

    if filename is not None:
        ani.save(filename, fps=30)
    else:
        plt.show()

    return ani