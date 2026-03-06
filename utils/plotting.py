"""
Shared plotting utilities for the RL notebooks.
Provides consistent, attractive charts across all notebooks.
"""
import matplotlib.pyplot as plt
import numpy as np

# Style defaults
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 12,
})

COLORS = {
    'primary': '#6366f1',    # indigo
    'secondary': '#14b8a6',  # teal
    'accent': '#f59e0b',     # amber
    'danger': '#ef4444',     # red
    'success': '#22c55e',    # green
    'muted': '#94a3b8',      # slate
}


def plot_rewards(rewards, window=50, title="Training Rewards", ax=None):
    """Plot per-episode rewards with a smoothed moving average overlay."""
    if ax is None:
        fig, ax = plt.subplots()

    episodes = np.arange(len(rewards))
    ax.plot(episodes, rewards, alpha=0.3, color=COLORS['muted'], label='Per episode')

    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(rewards)), smoothed,
                color=COLORS['primary'], linewidth=2, label=f'{window}-ep moving avg')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_values_grid(values, shape, title="State Values", ax=None, cmap='viridis'):
    """Plot a value function as a heatmap over a grid."""
    if ax is None:
        fig, ax = plt.subplots()

    grid = values.reshape(shape)
    im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Annotate each cell with the value
    for i in range(shape[0]):
        for j in range(shape[1]):
            ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center',
                    color='white' if grid[i, j] < grid.mean() else 'black', fontsize=10)

    ax.set_title(title)
    ax.set_xticks(range(shape[1]))
    ax.set_yticks(range(shape[0]))
    plt.tight_layout()
    return ax


def plot_comparison(results_dict, window=50, title="Algorithm Comparison"):
    """Plot multiple reward curves on the same axes for comparison.

    Args:
        results_dict: {'Algorithm Name': [rewards_list], ...}
    """
    fig, ax = plt.subplots()
    colors = list(COLORS.values())

    for i, (name, rewards) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        ax.plot(rewards, alpha=0.15, color=color)
        if len(rewards) >= window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, len(rewards)), smoothed,
                    color=color, linewidth=2, label=name)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_losses(losses, title="Training Loss", ax=None):
    """Plot a loss curve."""
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(losses, color=COLORS['danger'], alpha=0.6, linewidth=1)
    if len(losses) >= 50:
        smoothed = np.convolve(losses, np.ones(50)/50, mode='valid')
        ax.plot(np.arange(49, len(losses)), smoothed,
                color=COLORS['danger'], linewidth=2)

    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_policy_arrows(policy, shape, title="Policy"):
    """Plot a deterministic policy as arrows on a grid.

    Args:
        policy: array of action indices (0=up, 1=right, 2=down, 3=left)
        shape: (rows, cols)
    """
    fig, ax = plt.subplots(figsize=(shape[1]*1.5, shape[0]*1.5))

    arrow_map = {0: (0, 0.3), 1: (0.3, 0), 2: (0, -0.3), 3: (-0.3, 0)}

    for s in range(len(policy)):
        row, col = divmod(s, shape[1])
        dx, dy = arrow_map.get(policy[s], (0, 0))
        ax.arrow(col, row, dx, dy, head_width=0.1, head_length=0.05,
                 fc=COLORS['primary'], ec=COLORS['primary'])

    ax.set_xlim(-0.5, shape[1]-0.5)
    ax.set_ylim(shape[0]-0.5, -0.5)
    ax.set_xticks(range(shape[1]))
    ax.set_yticks(range(shape[0]))
    ax.grid(True, alpha=0.5)
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    return ax
