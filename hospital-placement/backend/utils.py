import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO

def generate_houses(m, n, num_houses, seed=None):
    if seed is not None:
        np.random.seed(seed)
    total = m * n
    if num_houses > total:
        raise ValueError("Número de casas mayor que celdas disponibles.")
    indices = np.random.choice(total, size=num_houses, replace=False)
    coords = np.vstack((indices // n, indices % n)).T
    return coords

def plot_map(m, n, houses_coords, hospital_positions, figsize=(7, 6), show_grid=False, title=None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(m - 0.5, -0.5)
    if show_grid:
        ax.set_xticks(np.arange(-0.5, n, 1))
        ax.set_yticks(np.arange(-0.5, m, 1))
        ax.grid(color='lightgray', linewidth=0.4)

    if len(houses_coords) > 0:
        ax.scatter(houses_coords[:, 1], houses_coords[:, 0], c='dodgerblue', s=25, label='Casas', alpha=0.7)
    hp = np.array(hospital_positions)
    if hp.size > 0:
        ax.scatter(hp[:, 1], hp[:, 0], c='crimson', s=80, marker='s', label='Hospitales', edgecolors='white', linewidth=0.8)

    ax.set_xlabel('Columna')
    ax.set_ylabel('Fila')
    if title:
        ax.set_title(title, fontsize=13, weight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
