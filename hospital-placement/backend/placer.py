"""
placer.py
Algoritmo para ubicar K hospitales sobre una grilla m x n.
Usa un enfoque tipo K-Means adaptado a grilla discreta,
asegurando que hospitales no queden sobre casas.
"""

import numpy as np
import random

def euclidean(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))

def pick_initial_centers(empty_cells, k, seed=None):
    if seed is not None:
        random.seed(seed)
    E = empty_cells.shape[0]
    k = min(k, E)
    chosen_idx = random.sample(range(E), k)
    return empty_cells[chosen_idx].astype(float)

def nearest_empty_to_point(point, empty_cells):
    dists = np.sqrt(((empty_cells - point) ** 2).sum(axis=1))
    idx = np.argmin(dists)
    return tuple(map(int, empty_cells[idx]))

def compute_assignments(centroids, houses_coords):
    if houses_coords.shape[0] == 0:
        return np.array([], dtype=int)
    dists = np.sqrt(((houses_coords[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2))
    return np.argmin(dists, axis=1)

def recompute_centroids(assignments, houses_coords, k):
    centroids = np.zeros((k, 2), dtype=float)
    for i in range(k):
        members = houses_coords[assignments == i]
        centroids[i] = np.nan if len(members) == 0 else members.mean(axis=0)
    return centroids

def find_hospital_positions(m, n, houses_coords, k, max_iters=50, seed=None, verbose=False):
    grid_idx = np.array([[i, j] for i in range(m) for j in range(n)])
    if houses_coords.shape[0] > 0:
        house_set = set((int(r), int(c)) for r, c in houses_coords)
        empties = np.array([p for p in grid_idx if (p[0], p[1]) not in house_set])
    else:
        empties = grid_idx

    if len(empties) == 0:
        raise ValueError("No hay celdas vacías donde colocar hospitales.")

    centroids = pick_initial_centers(empties, k, seed=seed)

    for _ in range(max_iters):
        assignments = compute_assignments(centroids, houses_coords)
        new_centroids = recompute_centroids(assignments, houses_coords, k)

        for i in range(k):
            if np.isnan(new_centroids[i]).any():
                new_centroids[i] = empties[random.randrange(len(empties))]

        if np.allclose(new_centroids, centroids, atol=1e-2):
            break
        centroids = new_centroids

    hospital_positions = []
    used = set()
    for c in centroids:
        pos = nearest_empty_to_point(c, empties)
        if pos in used:
            dists = np.sqrt(((empties - c) ** 2).sum(axis=1))
            for candidate in empties[np.argsort(dists)]:
                cand_t = tuple(map(int, candidate))
                if cand_t not in used:
                    pos = cand_t
                    break
        used.add(pos)
        hospital_positions.append(pos)

    return hospital_positions
