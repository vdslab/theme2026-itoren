"""Gravity-based edge bundling: core calculation, implemented independently of the Rust/WASM version.

Design notes
------------
Each node acts as a point mass generating a softened, radius-clipped
gravitational potential with three independent parameters:

  - gravity_param (G)  : overall coupling strength.
  - potential_max       : caps how deep/steep the well gets near the node
                           center, by setting a softening length
                           eps_i = G * m_i / potential_max.
  - gravity_alpha       : radius (scaled by node mass) around the node
                           center that is treated as a rigid, force-free
                           core -- points inside it feel zero force, which
                           keeps the relaxation stable near massive nodes.

Formally, with d = |p - node_i| and shifted distance d' = max(d - alpha*m_i, 0):

    denom_i(p)   = max(d', eps_i)
    phi_i(p)     = -G * m_i / denom_i(p)
    force_i(p)   = 0                                   if d' <= eps_i (inside the core)
                 = (G * m_i / denom_i(p)^2) * (p - node_i) / d   otherwise

This mirrors the physics used by the Rust/WASM simulation (same three
parameters, same core/softening behavior) so results can be compared
directly, while the surrounding code (data structures, integration loop,
control-point layout) is a fresh, independent design.

Edges are discretized into control points (polylines). Each interior control
point feels:
  - a discrete-Laplacian spring force pulling it straight (Hooke's law against
    its two neighbors on the same edge), and
  - the summed gravity force from every node, pulling it toward mass
    concentrations and causing nearby edges to bend together ("bundling").

Endpoints of every edge are pinned to their node position and never move.

All parameters (G, potential_max, gravity_alpha, spring_k, dt, damping,
spacing, steps, ...) are supplied by the caller -- nothing is hardcoded in
the calculation itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Polyline:
    """Control points for a single bundled edge."""

    edge_index: int
    source: int
    target: int
    points: np.ndarray  # shape (n, 2), points[0] and points[-1] are pinned


def build_control_points(
    nodes_xy: np.ndarray,
    edges: np.ndarray,
    spacing: float,
) -> list[Polyline]:
    """Create the initial (straight-line) control points for every edge.

    nodes_xy : (N, 2) float array of node positions.
    edges    : (E, 2) int array of (source_index, target_index) pairs.
    spacing  : target distance between consecutive control points; the
               number of points on an edge is chosen so points are spaced
               at roughly this distance (minimum 3 points per edge).
    """
    spacing = max(float(spacing), 1.0)
    polylines: list[Polyline] = []

    for e_idx, (s, t) in enumerate(edges):
        p0 = nodes_xy[s]
        p1 = nodes_xy[t]
        dist = float(np.linalg.norm(p1 - p0))
        n = max(int(dist // spacing) + 2, 3)
        tt = np.linspace(0.0, 1.0, n)[:, None]
        pts = p0[None, :] + tt * (p1[None, :] - p0[None, :])
        polylines.append(Polyline(e_idx, int(s), int(t), pts.astype(np.float64)))

    return polylines


def gravity_potential_and_force(
    points: np.ndarray,
    nodes_xy: np.ndarray,
    nodes_mass: np.ndarray,
    gravity_param: float,
    potential_max: float,
    gravity_alpha: float,
    chunk_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Radius-clipped, softened gravity from every node onto every point.

    points        : (M, 2) query positions.
    nodes_xy      : (N, 2) node positions.
    nodes_mass    : (N,) node masses.
    chunk_size    : points are processed in batches of this size so memory
                    stays bounded at O(chunk_size * N) regardless of M
                    (purely a performance knob, not a physical parameter).

    Returns (potential[M], force[M, 2]) summed over all nodes.
    """
    G = gravity_param
    M = points.shape[0]
    eps = (G * nodes_mass) / max(potential_max, 1e-8)   # (N,) softening length
    alpha_mass = gravity_alpha * nodes_mass             # (N,) force-free core radius

    potential = np.empty(M, dtype=np.float64)
    force = np.empty((M, 2), dtype=np.float64)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        diff = points[start:end, None, :] - nodes_xy[None, :, :]   # (m, N, 2)
        d2 = np.einsum("mnd,mnd->mn", diff, diff)                  # (m, N)
        d = np.sqrt(d2)

        d_shift = np.maximum(d - alpha_mass[None, :], 0.0)          # core cutout
        denom = np.maximum(d_shift, eps[None, :])

        potential[start:end] = -np.sum(G * nodes_mass[None, :] / denom, axis=1)

        outside_core = (d_shift > eps[None, :]) & (d > 0.0)
        force_mag = np.where(outside_core, G * nodes_mass[None, :] / (denom * denom), 0.0)
        inv_d = np.divide(1.0, d, out=np.zeros_like(d), where=d > 0.0)

        force[start:end, 0] = np.sum(force_mag * diff[:, :, 0] * inv_d, axis=1)
        force[start:end, 1] = np.sum(force_mag * diff[:, :, 1] * inv_d, axis=1)

    return potential, force


def spring_force(points: np.ndarray) -> np.ndarray:
    """Discrete-Laplacian spring force for interior points of one polyline.

    points: (n, 2). Returns (n, 2) with the force at endpoints set to zero
    (endpoints are pinned and never integrated).
    """
    force = np.zeros_like(points)
    if len(points) > 2:
        force[1:-1] = points[:-2] + points[2:] - 2.0 * points[1:-1]
    return force


def step(
    polylines: list[Polyline],
    nodes_xy: np.ndarray,
    nodes_mass: np.ndarray,
    gravity_param: float,
    potential_max: float,
    gravity_alpha: float,
    spring_k: float,
    dt: float,
    damping: float,
    max_displacement: float | None = None,
) -> None:
    """Advance every polyline's interior control points by one relaxation step, in place."""
    # Concatenate all interior points across all edges into one batch so the
    # gravity calculation is vectorized over every moving point at once.
    interior_slices = []
    interior_points = []
    for pl in polylines:
        n = len(pl.points)
        if n <= 2:
            interior_slices.append(None)
            continue
        interior_slices.append((len(interior_points), len(interior_points) + n - 2))
        interior_points.append(pl.points[1:-1])

    if not interior_points:
        return

    batch = np.concatenate(interior_points, axis=0)
    _, f_gravity = gravity_potential_and_force(
        batch, nodes_xy, nodes_mass, gravity_param, potential_max, gravity_alpha
    )

    offset = 0
    for pl, sl in zip(polylines, interior_slices):
        if sl is None:
            continue
        start, end = sl
        f_spring = spring_k * spring_force(pl.points)[1:-1]
        f_total = f_spring + f_gravity[start:end]

        disp = f_total * dt
        if max_displacement is not None:
            lengths = np.linalg.norm(disp, axis=1, keepdims=True)
            too_far = lengths[:, 0] > max_displacement
            if np.any(too_far):
                disp[too_far] *= max_displacement / lengths[too_far]

        pl.points[1:-1] += disp * damping
        offset += end - start


def simulate(
    nodes_xy: np.ndarray,
    nodes_mass: np.ndarray,
    edges: np.ndarray,
    *,
    spacing: float,
    gravity_param: float,
    potential_max: float,
    gravity_alpha: float,
    spring_k: float,
    dt: float,
    damping: float,
    n_steps: int,
    max_displacement: float | None = 5.0,
) -> list[Polyline]:
    """Run the full bundling calculation and return the final polylines.

    Every physical/numerical parameter is a required keyword argument --
    callers decide the values, this function only implements the calculation.
    """
    polylines = build_control_points(nodes_xy, edges, spacing)
    for _ in range(n_steps):
        step(
            polylines,
            nodes_xy,
            nodes_mass,
            gravity_param,
            potential_max,
            gravity_alpha,
            spring_k,
            dt,
            damping,
            max_displacement,
        )
    return polylines


def potential_grid(
    nodes_xy: np.ndarray,
    nodes_mass: np.ndarray,
    width: int,
    height: int,
    gravity_param: float,
    potential_max: float,
    gravity_alpha: float,
    grid_step: int = 1,
) -> np.ndarray:
    """Optional: sample the potential field on a regular grid, for visualization only.

    Returns an array of shape (height // grid_step, width // grid_step).
    """
    xs = np.arange(0, width, grid_step, dtype=np.float64)
    ys = np.arange(0, height, grid_step, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    potential, _ = gravity_potential_and_force(
        pts, nodes_xy, nodes_mass, gravity_param, potential_max, gravity_alpha
    )
    return potential.reshape(gy.shape)
