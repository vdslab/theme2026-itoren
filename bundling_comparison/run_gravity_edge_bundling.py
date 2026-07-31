"""Demo runner for gravity_bundling.simulate().

Loads a node/edge JSON file (e.g. public/eurosis.json), runs the calculation,
and optionally plots the result with matplotlib.

Edit the PARAMETERS block below, or import gravity_bundling directly and call
simulate(...) with your own values from another script / notebook.
"""

import json
from pathlib import Path

import numpy as np

import gravity_edge_bundling as gb

# --------------------------------------------------------------------------
# PARAMETERS -- change these to whatever values you want to test.
# --------------------------------------------------------------------------
DATA_PATH = r"C:\Users\lotus\theme2026-itoren\eurosis.json"

CANVAS_WIDTH = 960
CANVAS_HEIGHT = 720
PADDING = 40

PARAMS = dict(
    spacing=5.0,
    gravity_param=0.01563,
    potential_max=0.5,
    gravity_alpha=0.686,
    spring_k=0.165,
    dt=1.0,
    damping=0.95,
    n_steps=200,
    max_displacement=5.0,
)

DEFAULT_MASS = 10
# --------------------------------------------------------------------------


def load_graph(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    id_to_index = {n["id"]: i for i, n in enumerate(data["nodes"])}
    raw_xy = np.array([[n["x"], n["y"]] for n in data["nodes"]], dtype=np.float64)

    edges = np.array(
        [[id_to_index[e["source"]], id_to_index[e["target"]]] for e in data["edges"]],
        dtype=np.int64,
    )
    return raw_xy, edges


def scale_to_canvas(raw_xy: np.ndarray, width: int, height: int, padding: int) -> np.ndarray:
    min_xy = raw_xy.min(axis=0)
    max_xy = raw_xy.max(axis=0)
    span = max_xy - min_xy
    span[span == 0] = 1.0

    scale = min((width - 2 * padding) / span[0], (height - 2 * padding) / span[1])
    return padding + (raw_xy - min_xy) * scale


def plot(nodes_xy, edges, polylines, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7.5))
    for pl in polylines:
        ax.plot(pl.points[:, 0], pl.points[:, 1], color="steelblue", alpha=0.15, linewidth=0.8)
    ax.scatter(nodes_xy[:, 0], nodes_xy[:, 1], s=4, color="black", zorder=3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved: {out_path}")


def main():
    raw_xy, edges = load_graph(DATA_PATH)
    nodes_xy = scale_to_canvas(raw_xy, CANVAS_WIDTH, CANVAS_HEIGHT, PADDING)
    nodes_mass = np.full(len(nodes_xy), DEFAULT_MASS, dtype=np.float64)

    polylines = gb.simulate(nodes_xy, nodes_mass, edges, **PARAMS)

    out_path = Path(__file__).resolve().parent / "bundled_edges.png"
    plot(nodes_xy, edges, polylines, out_path)


if __name__ == "__main__":
    main()
