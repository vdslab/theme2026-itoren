import os
import json
import itertools
import time
import math
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = "param_sweep_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

KP_MAX = 0.5  # 密集地の短いエッジでkp = SPRING_CONSTANT/edge_lengthが暴走しないようにする上限

# -----------------------------
# 試すパラメータの候補値
# test2.pyでいい感じだった設定周辺を探索
# -----------------------------
PARAM_GRID = {
    "DEFAULT_NODE_MASS": [5000],
    "STEP_SIZE":         [50],
    "SPRING_CONSTANT":   [30],
    "STEP_SIZE_DECAY":   [0.9999],
    "N_ITERATIONS":      [50],
    "N_SAMPLES_PER_EDGE": [100],
    "WINDOW_SIZE":       [10000],
    "GRID_SIZE":         [10],
    "SIGMA":             [0, 0.3, 0.6],
    "DENSITY_RADIUS":    [300, 500, 800],
}

# -----------------------------
# データ読み込み
# -----------------------------
with open("airlines.json", "r") as f:
    data = json.load(f)

raw_x = [n["x"] for n in data["nodes"]]
raw_y = [n["y"] for n in data["nodes"]]
min_x, max_x_val = min(raw_x), max(raw_x)
min_y, max_y_val = min(raw_y), max(raw_y)
json_edges = data.get("edges", data.get("links", []))
edge_keys = [(str(e["source"]), str(e["target"])) for e in json_edges]


def build_nodes(window_size):
    scale = window_size / max(max_x_val - min_x, max_y_val - min_y)
    return {
        str(n["id"]): np.array([(n["x"] - min_x) * scale, (n["y"] - min_y) * scale])
        for n in data["nodes"]
    }


# (window_size, grid_size, node_mass) をキーにタイル勾配をキャッシュ
_tile_cache = {}


def get_tile_gradients(window_size, grid_size, node_mass, sigma, density_radius):
    """test2.py方式: WINDOW_SIZEをGRID_SIZEのタイルに分割してポテンシャル場を計算"""
    key = (window_size, grid_size, node_mass, sigma, density_radius)
    if key in _tile_cache:
        return _tile_cache[key]

    nodes = build_nodes(window_size)

    # 密集地ほど重力(質量)を弱める: 半径density_radius以内のノード数(自分を含む)で質量を割る
    node_ids = list(nodes.keys())
    coords = np.array([nodes[nid] for nid in node_ids])
    tree = cKDTree(coords)
    neighbor_counts = tree.query_ball_point(coords, r=density_radius, return_length=True)
    masses = {nid: node_mass / count for nid, count in zip(node_ids, neighbor_counts)}

    n_tiles_x = math.ceil(window_size / grid_size)
    n_tiles_y = math.ceil(window_size / grid_size)
    tile_cx = (np.arange(n_tiles_x) + 0.5) * grid_size
    tile_cy = (np.arange(n_tiles_y) + 0.5) * grid_size
    XX, YY = np.meshgrid(tile_cx, tile_cy)

    potential_field = np.zeros_like(XX)
    for name, pos in nodes.items():
        dist_sq = (XX - pos[0])**2 + (YY - pos[1])**2
        potential_field -= masses[name] / np.sqrt(dist_sq + 1e-9)

    if sigma > 0:
        potential_field = gaussian_filter(potential_field, sigma=sigma)

    tile_grad_y, tile_grad_x = np.gradient(potential_field)
    tile_grad_x /= grid_size
    tile_grad_y /= grid_size

    result = (nodes, potential_field, tile_grad_x, tile_grad_y, n_tiles_x, n_tiles_y)
    _tile_cache[key] = result
    return result


def run_bundling(nodes, tile_grad_x, tile_grad_y, n_tiles_x, n_tiles_y,
                 grid_size, n_samples, step_size, step_decay, spring_k, n_iterations):
    bundled_edges = []
    for start_key, end_key in edge_keys:
        start_pos, end_pos = nodes[start_key], nodes[end_key]
        edge = np.array([
            np.linspace(start_pos[0], end_pos[0], n_samples),
            np.linspace(start_pos[1], end_pos[1], n_samples),
        ]).T
        bundled_edges.append(edge)

    current_step = step_size
    for _ in range(n_iterations):
        for j, edge in enumerate(bundled_edges):
            new_edge = edge.copy()
            for k in range(1, len(edge) - 1):
                pos = edge[k]
                ti = int(np.clip(pos[0] / grid_size, 0, n_tiles_x - 1))
                tj = int(np.clip(pos[1] / grid_size, 0, n_tiles_y - 1))
                dx = tile_grad_x[tj, ti]
                dy = tile_grad_y[tj, ti]
                force = current_step * np.array([dx, dy])
                # タイル解像度以上に一気に動くと隣のタイルの力を無視して飛び出すのでクリップ
                force_mag = np.linalg.norm(force)
                if force_mag > grid_size:
                    force = force / force_mag * grid_size
                new_edge[k] += force
            edge_length = np.linalg.norm(edge[-1] - edge[0])
            # 密集地の短いエッジではkpが暴走して発散するのでクリップする
            kp = min(spring_k / edge_length, KP_MAX) if edge_length > 0 else 0
            for k in range(1, len(edge) - 1):
                current = new_edge[k]
                new_edge[k] += kp * ((new_edge[k-1] - current) + (new_edge[k+1] - current))
            bundled_edges[j] = new_edge
        current_step *= step_decay
    return bundled_edges


def save_result_image(nodes, potential_field, bundled_edges, window_size, params, out_path):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(potential_field, cmap="bone_r", origin="lower",
              extent=[0, window_size, 0, window_size], alpha=0.3)
    for edge in bundled_edges:
        ax.plot(edge[:, 0], edge[:, 1], color="cyan", linewidth=1.0, alpha=0.7)
    for pos in nodes.values():
        ax.scatter(pos[0], pos[1], s=15, c="magenta", zorder=10)
    ax.set_xlim(0, window_size)
    ax.set_ylim(0, window_size)
    ax.set_aspect("equal")
    ax.set_facecolor("black")

    param_text = "\n".join(f"{k} = {v}" for k, v in params.items())
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
            va="top", ha="left", fontsize=9, color="white", family="monospace",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="white", boxstyle="round,pad=0.4"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def make_filename(params):
    parts = [f"{k}-{v}" for k, v in params.items()]
    return "_".join(parts) + ".png"


def main():
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    total = len(combos)
    print(f"合計 {total} 通りのパラメータを実行します。")

    for idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        filename = make_filename(params)
        out_path = os.path.join(OUTPUT_DIR, filename)

        if os.path.exists(out_path):
            print(f"[{idx}/{total}] skip: {filename}")
            continue

        t0 = time.time()
        nodes, potential_field, tile_grad_x, tile_grad_y, n_tiles_x, n_tiles_y = get_tile_gradients(
            params["WINDOW_SIZE"], params["GRID_SIZE"], params["DEFAULT_NODE_MASS"], params["SIGMA"],
            params["DENSITY_RADIUS"]
        )
        bundled_edges = run_bundling(
            nodes, tile_grad_x, tile_grad_y, n_tiles_x, n_tiles_y,
            grid_size=params["GRID_SIZE"],
            n_samples=params["N_SAMPLES_PER_EDGE"],
            step_size=params["STEP_SIZE"],
            step_decay=params["STEP_SIZE_DECAY"],
            spring_k=params["SPRING_CONSTANT"],
            n_iterations=params["N_ITERATIONS"],
        )
        save_result_image(nodes, potential_field, bundled_edges, params["WINDOW_SIZE"], params, out_path)
        print(f"[{idx}/{total}] saved {filename} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
