import optuna
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# ---- データ読み込み（1回だけ） ----
with open('airlines.json', 'r') as f:
    data = json.load(f)

raw_x = [n["x"] for n in data["nodes"]]
raw_y = [n["y"] for n in data["nodes"]]
min_x, max_x_val = min(raw_x), max(raw_x)
min_y, max_y_val = min(raw_y), max(raw_y)
GRID_SIZE = max(max_x_val - min_x, max_y_val - min_y)
PADDING = 50
USABLE_GRID = GRID_SIZE - 2 * PADDING

nodes = {}
for n in data["nodes"]:
    node_id = str(n["id"])
    nodes[node_id] = np.array([
        PADDING + (n["x"] - min_x) * USABLE_GRID / GRID_SIZE,
        PADDING + (n["y"] - min_y) * USABLE_GRID / GRID_SIZE,
    ])

DEFAULT_NODE_MASS = 50
masses = {k: DEFAULT_NODE_MASS for k in nodes}

json_edges = data.get("edges", data.get("links", []))
edges_list = [(str(e["source"]), str(e["target"])) for e in json_edges]

N_SAMPLES_PER_EDGE = 30
N_ITERATIONS = 100


# ================================================================
# 評価指標
# ================================================================

def count_ink_pixels(edges):
    """matplotlibで描画してcyanピクセル数を返す"""
    dpi = 100
    fig_inches = GRID_SIZE / dpi
    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=dpi)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for edge in edges:
        ax.plot(edge[:, 0], edge[:, 1], color='cyan', linewidth=1.2, alpha=0.7)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return int(np.sum((r < 10) & (g > 50) & (b > 50)))


def ink_ratio(bundled_edges, original_pixels):
    """Ink Ratio = バンドリング後のcyanピクセル数 / 元のcyanピクセル数（低いほど良い）"""
    return count_ink_pixels(bundled_edges) / original_pixels


def distortion(bundled_edges, original_edges):
    """Distortion = バンドリング後のエッジ長 / 直線エッジ長 の平均（1.0に近いほど良い）"""
    ratios = []
    for bundled, orig in zip(bundled_edges, original_edges):
        direct_len = np.linalg.norm(orig[-1] - orig[0])
        if direct_len < 1e-9:
            continue
        bundled_len = float(np.sum(np.linalg.norm(np.diff(bundled, axis=0), axis=1)))
        ratios.append(bundled_len / direct_len)
    return float(np.mean(ratios)) if ratios else 1.0


def ambiguity(bundled_edges):
    """Ambiguity = 各ピクセルを通る異なるエッジ数の平均（低いほど追跡しやすい）"""
    pixel_edges = defaultdict(set)
    for j, edge in enumerate(bundled_edges):
        for i in range(len(edge) - 1):
            p0, p1 = edge[i], edge[i + 1]
            n_steps = max(int(np.linalg.norm(p1 - p0)) + 1, 2)
            pts = p0 + np.linspace(0, 1, n_steps)[:, None] * (p1 - p0)
            for pt in pts:
                pixel_edges[(int(round(pt[0])), int(round(pt[1])))].add(j)
    if not pixel_edges:
        return 1.0
    return float(np.mean([len(v) for v in pixel_edges.values()]))


# 元の直線エッジ（分母・比較基準、1回だけ計算）
original_edges = []
for start_key, end_key in edges_list:
    s, e = nodes[start_key], nodes[end_key]
    original_edges.append(np.array([
        np.linspace(s[0], e[0], N_SAMPLES_PER_EDGE),
        np.linspace(s[1], e[1], N_SAMPLES_PER_EDGE),
    ]).T)

original_pixels = count_ink_pixels(original_edges)
print(f"Original ink pixels: {original_pixels}")


# ================================================================
# ポテンシャル場 & バンドリング
# ================================================================

def make_potential_field(sigma, node_mass):
    x = np.arange(GRID_SIZE)
    y = np.arange(GRID_SIZE)
    xx, yy = np.meshgrid(x, y)
    phi = np.zeros_like(xx, dtype=float)
    for pos in nodes.values():
        dist_sq = (xx - pos[0])**2 + (yy - pos[1])**2
        phi -= node_mass / np.sqrt(dist_sq + 1e-9)
    return gaussian_filter(phi, sigma=sigma)


def run_bundling(step_size, step_decay, spring_constant, grad_x, grad_y):
    bundled_edges = []
    for start_key, end_key in edges_list:
        s, e = nodes[start_key], nodes[end_key]
        edge = np.array([
            np.linspace(s[0], e[0], N_SAMPLES_PER_EDGE),
            np.linspace(s[1], e[1], N_SAMPLES_PER_EDGE),
        ]).T
        bundled_edges.append(edge)

    current_step = step_size
    for _ in range(N_ITERATIONS):
        for j, edge in enumerate(bundled_edges):
            new_edge = edge.copy()

            for k in range(1, len(edge) - 1):
                pos = edge[k]
                dx = map_coordinates(grad_x, [[pos[1]], [pos[0]]], order=1)[0]
                dy = map_coordinates(grad_y, [[pos[1]], [pos[0]]], order=1)[0]
                new_edge[k] += current_step * np.array([dx, dy])

            F_spring_list = []
            for k in range(1, len(edge) - 1):
                cur = new_edge[k]
                vec_left  = new_edge[k - 1] - cur
                vec_right = new_edge[k + 1] - cur
                F_spring_list.append(spring_constant * (vec_left + vec_right))
            for k in range(1, len(edge) - 1):
                new_edge[k] += F_spring_list[k - 1]

            bundled_edges[j] = new_edge
        current_step *= step_decay

    return bundled_edges


# ================================================================
# 目的関数（3指標を同時最小化）
# ================================================================

INF3 = (float('inf'), float('inf'), float('inf'))


def objective(trial):
    step_size       = trial.suggest_float('step_size',       1.0,  200.0)
    step_decay      = trial.suggest_float('step_decay',      0.90, 0.999)
    spring_constant = trial.suggest_float('spring_constant', 0.01, 0.50)
    sigma           = trial.suggest_float('sigma',           10.0, 100.0)
    node_mass       = trial.suggest_float('node_mass',        1.0, 200.0)

    phi = make_potential_field(sigma, node_mass)
    grad_y, grad_x = np.gradient(phi)

    try:
        bundled = run_bundling(step_size, step_decay, spring_constant, grad_x, grad_y)
    except Exception:
        return INF3

    for edge in bundled:
        if np.any(~np.isfinite(edge)) or np.any(edge < -GRID_SIZE) or np.any(edge > 2 * GRID_SIZE):
            return INF3

    ir  = ink_ratio(bundled, original_pixels)
    dis = distortion(bundled, original_edges)
    amb = ambiguity(bundled)
    return ir, dis, amb


# ================================================================
# 実行
# ================================================================

if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        directions=['minimize', 'minimize', 'minimize'],
        study_name='edge_bundling_3obj'
    )
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    print("\n=== Pareto Front trials ===")
    print(f"{'Trial':>6}  {'InkRatio':>10}  {'Distortion':>10}  {'Ambiguity':>10}  params")
    for t in study.best_trials:
        ir, dis, amb = t.values
        params_str = '  '.join(f"{k}={v:.3f}" for k, v in t.params.items())
        print(f"{t.number:>6}  {ir:>10.4f}  {dis:>10.4f}  {amb:>10.4f}  {params_str}")
