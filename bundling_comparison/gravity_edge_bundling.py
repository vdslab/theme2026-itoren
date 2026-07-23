"""
likr-sandbox/gravity-edge-bundling (Rust/WebGPU実装) の物理演算部分(simulation.rs)を
忠実にPythonへ移植したもの。
https://github.com/likr-sandbox/gravity-edge-bundling/tree/master/gravity-edge-bundling/src

元実装との対応:
- Node.mass: 元コードにmassの計算方法(フロントエンド側)が含まれていなかったため、
  DEFAULT_MASSで全ノード一律とする(必要なら関数を差し替え可能)。
- 座標系: 元コードはノード座標=グリッド解像度(width/height)が同一スケールだったが、
  本ポートではWINDOW_SIZE=10000(他手法と共通)とGRID_RES(場の解像度)を分離し、
  シミュレーション自体はGRID_RES空間で行い、最後にWINDOW_SIZEへスケールし直す。
- control_point_spacingベースの分割数(FDEBのような倍化スケジュールではなく、
  エッジ長/間隔で決まる固定分割数)、spring_k・dt・dampingは元コードのデフォルト値。
- ポテンシャル/力場は「疑似ニュートン重力」: 質量に応じた特異点回避半径(gravity_alpha)と
  深さの上限(potential_max)を持つ、通常の1/r重力よりソフトなモデル。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_DATASET = PROJECT_ROOT / "airlines.json"

WINDOW_SIZE = 10000
GRID_RES = 1000          # ポテンシャル/力場を計算する解像度（この空間でシミュレーションを行う）
GRID_SCALE = WINDOW_SIZE / GRID_RES

# -----------------------------
# パラメータ（Rust版 SimulationState::new のデフォルト値そのまま）
# -----------------------------
SPRING_K = 0.05
DT = 0.5
DAMPING = 0.95
GRAVITY_PARAM = 0.03125
POTENTIAL_MAX = 16.0
GRAVITY_ALPHA = 1.0
CONTROL_POINT_SPACING = 10.0   # GRID_RES空間での間隔（GRID_RES=1000に対する目安値）
N_STEPS = 400                   # 元コードはインタラクティブに毎フレームstep()するので、
                                 # バッチ実行用に固定回数を採用（アニーリングなし、一定パラメータ）
N_SAMPLES = 100                  # 出力時にevaluate.pyの他手法と合わせて統一する点数
DEFAULT_MASS = 1.0               # 元コードにmass算出ロジックがないため一律とする
EPS = 1e-9


def load_nodes_edges(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    raw_x = [n["x"] for n in data["nodes"]]
    raw_y = [n["y"] for n in data["nodes"]]
    min_x, max_x_val = min(raw_x), max(raw_x)
    min_y, max_y_val = min(raw_y), max(raw_y)
    scale = WINDOW_SIZE / max(max_x_val - min_x, max_y_val - min_y)
    node_ids = [str(n["id"]) for n in data["nodes"]]
    positions = np.array([
        [(n["x"] - min_x) * scale, (n["y"] - min_y) * scale] for n in data["nodes"]
    ])
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    json_edges = data.get("edges", data.get("links", []))
    edges = [(id_to_idx[str(e["source"])], id_to_idx[str(e["target"])]) for e in json_edges]
    return node_ids, positions, edges


def resample_to_n_points(edge, n):
    diffs = np.diff(edge, axis=0)
    seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = arc[-1]
    if total < 1e-9:
        return np.tile(edge[0], (n, 1))
    target = np.linspace(0, total, n)
    fx = interp1d(arc, edge[:, 0])
    fy = interp1d(arc, edge[:, 1])
    return np.stack([fx(target), fy(target)], axis=1)


# -----------------------------
# update_physics_fields の移植（simulation.rs 98-144行目）
# -----------------------------
def compute_fields(grid_res, node_pos_grid, node_mass,
                    gravity_param=GRAVITY_PARAM, potential_max=POTENTIAL_MAX, gravity_alpha=GRAVITY_ALPHA):
    scale = grid_res / 256.0
    potential_max_scaled = max(potential_max / scale, 1e-5)

    xs = np.arange(grid_res, dtype=np.float64)
    XX, YY = np.meshgrid(xs, xs)  # XX[row,col]=col(=x), YY[row,col]=row(=y)

    potential = np.zeros((grid_res, grid_res))
    force_x = np.zeros((grid_res, grid_res))
    force_y = np.zeros((grid_res, grid_res))

    for (nx, ny), mass in zip(node_pos_grid, node_mass):
        dx = XX - nx
        dy = YY - ny
        d = np.sqrt(dx * dx + dy * dy)
        softening_scaled = (gravity_param * mass) / potential_max_scaled
        denom = np.maximum(d - gravity_alpha * mass, softening_scaled)

        potential -= (gravity_param * mass) / denom

        active = (d > 0.0) & (d - gravity_alpha * mass > softening_scaled)
        f_mag = np.where(active, (gravity_param * mass) / (denom * denom), 0.0)
        d_safe = np.where(d > 0, d, 1.0)
        force_x += np.where(active, f_mag * (dx / d_safe), 0.0)
        force_y += np.where(active, f_mag * (dy / d_safe), 0.0)

    potential *= scale
    force_x *= scale * scale
    force_y *= scale * scale
    return potential, force_x, force_y


def bilinear_sample(field, px, py, grid_res):
    px = np.clip(px, 0, grid_res - 1)
    py = np.clip(py, 0, grid_res - 1)
    col = np.floor(px).astype(int)
    row = np.floor(py).astype(int)
    next_col = np.minimum(col + 1, grid_res - 1)
    next_row = np.minimum(row + 1, grid_res - 1)
    fx = px - col
    fy = py - row

    f00 = field[row, col]
    f10 = field[row, next_col]
    f01 = field[next_row, col]
    f11 = field[next_row, next_col]
    return ((1 - fx) * (1 - fy) * f00 + fx * (1 - fy) * f10
            + (1 - fx) * fy * f01 + fx * fy * f11)


# -----------------------------
# reset_control_points の移植（simulation.rs 64-96行目）
# -----------------------------
def init_control_points(positions_grid, edges, spacing):
    spacing = max(spacing, 1.0)
    control_points = []
    for s, t in edges:
        p0, p1 = positions_grid[s], positions_grid[t]
        dist = np.linalg.norm(p1 - p0)
        n = max(int(dist // spacing) + 2, 3)
        t_vals = np.linspace(0, 1, n)
        pts = p0[None, :] + t_vals[:, None] * (p1 - p0)[None, :]
        control_points.append(pts)
    return control_points


# -----------------------------
# step の移植（simulation.rs 146-197行目）
# -----------------------------
def step(control_points, force_x, force_y, grid_res,
         spring_k=SPRING_K, dt=DT, damping=DAMPING, max_disp=5.0):
    new_control_points = []
    for pts in control_points:
        current = pts
        new_pts = current.copy()
        if len(current) > 2:
            prev = current[:-2]
            curr = current[1:-1]
            nxt = current[2:]

            f_spring = spring_k * (prev + nxt - 2.0 * curr)

            fgx = bilinear_sample(force_x, curr[:, 0], curr[:, 1], grid_res)
            fgy = bilinear_sample(force_y, curr[:, 0], curr[:, 1], grid_res)
            f_grav = np.stack([fgx, fgy], axis=1)

            f = f_spring + f_grav
            d = f * dt
            d_len = np.linalg.norm(d, axis=1, keepdims=True)
            scale_factor = np.minimum(1.0, max_disp / np.maximum(d_len, EPS))
            d = d * scale_factor

            updated = curr + d * damping
            updated = np.clip(updated, 0.0, grid_res - 1)
            new_pts[1:-1] = updated
        new_control_points.append(new_pts)
    return new_control_points


def run_gravity_eb(node_ids, positions, edges, verbose=True):
    positions_grid = positions / GRID_SCALE
    node_mass = np.full(len(node_ids), DEFAULT_MASS)

    if verbose:
        print("computing potential/force fields...")
    t0 = time.time()
    _, force_x, force_y = compute_fields(GRID_RES, positions_grid, node_mass)
    if verbose:
        print(f"   done [{time.time()-t0:.1f}s]")

    control_points = init_control_points(positions_grid, edges, CONTROL_POINT_SPACING)

    for i in range(N_STEPS):
        if verbose and ((i + 1) % 50 == 0 or i == 0):
            print(f"step {i+1}/{N_STEPS}")
        control_points = step(control_points, force_x, force_y, GRID_RES)

    bundled_edges = [pts * GRID_SCALE for pts in control_points]
    return bundled_edges


if __name__ == "__main__":
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    stem = dataset_path.stem
    out_name = "gravity_eb_result.json" if stem == "airlines" else f"{stem}_gravity_eb_result.json"

    print(f"Loading {dataset_path}...")
    node_ids, positions, edges = load_nodes_edges(dataset_path)
    print(f"nodes={len(node_ids)}, edges={len(edges)}")

    t0 = time.time()
    bundled = run_gravity_eb(node_ids, positions, edges)
    print(f"gravity-edge-bundling done in {time.time()-t0:.1f}s")

    output = [resample_to_n_points(e, N_SAMPLES).tolist() for e in bundled]
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / out_name, "w") as f:
        json.dump(output, f)
    print(f"{RESULTS_DIR / out_name} に保存しました。")
