"""
gravity_edge_bundling.py の比較用ベースラインには手を入れず、パラメータを強めに
チューニングした版を別出力するランナー。

デフォルト(GRAVITY_PARAM=0.03125, SPRING_K=0.05, max_disp=5.0)だと、直線に戻す力
(スプリング)に対して重力が弱く、曲がりが控えめだったため、
重力を強く・スプリングを弱く・1ステップの可動域を広げて、より積極的に曲げる。

compute_fields/stepが元々gravity_param/spring_k/max_dispをキーワード引数で
上書きできる作りになっているので、gravity_edge_bundling.pyをモジュールとして
importしてそれらを差し替えるだけでよい(物理演算部分の再実装は不要)。

使い方:
    python gravity_edge_bundling_tuned.py [データセット.json]
    (省略時は ../airlines.json)

出力: results/<データセット名>_gravity_eb_tuned_result.json
      (gravity_edge_bundling.pyの通常出力とは別名。比較実験のベースラインは上書きしない)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import gravity_edge_bundling as geb

RESULTS_DIR = geb.RESULTS_DIR
DEFAULT_DATASET = geb.DEFAULT_DATASET
GRID_RES = geb.GRID_RES
GRID_SCALE = geb.GRID_SCALE
N_SAMPLES = geb.N_SAMPLES

# チューニングしたパラメータ
TUNED = {
    "gravity_param": 0.15,  # デフォルト0.03125 → 重力(曲げる力)を強く
    "spring_k": 0.02,       # デフォルト0.05 → 張力(直線に戻す力)を弱く
    "n_steps": 600,         # デフォルト400 → 収束にかける時間を長く
    "max_disp": 8.0,        # デフォルト5.0 → 1ステップあたりの最大移動量を拡大
}


def run_gravity_eb_tuned(node_ids, positions, edges, verbose=True):
    positions_grid = positions / GRID_SCALE
    node_mass = np.full(len(node_ids), geb.DEFAULT_MASS)

    if verbose:
        print("computing potential/force fields...")
    t0 = time.time()
    _, force_x, force_y = geb.compute_fields(
        GRID_RES, positions_grid, node_mass,
        gravity_param=TUNED["gravity_param"],
    )
    if verbose:
        print(f"   done [{time.time()-t0:.1f}s]")

    control_points = geb.init_control_points(positions_grid, edges, geb.CONTROL_POINT_SPACING)

    n_steps = TUNED["n_steps"]
    for i in range(n_steps):
        if verbose and ((i + 1) % 50 == 0 or i == 0):
            print(f"step {i+1}/{n_steps}")
        control_points = geb.step(
            control_points, force_x, force_y, GRID_RES,
            spring_k=TUNED["spring_k"], max_disp=TUNED["max_disp"],
        )

    return [pts * GRID_SCALE for pts in control_points]


if __name__ == "__main__":
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    stem = dataset_path.stem
    out_name = f"{stem}_gravity_eb_tuned_result.json"

    print(f"Loading {dataset_path}...")
    node_ids, positions, edges = geb.load_nodes_edges(dataset_path)
    print(f"nodes={len(node_ids)}, edges={len(edges)}")

    print(f"Running gravity-edge-bundling (tuned: {TUNED})...")
    t0 = time.time()
    bundled = run_gravity_eb_tuned(node_ids, positions, edges)
    print(f"gravity-edge-bundling (tuned) done in {time.time()-t0:.1f}s")

    output = [geb.resample_to_n_points(e, N_SAMPLES).tolist() for e in bundled]
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / out_name, "w") as f:
        json.dump(output, f)
    print(f"{RESULTS_DIR / out_name} に保存しました。")
