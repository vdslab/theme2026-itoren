"""
datashader (HoloViz) の hammer_bundle を使ったKDEEB系実装を、他手法(run_fdeb.js,
gravity_edge_bundling.py等)と同じ入出力規約で実行するランナー。

hammer_bundleはHurter, Ersoy & Telea (EuroVis 2012, KDEEBの元論文) のバリアントで、
カーネル密度推定に基づくバンドリングをPure Pythonで実装したもの。
https://datashader.org/user_guide/Networks.html

使い方:
    python run_kdeeb.py [データセット.json]
    (省略時は ../airlines.json)

入力: airlines.jsonと同じスキーマ ({"nodes":[{id,x,y}], "edges":[{source,target}]})
出力: results/kdeeb_result.json (airlines.jsonの場合) または
      results/<データセット名>_kdeeb_result.json
      (evaluate.pyが読める形式: エッジごとの[x,y]点列のJSON配列、WINDOW_SIZE=10000空間)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from datashader.bundling import hammer_bundle

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_DATASET = PROJECT_ROOT / "airlines.json"

WINDOW_SIZE = 10000
N_SAMPLES = 100  # 出力時にevaluate.pyの他手法と合わせて統一する点数


def load_nodes_edges(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    raw_x = [n["x"] for n in data["nodes"]]
    raw_y = [n["y"] for n in data["nodes"]]
    min_x, max_x_val = min(raw_x), max(raw_x)
    min_y, max_y_val = min(raw_y), max(raw_y)
    scale = WINDOW_SIZE / max(max_x_val - min_x, max_y_val - min_y)
    positions = np.array([
        [(n["x"] - min_x) * scale, (n["y"] - min_y) * scale] for n in data["nodes"]
    ])
    node_ids = [str(n["id"]) for n in data["nodes"]]
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


def split_bundled_edges(bundled_df, n_edges):
    """hammer_bundleの出力(NaN区切りの1本のdataframe)をエッジごとの点列に分割する。"""
    xs = bundled_df["x"].to_numpy()
    ys = bundled_df["y"].to_numpy()
    nan_idx = np.where(np.isnan(xs))[0]
    if len(nan_idx) != n_edges:
        raise ValueError(f"想定エッジ数と区切り数が一致しません ({len(nan_idx)} != {n_edges})")
    edges = []
    start = 0
    for end in nan_idx:
        edges.append(np.stack([xs[start:end], ys[start:end]], axis=1))
        start = end + 1
    return edges


def run_kdeeb(positions, edges, verbose=True):
    nodes_df = pd.DataFrame({"x": positions[:, 0], "y": positions[:, 1]})
    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    if verbose:
        print("running hammer_bundle (KDEEB)...")
    bundled_df = hammer_bundle(nodes_df, edges_df)
    return split_bundled_edges(bundled_df, len(edges))


if __name__ == "__main__":
    dataset_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATASET
    stem = dataset_path.stem
    out_name = "kdeeb_result.json" if stem == "airlines" else f"{stem}_kdeeb_result.json"

    print(f"Loading {dataset_path}...")
    node_ids, positions, edges = load_nodes_edges(dataset_path)
    print(f"nodes={len(node_ids)}, edges={len(edges)}")

    t0 = time.time()
    bundled = run_kdeeb(positions, edges)
    print(f"KDEEB (hammer_bundle) done in {time.time()-t0:.1f}s")

    output = [resample_to_n_points(e, N_SAMPLES).tolist() for e in bundled]
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / out_name, "w") as f:
        json.dump(output, f)
    print(f"{RESULTS_DIR / out_name} に保存しました。")
