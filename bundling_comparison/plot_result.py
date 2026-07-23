"""
バンドリング結果のJSON(evaluate.pyと同じ形式: エッジごとの[x,y]点列の配列)を
PNG画像として可視化する汎用スクリプト。

使い方:
    python plot_result.py [--dataset <データセット.json>] [--out <出力png>] <結果ファイル.json>

--dataset を省略すると airlines.json を使い、ノード位置も薄く重ねて描画する。
--out を省略すると results/<結果ファイル名(拡張子抜き)>.png に保存する。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_DATASET = PROJECT_ROOT / "airlines.json"

WINDOW_SIZE = 10000


def resolve_result_path(path):
    p = Path(path)
    if p.exists():
        return p
    if (RESULTS_DIR / p.name).exists():
        return RESULTS_DIR / p.name
    raise FileNotFoundError(f"{path} が見つかりません（{RESULTS_DIR} も探索しましたが見つかりませんでした）")


def load_nodes(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    raw_x = [n["x"] for n in data["nodes"]]
    raw_y = [n["y"] for n in data["nodes"]]
    min_x, max_x_val = min(raw_x), max(raw_x)
    min_y, max_y_val = min(raw_y), max(raw_y)
    scale = WINDOW_SIZE / max(max_x_val - min_x, max_y_val - min_y)
    return np.array([[(n["x"] - min_x) * scale, (n["y"] - min_y) * scale] for n in data["nodes"]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                         help="ノード位置を重ねて描画するデータセットJSON（省略時はairlines.json）")
    parser.add_argument("--out", default=None, help="出力png（省略時はresults/<結果ファイル名>.png）")
    parser.add_argument("result_file", help="可視化する結果ファイル")
    args = parser.parse_args()

    result_path = resolve_result_path(args.result_file)
    print(f"Loading {result_path}...")
    with open(result_path, "r") as f:
        edges = json.load(f)
    edges = [np.array(e) for e in edges]
    print(f"edges={len(edges)}")

    nodes = load_nodes(args.dataset)

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{Path(args.result_file).stem}.png"

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    for edge in edges:
        ax.plot(edge[:, 0], edge[:, 1], color="cyan", linewidth=0.6, alpha=0.5)
    ax.scatter(nodes[:, 0], nodes[:, 1], s=6, c="magenta", zorder=10, alpha=0.8)
    ax.set_xlim(0, WINDOW_SIZE)
    ax.set_ylim(0, WINDOW_SIZE)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="black")
    plt.close(fig)
    print(f"{out_path} に保存しました。")


if __name__ == "__main__":
    main()
