import os
import json
import itertools
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 画面表示せずファイル保存だけ行う(ウィンドウを出さない)
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates

# 結果画像の保存先フォルダ
OUTPUT_DIR = "param_sweep_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 試すパラメータの候補値
# ここに書いた値の全組み合わせ(直積)を総当たりで実行する
# -----------------------------
PARAM_GRID = {
    "SPRING_CONSTANT": [10, 40, 100],
    "STEP_SIZE": [50, 100, 200],
    "STEP_SIZE_DECAY": [0.995, 0.999],
    "DEFAULT_NODE_MASS": [1, 4, 10],
    "N_ITERATIONS": [50, 100],
    "N_SAMPLES_PER_EDGE": [30, 60],
    "GRID_SIZE": [10000],
}

# -----------------------------
# ノード・エッジデータの読み込み(test2.pyと同じデータソース)
# -----------------------------
with open("airlines.json", "r") as f:
    data = json.load(f)

raw_x = [n["x"] for n in data["nodes"]]
raw_y = [n["y"] for n in data["nodes"]]
min_x, max_x_val = min(raw_x), max(raw_x)
min_y, max_y_val = min(raw_y), max(raw_y)

json_edges = data.get("edges", data.get("links", []))
edge_keys = [(str(e["source"]), str(e["target"])) for e in json_edges]


def build_nodes(grid_size):
    """元座標をgrid_sizeの範囲にスケールしたノード座標を作る"""
    scale = grid_size / max(max_x_val - min_x, max_y_val - min_y)
    return {
        str(n["id"]): np.array([(n["x"] - min_x) * scale, (n["y"] - min_y) * scale])
        for n in data["nodes"]
    }


# ポテンシャル場はGRID_SIZEとDEFAULT_NODE_MASSだけで決まり、計算コストが一番高い。
# 他のパラメータ(SPRING_CONSTANTなど)を変えるたびに毎回作り直すと無駄なので、
# (grid_size, node_mass)をキーにキャッシュして使い回す。
_potential_cache = {}


def get_potential_field(grid_size, node_mass):
    """指定したgrid_size・node_massのポテンシャル場と勾配を計算(キャッシュあり)"""
    key = (grid_size, node_mass)
    if key in _potential_cache:
        return _potential_cache[key]

    nodes = build_nodes(grid_size)
    masses = {k: node_mass for k in nodes.keys()}

    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    potential_field = np.zeros_like(xx, dtype=float)
    for name, pos in nodes.items():
        dist_sq = (xx - pos[0]) ** 2 + (yy - pos[1]) ** 2
        potential_field -= masses[name] / np.sqrt(dist_sq + 1e-9)
    grad_y, grad_x = np.gradient(potential_field)

    result = (nodes, potential_field, grad_x, grad_y)
    _potential_cache[key] = result
    return result


def run_bundling(nodes, grad_x, grad_y, n_samples, step_size, step_decay, spring_k, n_iterations):
    """力場+ばねモデルでエッジをバンドリングし、最終的なエッジ座標を返す(test2.pyの反復計算部分)"""
    # 各エッジをn_samples個の点で初期化(始点-終点の直線)
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
            # 重力ポテンシャル場の勾配方向に各点を引き寄せる
            for k in range(1, len(edge) - 1):
                pos = edge[k]
                dx = map_coordinates(grad_x, [[pos[1]], [pos[0]]], order=1)[0]
                dy = map_coordinates(grad_y, [[pos[1]], [pos[0]]], order=1)[0]
                new_edge[k] += current_step * np.array([dx, dy])
            # 両隣の点とのばね力で直線に戻そうとする
            edge_length = np.linalg.norm(edge[-1] - edge[0])
            kp = spring_k / edge_length
            for k in range(1, len(edge) - 1):
                current = new_edge[k]
                left = new_edge[k - 1]
                right = new_edge[k + 1]
                new_edge[k] += kp * ((left - current) + (right - current))
            bundled_edges[j] = new_edge
        current_step *= step_decay  # ステップ幅を徐々に減衰させて収束させる
    return bundled_edges


def save_result_image(nodes, potential_field, bundled_edges, grid_size, params, out_path):
    """バンドリング結果を1枚の画像として保存し、使ったパラメータをテキストで焼き込む"""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(potential_field, cmap="bone_r", origin="lower",
              extent=[0, grid_size, 0, grid_size], alpha=0.3)
    for edge in bundled_edges:
        ax.plot(edge[:, 0], edge[:, 1], color="cyan", linewidth=1.0, alpha=0.7)
    for pos in nodes.values():
        ax.scatter(pos[0], pos[1], s=15, c="magenta", zorder=10)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_aspect("equal")
    ax.set_facecolor("black")

    # 左上にこの画像で使ったパラメータ一覧を表示
    param_text = "\n".join(f"{k} = {v}" for k, v in params.items())
    ax.text(
        0.02, 0.98, param_text,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=9, color="white", family="monospace",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="white", boxstyle="round,pad=0.4"),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)  # 開いたままだとメモリを圧迫するので必ず閉じる


def make_filename(params):
    """パラメータの値をそのままファイル名にする(内容が一目でわかるように)"""
    parts = [f"{k}-{v}" for k, v in params.items()]
    return "_".join(parts) + ".png"


def main():
    keys = list(PARAM_GRID.keys())
    value_lists = [PARAM_GRID[k] for k in keys]
    combos = list(itertools.product(*value_lists))  # 全パラメータの直積(総当たり)
    total = len(combos)
    print(f"合計 {total} 通りのパラメータの組み合わせを実行します。")

    for idx, combo in enumerate(combos, start=1):
        params = dict(zip(keys, combo))
        filename = make_filename(params)
        out_path = os.path.join(OUTPUT_DIR, filename)

        # 既に保存済みならスキップ(途中で止めても再開できるようにするため)
        if os.path.exists(out_path):
            print(f"[{idx}/{total}] skip (already exists): {filename}")
            continue

        t0 = time.time()
        nodes, potential_field, grad_x, grad_y = get_potential_field(
            params["GRID_SIZE"], params["DEFAULT_NODE_MASS"]
        )
        bundled_edges = run_bundling(
            nodes, grad_x, grad_y,
            n_samples=params["N_SAMPLES_PER_EDGE"],
            step_size=params["STEP_SIZE"],
            step_decay=params["STEP_SIZE_DECAY"],
            spring_k=params["SPRING_CONSTANT"],
            n_iterations=params["N_ITERATIONS"],
        )
        save_result_image(nodes, potential_field, bundled_edges, params["GRID_SIZE"], params, out_path)
        elapsed = time.time() - t0
        print(f"[{idx}/{total}] saved {filename} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
