"""
バンドリング結果を評価する汎用スクリプト。

使い方:
    python evaluate.py [--dataset <データセット.json>] <結果ファイル1.json> [<結果ファイル2.json> ...]

結果ファイルの形式(全手法共通):
    JSON配列。要素はエッジ1本ごとの [x, y] 点列（--datasetのedges順、
    WINDOW_SIZE=10000にスケールした座標系）。
    例: ours_result.json, fdeb_github_result.json, kdeeb_result.json

直線(バンドリング前)は --dataset (省略時 airlines.json) から自動的に基準として計算される。
評価対象を変えたいときはコマンドライン引数のファイル名を変えるだけでよい。
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_DATASET = PROJECT_ROOT / "airlines.json"

WINDOW_SIZE = 10000
N_SAMPLES = 100     # 指標計算を統一するため、全手法をこの点数にリサンプリングしてから比較する
CANVAS_RES = 1000    # Ink/Ambiguity計算用のラスタ解像度


def resolve_result_path(path):
    """そのままのパス、または results/ フォルダ内を探す（ファイル名だけ渡せばよいように）"""
    p = Path(path)
    if p.exists():
        return p
    if (RESULTS_DIR / p.name).exists():
        return RESULTS_DIR / p.name
    raise FileNotFoundError(f"{path} が見つかりません（{RESULTS_DIR} も探索しましたが見つかりませんでした）")


# -----------------------------
# データ読み込み
# -----------------------------
def load_nodes_edges(dataset_path):
    """データセットJSONを読み込み、ノード座標を原点合わせ＋WINDOW_SIZEスケールに正規化する。
    これにより、生の座標がどんな単位・範囲でも全データセットを同じ座標系で比較できる。"""
    with open(dataset_path, "r") as f:
        data = json.load(f)
    raw_x = [n["x"] for n in data["nodes"]]
    raw_y = [n["y"] for n in data["nodes"]]
    min_x, max_x_val = min(raw_x), max(raw_x)
    min_y, max_y_val = min(raw_y), max(raw_y)
    # x/yのうち広い方の幅がWINDOW_SIZEに収まるように統一スケールを決める（縦横比を保つため）
    scale = WINDOW_SIZE / max(max_x_val - min_x, max_y_val - min_y)
    nodes = {
        str(n["id"]): np.array([(n["x"] - min_x) * scale, (n["y"] - min_y) * scale])
        for n in data["nodes"]
    }
    edges = []
    # データセットによってキー名が"edges"/"links"どちらの場合もあるので両対応
    json_edges = data.get("edges", data.get("links", []))
    for e in json_edges:
        edges.append((str(e["source"]), str(e["target"])))
    return nodes, edges


def make_straight_edges(nodes, edges, n_samples):
    """バンドリング前の基準となる直線エッジを、始点-終点をn_samples点で等間隔に補間して生成する"""
    straight_edges = []
    for start_key, end_key in edges:
        s, t = nodes[start_key], nodes[end_key]
        edge = np.array([
            np.linspace(s[0], t[0], n_samples),
            np.linspace(s[1], t[1], n_samples),
        ]).T
        straight_edges.append(edge)
    return straight_edges


def resample_to_n_points(edge, n):
    """折れ線エッジを弧長パラメータ化し、n点に等間隔でリサンプリングする。
    手法ごとに出力される点数が異なっていても指標計算を揃えられるようにするため。"""
    edge = np.asarray(edge, dtype=float)
    diffs = np.diff(edge, axis=0)
    seg_len = np.sqrt(np.sum(diffs ** 2, axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = arc[-1]
    if total < 1e-9:
        # 始点と終点が同じ（長さ0）の退化エッジは補間できないので同じ点を複製する
        return np.tile(edge[0], (n, 1))
    target = np.linspace(0, total, n)
    fx = interp1d(arc, edge[:, 0])
    fy = interp1d(arc, edge[:, 1])
    return np.stack([fx(target), fy(target)], axis=1)


def load_result_file(path, n_edges):
    """バンドリング手法の出力JSON(エッジごとの点列)を読み込み、N_SAMPLES点にリサンプリングする。
    エッジ数がデータセットと一致しない場合は結果ファイルの取り違えとみなしエラーにする。"""
    with open(path, "r") as f:
        raw = json.load(f)
    if len(raw) != n_edges:
        raise ValueError(f"{path}: エッジ数が airlines.json と一致しません ({len(raw)} != {n_edges})")
    return [resample_to_n_points(np.array(e), N_SAMPLES) for e in raw]


# -----------------------------
# 評価指標
# -----------------------------
def to_canvas(edge_list, window_size=WINDOW_SIZE, canvas_res=CANVAS_RES):
    """WINDOW_SIZE座標系からラスタ描画用のCANVAS_RES解像度座標系へ縮小する"""
    s = canvas_res / window_size
    return [e * s for e in edge_list]


def count_ink_pixels(edge_list, canvas_res=CANVAS_RES):
    """全エッジを黒背景に描画し、線が乗っている（=インクがある）ピクセル数を数える。
    Ink Reductionの計算に使う。matplotlibで実際にラスタ化するのはアンチエイリアス・
    線の重なりを含めた「見た目上の面積」を再現するため。"""
    dpi = 100
    fig_inches = canvas_res / dpi
    fig, ax = plt.subplots(figsize=(fig_inches, fig_inches), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(0, canvas_res)
    ax.set_ylim(0, canvas_res)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    for edge in edge_list:
        ax.plot(edge[:, 0], edge[:, 1], color="cyan", linewidth=1.0, alpha=0.7)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return int(np.sum((r < 10) & (g > 50) & (b > 50)))


def ink_reduction(bundled_edges, straight_edges):
    """バンドリングによって描画インク（占有ピクセル数）がどれだけ減ったかを割合で返す。
    1.0に近いほど、直線描画時より視覚的な線の量が大きく減った（＝バンドリング効果が大きい）ことを示す。"""
    straight_canvas = to_canvas(straight_edges)
    bundled_canvas = to_canvas(bundled_edges)
    ink_before = count_ink_pixels(straight_canvas)
    ink_after = count_ink_pixels(bundled_canvas)
    return 1.0 - ink_after / ink_before, ink_before, ink_after


def distortion(bundled_edges, straight_edges):
    """バンドル後のエッジ長が直線距離の何倍に伸びたかの平均値。1.0に近いほど歪みが小さい。"""
    ratios = []
    for bundled, orig in zip(bundled_edges, straight_edges):
        direct_len = np.linalg.norm(orig[-1] - orig[0])
        if direct_len < 1e-9:
            continue
        bundled_len = float(np.sum(np.linalg.norm(np.diff(bundled, axis=0), axis=1)))
        ratios.append(bundled_len / direct_len)
    return float(np.mean(ratios)) if ratios else 1.0


def ambiguity(edge_list):
    """各ピクセルに何本のエッジが重なって通っているかの平均値。
    値が大きいほど「どのエッジがどれだか区別しにくい」＝視覚的な曖昧さが高いことを示す。"""
    edge_list_canvas = to_canvas(edge_list)
    pixel_edges = defaultdict(set)
    for j, edge in enumerate(edge_list_canvas):
        for i in range(len(edge) - 1):
            p0, p1 = edge[i], edge[i + 1]
            n_steps = max(int(np.linalg.norm(p1 - p0)) + 1, 2)
            pts = p0 + np.linspace(0, 1, n_steps)[:, None] * (p1 - p0)
            for pt in pts:
                pixel_edges[(int(round(pt[0])), int(round(pt[1])))].add(j)
    if not pixel_edges:
        return 1.0
    return float(np.mean([len(v) for v in pixel_edges.values()]))


def node_edge_min_distances(nodes_dict, node_id_list, edge_keys, edge_list):
    """v がエッジ e の端点でない (v, e) ペア全てについて最短距離 d_min(v,e) を計算"""
    n_edges = len(edge_list)
    seg_per_edge = edge_list[0].shape[0] - 1

    # 全エッジの全線分の始点(A)・終点(B)を1つの配列にまとめ、ノードごとにベクトル化して
    # 最近傍点への距離を一括計算する（ノード数×線分数のループを避けるため）
    A = np.concatenate([e[:-1] for e in edge_list], axis=0)
    B = np.concatenate([e[1:] for e in edge_list], axis=0)
    AB = B - A
    AB_len_sq = np.sum(AB ** 2, axis=1)
    AB_len_sq = np.where(AB_len_sq < 1e-12, 1e-12, AB_len_sq)

    sources = np.array([s for s, t in edge_keys])
    targets = np.array([t for s, t in edge_keys])

    all_distances = []
    for v in node_id_list:
        p = nodes_dict[v]
        # 点pから各線分ABへの垂線の足の位置tを求め、[0,1]にクランプして線分上の最近傍点を得る
        t = np.clip(np.sum((p - A) * AB, axis=1) / AB_len_sq, 0, 1)
        closest = A + t[:, None] * AB
        d = np.linalg.norm(p - closest, axis=1)
        d_min_per_edge = d.reshape(n_edges, seg_per_edge).min(axis=1)
        # vが端点になっているエッジ自身との距離は0になって当然なので比較対象から除外する
        own_mask = (sources == v) | (targets == v)
        all_distances.append(d_min_per_edge[~own_mask])
    return np.concatenate(all_distances)


# -----------------------------
# メイン
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET),
                         help="ノード/エッジのデータセットJSON（省略時は airlines.json）")
    parser.add_argument("--label", default=None,
                         help="出力ファイル名の接頭辞（省略時はデータセット名から自動決定）")
    parser.add_argument("result_files", nargs="+", help="評価する結果ファイル（複数可）")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    stem = dataset_path.stem
    # airlines(デフォルト)以外のデータセットを使うときは出力ファイル名が衝突しないよう接頭辞を付ける
    label = args.label or ("" if stem == "airlines" else f"{stem}_")
    result_paths = args.result_files

    print(f"Loading {dataset_path}...")
    nodes, edges = load_nodes_edges(dataset_path)
    node_ids = list(nodes.keys())
    n_edges = len(edges)
    straight_edges = make_straight_edges(nodes, edges, N_SAMPLES)

    labels = ["Straight"] + [p for p in result_paths]
    all_dists = {}
    summary = {}

    # バンドリング無しの直線描画を基準（distortion=1.0固定）として先に評価しておく
    print("Evaluating Straight (baseline)...")
    all_dists["Straight"] = node_edge_min_distances(nodes, node_ids, edges, straight_edges)
    summary["Straight"] = {
        "distortion": 1.0,
        "ambiguity": ambiguity(straight_edges),
    }

    # 引数で渡された各手法の結果ファイルについて指標を計算
    for path in result_paths:
        resolved = resolve_result_path(path)
        print(f"Evaluating {path} ({resolved})...")
        bundled = load_result_file(resolved, n_edges)
        ir, ink_before, ink_after = ink_reduction(bundled, straight_edges)
        dis = distortion(bundled, straight_edges)
        amb = ambiguity(bundled)
        dist = node_edge_min_distances(nodes, node_ids, edges, bundled)
        all_dists[path] = dist
        summary[path] = {
            "ink_reduction": ir,
            "ink_pixels_before": ink_before,
            "ink_pixels_after": ink_after,
            "distortion": dis,
            "ambiguity": amb,
            "node_edge_dist_median": float(np.median(dist)),
            "node_edge_dist_q1": float(np.percentile(dist, 25)),
            "node_edge_dist_q3": float(np.percentile(dist, 75)),
            "node_edge_dist_min": float(dist.min()),
        }

    # 結果を整形してコンソールに表として出力
    print("\n=== Summary ===")
    header = f"{'File':<28} {'InkReduction':>12} {'Distortion':>11} {'Ambiguity':>10} {'Dist(min)':>13}"
    print(header)
    print("-" * len(header))
    s = summary["Straight"]
    print(f"{'Straight':<28} {'--':>12} {s['distortion']:>11.4f} {s['ambiguity']:>10.4f} "
          f"{np.median(all_dists['Straight']):>13.2f}")
    for path in result_paths:
        r = summary[path]
        print(f"{path:<28} {r['ink_reduction']*100:>11.2f}% {r['distortion']:>11.4f} "
              f"{r['ambiguity']:>10.4f} {r['node_edge_dist_min']:>13.2f}")

    # 指標のサマリーをJSONとして保存（他ツールでの再利用・比較のため）
    result_json_path = RESULTS_DIR / f"{label}evaluate_result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{result_json_path} に保存しました。")

    # ノード-エッジ最短距離の分布を手法ごとに箱ひげ図で可視化（0に近い外れ値＝ノードへの衝突を示す）
    fig, ax = plt.subplots(figsize=(2.5 + 2 * len(labels), 8))
    ax.boxplot([all_dists[l] for l in labels], labels=labels, showfliers=True, widths=0.5)
    ax.set_ylabel("Node-Edge Minimum Distance (px)")
    ax.set_title("Node-Edge Proximity Comparison\n(lower whisker/outliers near 0 = collision)")
    ax.set_yscale("symlog")
    ax.grid(True, axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    boxplot_path = RESULTS_DIR / f"{label}evaluate_boxplot.png"
    fig.savefig(boxplot_path, dpi=120)
    plt.close(fig)
    print(f"{boxplot_path} に保存しました。")


if __name__ == "__main__":
    main()
