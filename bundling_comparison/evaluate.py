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
from scipy.spatial import cKDTree

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


def fit_scale_transform(raw_edges, nodes_dict, edge_keys):
    """各エッジの始点・終点は本来対応するノードの座標と一致するはずという制約を使い、
    出力座標系をノード座標系(WINDOW_SIZE基準)へ合わせる一様スケール+平行移動を最小二乗で推定する。
    手法によって内部で使うキャンバスサイズ・原点がまちまちで、座標系が食い違うことがある
    （例: 別のcanvas_width/heightで計算してJSON化した結果）ため、読み込み時に必ず補正する。
    既に正しい座標系の結果ファイルではscale≈1.0, translation≈0になり実質何も変わらない。"""
    pred_pts, true_pts = [], []
    for raw, (s_id, t_id) in zip(raw_edges, edge_keys):
        raw = np.asarray(raw, dtype=float)
        pred_pts.append(raw[0])
        true_pts.append(nodes_dict[s_id])
        pred_pts.append(raw[-1])
        true_pts.append(nodes_dict[t_id])
    pred = np.array(pred_pts)
    true = np.array(true_pts)
    pred_mean = pred.mean(axis=0)
    true_mean = true.mean(axis=0)
    pred_c = pred - pred_mean
    true_c = true - true_mean
    denom = np.sum(pred_c * pred_c)
    if denom < 1e-9:
        return 1.0, np.zeros(2)
    scale = float(np.sum(pred_c * true_c) / denom)
    translation = true_mean - scale * pred_mean
    return scale, translation


def load_result_file(path, n_edges, nodes_dict, edge_keys):
    """バンドリング手法の出力JSON(エッジごとの点列)を読み込み、ノード座標系に
    スケール補正した上でN_SAMPLES点にリサンプリングする。
    エッジ数がデータセットと一致しない場合は結果ファイルの取り違えとみなしエラーにする。"""
    with open(path, "r") as f:
        raw = json.load(f)
    if len(raw) != n_edges:
        raise ValueError(f"{path}: エッジ数が airlines.json と一致しません ({len(raw)} != {n_edges})")
    scale, translation = fit_scale_transform(raw, nodes_dict, edge_keys)
    if abs(scale - 1.0) > 1e-3 or np.linalg.norm(translation) > 1e-3:
        print(f"  座標スケールを自動補正しました: scale={scale:.4f}, translation={translation}")
    raw = [np.asarray(e, dtype=float) * scale + translation for e in raw]
    return [resample_to_n_points(e, N_SAMPLES) for e in raw]


def compute_data_bounds(nodes_dict, edge_lists, padding_frac=0.02):
    """ノード＋全手法のエッジ座標を合わせた実際の座標範囲(左下min_xy, 一辺の長さextent)を返す。
    バンドリング手法によっては制御点がノードの外接範囲(WINDOW_SIZE)より外側に膨らむ
    （オーバーシュートする）ことがあり、固定のWINDOW_SIZEを前提にすると、はみ出た部分が
    ラスタ化・グリッド集計から黙って切り捨てられてしまう。ink/ambiguity/grid系の指標と
    可視化はすべてこの実データ範囲を共通の基準にすることで、全手法を同じ枠内で公平に
    比較しつつ、何も画面の外に切り捨てられないようにする。縦横比を保つため、
    範囲は正方形（1辺=extent）にそろえる。"""
    pts = [np.array(list(nodes_dict.values()))]
    for edges in edge_lists:
        pts.append(np.concatenate(edges, axis=0))
    all_pts = np.concatenate(pts, axis=0)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)
    extent = float(np.max(max_xy - min_xy))
    pad = extent * padding_frac
    return min_xy - pad, extent + 2 * pad


def compute_typical_spacing(nodes_dict):
    """ノードごとの最近傍ノードまでの距離の中央値を返す。レイアウト上で「普通の間隔」が
    どれくらいかを表す、ノード座標だけから自動的に決まるスケール
    （data_extentのようなキャンバス全体のサイズとは違い、ユーザーが指定するパラメータでもない）。
    平均ではなく中央値を使うのは、同一・近接座標のノードが少数でも混ざっていると
    平均は0近くに強く引っ張られてしまい、どんなデータセットでも安定して「普通の間隔」を
    表せなくなるため。node_confusion_scoresで「近い」「近くない」を判定する基準として使う。"""
    pts = np.array(list(nodes_dict.values()))
    tree = cKDTree(pts)
    # k=1は自分自身(距離0)なのでk=2で最近傍の"他"ノードまでの距離を取る
    dists, _ = tree.query(pts, k=2)
    return float(np.median(dists[:, 1]))


# -----------------------------
# 評価指標
# -----------------------------
def to_canvas(edge_list, data_min, data_extent, canvas_res=CANVAS_RES):
    """実データ範囲[data_min, data_min+data_extent]からラスタ描画用のCANVAS_RES解像度座標系へ縮小する"""
    s = canvas_res / data_extent
    return [(e - data_min) * s for e in edge_list]


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


def ink_reduction(bundled_edges, straight_edges, data_min, data_extent):
    """バンドリングによって描画インク（占有ピクセル数）がどれだけ減ったかを割合で返す。
    1.0に近いほど、直線描画時より視覚的な線の量が大きく減った（＝バンドリング効果が大きい）ことを示す。"""
    straight_canvas = to_canvas(straight_edges, data_min, data_extent)
    bundled_canvas = to_canvas(bundled_edges, data_min, data_extent)
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


def ambiguity(edge_list, data_min, data_extent):
    """各ピクセルに何本のエッジが重なって通っているかの平均値。
    値が大きいほど「どのエッジがどれだか区別しにくい」＝視覚的な曖昧さが高いことを示す。"""
    edge_list_canvas = to_canvas(edge_list, data_min, data_extent)
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


def node_confusion_scores(nodes_dict, node_id_list, edge_keys, edge_list, typical_spacing):
    """各ノードについて、非接続の全エッジとの「紛らわしさ」を合計したスコアを返す（{ノードID: 合計スコア}）。
        x = max(0, typical_spacing - d(v, e))   # エッジ全体への最短距離がtypical_spacingより近ければ大きい
        y = d(v, 最寄りの端点)                    # エッジの端点に近いほど小さい
        score(v, e) = x * y
    を、そのノードにとって非接続な全エッジで合計する。端点のすぐそばにいる
    （＝グラフ構造的に当然近い）ケースはyが小さく効かなくなり、typical_spacing
    （レイアウト上の「普通の間隔」）より遠いエッジはxが0になって完全に効かなくなるので、
    「端点からは遠いのに経路自体には異様に近い」ケース（バンドリングが無関係なノードの
    すぐそばを通ってしまっている状態）だけが強調される。data_extent（キャンバス全体の
    サイズ）を基準にするとほとんどのエッジのxが大きい値のまま残ってしまい合計が
    「何でもない普通のエッジ」の数で決まってしまうため、代わりにノード間の実際の間隔
    (typical_spacing)を基準にする。割り算を使わないため、d(v,e)が0に近くても発散しない。"""
    n_edges = len(edge_list)
    seg_per_edge = edge_list[0].shape[0] - 1

    A = np.concatenate([e[:-1] for e in edge_list], axis=0)
    B = np.concatenate([e[1:] for e in edge_list], axis=0)
    AB = B - A
    AB_len_sq = np.sum(AB ** 2, axis=1)
    AB_len_sq = np.where(AB_len_sq < 1e-12, 1e-12, AB_len_sq)

    sources = np.array([s for s, t in edge_keys])
    targets = np.array([t for s, t in edge_keys])
    source_pts = np.array([nodes_dict[s] for s in sources])
    target_pts = np.array([nodes_dict[t] for t in targets])

    scores = {}
    for v in node_id_list:
        p = nodes_dict[v]
        t = np.clip(np.sum((p - A) * AB, axis=1) / AB_len_sq, 0, 1)
        closest = A + t[:, None] * AB
        d = np.linalg.norm(p - closest, axis=1)
        d_min_per_edge = d.reshape(n_edges, seg_per_edge).min(axis=1)

        d_nearest_endpoint = np.minimum(
            np.linalg.norm(p - source_pts, axis=1),
            np.linalg.norm(p - target_pts, axis=1),
        )

        own_mask = (sources == v) | (targets == v)
        x = np.maximum(0.0, typical_spacing - d_min_per_edge[~own_mask])
        y = d_nearest_endpoint[~own_mask]
        scores[v] = float(np.sum(x * y))
    return scores


DIST_THRESHOLDS = [10, 100, 1000]  # node_edge_min_distancesの単位(px, WINDOW_SIZE=10000基準)でのしきい値


def distance_threshold_fractions(dist, thresholds=DIST_THRESHOLDS):
    """(ノード, エッジ)最短距離のうち、各しきい値以下のペアが占める割合を返す。
    グリッドに区切らず実際の距離そのもので集計するので、グリッド境界に依存した
    アーティファクトが出ない。値が大きいほど、そのしきい値以内までノードに接近している
    （＝視覚的に衝突・誤読しやすい）ペアが多いことを示す。"""
    return {t: float(np.mean(dist <= t)) for t in thresholds}


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
    edge_lists = {"Straight": straight_edges}

    # 先に全手法のエッジ(スケール補正済み)を読み込んでおく。バンドリングで制御点が
    # ノード範囲より外側に膨らむ(オーバーシュートする)手法があるため、ink/ambiguity/grid系の
    # 指標や可視化で使う描画範囲は、ノードと全手法の実際の座標を合わせてから決める必要がある。
    for path in result_paths:
        resolved = resolve_result_path(path)
        print(f"Loading {path} ({resolved})...")
        edge_lists[path] = load_result_file(resolved, n_edges, nodes, edges)

    data_min, data_extent = compute_data_bounds(nodes, edge_lists.values())
    typical_spacing = compute_typical_spacing(nodes)

    all_dists = {}
    all_confusion = {}
    summary = {}

    # バンドリング無しの直線描画を基準（distortion=1.0固定）として先に評価しておく
    print("Evaluating Straight (baseline)...")
    dist_straight = node_edge_min_distances(nodes, node_ids, edges, straight_edges)
    conf_straight = node_confusion_scores(nodes, node_ids, edges, straight_edges, typical_spacing)
    all_dists["Straight"] = dist_straight
    all_confusion["Straight"] = conf_straight
    summary["Straight"] = {
        "distortion": 1.0,
        "ambiguity": ambiguity(straight_edges, data_min, data_extent),
        "dist_threshold_fractions": distance_threshold_fractions(dist_straight),
        "confusion_score_mean": float(np.mean(list(conf_straight.values()))),
        "confusion_score_median": float(np.median(list(conf_straight.values()))),
    }

    # 引数で渡された各手法の結果ファイルについて指標を計算
    for path in result_paths:
        print(f"Evaluating {path}...")
        bundled = edge_lists[path]
        ir, ink_before, ink_after = ink_reduction(bundled, straight_edges, data_min, data_extent)
        dis = distortion(bundled, straight_edges)
        amb = ambiguity(bundled, data_min, data_extent)
        dist = node_edge_min_distances(nodes, node_ids, edges, bundled)
        conf = node_confusion_scores(nodes, node_ids, edges, bundled, typical_spacing)
        all_dists[path] = dist
        all_confusion[path] = conf
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
            "dist_threshold_fractions": distance_threshold_fractions(dist),
            "confusion_score_mean": float(np.mean(list(conf.values()))),
            "confusion_score_median": float(np.median(list(conf.values()))),
        }

    # 結果を整形してコンソールに表として出力
    print("\n=== Summary ===")
    thresh_headers = " ".join(f"{'<=' + str(t):>10}" for t in DIST_THRESHOLDS)
    header = (f"{'File':<28} {'InkReduction':>12} {'Distortion':>11} {'Ambiguity':>10} "
              f"{'Dist(min)':>13} {thresh_headers} {'Confusion(mean)':>16}")
    print(header)
    print("-" * len(header))
    s = summary["Straight"]
    s_thresh = " ".join(f"{s['dist_threshold_fractions'][t] * 100:>9.2f}%" for t in DIST_THRESHOLDS)
    print(f"{'Straight':<28} {'--':>12} {s['distortion']:>11.4f} {s['ambiguity']:>10.4f} "
          f"{np.median(all_dists['Straight']):>13.2f} {s_thresh} {s['confusion_score_mean']:>16.2f}")
    for path in result_paths:
        r = summary[path]
        r_thresh = " ".join(f"{r['dist_threshold_fractions'][t] * 100:>9.2f}%" for t in DIST_THRESHOLDS)
        print(f"{path:<28} {r['ink_reduction']*100:>11.2f}% {r['distortion']:>11.4f} "
              f"{r['ambiguity']:>10.4f} {r['node_edge_dist_min']:>13.2f} {r_thresh} "
              f"{r['confusion_score_mean']:>16.2f}")

    # 指標のサマリーをJSONとして保存（他ツールでの再利用・比較のため）
    result_json_path = RESULTS_DIR / f"{label}evaluate_result.json"
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{result_json_path} に保存しました。")

    # ノード-エッジ最短距離のECDF（経験累積分布関数）。x軸をlogにすることで、
    # 衝突に近い小さい距離域での手法間の差を箱ひげ図より詳細に見比べられるようにする。
    fig, ax = plt.subplots(figsize=(8, 6))
    for l in labels:
        d = np.sort(all_dists[l])
        # log軸に0は乗らないため、正確に0の距離（完全な重なり）は極小値に丸めてプロットする
        d = np.where(d <= 0, 1e-6, d)
        y = np.arange(1, len(d) + 1) / len(d)
        # 大規模データセットだと点数が数百万になりメモリ・描画コストが跳ね上がるため、
        # 曲線の見た目を保ったまま等間隔に間引く（ソート済みなので形状は変わらない）
        if len(d) > 20000:
            idx = np.linspace(0, len(d) - 1, 20000).astype(int)
            d, y = d[idx], y[idx]
        ax.plot(d, y, label=l, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Node-Edge Minimum Distance (px, log scale)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Node-Edge Proximity ECDF\n(curves further left/up = more collisions)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    ecdf_path = RESULTS_DIR / f"{label}evaluate_ecdf.png"
    fig.savefig(ecdf_path, dpi=120)
    plt.close(fig)
    print(f"{ecdf_path} に保存しました。")

    # ノードごとの「紛らわしさ」合計スコアの分布を手法ごとに箱ひげ図で可視化
    # （中央値は低くても裾の外れ値が突出して伸びている手法＝一部のノードだけ異常に
    # 混雑している、という見方をする。中央値だけで「マシ」と判断しないよう注意）
    fig, ax = plt.subplots(figsize=(2.5 + 2 * len(labels), 8))
    ax.boxplot([list(all_confusion[l].values()) for l in labels], labels=labels,
               showfliers=True, widths=0.5)
    ax.set_ylabel("Node Confusion Score (sum over edges)")
    ax.set_title("Node Confusion Score Comparison\n(higher = more nodes crowded by unrelated edge paths)")
    ax.grid(True, axis="y", alpha=0.3, which="both")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    confusion_boxplot_path = RESULTS_DIR / f"{label}evaluate_boxplot.png"
    fig.savefig(confusion_boxplot_path, dpi=120)
    plt.close(fig)
    print(f"{confusion_boxplot_path} に保存しました。")


if __name__ == "__main__":
    main()
