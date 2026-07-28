import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
import random
import json
import math

# -----------------------------
# パラメータ (Balanced Version)
# -----------------------------
N_ITERATIONS = 50
N_SAMPLES_PER_EDGE = 50
STEP_SIZE = 50
STEP_SIZE_DECAY = 0.99
SPRING_CONSTANT = 300
DEFAULT_NODE_MASS = 3000
WINDOW_SIZE = 10000  # キャンバス全体のサイズ
GRID_SIZE = 10      # タイル（グリッド）1辺のサイズ
SIGMA = 0            # ポテンシャル場に掛けるガウシアンフィルタのsigma（0で無効）
MAX_POTENTIAL = 5000  # ポテンシャル場の下限(絶対値の上限)。ノード直近での発散を防ぐ。Noneで無効
DENSITY_RADIUS = 150   # この半径内のノード数で質量を割り、密集地の重力を弱める
SPRING_SUBSTEPS = 3     # kp=0.5(安定限界)を1 iterationあたり複数回適用してテンションを強化する回数
GRAVITY_DECAY = 10

# -----------------------------
# ノードデータの生成 (Node data generation) - JSONから読み込み
# -----------------------------
print("0. Loading and normalizing node positions...")
with open('eurosis.json', 'r') as f:
    data = json.load(f)

# 1. すべてのノードからXとYの最小値・最大値を取得
raw_x = [n["x"] for n in data["nodes"]]
raw_y = [n["y"] for n in data["nodes"]]
min_x, max_x_val = min(raw_x), max(raw_x)
min_y, max_y_val = min(raw_y), max(raw_y)
# WINDOW_SIZEはパラメータで指定。元座標をWINDOW_SIZEにスケール（アスペクト比保持）
scale = WINDOW_SIZE / max(max_x_val - min_x, max_y_val - min_y)

# print("0. Generating parallel edges and a gravity well...")

# GRID_SIZE = 600  # 画面のサイズを固定

# # ノードの配置（左側に3つ、右側に3つ、中央に巨大な重力源を1つ）
# nodes = {
#     # 左側のスタート地点
#     "S1": np.array([100.0, 290.0]),
#     "S2": np.array([300.0, 280.0]),
#     "S3": np.array([100.0, 310.0]),
    
#     # 右側のゴール地点
#     "T1": np.array([500.0, 290.0]),
#     "T2": np.array([300.0, 320.0]),
#     "T3": np.array([500.0, 310.0]),
    
#     # 中央の重力源（これがないと平行なまま曲がらない）
#     "BlackHole": np.array([300.0, 300.0])
# }

# # 質量の設定（中央のノードだけ圧倒的に重くして引き寄せる）
# masses = {
#     "S1": 60, "S2": 0, "S3": 10,
#     "T1": 10, "T2": 0, "T3": 10,
#     "BlackHole": 0  # ここを大きくするともっと強く束ねられます
# }

# # エッジ（平行な3本の線）の接続定義
# edges = [
#     ("S1", "T1"),
#     ("S2", "T2"),
#     ("S3", "T3")
# ]
# ノードデータの正規化と配置
nodes = {
    str(n["id"]): np.array([(n["x"] - min_x) * scale, (n["y"] - min_y) * scale])
    for n in data["nodes"]
}


# 密集地ほど重力(質量)を弱める: 半径DENSITY_RADIUS以内にあるノード数(自分を含む)で質量を割る
node_ids = list(nodes.keys())
for nid in node_ids:
    print(f"Node {nid}: position={nodes[nid]}")
coords = np.array([nodes[nid] for nid in node_ids])
tree = cKDTree(coords)
neighbor_counts = tree.query_ball_point(coords, r=DENSITY_RADIUS, return_length=True)
masses = {nid: DEFAULT_NODE_MASS / ((count-1)*GRAVITY_DECAY+1) for nid, count in zip(node_ids, neighbor_counts)}

# エッジデータの読み込み（"edges" または "links" のキーに対応）
edges = []
json_edges = data.get("edges", data.get("links", []))
for e in json_edges:
    edges.append((str(e["source"]), str(e["target"])))

# -----------------------------
# タイル分割ポテンシャル場 & 勾配計算
# WINDOW_SIZE空間をGRID_SIZEのタイルに分割し、
# 各タイル中心でポテンシャルを計算 → タイルの傾き（勾配）を定数として使う
# -----------------------------
print("1. タイルポテンシャル場を計算中...")
n_tiles_x = math.ceil(WINDOW_SIZE / GRID_SIZE)
n_tiles_y = math.ceil(WINDOW_SIZE / GRID_SIZE)

# 各タイル中心の座標 (WINDOW_SIZE座標系)
tile_cx = (np.arange(n_tiles_x) + 0.5) * GRID_SIZE
tile_cy = (np.arange(n_tiles_y) + 0.5) * GRID_SIZE
XX, YY = np.meshgrid(tile_cx, tile_cy)  # shape: (n_tiles_y, n_tiles_x)

# タイル中心でのポテンシャルを計算
potential_field = np.zeros_like(XX)
for name, pos in nodes.items():
    dist_sq = (XX - pos[0])**2 + (YY - pos[1])**2
    potential_field -= masses[name] / np.sqrt(dist_sq + 1e-9)
print(potential_field)

# ノード直近でポテンシャルが発散しすぎないよう下限をクリップ(絶対値をMAX_POTENTIALで頭打ち)
if MAX_POTENTIAL is not None:
    potential_field = np.maximum(potential_field, -MAX_POTENTIAL)

# 密集地の勾配が急峻になりすぎないようガウシアンフィルタで均す
if SIGMA > 0:
    potential_field = gaussian_filter(potential_field, sigma=SIGMA)

# タイルごとの勾配（傾き）を計算
# np.gradientはタイルインデックス単位なのでGRID_SIZEで割ってpx単位に変換
tile_grad_y, tile_grad_x = np.gradient(potential_field)
tile_grad_x /= GRID_SIZE
tile_grad_y /= GRID_SIZE

# -----------------------------
# エッジ初期化
# -----------------------------
print("2. エッジを初期化中...")
bundled_edges = []
edge_natural_length=[]
for start_key, end_key in edges:
    start_pos, end_pos = nodes[start_key], nodes[end_key]
    edge = np.array([
        np.linspace(start_pos[0], end_pos[0], N_SAMPLES_PER_EDGE),
        np.linspace(start_pos[1], end_pos[1], N_SAMPLES_PER_EDGE)
    ]).T
    lengths = np.sqrt(np.sum(np.diff(edge, axis=0)**2, axis=1))
    edge_natural_length.append(lengths[0])
    bundled_edges.append(edge)

# -----------------------------
# 反復計算 (Iterative calculation)
# -----------------------------
print(f"3. {N_ITERATIONS}回の反復計算でバンドリングを実行中...")
current_step_size = STEP_SIZE
iteration_log = []

for i in range(N_ITERATIONS):
    print(f"   Iteration {i+1}/{N_ITERATIONS}, Current Step Size: {current_step_size:.2f}")
    for j, edge in enumerate(bundled_edges):
        new_edge = edge.copy()
        # 重力の1ステップ移動量は、このエッジ自身の隣接サンプル間隔(natural_length)を
        # 超えないようクリップする。GRID_SIZEなど一律の値だと短いエッジ(間隔が狭い)で
        # 重力が張力に対して相対的に大きくなりすぎ、テンションが追いつかなくなるため。
        max_step = edge_natural_length[j]
        for k in range(1, len(edge) - 1):
            pos = edge[k]
            # エッジ点がどのタイルに属するか特定し、そのタイルの傾きを力として使う
            ti = int(np.clip(pos[0] / GRID_SIZE, 0, n_tiles_x - 1))
            tj = int(np.clip(pos[1] / GRID_SIZE, 0, n_tiles_y - 1))
            dx = tile_grad_x[tj, ti]
            dy = tile_grad_y[tj, ti]
            force = current_step_size * np.array([dx, dy])
            force_mag = np.linalg.norm(force)
            if force_mag > max_step:
                force = force / force_mag * max_step
            new_edge[k] += force
        # kp*(vec_left+vec_right)は隣接点の平均に何%寄せるかという比率の力なので
        # 本来edge_lengthで割る必要はない。ただ既存のチューニング(短〜中距離エッジ)を
        # 崩さないよう、長いエッジだけkpがゼロに近づかないよう下限(KP_MIN)を設ける
        edge_length = np.linalg.norm(edge[-1] - edge[0])
        KP_MAX = 0.5  # 安定限界(これ以上はJacobi更新が振動する)
        KP_MIN = 0.2 * KP_MAX  # 最長エッジでもKP_MAXの20%は保証する
        kp = min(SPRING_CONSTANT / edge_length, KP_MAX) if edge_length > 0 else 0
        kp = max(kp, KP_MIN)
        # kp自体は0.5(安定限界)で頭打ちなので、安定した1手を複数回連続で適用して
        # テンションの実効的な効き目を強化する
        for _ in range(SPRING_SUBSTEPS):
            for k in range(1, len(edge) - 1):
                current = new_edge[k]
                vec_left = new_edge[k-1] - current
                vec_right = new_edge[k+1] - current
                new_edge[k] += kp * (vec_left + vec_right)

        bundled_edges[j] = new_edge

    iteration_log.append({
        "iteration": i + 1,
        "step_size": round(float(current_step_size), 4),
        "edges": [
            {
                "source": edges[j][0],
                "target": edges[j][1],
                "points": [[round(x, 2), round(y, 2)] for x, y in edge]
            }
            for j, edge in enumerate(bundled_edges)
        ]
    })

    current_step_size *= STEP_SIZE_DECAY

with open("bundling_log.json", "w", encoding="utf-8") as f:
    json.dump(iteration_log, f, indent=2, ensure_ascii=False)
print("bundling_log.json に保存しました。")

# evaluate.py で他手法と比較できる共通フォーマット(エッジごとの[x,y]点列)で最終結果を保存
with open("test2_result.json", "w") as f:
    json.dump([edge.tolist() for edge in bundled_edges], f)
print("test2_result.json に保存しました。")

# -----------------------------
# 全エッジ長の計算
# -----------------------------
total_length = 0.0

for edge in bundled_edges:
    segment_lengths = np.linalg.norm(
        np.diff(edge, axis=0),
        axis=1
    )
    total_length += np.sum(segment_lengths)

print(f"Total edge length: {total_length:.2f}")
# -----------------------------
# 可視化 (param_sweep.pyと同じスタイルで1枚に保存)
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 9))
ax.imshow(potential_field, cmap="bone_r", origin="lower",
          extent=[0, WINDOW_SIZE, 0, WINDOW_SIZE], alpha=0.3)
for edge in bundled_edges:
    ax.plot(edge[:, 0], edge[:, 1], color="cyan", linewidth=1.0, alpha=0.7)
for name, pos in nodes.items():
    ax.scatter(pos[0], pos[1], s=15, c="magenta", zorder=10)
ax.set_xlim(0, WINDOW_SIZE)
ax.set_ylim(0, WINDOW_SIZE)
ax.set_aspect("equal")
ax.set_facecolor("black")

params = {
    "DEFAULT_NODE_MASS": DEFAULT_NODE_MASS,
    "STEP_SIZE": STEP_SIZE,
    "SPRING_CONSTANT": SPRING_CONSTANT,
    "STEP_SIZE_DECAY": STEP_SIZE_DECAY,
    "N_ITERATIONS": N_ITERATIONS,
    "N_SAMPLES_PER_EDGE": N_SAMPLES_PER_EDGE,
    "WINDOW_SIZE": WINDOW_SIZE,
    "GRID_SIZE": GRID_SIZE,
    "SIGMA": SIGMA,
}
param_text = "\n".join(f"{k} = {v}" for k, v in params.items())
ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
        va="top", ha="left", fontsize=9, color="white", family="monospace",
        bbox=dict(facecolor="black", alpha=0.6, edgecolor="white", boxstyle="round,pad=0.4"))

fig.tight_layout()
fig.savefig("test2_result.png", dpi=120)
plt.close(fig)
print("test2_result.png に保存しました。")

# -----------------------------
# ポテンシャル場のみを可視化（勾配が急峻な場所を確認するデバッグ用）
# -----------------------------
fig2, ax2 = plt.subplots(figsize=(9, 9))
display_field = -potential_field  # 正にする
display_field = np.log1p(display_field)  # log圧縮でダイナミックレンジを圧縮
cs = ax2.contourf(
    XX, YY, display_field,
    levels=50,
    cmap="viridis",
    origin="lower",
)
fig2.colorbar(cs, ax=ax2, label="log1p(-potential)")
for name, pos in nodes.items():
    ax2.scatter(pos[0], pos[1], c="red", s=15, zorder=10)
ax2.set_xlim(0, WINDOW_SIZE)
ax2.set_ylim(0, WINDOW_SIZE)
ax2.set_aspect("equal")
ax2.set_title(f"Potential Field (GRID_SIZE={GRID_SIZE}, SIGMA={SIGMA})")

fig2.tight_layout()
fig2.savefig("test2_potential_field.png", dpi=120)
plt.close(fig2)
print("test2_potential_field.png に保存しました。")