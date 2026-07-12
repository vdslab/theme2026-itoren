import numpy as np 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import random
from scipy.ndimage import map_coordinates
import json  # JSON読み込みのために追加

# -----------------------------
# パラメータ (Balanced Version)
# -----------------------------
N_ITERATIONS = 100
N_SAMPLES_PER_EDGE = 30
STEP_SIZE = 35 
STEP_SIZE_DECAY = 0.97
SPRING_CONSTANT = 0.1
DEFAULT_NODE_MASS = 1

# # -----------------------------
# # ノードデータの生成 (Node data generation) - JSONから読み込み
# # -----------------------------
# print("0. Loading and normalizing node positions...")
# with open('airlines.json', 'r') as f:
#     data = json.load(f)

# # 1. すべてのノードからXとYの最小値・最大値を取得
# raw_x = [n["x"] for n in data["nodes"]]
# raw_y = [n["y"] for n in data["nodes"]]
# min_x, max_x_val = min(raw_x), max(raw_x)
# min_y, max_y_val = min(raw_y), max(raw_y)
# if max_x_val-min_x >max_y_val-min_y:
#     GRID_SIZE = max_x_val - min_x
# else:    
#     GRID_SIZE = max_y_val - min_y

# # 3. 画面端にノードが隠れないようにパディング（余白）を設定
# PADDING = 50
# USABLE_GRID = GRID_SIZE - (2 * PADDING)

print("0. Generating parallel edges and a gravity well...")

GRID_SIZE = 600  # 画面のサイズを固定

# ノードの配置（左側に3つ、右側に3つ、中央に巨大な重力源を1つ）
nodes = {
    # 左側のスタート地点
    "S1": np.array([100.0, 290.0]),
    "S2": np.array([300.0, 280.0]),
    "S3": np.array([100.0, 310.0]),
    
    # 右側のゴール地点
    "T1": np.array([500.0, 290.0]),
    "T2": np.array([300.0, 320.0]),
    "T3": np.array([500.0, 310.0]),
    
    # 中央の重力源（これがないと平行なまま曲がらない）
    "BlackHole": np.array([300.0, 300.0])
}

# 質量の設定（中央のノードだけ圧倒的に重くして引き寄せる）
masses = {
    "S1": 60, "S2": 0, "S3": 60,
    "T1": 60, "T2": 0, "T3": 60,
    "BlackHole": 0  # ここを大きくするともっと強く束ねられます
}

# エッジ（平行な3本の線）の接続定義
edges = [
    ("S1", "T1"),
    ("S2", "T2"),
    ("S3", "T3")
]
# # ノードデータの正規化と配置
# nodes = {}
# for n in data["nodes"]:
#     node_id = str(n["id"])
#     # [PADDING, PADDING+USABLE_GRID] の範囲に正規化（アスペクト比を保持）
#     norm_x = PADDING + (n["x"] - min_x) * USABLE_GRID / GRID_SIZE
#     norm_y = PADDING + (n["y"] - min_y) * USABLE_GRID / GRID_SIZE

#     nodes[node_id] = np.array([
#         norm_x,
#         norm_y
#     ])

# masses = {k: DEFAULT_NODE_MASS for k in nodes.keys()}

# # エッジデータの読み込み（"edges" または "links" のキーに対応）
# edges = []
# json_edges = data.get("edges", data.get("links", []))
# for e in json_edges:
#     edges.append((str(e["source"]), str(e["target"])))

# -----------------------------
# 力場計算関数 (Force field calculation function)
# -----------------------------
def calculate_potential_field(nodes, masses, grid_size):
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)

    potential_field = np.zeros_like(xx, dtype=float)

    for name, pos in nodes.items():
        dist_sq = (xx - pos[0])**2 + (yy - pos[1])**2
        potential_field -= masses[name] / np.sqrt(dist_sq + 1e-9)

    return gaussian_filter(potential_field, sigma=60)

# -----------------------------
# ポテンシャル場 & 勾配
# -----------------------------
print("1. 重力ポテンシャル場を計算中...")
potential_field = calculate_potential_field(nodes, masses, GRID_SIZE)
grad_y, grad_x = np.gradient(potential_field)

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
# 可視化準備 (Visualization preparation)
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# --- Before Bundling ---
ax_before = axes[0]
ax_before.imshow(potential_field, cmap='bone_r', origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE], alpha=0.3)
for edge in bundled_edges:
    ax_before.plot(edge[:, 0], edge[:, 1], color='cyan', linewidth=1.2, alpha=0.7)
for name, pos in nodes.items():
    ax_before.scatter(pos[0], pos[1], s=30, c='magenta', zorder=10)
ax_before.set_xlim(0, GRID_SIZE)
ax_before.set_ylim(0, GRID_SIZE)
ax_before.set_aspect('equal')
ax_before.set_facecolor('black')
ax_before.set_title("Edges Before Bundling (Random Nodes)")

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
        edge_length = np.linalg.norm(edge[-1] - edge[0])
        kp = SPRING_CONSTANT / (edge_length * (N_SAMPLES_PER_EDGE - 1) + 1e-8)
        for k in range(1, len(edge) - 1):
            pos = edge[k]
            dx = map_coordinates(grad_x, [[pos[1]], [pos[0]]], order=1)[0]
            dy = map_coordinates(grad_y, [[pos[1]], [pos[0]]], order=1)[0]
            F_attr = np.array([dx, dy])

            # FDEBばね：旧座標から計算（Jacobi法）
            vec_left  = edge[k - 1] - pos
            vec_right = edge[k + 1] - pos
            F_spring = kp * (
                np.dot(vec_left,  vec_left)  * vec_left +
                np.dot(vec_right, vec_right) * vec_right
            )

            new_edge[k] += current_step_size * (F_attr + F_spring)

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

# -----------------------------
# バンドリング後の可視化 (Visualization after bundling)
# -----------------------------
ax_after = axes[1]
ax_after.imshow(potential_field, cmap='bone_r', origin='lower', extent=[0, GRID_SIZE, 0, GRID_SIZE], alpha=0.3)
display_field = -potential_field  # 正にする

display_field = np.log1p(display_field)  # log圧縮

display_field = (display_field - display_field.min()) / (
    display_field.max() - display_field.min()
)

cs = ax_after.contourf(
    display_field,
    levels=50,
    cmap="viridis",
    origin="lower"
)

# ノード表示
for name, pos in nodes.items():
    ax_after.scatter(pos[0], pos[1], c="red", s=40)

fig.colorbar(cs, ax=ax_before)

ax_after.set_xlim(0, GRID_SIZE)
ax_after.set_ylim(0, GRID_SIZE)
ax_after.set_aspect("equal")
ax_after.set_title("Potential Field (Contour)")

for edge in bundled_edges:
    ax_after.plot(edge[:, 0], edge[:, 1], color='cyan', linewidth=1.2, alpha=0.7)

for name, pos in nodes.items():
    ax_after.scatter(pos[0], pos[1], s=30, c='magenta', zorder=10)
ax_after.set_xlim(0, GRID_SIZE)
ax_after.set_ylim(0, GRID_SIZE)
ax_after.set_aspect('equal')
ax_after.set_facecolor('black')
ax_after.set_title("Edges After Bundling (Random Nodes)")

plt.tight_layout()
plt.show()