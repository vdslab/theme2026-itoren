import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import ConvexHull
import json

# -----------------------------
# パラメータ
# -----------------------------
N_ITERATIONS = 200
N_SAMPLES_PER_EDGE = 30
STEP_SIZE = 30
STEP_SIZE_DECAY = 0.98
SPRING_CONSTANT = 0.15
DEFAULT_NODE_MASS = 50
HULL_MASS = 80           # 凸包上の重力源の質量
HULL_OFFSET_PX = 20.0     # 凸包を外側に拡張するpx数
HULL_SAMPLE_SPACING = 5.0  # 凸包辺上のサンプリング間隔(px)

# -----------------------------
# データ読み込み・正規化
# -----------------------------
print("0. Loading and normalizing node positions...")
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

masses = {k: DEFAULT_NODE_MASS for k in nodes}

json_edges = data.get("edges", data.get("links", []))
edges = [(str(e["source"]), str(e["target"])) for e in json_edges]

# -----------------------------
# 凸包 + HULL_OFFSET_PX の重力点を生成
# -----------------------------
def get_hull_gravity_points(nodes_dict, offset_px, sample_spacing):
    positions = np.array(list(nodes_dict.values()))
    hull = ConvexHull(positions)
    vertices = positions[hull.vertices]
    centroid = vertices.mean(axis=0)

    # 各頂点を重心から offset_px だけ外側へ移動
    expanded = []
    for v in vertices:
        d = v - centroid
        expanded.append(v + offset_px * d / (np.linalg.norm(d) + 1e-9))
    expanded = np.array(expanded)

    # 辺に沿って sample_spacing 間隔でサンプリング
    points = []
    n = len(expanded)
    for i in range(n):
        p0 = expanded[i]
        p1 = expanded[(i + 1) % n]
        dist = np.linalg.norm(p1 - p0)
        n_samples = max(int(dist / sample_spacing), 2)
        for t in np.linspace(0, 1, n_samples, endpoint=False):
            points.append(p0 + t * (p1 - p0))

    return np.array(points), hull, positions

hull_gravity_points, hull, node_positions = get_hull_gravity_points(
    nodes, HULL_OFFSET_PX, HULL_SAMPLE_SPACING
)
print(f"  Hull gravity points: {len(hull_gravity_points)}")

# -----------------------------
# ポテンシャル場（ノード + 凸包重力源）
# -----------------------------
def calculate_potential_field(nodes, masses, hull_points, hull_mass, grid_size):
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    potential_field = np.zeros_like(xx, dtype=float)

    for name, pos in nodes.items():
        dist_sq = (xx - pos[0])**2 + (yy - pos[1])**2
        potential_field -= masses[name] / np.sqrt(dist_sq + 1e-9)

    for pos in hull_points:
        dist_sq = (xx - pos[0])**2 + (yy - pos[1])**2
        potential_field -= hull_mass / np.sqrt(dist_sq + 1e-9)

    return gaussian_filter(potential_field, sigma=60)

print("1. 重力ポテンシャル場を計算中...")
potential_field = calculate_potential_field(
    nodes, masses, hull_gravity_points, HULL_MASS, GRID_SIZE
)
grad_y, grad_x = np.gradient(potential_field)

# -----------------------------
# エッジ初期化
# -----------------------------
print("2. エッジを初期化中...")
bundled_edges = []
edge_natural_length = []
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
# 可視化準備
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

ax_before = axes[0]
ax_before.imshow(potential_field, cmap='bone_r', origin='lower',
                 extent=[0, GRID_SIZE, 0, GRID_SIZE], alpha=0.3)
for edge in bundled_edges:
    ax_before.plot(edge[:, 0], edge[:, 1], color='cyan', linewidth=1.2, alpha=0.7)
for name, pos in nodes.items():
    ax_before.scatter(pos[0], pos[1], s=30, c='magenta', zorder=10)
# 凸包を描画
hull_loop = np.append(hull.vertices, hull.vertices[0])
ax_before.plot(node_positions[hull_loop, 0], node_positions[hull_loop, 1],
               color='yellow', linewidth=1.0, linestyle='--', alpha=0.6, label='convex hull')
ax_before.scatter(hull_gravity_points[:, 0], hull_gravity_points[:, 1],
                  s=5, c='yellow', alpha=0.4, zorder=5)
ax_before.set_xlim(0, GRID_SIZE)
ax_before.set_ylim(0, GRID_SIZE)
ax_before.set_aspect('equal')
ax_before.set_facecolor('black')
ax_before.set_title("Before Bundling + Convex Hull")

# -----------------------------
# 反復計算
# -----------------------------
print(f"3. {N_ITERATIONS}回の反復計算でバンドリングを実行中...")
current_step_size = STEP_SIZE
iteration_log = []

for i in range(N_ITERATIONS):
    print(f"   Iteration {i+1}/{N_ITERATIONS}, step={current_step_size:.2f}")
    for j, edge in enumerate(bundled_edges):
        new_edge = edge.copy()
        new_edge_spring = new_edge.copy()

        for k in range(1, len(edge) - 1):
            F_attr = np.array([0.0, 0.0])
            pos = edge[k]
            dx = map_coordinates(grad_x, [[pos[1]], [pos[0]]], order=1)[0]
            dy = map_coordinates(grad_y, [[pos[1]], [pos[0]]], order=1)[0]
            F_attr += np.array([dx, dy])
            new_edge[k] += current_step_size * F_attr

        F_spring_list = []
        for k in range(1, len(edge) - 1):
            current = new_edge[k]
            vec_left  = new_edge[k - 1] - current
            vec_right = new_edge[k + 1] - current
            F_spring_list.append(SPRING_CONSTANT * (vec_left + vec_right))
        for k in range(1, len(edge) - 1):
            new_edge[k] += F_spring_list[k - 1]

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

with open("bundling_log_totu.json", "w", encoding="utf-8") as f:
    json.dump(iteration_log, f, indent=2, ensure_ascii=False)
print("bundling_log_totu.json に保存しました。")

# -----------------------------
# 全エッジ長の計算
# -----------------------------
total_length = sum(
    np.sum(np.linalg.norm(np.diff(edge, axis=0), axis=1))
    for edge in bundled_edges
)
print(f"Total edge length: {total_length:.2f}")

# -----------------------------
# バンドリング後の可視化
# -----------------------------
ax_after = axes[1]
ax_after.imshow(potential_field, cmap='bone_r', origin='lower',
                extent=[0, GRID_SIZE, 0, GRID_SIZE], alpha=0.3)
display_field = -potential_field
display_field = np.log1p(display_field)
display_field = (display_field - display_field.min()) / (display_field.max() - display_field.min())

cs = ax_after.contourf(display_field, levels=50, cmap="viridis", origin="lower")
fig.colorbar(cs, ax=axes[0])

for edge in bundled_edges:
    ax_after.plot(edge[:, 0], edge[:, 1], color='cyan', linewidth=1.2, alpha=0.7)
for name, pos in nodes.items():
    ax_after.scatter(pos[0], pos[1], s=30, c='magenta', zorder=10)
ax_after.plot(node_positions[hull_loop, 0], node_positions[hull_loop, 1],
              color='yellow', linewidth=1.0, linestyle='--', alpha=0.6)

ax_after.set_xlim(0, GRID_SIZE)
ax_after.set_ylim(0, GRID_SIZE)
ax_after.set_aspect('equal')
ax_after.set_facecolor('black')
ax_after.set_title("After Bundling + Convex Hull Gravity")

plt.tight_layout()
plt.show()
