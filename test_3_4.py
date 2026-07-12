import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import json

# --- shared setup (same as test.py) ---
with open('airlines.json', 'r') as f:
    data = json.load(f)

raw_x = [n["x"] for n in data["nodes"]]
raw_y = [n["y"] for n in data["nodes"]]
min_x, max_x_val = min(raw_x), max(raw_x)
min_y, max_y_val = min(raw_y), max(raw_y)
GRID_SIZE = max_x_val - min_x if max_x_val - min_x > max_y_val - min_y else max_y_val - min_y
PADDING = 50
USABLE_GRID = GRID_SIZE - 2 * PADDING

nodes = {}
for n in data["nodes"]:
    node_id = str(n["id"])
    nodes[node_id] = np.array([
        PADDING + (n["x"] - min_x) * USABLE_GRID / GRID_SIZE,
        PADDING + (n["y"] - min_y) * USABLE_GRID / GRID_SIZE,
    ])

DEFAULT_NODE_MASS = 20
masses = {k: DEFAULT_NODE_MASS for k in nodes}
json_edges = data.get("edges", data.get("links", []))
edges = [(str(e["source"]), str(e["target"])) for e in json_edges]

N_SAMPLES_PER_EDGE = 30
SPRING_CONSTANT = 0.6
STEP_SIZE = 3.5

def calculate_potential_field(nodes, masses, grid_size):
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y)
    potential_field = np.zeros_like(xx, dtype=float)
    for name, pos in nodes.items():
        dist_sq = (xx - pos[0])**2 + (yy - pos[1])**2
        potential_field -= masses[name] / np.sqrt(dist_sq + 1e-9)
    return gaussian_filter(potential_field, sigma=18)

potential_field = calculate_potential_field(nodes, masses, GRID_SIZE)
grad_y, grad_x = np.gradient(potential_field)

bundled_edges = []
edge_natural_length = []
for start_key, end_key in edges:
    s, e = nodes[start_key], nodes[end_key]
    edge = np.array([
        np.linspace(s[0], e[0], N_SAMPLES_PER_EDGE),
        np.linspace(s[1], e[1], N_SAMPLES_PER_EDGE),
    ]).T
    lengths = np.sqrt(np.sum(np.diff(edge, axis=0)**2, axis=1))
    edge_natural_length.append(lengths[0])
    bundled_edges.append(edge)

# ==============================
# Test 3: endpoints do not move
# ==============================
print("=== Test 3: endpoints do not move after 1 iteration ===")

endpoints_before = [(edge[0].copy(), edge[-1].copy()) for edge in bundled_edges]

# run 1 iteration (same logic as test.py)
current_step_size = STEP_SIZE
for j, edge in enumerate(bundled_edges):
    new_edge = edge.copy()
    for k in range(1, len(edge) - 1):
        pos = edge[k]
        dx = map_coordinates(grad_x, [[pos[1]], [pos[0]]], order=1)[0]
        dy = map_coordinates(grad_y, [[pos[1]], [pos[0]]], order=1)[0]
        F_attr = np.array([dx, dy])
        new_edge[k] += current_step_size * F_attr

        current = edge[k]
        left, right = edge[k-1], edge[k+1]
        vec_left  = left  - current
        dist_left = np.linalg.norm(vec_left)
        F_left    = (dist_left  - edge_natural_length[j]) * (vec_left  / (dist_left  + 1e-9))
        vec_right = right - current
        dist_right = np.linalg.norm(vec_right)
        F_right   = (dist_right - edge_natural_length[j]) * (vec_right / (dist_right + 1e-9))
        new_edge[k] += current_step_size * SPRING_CONSTANT * (F_left + F_right)

    bundled_edges[j] = new_edge

fail3 = 0
for j, (start, end) in enumerate(endpoints_before):
    if not np.allclose(bundled_edges[j][0], start):
        print(f"  FAIL edge {j}: start moved {bundled_edges[j][0]} != {start}")
        fail3 += 1
    if not np.allclose(bundled_edges[j][-1], end):
        print(f"  FAIL edge {j}: end   moved {bundled_edges[j][-1]} != {end}")
        fail3 += 1

if fail3 == 0:
    print(f"  PASS: all {len(bundled_edges)} edges kept endpoints fixed")
else:
    print(f"  FAIL: {fail3} endpoint(s) moved")

# ==============================
# Test 4: all nodes within grid
# ==============================
print("\n=== Test 4: all nodes within [PADDING, GRID_SIZE-PADDING] ===")

fail4 = 0
for node_id, pos in nodes.items():
    if not (PADDING <= pos[0] <= GRID_SIZE - PADDING):
        print(f"  FAIL node {node_id}: x={pos[0]:.2f} out of [{PADDING}, {GRID_SIZE-PADDING:.1f}]")
        fail4 += 1
    if not (PADDING <= pos[1] <= GRID_SIZE - PADDING):
        print(f"  FAIL node {node_id}: y={pos[1]:.2f} out of [{PADDING}, {GRID_SIZE-PADDING:.1f}]")
        fail4 += 1

if fail4 == 0:
    print(f"  PASS: all {len(nodes)} nodes within grid bounds")
    xs = [pos[0] for pos in nodes.values()]
    ys = [pos[1] for pos in nodes.values()]
    print(f"  x range: [{min(xs):.2f}, {max(xs):.2f}]  (valid: [{PADDING}, {GRID_SIZE-PADDING:.1f}])")
    print(f"  y range: [{min(ys):.2f}, {max(ys):.2f}]  (valid: [{PADDING}, {GRID_SIZE-PADDING:.1f}])")
else:
    print(f"  FAIL: {fail4} node(s) out of bounds")
