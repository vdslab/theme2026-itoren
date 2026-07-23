"""
共有された画像(複数の密なクラスタが疎な線で連結された「ビッグデータ」風の図)と
似た構造を持つ合成グラフを生成し、airlines.jsonと同じスキーマで保存する。

Stochastic Block Model: 10個のクラスタ(サイズはランダム)を作り、
クラスタ内は高確率、クラスタ間は低確率でエッジを張る。
レイアウトは、各クラスタの中心を円周上に配置してから局所的にspring_layoutで
緩和することで、画像のように分離した塊として見えるようにする。
"""
import json
from pathlib import Path

import networkx as nx
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_PATH = PROJECT_ROOT / "multicluster_graph.json"

N_CLUSTERS = 10
CLUSTER_SIZE_RANGE = (20, 55)
P_IN = 0.25       # クラスタ内の接続確率
P_OUT = 0.0015    # クラスタ間の接続確率
SEED = 7


def main():
    rng = np.random.default_rng(SEED)
    sizes = rng.integers(CLUSTER_SIZE_RANGE[0], CLUSTER_SIZE_RANGE[1] + 1, size=N_CLUSTERS).tolist()
    print(f"cluster sizes: {sizes} (total {sum(sizes)} nodes)")

    p_matrix = np.full((N_CLUSTERS, N_CLUSTERS), P_OUT)
    np.fill_diagonal(p_matrix, P_IN)

    G = nx.stochastic_block_model(sizes, p_matrix.tolist(), seed=SEED)
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # 各クラスタの中心を円周上に配置し、そこを初期位置としてspring_layoutで緩和する
    # （画像のように塊が分離しつつ、間の接続エッジも自然に描けるようにする）
    block = G.graph["partition"]
    node_block = {}
    for b_idx, members in enumerate(block):
        for n in members:
            if n in G:
                node_block[n] = b_idx

    radius = 12.0
    centers = {
        b: (radius * np.cos(2 * np.pi * b / N_CLUSTERS), radius * np.sin(2 * np.pi * b / N_CLUSTERS))
        for b in range(N_CLUSTERS)
    }
    init_pos = {}
    for n in G.nodes():
        cx, cy = centers[node_block[n]]
        jitter = rng.normal(0, 0.8, size=2)
        init_pos[n] = (cx + jitter[0], cy + jitter[1])

    print("Computing spring_layout...")
    pos = nx.spring_layout(G, pos=init_pos, seed=SEED, iterations=100, k=0.3)

    nodes_out = [{"id": str(n), "x": float(p[0]), "y": float(p[1])} for n, p in pos.items()]
    edges_out = [{"source": str(u), "target": str(v)} for u, v in G.edges()]

    data = {"nodes": nodes_out, "edges": edges_out}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f)
    print(f"nodes={len(nodes_out)}, edges={len(edges_out)} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
