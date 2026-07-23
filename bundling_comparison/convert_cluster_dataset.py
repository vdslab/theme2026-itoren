"""
bio-yeast(酵母タンパク質相互作用ネットワーク, networkrepository.com)を
airlines.jsonと同じスキーマ({"nodes":[{id,x,y}], "edges":[{source,target}]})に変換する。

出典: Network Repository (Ryan Rossi and Nesreen Ahmed, Purdue University, 2013)
      http://networkrepository.com/bio-yeast.php
      ノード数1458, エッジ数1948（無向・単純グラフ）

生の座標を持たないグラフなので、networkxのspring_layout(Fruchterman-Reingold)で
クラスタ構造が見えるレイアウトを計算してから座標として使う。
"""
import json
from pathlib import Path

import networkx as nx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MTX_PATH = Path(r"C:\Users\lotus\AppData\Local\Temp\claude\c--Users-lotus-theme2026-itoren\e246875a-358c-47a5-9e12-8b2281896d76\scratchpad\bio-yeast\bio-yeast.mtx")
OUTPUT_PATH = PROJECT_ROOT / "cluster_graph.json"


def load_mtx_edges(path):
    edges = []
    with open(path, "r") as f:
        lines = f.readlines()
    # 1行目: コメント、2行目: "rows cols nnz"、以降: "u v" (1-indexed)
    header_idx = next(i for i, line in enumerate(lines) if not line.startswith("%"))
    for line in lines[header_idx + 1:]:
        parts = line.split()
        if len(parts) < 2:
            continue
        u, v = int(parts[0]), int(parts[1])
        if u != v:
            edges.append((u, v))
    return edges


def main():
    print(f"Loading {MTX_PATH}...")
    edges = load_mtx_edges(MTX_PATH)
    print(f"raw edges (with duplicates): {len(edges)}")

    G = nx.Graph()
    G.add_edges_from(edges)
    print(f"nodes={G.number_of_nodes()}, simple edges={G.number_of_edges()}")

    print("Computing spring_layout (this may take a bit)...")
    pos = nx.spring_layout(G, seed=42, k=None, iterations=200)

    nodes_out = [{"id": str(n), "x": float(p[0]), "y": float(p[1])} for n, p in pos.items()]
    edges_out = [{"source": str(u), "target": str(v)} for u, v in G.edges()]

    data = {"nodes": nodes_out, "edges": edges_out}
    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f)
    print(f"nodes={len(nodes_out)}, edges={len(edges_out)} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
