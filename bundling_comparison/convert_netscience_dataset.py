"""
netscience(Newmanの共著関係ネットワーク, Network Science論文の著者間共著グラフ)を
airlines.jsonと同じスキーマ({"nodes":[{id,x,y}], "edges":[{source,target}]})に変換する。

出典: M. E. J. Newman, "Finding community structure in networks using the
      eigenvectors of matrices", Phys. Rev. E 74, 036104 (2006)
      https://raw.githubusercontent.com/gephi/gephi.github.io/master/datasets/netscience.gml.zip
      ノード数1589, エッジ数2742（無向、value=共著回数だが未使用）

生の座標を持たないグラフなので、networkxのspring_layout(Fruchterman-Reingold)で
クラスタ構造が見えるレイアウトを計算してから座標として使う。
"""
import json
from pathlib import Path

import networkx as nx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GML_PATH = Path(
    r"C:\Users\lotus\AppData\Local\Temp\claude\c--Users-lotus-theme2026-itoren"
    r"\53c27554-7bbf-4869-90a0-c6cde0c16146\scratchpad\netscience\netscience.gml"
)
OUTPUT_PATH = PROJECT_ROOT / "netscience.json"


def main():
    print(f"Loading {GML_PATH}...")
    G = nx.read_gml(GML_PATH, label="id")
    print(f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

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
