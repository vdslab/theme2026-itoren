"""
EuroSiS Generale Pays.gexf(欧州の社会イノベーション関連アクター・ネットワーク,
Gephiのチュートリアルデータセットとして有名)を airlines.json と同じスキーマ
({"nodes":[{id,x,y}], "edges":[{source,target}]})に変換する。

出典: プロジェクトルートの "EuroSiS Generale Pays.gexf"
      (Gephiのviz:position属性として既にレイアウト済みの座標を持っているので、
       netscience/migrationsと違ってレイアウト計算は不要。そのまま座標を使う)

GEXFのxmlns("http://www.gephi.org/gexf")が標準のgexf.net名前空間と異なるため、
networkxのread_gexf()では読み込めない(No <graph> element エラー)。
そのためxml.etree.ElementTreeで直接パースする。
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
GEXF_PATH = PROJECT_ROOT / "EuroSiS Generale Pays.gexf"
OUTPUT_PATH = PROJECT_ROOT / "eurosis.json"

NS = {"g": "http://www.gephi.org/gexf", "viz": "http://www.gephi.org/gexf/viz"}


def main():
    print(f"Loading {GEXF_PATH}...")
    root = ET.parse(GEXF_PATH).getroot()

    nodes_out = []
    for node in root.findall(".//g:node", NS):
        pos = node.find("viz:position", NS)
        nodes_out.append({
            "id": node.get("id"),
            "x": float(pos.get("x")),
            "y": float(pos.get("y")),
        })

    edges_out = [
        {"source": edge.get("source"), "target": edge.get("target")}
        for edge in root.findall(".//g:edge", NS)
    ]

    data = {"nodes": nodes_out, "edges": edges_out}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"nodes={len(nodes_out)}, edges={len(edges_out)} -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
