/*
 d3-ForceEdgeBundling.js (upphiminn版FDEB実装) を他手法(gravity_edge_bundling.py等)と
 同じ入出力規約で実行するランナー。

 使い方:
   node run_fdeb.js [データセット.json]
   (省略時は ../airlines.json)

 入力: airlines.jsonと同じスキーマ ({"nodes":[{id,x,y}], "edges":[{source,target}]})
 出力: results/fdeb_github_result.json (airlines.jsonの場合) または
       results/<データセット名>_fdeb_result.json
       (evaluate.pyが読める形式: エッジごとの[x,y]点列のJSON配列、WINDOW_SIZE=10000空間)
*/
"use strict";

const fs = require("fs");
const path = require("path");

const SCRIPT_DIR = __dirname;
const PROJECT_ROOT = path.dirname(SCRIPT_DIR);
const RESULTS_DIR = path.join(SCRIPT_DIR, "results");
const DEFAULT_DATASET = path.join(PROJECT_ROOT, "airlines.json");

const WINDOW_SIZE = 10000; // 他手法(gravity_edge_bundling.py等)と同じ座標空間。FDEB自体のパラメータは弄らない。
const N_SAMPLES = 100;

// d3-ForceEdgeBundling.js は `d3.ForceEdgeBundling = ...` という形で
// グローバルのd3オブジェクトに生やす作りなので、先にダミーのd3を用意してから読み込む。
global.d3 = {};
require("./d3-ForceEdgeBundling.js");

function loadNodesEdges(datasetPath) {
  const data = JSON.parse(fs.readFileSync(datasetPath, "utf8"));
  const xs = data.nodes.map((n) => n.x);
  const ys = data.nodes.map((n) => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const scale = WINDOW_SIZE / Math.max(maxX - minX, maxY - minY);

  const dataNodes = {};
  for (const n of data.nodes) {
    dataNodes[String(n.id)] = {
      x: (n.x - minX) * scale,
      y: (n.y - minY) * scale,
    };
  }
  const rawEdges = data.edges || data.links || [];
  const dataEdges = rawEdges.map((e) => ({
    source: String(e.source),
    target: String(e.target),
  }));
  return { dataNodes, dataEdges };
}

function resampleToNPoints(points, n) {
  const seg = [];
  let total = 0;
  seg.push(0);
  for (let i = 1; i < points.length; i++) {
    const dx = points[i].x - points[i - 1].x;
    const dy = points[i].y - points[i - 1].y;
    total += Math.sqrt(dx * dx + dy * dy);
    seg.push(total);
  }
  if (total < 1e-9) {
    return Array.from({ length: n }, () => [points[0].x, points[0].y]);
  }
  const out = [];
  let segIdx = 0;
  for (let i = 0; i < n; i++) {
    const target = (total * i) / (n - 1);
    while (segIdx < seg.length - 2 && seg[segIdx + 1] < target) segIdx++;
    const t0 = seg[segIdx], t1 = seg[segIdx + 1];
    const frac = t1 > t0 ? (target - t0) / (t1 - t0) : 0;
    const p0 = points[segIdx], p1 = points[segIdx + 1];
    out.push([
      p0.x + frac * (p1.x - p0.x),
      p0.y + frac * (p1.y - p0.y),
    ]);
  }
  return out;
}

function main() {
  const datasetPath = process.argv[2] ? path.resolve(process.argv[2]) : DEFAULT_DATASET;
  const stem = path.basename(datasetPath, ".json");
  const outName = stem === "airlines" ? "fdeb_github_result.json" : `${stem}_fdeb_result.json`;

  console.log(`Loading ${datasetPath}...`);
  const { dataNodes, dataEdges } = loadNodesEdges(datasetPath);
  console.log(`nodes=${Object.keys(dataNodes).length}, edges=${dataEdges.length}`);

  console.log("Running FDEB (this may take a bit)...");
  const t0 = Date.now();
  const fbundling = d3.ForceEdgeBundling().nodes(dataNodes).edges(dataEdges);
  const results = fbundling();
  console.log(`FDEB done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const output = results.map((pts) => resampleToNPoints(pts, N_SAMPLES));

  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  const outPath = path.join(RESULTS_DIR, outName);
  fs.writeFileSync(outPath, JSON.stringify(output));
  console.log(`nodes=${Object.keys(dataNodes).length}, edges=${output.length} -> ${outPath}`);
}

main();
