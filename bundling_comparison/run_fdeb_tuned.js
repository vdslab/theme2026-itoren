/*
 run_fdeb.js の比較用ベースラインには手を入れず、FDEBのパラメータを強めに
 チューニングした版を別出力するランナー。

 デフォルトパラメータ(K=0.1, compatibility_threshold=0.6等)だと、eurosis.json
 のような密なグラフではほとんど束ならず直線に近いままだったため、
 bundling_stiffnessを上げ、compatibility_thresholdを下げてより積極的に束ねる。

 使い方:
   node run_fdeb_tuned.js [データセット.json]
   (省略時は ../airlines.json)

 出力: results/<データセット名>_fdeb_tuned_result.json
       (run_fdeb.jsの通常出力とは別名。比較実験のベースラインは上書きしない)
*/
"use strict";

const fs = require("fs");
const path = require("path");

const SCRIPT_DIR = __dirname;
const PROJECT_ROOT = path.dirname(SCRIPT_DIR);
const RESULTS_DIR = path.join(SCRIPT_DIR, "results");
const DEFAULT_DATASET = path.join(PROJECT_ROOT, "airlines.json");

const WINDOW_SIZE = 10000;
const N_SAMPLES = 100;

// チューニングしたFDEBパラメータ
const TUNED = {
  bundling_stiffness: 1000.0,        // K: デフォルト0.1 → 束ねる力をさらに強く
  compatibility_threshold: 0.2,   // デフォルト0.6 → より多くのエッジ対を束ねる仲間とみなす
  cycles: 10,                     // デフォルト6 → 収束をより丁寧に
};

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
  const outName = `${stem}_fdeb_tuned_result.json`;

  console.log(`Loading ${datasetPath}...`);
  const { dataNodes, dataEdges } = loadNodesEdges(datasetPath);
  console.log(`nodes=${Object.keys(dataNodes).length}, edges=${dataEdges.length}`);

  console.log(`Running FDEB (tuned: ${JSON.stringify(TUNED)})...`);
  const t0 = Date.now();
  const fbundling = d3.ForceEdgeBundling()
    .nodes(dataNodes)
    .edges(dataEdges)
    .bundling_stiffness(TUNED.bundling_stiffness)
    .compatibility_threshold(TUNED.compatibility_threshold)
    .cycles(TUNED.cycles);
  const results = fbundling();
  console.log(`FDEB (tuned) done in ${((Date.now() - t0) / 1000).toFixed(1)}s`);

  const output = results.map((pts) => resampleToNPoints(pts, N_SAMPLES));

  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  const outPath = path.join(RESULTS_DIR, outName);
  fs.writeFileSync(outPath, JSON.stringify(output));
  console.log(`nodes=${Object.keys(dataNodes).length}, edges=${output.length} -> ${outPath}`);
}

main();
