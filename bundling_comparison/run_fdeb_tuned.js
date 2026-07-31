/*
 run_fdeb.js の比較用ベースラインには手を入れず、FDEBのパラメータを
 チューニングした版を別出力するランナー。

 デフォルトパラメータ(K=0.1, compatibility_threshold=0.6等)だと、eurosis.json
 のような密なグラフではほとんど束ならず直線に近いままだったため調整した。

 パラメータ探索で分かったこと(eurosis.jsonでink_reduction/distortion/ambiguityを
 計測して比較。詳細はresults/eurosis_sweep_*の実験結果を参照):
   - bundling_stiffness(K)は「束ねる力」ではなく、d3-ForceEdgeBundling.js内の
     コメント通り"edge stiffness"（直線に戻そうとするバネの強さ）。したがって
     K=1000のように大きくするとむしろ曲がりにくくなり束ねが弱まる上、
     数値的に不安定化して座標が発散する(実測でx座標が-3580億まで発散)。
     K=0.1〜1程度の小さい値の方がink_reductionが高く安定する。
   - 効果が一番大きいのはcompatibility_threshold。eurosisのエッジペアは
     visibility_compatibilityがほぼ0(97.8%のペアで0)になりやすく、
     デフォルト0.6ではエッジ1本あたり平均6本程度しか束ね相手が見つからない。
     0.1まで下げると平均100本程度に増え、ink_reductionが約1.9%→約13%に改善。
     0.05まで下げても平均128本程度で頭打ちになり、むしろdistortionが悪化する。
   - cyclesは6を超えても(8, 10で試験)ink_reduction/distortionはほぼ変化せず、
     計算時間だけが増える。step_size(移動量)を初期値より大きくすると
     distortionが悪化するだけでink_reductionは伸びない。
   - この結果、FDEBの束ね方式(似た向き・近い位置のエッジ同士を電気的に引き寄せる)
     はkdeeb(ink_reduction約69%)ほど強くは束ねられない。visibility_compatibilityの
     制約がボトルネックであり、K/cycles/step_sizeを更に強めても超えられない上限。
   - compatibility_thresholdを0.1までさらに下げると(全ペアの0%扱いに近い)
     ink_reductionは微増するがdistortion/ambiguityが悪化するので、0.15の方が
     ink_reductionをほぼ落とさずdistortion/ambiguityを抑えられる(Pareto良好)。
     0にすると無関係なエッジ同士まで引き合ってdistortionが15超まで破綻する。

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
  bundling_stiffness: 0.1,        // デフォルトのまま(上げると発散・束ね弱化のリスクがあるだけで得はない)
  compatibility_threshold: 0.15,  // デフォルト0.6 → 束ね相手とみなすエッジ対を大幅に増やす(効果の本命)
  cycles: 6,                      // デフォルトのまま(増やしても改善なし)
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
