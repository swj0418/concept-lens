// Compare the JavaScript port against fixtures recorded from the original Django views.
// usage: node parity.mjs <bundle dir> <fixtures.json> [--verbose]
//    or: deno run --allow-read parity.mjs <bundle dir> <fixtures.json> [--verbose]
import {readFileSync} from "node:fs";
import {ConceptLens, loadBundle, selectExperiments} from "../conceptlens-core.js";

const [bundleDir, fixturesPath] = process.argv.slice(2);
const verbose = process.argv.includes("--verbose");
// Read the bundle from disk through the same loader the notebook uses.
const open = name => {
  const bytes = () => { const b = readFileSync(`${bundleDir}/${name}`); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
  return {json: async () => JSON.parse(readFileSync(`${bundleDir}/${name}`, "utf8")), arrayBuffer: async () => bytes(), url: async () => `${bundleDir}/${name}`};
};
const fixtures0 = JSON.parse(readFileSync(fixturesPath, "utf8").replace(/\bNaN\b/g, "null"));
const bundle = await selectExperiments(await loadBundle(open), fixtures0.experiments);
const cl = new ConceptLens(bundle);
const fixtures = fixtures0; // Python writes NaN, hence the replace above

// ── comparison
const stats = {};
const note = (key, value) => { stats[key] = Math.max(stats[key] ?? 0, value); };
const relative = (a, b) => (a === b ? 0 : Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-12));
function compareTree(py, js, path, out) {
  for (const k of ["name", "leaf", "visleaf", "parent", "depth", "quantity"])
    if (py[k] !== js[k]) out.push(`${path}.${k}: ${py[k]} vs ${js[k]}`);
  if ("treeID" in py && (py.treeID !== js.treeID || py.expName !== js.expName || py.flatIdx !== js.flatIdx)) out.push(`${path}: experiment ids differ`);
  if (JSON.stringify(py.leaf_indices) !== JSON.stringify(js.leaf_indices)) out.push(`${path}.leaf_indices differ`);
  note("magnitude abs diff", Math.abs(py.magnitude - js.magnitude));
  if (py.magnitude !== js.magnitude) note("magnitude mismatches", (stats["magnitude mismatches"] ?? 0) + 1);
  note("var rel diff", relative(py.var, js.var));
  if ("weaveScore" in py) note("weave abs diff", Math.abs(py.weaveScore - js.weaveScore));
  if (py.children.length !== js.children.length) { out.push(`${path}: ${py.children.length} vs ${js.children.length} children`); return; }
  py.children.forEach((c, i) => compareTree(c, js.children[i], `${path}/${c.name}`, out));
}
const leafOrder = t => (t.children.length ? t.children.flatMap(leafOrder) : [t.name]);

let failures = 0;
const report = (label, problems, extra = "") => {
  if (problems.length) { failures++; console.log(`FAIL ${label}: ${problems.length} differences, e.g. ${problems.slice(0, 3).join("; ")}`); }
  else if (verbose) console.log(`ok   ${label} ${extra}`);
};

const js = {code: {}, direction: {}};
const t0 = performance.now();
const timings = {};
const time = (key, fn) => { const s = performance.now(); const r = fn(); (timings[key] ??= []).push(performance.now() - s); return r; };
for (const c of fixtures.cases) {
  const r = c.request, py = c.response, problems = [];
  if (c.endpoint === "code_initialization") {
    const out = time(c.endpoint, () => cl.codeInitialization({clusteringMethod: r.clustering_method}));
    js.code[r.clustering_method] = out;
    compareTree(py.rawtree, out.rawtree, "raw", problems);
    compareTree(py.tree, out.tree, "trim", problems);
    note("init stat rel diff", Math.max(relative(py.avgMagnitude, out.avgMagnitude), relative(py.avgStd, out.avgStd)));
    report(`${c.endpoint} ${r.clustering_method}`, problems);
  } else if (c.endpoint === "direction_initialization") {
    const out = time(c.endpoint, () => cl.directionInitialization({clusteringMethod: r.clustering_method, pairwiseMetric: r.pairwise_metric}));
    js.direction[`${r.clustering_method}-${r.pairwise_metric}`] = out;
    compareTree(py.rawtree, out.rawtree, "raw", problems);
    compareTree(py.tree, out.tree, "trim", problems);
    for (const k of ["avgMagnitude", "avgStd", "magsStd", "stdsStd"]) note("init stat rel diff", relative(py[k], out[k]));
    report(`${c.endpoint} ${r.clustering_method}/${r.pairwise_metric}`, problems);
  } else {
    const dir = js.direction[`complete-${r.metric}`], code = js.code.complete;
    let out, tree;
    if (c.endpoint === "direction_selection") {
      out = time(c.endpoint, () => cl.directionSelection({codeTree: code.rawtree, codeIndices: r.code_indices, directionIndices: r.direction_indices}));
      tree = ["codeTree", out.codeTree];
    } else if (c.endpoint === "code_selection") {
      out = time(c.endpoint, () => cl.codeSelection({directionTree: dir.rawtree, codeIndices: r.code_indices, directionIndices: r.direction_indices}));
      tree = ["directionTree", out.directionTree];
    } else {
      out = time(c.endpoint, () => cl.directionRecluster({directionTree: dir.tree, codeIndicesClustering: r.code_indices,
        codeIndicesCoherence: r.code_indices, pairwiseMetric: r.metric, clusteringMethod: r.clustering_method}));
      tree = ["tree", out.tree];
      compareTree(py.rawtree, out.rawtree, "raw", problems);
      // The server pops every score off weaveScores while attaching them to the tree, so it always
      // returns []; the per-node weaveScore values are compared in compareTree instead.
      if (py.weaveScores.length && JSON.stringify(py.weaveScores) !== JSON.stringify(out.weaveScores)) {
        const d = Math.max(...py.weaveScores.map((v, i) => Math.abs(v - out.weaveScores[i])));
        if (!(d < 1e-9) || py.weaveScores.length !== out.weaveScores.length) problems.push(`weaveScores differ (max ${d})`);
      }
      if (JSON.stringify(leafOrder(py.rawtree)) !== JSON.stringify(leafOrder(out.rawtree))) problems.push("leaf order differs");
    }
    compareTree(py[tree[0]], tree[1], "tree", problems);
    if ("magnitude" in py) {
      note("selection magnitude rel diff", relative(py.magnitude, out.magnitude));
      if (py.variance === null || Number.isNaN(py.variance)) { if (!Number.isNaN(out.variance)) problems.push("variance should be NaN"); }
      else note("selection variance rel diff", relative(py.variance, out.variance));
      if (py.magmin !== out.magmin || py.magmax !== out.magmax) problems.push("magmin/magmax differ");
      const a = JSON.stringify(py.contributions.map(x => [x.code, x.direction, x.mag_contribution]));
      const b = JSON.stringify(out.contributions.map(x => [x.code, x.direction, x.mag_contribution]));
      if (a !== b) problems.push(`contributions differ (${py.contributions.length} vs ${out.contributions.length})`);
    }
    report(`${c.endpoint} (${r.code_indices.length} codes, ${(r.direction_indices ?? []).length} directions)`, problems);
  }
}
console.log(`${fixtures.cases.length} cases, ${failures} failed — ${(performance.now() - t0).toFixed(0)} ms total`);
for (const [k, v] of Object.entries(stats)) console.log(`  ${k}: ${typeof v === "number" && v < 1 ? v.toExponential(2) : v}`);
for (const [k, v] of Object.entries(timings)) console.log(`  ${k}: median ${v.sort((a, b) => a - b)[v.length >> 1].toFixed(1)} ms (n=${v.length})`);
process.exitCode = failures ? 1 : 0;
