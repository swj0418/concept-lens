// ConceptLens — Observable notebook, no server
// ---------------------------------------------------------------------------
// Each block between the "═══" lines is ONE Observable cell. Paste each into its own
// cell (order doesn't matter to Observable; this is the suggested reading order).
// Then upload the files from export_for_observable.py (drag them onto the Files pane)
// and replace the `files` cell with the attachments.js it wrote.
// ---------------------------------------------------------------------------

// ═══ Cell 1 · Title ═══════════════════════════════════════════════════════
md`# ConceptLens

Latent directions from several discovery methods are clustered by the edits they make to generated images, measured with CLIP features of the edited images. Codes (the images being edited) are clustered by their originals. Color combines two measures: **edit magnitude** (yellow for weak edits, deep purple for strong ones) and **inconsistency** (fading toward white when an edit varies more across images).

- **Directions** run across the top and **codes** down the left, as icicles. Click a cluster to add it to the selection, or drag along the thin bar at an icicle's edge to select a range.
- Selecting directions recolors the codes for just those directions; selecting codes recolors the directions for just those codes. The dot in the legend marks the selection as a whole.
- The **bi-hierarchy view** crosses the selected clusters: a heat map while cells are small, generated images (picked at equal spacing in tree order) once there is room, with bars summarizing each image row's and column's edit magnitudes and the original images on the right.
- **Recluster** regroups the directions using only the selected codes. Pink dots show how much each new cluster gathers directions from different, distant clusters of the previous hierarchy (displacement).

Everything runs in the browser from files attached to this notebook — there is no server.`

// ═══ Cell 2 · Methods ═════════════════════════════════════════════════════
viewof methods = Inputs.checkbox(bundle.manifest.experiments, {
  label: "Methods",
  format: e => methodLabel(e.name),
  valueof: e => e.name,
  value: bundle.manifest.experiments.map(e => e.name)
})

// ═══ Cell 3 · Clustering ══════════════════════════════════════════════════
viewof clustering = Inputs.form({
  method: Inputs.radio(new Map([["Complete", "complete"], ["Ward", "ward"]]), {label: "Linkage", value: "complete"}),
  metric: Inputs.radio(new Map([["Cosine", "cosine"], ["Euclidean", "euclidean"]]), {label: "Direction distance", value: "cosine"})
})

// ═══ Cell 4 · The linked views ════════════════════════════════════════════
viewof selection = ConceptLensView(model, {width})

// ═══ Cell 5 · Selection heading ═══════════════════════════════════════════
md`### Selected edits

${selection.codes.length.toLocaleString()} codes × ${selection.directions.length.toLocaleString()} directions${
  selection.magnitude == null ? "" : ` · mean edit magnitude ${selection.magnitude.toFixed(3)} · inconsistency ${selection.variance.toFixed(4)}`}`

// ═══ Cell 6 · Selection table ═════════════════════════════════════════════
Inputs.table(selectedEdits, {
  columns: ["code", "method", "direction", "magnitude"],
  header: {code: "Code", method: "Method", direction: "Direction", magnitude: "Edit magnitude"},
  format: {magnitude: d3.format(".3f")},
  sort: "magnitude",
  reverse: true,
  rows: 12
})

// ═══ Cell 7 · Selected edits as rows ══════════════════════════════════════
selectedEdits = {
  const {cl, bundle} = model;
  const experiments = bundle.manifest.experiments;
  const rows = [];
  for (const code of selection.codes) {
    for (const d of selection.directions) {
      const e = experiments.find(e => d < e.offset + e.count);
      rows.push({code, method: methodLabel(e.name), direction: d - e.offset, magnitude: cl.magnitude[code * cl.nDirection + d]});
    }
  }
  return rows;
}

// ═══ Cell 8 · Heading ═════════════════════════════════════════════════════
md`## Data`

// ═══ Cell 9 · Attached files ══════════════════════════════════════════════
// Replace this cell with the attachments.js written by export_for_observable.py.
files = ({
  "manifest.json": FileAttachment("manifest.json")
})

// ═══ Cell 10 · Bundle ═════════════════════════════════════════════════════
bundle = core.loadBundle(core.fromAttachments(files))

// ═══ Cell 11 · Model ══════════════════════════════════════════════════════
// The two initial hierarchies for the chosen methods and clustering settings —
// what the Django server's *_initialization endpoints used to return.
model = {
  const lens = await core.selectExperiments(bundle, methods);
  const cl = new core.ConceptLens(lens);
  return {
    cl,
    bundle: lens,
    clustering,
    direction: cl.directionInitialization({clusteringMethod: clustering.method, pairwiseMetric: clustering.metric}),
    code: cl.codeInitialization({clusteringMethod: clustering.method})
  };
}

// ═══ Cell 12 · Heading ════════════════════════════════════════════════════
md`## Implementation`

// ═══ Cell 13 · Method names ═══════════════════════════════════════════════
// Experiment folders are named <domain>-<method>-<layer>-<application>.
methodLabel = {
  const names = {
    sefa: "SeFA", sefakmc: "SeFA", ganspace: "GANSpace", ganspacekmc: "GANSpace",
    va: "Vector arithmetic", vac: "Vector arithmetic", vector_arithmetic: "Vector arithmetic",
    svm: "Supervised", svmw: "Supervised", supervised: "Supervised"
  };
  return experiment => {
    const [, method, layer] = experiment.split("-");
    if (!method) return experiment;
    return `${names[method] ?? method}${layer ? ` · ${layer}` : ""}`;
  };
}

// ═══ Cell 14 · VSUP color scale ═══════════════════════════════════════════
// Value-suppressing uncertainty palette, as vsup.quantization(2, 4) + vsup.scale() with the
// reversed viridis ramp the original client used: a tree of bins (1, 2, 4, 8 from most to
// least uncertain), each blended toward white by its uncertainty.
function vsup({valueDomain, uncertaintyDomain, branching = 2, layers = 4,
               ramp = t => d3.interpolateViridis(Math.abs(1 - t))}) {
  const vscale = d3.scaleLinear().domain(valueDomain).nice(branching ** (layers - 1));
  const uscale = d3.scaleLinear().domain(uncertaintyDomain).nice(layers);
  const tree = [[{layer: 0, index: 0, u: uscale.invert((layers - 1) / layers), v: vscale.invert(0.5)}]];
  for (let i = 1; i < layers; ++i) {
    const n = 2 * branching ** i;
    tree[i] = [];
    for (let j = 1; j < n; j += 2) tree[i].push({layer: i, index: (j - 1) / 2, u: uscale.invert(1 - (i + 1) / layers), v: vscale.invert(j / n)});
  }
  const leaf = (v, u) => {
    let i = 0;
    while (i < tree.length - 1 && uscale(u) < 1 - (i + 1) / layers - 1e-9) i++;
    const row = tree[i], half = row.length > 1 ? (row[1].v - row[0].v) / 2 : 0;
    let j = 0;
    while (j < row.length - 1 && v > row[j].v + half) j++;
    return row[j];
  };
  const vNorm = d3.scaleLinear(vscale.domain(), [0, 1]), uNorm = d3.scaleLinear(uscale.domain(), [0, 1]);
  const colorOf = node => d3.interpolateLab(ramp(vNorm(node.v)), "#fff")(uNorm(node.u));
  const color = (v, u) => colorOf(leaf(v, u));
  return Object.assign(color, {tree, leaf, colorOf, layers, valueDomain: vscale.domain(), uncertaintyDomain: uscale.domain()});
}

// ═══ Cell 15 · VSUP legend ════════════════════════════════════════════════
// The wedge legend: angle = edit magnitude, radius = inconsistency (outer = consistent).
function vsupLegend(g, scale, {radius = 96, marker = null} = {}) {
  const span = Math.PI / 3, layers = scale.layers;
  const angle = d3.scaleLinear(scale.valueDomain, [-span, span]).clamp(true);
  const ring = d3.scaleLinear(scale.uncertaintyDomain, [radius, 0]).clamp(true);
  const arc = d3.arc();
  const muted = "#898781", ink = "#0b0b0b";
  const polar = (a, r) => [r * Math.sin(a), -r * Math.cos(a)];
  g.selectAll("*").remove();
  g.append("g").selectAll("path").data(scale.tree.flat()).join("path")
      .attr("d", d => {
        const n = scale.tree[d.layer].length;
        return arc({
          innerRadius: (radius * d.layer) / layers, outerRadius: (radius * (d.layer + 1)) / layers,
          startAngle: -span + (2 * span * d.index) / n, endAngle: -span + (2 * span * (d.index + 1)) / n
        });
      })
      .attr("fill", d => scale.colorOf(d))
      .attr("stroke", "#fff")
      .attr("stroke-width", 0.75);
  const fmtV = d3.format(".2~f"), fmtU = d3.format(".3~f");
  const [v0, v1] = scale.valueDomain, [u0, u1] = scale.uncertaintyDomain;
  // Edit magnitude along the outer arc
  g.append("g").attr("fill", muted).attr("font-size", 9)
    .selectAll("text").data([[v0, -span], [(v0 + v1) / 2, 0], [v1, span]]).join("text")
      .attr("transform", ([, a]) => `translate(${polar(a, radius + 6)}) rotate(${(a * 180) / Math.PI})`)
      .attr("text-anchor", "middle")
      .text(([v]) => fmtV(v));
  g.append("text").attr("y", -radius - 17).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", ink).text("Edit magnitude");
  // Inconsistency along the right edge (consistent at the rim, inconsistent at the centre)
  const edge = (r, off) => { const [x, y] = polar(span, r); return [x + off * Math.cos(span), y + off * Math.sin(span)]; };
  g.append("g").attr("fill", muted).attr("font-size", 9)
    .selectAll("text").data([[u0, radius - 22], [u1, 8]]).join("text")
      .attr("transform", ([, r]) => `translate(${edge(r, 5)}) rotate(${(span * 180) / Math.PI - 90})`)
      .attr("dy", "0.7em")
      .text(([u]) => fmtU(u));
  g.append("text").attr("transform", `translate(${edge(radius / 2, 20)}) rotate(${(span * 180) / Math.PI - 90})`)
      .attr("text-anchor", "middle").attr("dy", "0.7em").attr("font-size", 11).attr("fill", ink).text("Inconsistency");
  if (marker && Number.isFinite(marker.magnitude) && Number.isFinite(marker.variance)) {
    const [x, y] = polar(angle(marker.magnitude), ring(marker.variance));
    g.append("circle").attr("cx", x).attr("cy", y).attr("r", 5).attr("fill", "#d03b3b").attr("stroke", "#fff").attr("stroke-width", 2);
  }
}

// ═══ Cell 16 · Icicle layout ══════════════════════════════════════════════
// Nodes of a (trimmed) hierarchy down to `depth`. Each node covers a contiguous run of
// leaves [start, end) in display order; `bottom` nodes are drawn down to the last row.
function icicleLayout(tree, depth) {
  const nodes = [], order = [];
  (function visit(node, level) {
    const start = order.length;
    const bottom = level >= depth || node.leaf || !node.children.length;
    if (bottom) order.push(...(node.leaf_indices.length ? node.leaf_indices : [node.name]));
    else for (const child of node.children) visit(child, level + 1);
    if (level > 0 || bottom) nodes.push({data: node, name: node.name, level: Math.max(1, level), start, end: order.length, bottom});
  })(tree, 0);
  return {nodes, order, bottom: nodes.filter(n => n.bottom).sort((a, b) => a.start - b.start)};
}

// ═══ Cell 17 · ConceptLensView ════════════════════════════════════════════
// The direction icicle (top), code icicle (left) and the matrix that crosses them.
// All interaction goes through `state`; render() redraws from it. The value is the
// current selection: {codes, directions, magnitude, variance, reclustered}.
function ConceptLensView(model, {width = 1000, height = 560, depth: initialDepth = 7, thumb = 64} = {}) {
  const {cl, bundle, clustering} = model;
  const experiments = bundle.manifest.experiments;
  const D = cl.nDirection, C = cl.nCode;
  const images = bundle.manifest.images;
  const ink = "#0b0b0b", secondary = "#52514e", muted = "#898781", hairline = "#e1e0d9", surface = "#ffffff";
  const selectColor = "#eb6834", displacementColor = d3.hcl(360, 100, 70).formatHex();
  const methodColor = d3.scaleOrdinal(bundle.manifest.all_experiments ?? experiments.map(e => e.name),
    ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]);
  const experimentOf = d => experiments.find(e => d < e.offset + e.count);
  const fmt = {m: d3.format(".3f"), v: d3.format(".4f"), n: d3.format(",")};

  // Colors: the value/uncertainty domains come from the initial direction tree, as in the paper.
  const {avgMagnitude, magsStd, avgStd, stdsStd} = model.direction;
  const color = vsup({valueDomain: [avgMagnitude - magsStd, avgMagnitude + magsStd],
                      uncertaintyDomain: [avgStd - stdsStd, avgStd + stdsStd]});
  const nodeColor = n => color(n.magnitude, n.var);

  // Geometry
  width = Math.max(width, 640);
  const pad = 20, L = 176, T = 176, band = 16, strip = 8, gap = 8;
  const mx = pad + L + gap, my = pad + T + strip + gap, mw = width - mx, mh = height;

  const state = {
    base: model.direction,             // initial or reclustered direction hierarchy
    dirTree: model.direction.tree,     // colored for the selected codes
    codeTree: model.code.tree,         // colored for the selected directions
    dirSel: new Set(), codeSel: new Set(),
    depth: initialDepth, summary: null, weave: new Map(), reclusteredWith: 0,
    keys: {dirs: "", codes: ""}
  };

  // ── DOM
  const svg = d3.create("svg").attr("width", width).attr("height", my + mh).attr("viewBox", [0, 0, width, my + mh]);
  const legendG = svg.append("g").attr("transform", `translate(${pad + L / 2},${pad + T - 36})`);
  const dirG = svg.append("g").attr("transform", `translate(${mx},${pad})`);
  const stripG = svg.append("g").attr("transform", `translate(${mx},${pad + T})`);
  const codeG = svg.append("g").attr("transform", `translate(${pad},${my})`);
  const matrixG = svg.append("g").attr("transform", `translate(${mx},${my})`);
  const dirBrushG = svg.append("g").attr("transform", `translate(${mx},${pad})`);
  const codeBrushG = svg.append("g").attr("transform", `translate(${pad},${my})`);

  const depthInput = html`<input type="range" min="2" max="12" step="1" value=${initialDepth}>`;
  const reclusterButton = html`<button type="button">Recluster directions</button>`;
  const resetButton = html`<button type="button">Reset</button>`;
  const status = html`<div class="cl-status"></div>`;
  const tip = html`<div class="cl-tip" hidden></div>`;
  const root = html`<div class="conceptlens" tabindex="0">
    <style>
      .conceptlens { position: relative; font: 12px/1.35 system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; outline: none; }
      .conceptlens > svg { display: block; overflow: visible; max-width: none; }
      .conceptlens svg svg { overflow: hidden; }
      .conceptlens svg text { font-family: inherit; }
      .conceptlens .cl-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 10px 18px; padding: 4px 0 10px; margin-bottom: 10px; border-bottom: 1px solid #e1e0d9; }
      .conceptlens .cl-status { flex: 1 1 100%; color: #52514e; font-variant-numeric: tabular-nums; }
      .conceptlens .cl-status b { color: #0b0b0b; font-weight: 600; }
      .conceptlens .cl-methods { display: flex; flex-wrap: wrap; gap: 4px 12px; color: #52514e; }
      .conceptlens .cl-methods span { display: inline-flex; align-items: center; gap: 5px; }
      .conceptlens .cl-methods i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
      .conceptlens label { display: inline-flex; align-items: center; gap: 6px; color: #52514e; }
      .conceptlens input[type=range] { width: 90px; }
      .conceptlens button { font: inherit; padding: 3px 10px; border: 1px solid #c3c2b7; border-radius: 4px; background: #fff; color: inherit; cursor: pointer; }
      .conceptlens button:disabled { opacity: 0.4; cursor: default; }
      .conceptlens .node { cursor: pointer; }
      .conceptlens .selection { fill: #0b0b0b; fill-opacity: 0.05; stroke: #52514e; shape-rendering: crispEdges; }
      .conceptlens .cl-tip { position: absolute; z-index: 3; pointer-events: none; padding: 7px 9px; background: #fff; border: 1px solid rgba(11,11,11,.12); border-radius: 6px; box-shadow: 0 2px 10px rgba(0,0,0,.1); max-width: 300px; }
      .conceptlens .cl-tip .title { font-weight: 600; margin-bottom: 3px; }
      .conceptlens .cl-tip .row { display: flex; gap: 10px; }
      .conceptlens .cl-tip strong { font-weight: 600; font-variant-numeric: tabular-nums; min-width: 3.6em; }
      .conceptlens .cl-tip span { color: #52514e; }
      .conceptlens .cl-tip .note { color: #898781; margin-top: 3px; white-space: pre-line; }
      .conceptlens .cl-tip .pair { display: flex; gap: 6px; margin-top: 6px; }
      .conceptlens .cl-tip .pair figure { margin: 0; text-align: center; color: #898781; font-size: 11px; }
      .conceptlens .cl-tip .pair div { border-radius: 3px; }
    </style>
    <div class="cl-bar">
      ${status}
      <div class="cl-methods">${experiments.map(e => html`<span><i style=${`background:${methodColor(e.name)}`}></i>${methodLabel(e.name)}</span>`)}</div>
      <label>Depth ${depthInput}</label>
      ${reclusterButton}${resetButton}
    </div>
    ${svg.node()}
    ${tip}
  </div>`;

  // Static labels
  svg.append("text").attr("x", mx + mw / 2).attr("y", 13).attr("text-anchor", "middle").attr("font-weight", 600)
      .text(`Concept direction hierarchy · ${fmt.n(D)} directions`);
  svg.append("text").attr("transform", `translate(13,${my + mh / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("font-weight", 600)
      .text(`Code hierarchy · ${fmt.n(C)} codes`);

  // ── Icicles
  // Rows 1..depth-1 share the icicle's thickness; the last row is the thin brushable band.
  function levelBox(n, thickness) {
    const rows = Math.max(1, state.depth - 1), rh = (thickness - band) / rows;
    const a = n.level >= state.depth ? thickness - band : (n.level - 1) * rh;
    const b = n.bottom ? thickness : a + rh;
    return [a, b];
  }

  function drawIcicle(g, layout, {horizontal, unit, thickness, selected}) {
    g.selectAll("rect.node").data(layout.nodes, n => n.name).join("rect")
        .attr("class", "node")
        .each(function(n) {
          const [a, b] = levelBox(n, thickness);
          const s = n.start * unit, e = n.end * unit, inset = e - s > 2 ? 0.5 : 0;
          const box = horizontal ? {x: s + inset, y: a + 0.5, width: e - s - 2 * inset, height: b - a - 1}
                                 : {x: a + 0.5, y: s + inset, width: b - a - 1, height: e - s - 2 * inset};
          d3.select(this).attr("x", box.x).attr("y", box.y).attr("width", Math.max(0, box.width)).attr("height", Math.max(0, box.height));
        })
        .attr("rx", 2)
        .attr("fill", n => nodeColor(n.data))
        .attr("stroke", n => (selected.has(n.name) ? selectColor : "none"))
        .attr("stroke-width", 2.5);
    g.selectAll("rect.node").filter(n => selected.has(n.name)).raise();
  }

  function drawWeave(layout) {
    const leafScores = n => {
      let best = state.weave.get(n.name);
      if (best == null) cl_walk(n.data, m => { const s = state.weave.get(m.name); if (s != null) best = Math.max(best ?? 0, s); });
      return best;
    };
    const dots = state.weave.size ? layout.bottom.map(n => ({n, w: leafScores(n)})).filter(d => d.w > 0) : [];
    const wmax = d3.max(dots, d => d.w) || 1;
    dirG.selectAll("circle.weave").data(dots, d => d.n.name).join("circle")
        .attr("class", "weave")
        .attr("pointer-events", "none")
        .attr("cx", d => ((d.n.start + d.n.end) / 2) * (mw / D))
        .attr("cy", T - band / 2)
        .attr("r", d => 1 + (4 * d.w) / wmax)
        .attr("fill", displacementColor).attr("stroke", "#fff").attr("stroke-width", 1);
  }
  const cl_walk = (node, visit) => { visit(node); node.children.forEach(ch => cl_walk(ch, visit)); };

  // ── Selection helpers
  const leavesOf = (layout, selected) => {
    const out = [];
    for (const n of layout.nodes) if (selected.has(n.name)) out.push(...layout.order.slice(n.start, n.end));
    return [...new Set(out)].sort((a, b) => a - b);
  };
  const focus = (layout, selected) => {
    if (!selected.size) return layout.bottom;
    const ranges = layout.nodes.filter(n => selected.has(n.name));
    return layout.bottom.filter(b => ranges.some(r => r.start <= b.start && b.end <= r.end));
  };

  // Recompute colors and the summary when the selected leaves change (the old endpoints).
  function applySelection(dirLayout, codeLayout) {
    const dirs = leavesOf(dirLayout, state.dirSel), codes = leavesOf(codeLayout, state.codeSel);
    const dk = dirs.join(","), ck = codes.join(",");
    if (dk !== state.keys.dirs) {
      if (dirs.length) {
        const r = cl.directionSelection({codeTree: model.code.rawtree, codeIndices: codes, directionIndices: dirs});
        state.codeTree = r.codeTree;
        state.summary = {magnitude: r.magnitude, variance: r.variance};
      } else state.codeTree = model.code.tree;
    }
    if (ck !== state.keys.codes) {
      if (codes.length) {
        const r = cl.codeSelection({directionTree: state.base.rawtree, codeIndices: codes, directionIndices: dirs});
        state.dirTree = r.directionTree;
        state.summary = {magnitude: r.magnitude, variance: r.variance};
      } else state.dirTree = state.base.tree;
    }
    if (!dirs.length && !codes.length) state.summary = null;
    state.keys = {dirs: dk, codes: ck};
    return {dirs, codes};
  }

  // ── Matrix
  const varianceCache = new Map();
  function columnVariance(dirs) {
    const key = dirs.join(",");
    if (!varianceCache.has(key)) varianceCache.set(key, dirs.length > 1 ? cl.codeVariances(dirs) : new Float64Array(C));
    return varianceCache.get(key);
  }
  const sample = (items, n) => (n >= items.length ? items : Array.from({length: n}, (_, i) => items[Math.floor((i * items.length) / n)]));
  function spread(clusters, total, gapPx) {
    if (!clusters.length) return [];
    const size = d3.sum(clusters, c => c.end - c.start), free = total - gapPx * (clusters.length - 1);
    let at = 0;
    return clusters.map(c => {
      const extent = (free * (c.end - c.start)) / size;
      const box = {c, at, extent};
      at += extent + gapPx;
      return box;
    });
  }
  // Zoomed layout: as many selected clusters as fit get image slots (largest first), the rest
  // become thin heat strips; spare slots go to clusters in proportion to their size, never more
  // than they hold. With far too many clusters, cells are sized by cluster size (heat map only).
  function slots(clusters, total, gapPx, pad, strip = 10) {
    const n = clusters.length;
    if (!n) return [];
    const sizes = clusters.map(c => c.end - c.start);
    let k = n;
    while (k > 0 && k * (thumb + pad) + (n - k) * strip + (n - 1) * gapPx > total) k--;
    if (!k) return spread(clusters, total, gapPx).map(b => ({...b, fit: 0}));
    const ranked = sizes.map((_, i) => i).sort((a, b) => sizes[b] - sizes[a] || a - b);
    const give = sizes.map(() => 0);
    ranked.slice(0, k).forEach(i => (give[i] = 1));
    let free = Math.floor((total - (n - 1) * gapPx - (n - k) * strip - k * pad) / thumb) - k;
    while (free > 0) {
      let best = -1, score = -Infinity;
      sizes.forEach((size, i) => { if (give[i] && give[i] < size && size / give[i] > score) { score = size / give[i]; best = i; } });
      if (best < 0) break;
      give[best]++;
      free--;
    }
    let at = 0;
    return clusters.map((c, i) => {
      const box = {c, at, extent: give[i] ? give[i] * thumb + pad : strip, fit: give[i]};
      at += box.extent + gapPx;
      return box;
    });
  }

  const spriteUrls = new Map();
  const spriteUrl = code => {
    if (!spriteUrls.has(code)) spriteUrls.set(code, bundle.sprite(code));
    return spriteUrls.get(code);
  };
  const cellOf = d => (d == null ? 0 : 1 + bundle.global[d]); // sprite cell: 0 = original image
  function thumbnail(g, code, d, x, y, size) {
    const {cell, columns, rows} = images, index = cellOf(d);
    const view = g.append("svg").attr("x", x).attr("y", y).attr("width", size).attr("height", size)
        .attr("viewBox", [(index % columns) * cell, Math.floor(index / columns) * cell, cell, cell]);
    view.append("rect").attr("x", (index % columns) * cell).attr("y", Math.floor(index / columns) * cell)
        .attr("width", cell).attr("height", cell).attr("fill", "#ebeae4");
    const image = view.append("image").attr("width", columns * cell).attr("height", rows * cell).attr("preserveAspectRatio", "none");
    spriteUrl(code).then(url => image.attr("href", url));
    return view;
  }

  // The bi-hierarchy view: the Cartesian product of the selected code and direction clusters.
  // Cells are sized by how many codes/directions they hold; small cells show the mean edit
  // (heat map), large ones show images picked at equal spacing in tree order. As in the
  // paper, each image row gets a horizontal bar summary of its edit magnitudes on the left,
  // each image column a vertical one on top, and original images sit on the right.
  function drawMatrix(dirLayout, codeLayout) {
    matrixG.selectAll("*").remove();
    const zoomed = state.dirSel.size > 0 || state.codeSel.size > 0;
    // With only one hierarchy selected, the other one contributes all of its leaves as a single
    // group, sampled evenly across the tree (so a direction can be judged across the latent space).
    const everything = n => [{name: "all", start: 0, end: n}];
    const colsC = zoomed && !state.dirSel.size ? everything(D) : focus(dirLayout, state.dirSel);
    const rowsC = zoomed && !state.codeSel.size ? everything(C) : focus(codeLayout, state.codeSel);
    const showImages = zoomed && !!images;
    const G = showImages ? 28 : 0, dots = showImages ? 10 : 0, originW = showImages ? thumb + 14 : 0;
    const left = G + (showImages ? 2 : 0), top = dots + G + (showImages ? 2 : 0);
    const cols = zoomed ? slots(colsC, mw - left - originW, 6, 4).map(b => ({...b, at: b.at + left}))
                        : colsC.map(c => ({c, at: c.start * (mw / D), extent: (c.end - c.start) * (mw / D), fit: 0}));
    const rows = zoomed ? slots(rowsC, mh - top, 6, 6).map(b => ({...b, at: b.at + top}))
                        : rowsC.map(c => ({c, at: c.start * (mh / C), extent: (c.end - c.start) * (mh / C), fit: 0}));
    const M = cl.magnitude;
    for (const col of cols) {
      col.dirs = dirLayout.order.slice(col.c.start, col.c.end);
      col.perCode = columnVariance(col.dirs);
      col.sampled = sample(col.dirs, showImages ? col.fit : 0);
    }
    for (const row of rows) {
      row.codes = codeLayout.order.slice(row.c.start, row.c.end);
      row.sampled = sample(row.codes, showImages ? row.fit : 0);
    }

    // Cells: mean edit magnitude and inconsistency of every code × direction pair inside.
    const grid = matrixG.append("g");
    for (const col of cols) for (const row of rows) {
      let m = 0, v = 0;
      for (const c of row.codes) { for (const d of col.dirs) m += M[c * D + d]; v += col.perCode[c]; }
      m /= row.codes.length * col.dirs.length;
      v /= row.codes.length;
      const withImages = row.sampled.length > 0 && col.sampled.length > 0;
      grid.append("rect")
          .attr("x", col.at).attr("y", row.at).attr("width", Math.max(0, col.extent)).attr("height", Math.max(0, row.extent))
          .attr("fill", withImages ? "#f5f4f0" : color(m, v))
          .attr("stroke", zoomed ? surface : "none")
          .on("pointerenter pointermove", event => showTip(event, {
            title: `${fmt.n(row.codes.length)} codes × ${fmt.n(col.dirs.length)} directions`,
            rows: [[fmt.m(m), "mean edit magnitude"], [fmt.v(v), "inconsistency"]],
            note: withImages ? null : zoomed ? "Too small for images; select fewer clusters" : "Select clusters to compare their images"
          }))
          .on("pointerleave", hideTip);
      if (withImages) grid.append("rect").attr("x", col.at).attr("y", row.at).attr("width", col.extent).attr("height", 3).attr("fill", color(m, v));
    }
    if (!showImages) return;

    // Images, with a key per (code, direction) so hovering one lights up its summary bars.
    const shown = [];
    const imagesG = matrixG.append("g");
    for (const row of rows) row.sampled.forEach((c, i) => {
      const y = row.at + 4 + i * thumb;
      for (const col of cols) col.sampled.forEach((d, j) => {
        const x = col.at + 2 + j * thumb;
        shown.push({c, d, x, y, m: M[c * D + d]});
        thumbnail(imagesG, c, d, x + 1, y + 1, thumb - 2);
      });
      thumbnail(imagesG, c, null, mw - thumb, y + 1, thumb - 2); // original image, right column
      imagesG.append("rect").attr("x", mw - thumb - 1).attr("y", y).attr("width", thumb).attr("height", thumb).attr("fill", "transparent")
          .on("pointerenter pointermove", event => showTip(event, imageTip(c, null))).on("pointerleave", hideTip);
    });
    if (!shown.length) return;
    matrixG.append("text").attr("x", mw - thumb / 2).attr("y", top - 4).attr("text-anchor", "middle").attr("fill", muted).attr("font-size", 10).text("Original");

    // Method dots above each image column (as in the paper's method comparisons).
    for (const col of cols) col.sampled.forEach((d, j) => matrixG.append("circle")
      .attr("cx", col.at + 2 + j * thumb + thumb / 2).attr("cy", dots / 2).attr("r", 3.5)
      .attr("fill", methodColor(experimentOf(d).name)));

    // Magnitude summaries: bars scaled to the range of the images on screen.
    const [lo, hi] = d3.extent(shown, s => s.m);
    const len = d3.scaleLinear([lo, hi === lo ? lo + 1e-9 : hi], [3, G - 2]);
    const bars = matrixG.append("g").attr("fill", secondary);
    const byRow = d3.group(shown, s => s.y), byCol = d3.group(shown, s => s.x);
    for (const [y, items] of byRow) {           // left: one horizontal bar per image in the row; bottom bar = leftmost image
      const h = Math.min(6, (thumb - 4) / items.length);
      items.forEach((s, k) => bars.append("rect").datum(s).attr("class", "sum")
        .attr("x", G - len(s.m)).attr("width", len(s.m)).attr("height", Math.max(1, h - 1))
        .attr("y", y + thumb - 2 - (k + 1) * h));
    }
    for (const [x, items] of byCol) {           // top: one vertical bar per image in the column; left bar = top image
      const w = Math.min(6, (thumb - 4) / items.length);
      items.forEach((s, k) => bars.append("rect").datum(s).attr("class", "sum")
        .attr("x", x + 2 + k * w).attr("width", Math.max(1, w - 1))
        .attr("y", dots + G - len(s.m)).attr("height", len(s.m)));
    }
    const highlight = key => bars.selectAll("rect.sum").attr("fill", s => (key && s.c === key.c && s.d === key.d ? selectColor : null));
    imagesG.selectAll(null).data(shown).join("rect")
        .attr("x", s => s.x).attr("y", s => s.y).attr("width", thumb).attr("height", thumb).attr("fill", "transparent")
        .on("pointerenter pointermove", (event, s) => { highlight(s); showTip(event, imageTip(s.c, s.d, s.m)); })
        .on("pointerleave", () => { highlight(null); hideTip(); });
    bars.selectAll("rect.sum")
        .on("pointerenter pointermove", (event, s) => { highlight(s); showTip(event, imageTip(s.c, s.d, s.m)); })
        .on("pointerleave", () => { highlight(null); hideTip(); });
  }

  // ── Tooltips (text via textContent; images as sprite crops)
  function imageTip(code, d, mag) {
    const e = d == null ? null : experimentOf(d);
    return {
      title: d == null ? `Code ${code} · original` : `Code ${code} · ${methodLabel(e.name)} direction ${d - e.offset}`,
      rows: d == null ? [] : [[fmt.m(mag), "edit magnitude"]],
      images: d == null ? [[code, null, "original"]] : [[code, null, "original"], [code, d, "edited"]]
    };
  }
  function crop(code, d, size) {
    const {cell, columns, rows} = images, index = cellOf(d), k = size / cell;
    const div = document.createElement("div");
    Object.assign(div.style, {width: `${size}px`, height: `${size}px`, background: "#ebeae4",
      backgroundSize: `${columns * cell * k}px ${rows * cell * k}px`,
      backgroundPosition: `${-(index % columns) * cell * k}px ${-Math.floor(index / columns) * cell * k}px`});
    spriteUrl(code).then(url => (div.style.backgroundImage = `url("${url}")`));
    return div;
  }
  function showTip(event, {title, rows = [], note, images: pics}) {
    const el = (tag, className, text) => Object.assign(document.createElement(tag), {className, textContent: text ?? ""});
    tip.replaceChildren(el("div", "title", title));
    for (const [value, label] of rows) {
      const r = tip.appendChild(el("div", "row"));
      r.append(el("strong", "", value), el("span", "", label));
    }
    if (pics && images) {
      const pair = tip.appendChild(el("div", "pair"));
      for (const [c, d, label] of pics) {
        const fig = pair.appendChild(el("figure", ""));
        fig.append(crop(c, d, Math.min(images.cell, 128)), el("figcaption", "", label));
      }
    }
    if (note) tip.appendChild(el("div", "note", note));
    tip.hidden = false;
    const [px, py] = d3.pointer(event, root);
    const w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = `${px + 14 + w > width ? px - 14 - w : px + 14}px`;
    tip.style.top = `${py + 14 + h > my + mh + 60 ? py - 14 - h : py + 14}px`;
  }
  function hideTip() { tip.hidden = true; }

  function nodeTip(n, kind) {
    const leaves = n.end - n.start, rows = [[fmt.m(n.data.magnitude), "edit magnitude"], [fmt.v(n.data.var), "inconsistency"]];
    let note = "Click to select · drag along the bar to select a range";
    if (kind === "direction") {
      const counts = d3.rollups(dirLayoutNow().order.slice(n.start, n.end), v => v.length, d => experimentOf(d).name);
      note = counts.map(([name, k]) => `${methodLabel(name)} ${k}`).join(" · ") + "\n" + note;
      const w = state.weave.get(n.name);
      if (w != null) rows.push([d3.format(".2f")(w), "displacement"]);
    }
    return {title: `${fmt.n(leaves)} ${kind === "direction" ? "directions" : "codes"}`, rows, note};
  }

  // ── Brushes on the thin bands
  let dirLayoutNow, codeLayoutNow;
  const dirBrush = d3.brushX().extent([[0, T - band], [mw, T]]).on("brush end", brushed("direction"));
  const codeBrush = d3.brushY().extent([[L - band, 0], [L, mh]]).on("brush end", brushed("code"));
  dirBrushG.call(dirBrush);
  codeBrushG.call(codeBrush);
  function brushed(kind) {
    return ({selection: s, type, sourceEvent}) => {
      if (!sourceEvent) return;
      const layout = kind === "direction" ? dirLayoutNow() : codeLayoutNow();
      const unit = kind === "direction" ? mw / D : mh / C;
      const set = kind === "direction" ? state.dirSel : state.codeSel;
      const hits = s ? layout.bottom.filter(n => n.end * unit > s[0] && n.start * unit < s[1]).map(n => n.name) : [];
      set.clear();
      hits.forEach(h => set.add(h));
      if (type === "end") {
        (kind === "direction" ? dirBrushG : codeBrushG).call((kind === "direction" ? dirBrush : codeBrush).move, null);
        update();
      } else render(false);
    };
  }

  // ── Render
  function render(recompute = true) {
    const dirLayout = icicleLayout(state.dirTree, state.depth), codeLayout = icicleLayout(state.codeTree, state.depth);
    dirLayoutNow = () => dirLayout;
    codeLayoutNow = () => codeLayout;
    drawIcicle(dirG, dirLayout, {horizontal: true, unit: mw / D, thickness: T, selected: state.dirSel});
    drawIcicle(codeG, codeLayout, {horizontal: false, unit: mh / C, thickness: L, selected: state.codeSel});
    dirG.selectAll("rect.node").on("click", (e, n) => toggle(state.dirSel, n.name))
        .on("pointerenter pointermove", (e, n) => showTip(e, nodeTip(n, "direction"))).on("pointerleave", hideTip);
    codeG.selectAll("rect.node").on("click", (e, n) => toggle(state.codeSel, n.name))
        .on("pointerenter pointermove", (e, n) => showTip(e, nodeTip(n, "code"))).on("pointerleave", hideTip);
    drawWeave(dirLayout);
    stripG.selectAll("rect").data(dirLayout.order).join("rect")
        .attr("x", (d, i) => (i * mw) / D).attr("width", Math.max(0.5, mw / D - (mw / D > 3 ? 0.5 : 0))).attr("height", strip)
        .attr("fill", d => methodColor(experimentOf(d).name));
    if (recompute) drawMatrix(dirLayout, codeLayout);
    vsupLegend(legendG, color, {radius: 62, marker: state.summary});
    const dirs = leavesOf(dirLayout, state.dirSel), codes = leavesOf(codeLayout, state.codeSel);
    status.replaceChildren(html`${state.dirSel.size
      ? html`<b>${fmt.n(dirs.length)}</b> directions` : html`<b>${fmt.n(D)}</b> directions`} × ${state.codeSel.size
      ? html`<b>${fmt.n(codes.length)}</b> codes` : html`<b>${fmt.n(C)}</b> codes`}${state.summary
      ? html` · edit magnitude <b>${fmt.m(state.summary.magnitude)}</b> · inconsistency <b>${fmt.v(state.summary.variance)}</b>` : ""}${state.reclusteredWith
      ? html` · directions reclustered on <b>${state.reclusteredWith}</b> codes` : ""}`);
    reclusterButton.disabled = !state.codeSel.size;
    resetButton.disabled = !(state.dirSel.size || state.codeSel.size || state.reclusteredWith);
    return {dirLayout, codeLayout};
  }

  function update() {
    const {dirLayout, codeLayout} = render(false);
    const {dirs, codes} = applySelection(dirLayout, codeLayout);
    render(true);
    root.value = {codes, directions: dirs, magnitude: state.summary?.magnitude ?? null, variance: state.summary?.variance ?? null,
                  reclustered: state.reclusteredWith > 0};
    root.dispatchEvent(new Event("input", {bubbles: true}));
  }

  function toggle(set, name) {
    if (set.has(name)) set.delete(name); else set.add(name);
    update();
  }

  function clearSelection() {
    state.dirSel.clear();
    state.codeSel.clear();
    update();
  }

  reclusterButton.onclick = () => {
    const codes = leavesOf(codeLayoutNow(), state.codeSel);
    if (!codes.length) return;
    status.textContent = `Reclustering ${D} directions on ${codes.length} codes…`;
    reclusterButton.disabled = true;
    setTimeout(() => {
      const r = cl.directionRecluster({directionTree: state.dirTree, codeIndicesClustering: codes, codeIndicesCoherence: codes,
                                       pairwiseMetric: clustering.metric, clusteringMethod: clustering.method});
      state.base = r;
      state.dirTree = r.tree;
      state.weave = new Map();
      cl_walk(r.tree, n => { if (n.weaveScore != null) state.weave.set(n.name, n.weaveScore); });
      state.reclusteredWith = codes.length;
      state.dirSel.clear();
      state.keys = {dirs: null, codes: null}; // recolor both trees for the new hierarchy
      update();
    }, 20);
  };
  resetButton.onclick = () => {
    Object.assign(state, {base: model.direction, dirTree: model.direction.tree, codeTree: model.code.tree,
                          weave: new Map(), reclusteredWith: 0, summary: null, keys: {dirs: "", codes: ""}});
    state.dirSel.clear();
    state.codeSel.clear();
    update();
  };
  depthInput.oninput = () => {
    state.depth = +depthInput.value;
    state.dirSel.clear();
    state.codeSel.clear();
    update();
  };
  root.addEventListener("keydown", event => { if (event.key === "Escape") clearSelection(); });

  const {dirLayout, codeLayout} = render(true);
  root.value = {codes: [], directions: [], magnitude: null, variance: null, reclustered: false};
  return root;
}

// ═══ Cell 18 · ConceptLens core ═══════════════════════════════════════════
// The Django server's computations, ported to the browser (conceptlens-core.js).
// Verified against the original views on 42 recorded requests; see parity/ in the repo folder.
core = {
  // ConceptLens core — the Django server's computations, ported to run in the browser.
  //
  // Mirrors server/conceptlens/views.py, utilities/features.py and utilities/trees.py.
  // Trees use the same fields as the server's JSON (name, leaf, visleaf, leaf_indices,
  // quantity, depth, magnitude, var, flatIdx, treeID, expName, weaveScore, ...), so code
  // written against the server responses works unchanged.

  // ─────────────────────────────────────────────────────────────── numeric helpers

  const f32 = Math.fround;
  const scratchF32 = new Float32Array(1);
  const scratchU32 = new Uint32Array(scratchF32.buffer);

  // Round to bfloat16 (round-to-nearest-even), as torch.mean(..., dtype=torch.bfloat16) does.
  function bf16(x) {
    if (!Number.isFinite(x)) return x;
    scratchF32[0] = x;
    const u = scratchU32[0];
    scratchU32[0] = ((u + 0x7fff + ((u >>> 16) & 1)) >>> 0) & 0xffff0000;
    return scratchF32[0];
  }

  // Lookup table from float16 bit patterns to values.
  function float16Table() {
    const table = new Float32Array(65536);
    for (let h = 0; h < 65536; ++h) {
      const sign = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
      table[h] = e === 0 ? sign * m * 2 ** -24 : e === 31 ? (m ? NaN : sign * Infinity) : sign * (1 + m / 1024) * 2 ** (e - 15);
    }
    return table;
  }

  const ascending = values => [...values].sort((a, b) => a - b);
  const mean = values => {
    let s = 0;
    for (const v of values) s += v;
    return s / values.length;
  };
  const populationStd = values => {
    const m = mean(values);
    let s = 0;
    for (const v of values) s += (v - m) ** 2;
    return Math.sqrt(s / values.length);
  };

  // Index into a condensed (upper-triangle) distance vector, as scipy stores it.
  const condensedIndex = (n, i, j) =>
    i < j ? n * i - (i * (i + 1)) / 2 + (j - i - 1) : n * j - (j * (j + 1)) / 2 + (i - j - 1);

  // ─────────────────────────────────────────────────────────────── hierarchical clustering
  // Port of scipy.cluster.hierarchy.linkage(y, method, optimal_ordering=True) for a condensed
  // distance vector y. nn_chain + stable sort + relabel + optimal leaf ordering, including the
  // float32 arithmetic scipy's optimal-leaf-ordering kernel uses, so ties resolve identically.

  const distanceUpdates = {
    complete: (dxi, dyi) => Math.max(dxi, dyi),
    average: (dxi, dyi, dxy, nx, ny) => (nx * dxi + ny * dyi) / (nx + ny),
    weighted: (dxi, dyi) => 0.5 * (dxi + dyi),
    ward: (dxi, dyi, dxy, nx, ny, ni) => {
      const t = 1.0 / (nx + ny + ni);
      return Math.sqrt((ni + nx) * t * dxi * dxi + (ni + ny) * t * dyi * dyi - ni * t * dxy * dxy);
    }
  };

  function nnChain(dists, n, method) {
    const update = distanceUpdates[method];
    if (!update) throw new Error(`Unsupported linkage method: ${method}`);
    const D = Float64Array.from(dists);
    const size = new Int32Array(n).fill(1);
    const chain = new Int32Array(n);
    const Z = [];
    let chainLength = 0, x = 0, y = 0;
    for (let k = 0; k < n - 1; ++k) {
      if (chainLength === 0) {
        chainLength = 1;
        for (let i = 0; i < n; ++i) if (size[i] > 0) { chain[0] = i; break; }
      }
      let currentMin;
      for (;;) {
        x = chain[chainLength - 1];
        if (chainLength > 1) {
          y = chain[chainLength - 2];
          currentMin = D[condensedIndex(n, x, y)];
        } else currentMin = Infinity;
        for (let i = 0; i < n; ++i) {
          if (size[i] === 0 || x === i) continue;
          const dist = D[condensedIndex(n, x, i)];
          if (dist < currentMin) { currentMin = dist; y = i; }
        }
        if (chainLength > 1 && y === chain[chainLength - 2]) break;
        chain[chainLength++] = y;
      }
      chainLength -= 2;
      if (x > y) [x, y] = [y, x];
      const nx = size[x], ny = size[y];
      Z.push([x, y, currentMin, nx + ny]);
      size[x] = 0;
      size[y] = nx + ny;
      for (let i = 0; i < n; ++i) {
        const ni = size[i];
        if (ni === 0 || i === y) continue;
        D[condensedIndex(n, i, y)] = update(D[condensedIndex(n, i, x)], D[condensedIndex(n, i, y)], currentMin, nx, ny, ni);
      }
    }
    const order = Z.map((_, i) => i).sort((a, b) => Z[a][2] - Z[b][2]); // stable, like argsort(kind="mergesort")
    const sorted = order.map(i => Z[i].slice());
    relabel(sorted, n);
    return sorted;
  }

  function relabel(Z, n) {
    const parent = Int32Array.from({length: 2 * n - 1}, (_, i) => i);
    const size = new Int32Array(2 * n - 1).fill(1);
    let next = n;
    const find = x => {
      let p = x;
      while (parent[x] !== x) x = parent[x];
      while (parent[p] !== x) { const q = parent[p]; parent[p] = x; p = q; }
      return x;
    };
    for (const row of Z) {
      const xr = find(row[0]), yr = find(row[1]);
      [row[0], row[1]] = xr < yr ? [xr, yr] : [yr, xr];
      parent[xr] = next;
      parent[yr] = next;
      row[3] = size[next] = size[xr] + size[yr];
      ++next;
    }
  }

  function leavesList(Z, n) {
    const out = [], stack = [2 * n - 2];
    while (stack.length) {
      const c = stack.pop();
      if (c < n) out.push(c);
      else stack.push(Z[c - n][1], Z[c - n][0]);
    }
    return out;
  }

  // scipy's _simultaneous_sort (quicksort on float32 values); only the order of equal values
  // can differ between sorts, and the kernel's result does not depend on it, but we keep it.
  function sortSlice(M, n, lo, hi, col, vals, idx) {
    const len = hi - lo;
    for (let i = 0; i < len; ++i) { vals[i] = M[(lo + i) * n + col]; idx[i] = lo + i; }
    const swap = (i, j) => {
      const v = vals[i]; vals[i] = vals[j]; vals[j] = v;
      const t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    };
    const sort = (off, size) => {
      if (size <= 1) return;
      if (size === 2) { if (vals[off] > vals[off + 1]) swap(off, off + 1); return; }
      if (size === 3) {
        if (vals[off] > vals[off + 1]) swap(off, off + 1);
        if (vals[off + 1] > vals[off + 2]) { swap(off + 1, off + 2); if (vals[off] > vals[off + 1]) swap(off, off + 1); }
        return;
      }
      let pivot = off + (size >> 1);
      const last = off + size - 1;
      if (vals[off] > vals[last]) swap(off, last);
      if (vals[last] > vals[pivot]) { swap(last, pivot); if (vals[off] > vals[last]) swap(off, last); }
      const pivotVal = vals[last];
      let store = off;
      for (let i = off; i < last; ++i) if (vals[i] < pivotVal) swap(i, store++);
      swap(store, last);
      pivot = store - off;
      if (pivot > 1) sort(off, pivot);
      if (pivot + 2 < size) sort(off + pivot + 1, size - pivot - 1);
    };
    sort(0, len);
  }

  function optimalLeafOrdering(Z, dists, n) {
    const leaves = leavesList(Z, n);
    const pos = new Int32Array(n);
    leaves.forEach((leaf, i) => (pos[leaf] = i));
    const sZ = Z.map(([l, r]) => [l < n ? pos[l] : l, r < n ? pos[r] : r]);
    const SD = new Float64Array(n * n);
    for (let i = 0; i < n; ++i)
      for (let j = 0; j < n; ++j) if (i !== j) SD[i * n + j] = dists[condensedIndex(n, leaves[i], leaves[j])];
    const ranges = new Int32Array(2 * (2 * n - 1));
    for (let i = 0; i < n; ++i) { ranges[2 * i] = i; ranges[2 * i + 1] = i + 1; }
    sZ.forEach(([l, r], i) => { ranges[2 * (i + n)] = ranges[2 * l]; ranges[2 * (i + n) + 1] = ranges[2 * r + 1]; });

    const M = new Float32Array(n * n);
    const swapStatus = new Int32Array(n * n * 2);
    const mustSwap = new Int32Array(n - 1);
    const mVals = new Float32Array(n), mIdx = new Int32Array(n), kVals = new Float32Array(n), kIdx = new Int32Array(n);
    const BIG = 1073741824;
    let bestU = 0, bestW = 0;
    for (let i = 0; i < n - 1; ++i) {
      const [vl, vr] = sZ[i];
      const uC = [], mC = [], wC = [], kC = [];
      if (vl < n) { uC.push(vl); mC.push(vl); } else { const [a, b] = sZ[vl - n]; uC.push(a, b); mC.push(b, a); }
      if (vr < n) { wC.push(vr); kC.push(vr); } else { const [a, b] = sZ[vr - n]; wC.push(b, a); kC.push(a, b); }
      for (let sl = 0; sl < uC.length; ++sl) {
        for (let sr = 0; sr < wC.length; ++sr) {
          const uMin = ranges[2 * uC[sl]], uMax = ranges[2 * uC[sl] + 1];
          const mMin = ranges[2 * mC[sl]], mMax = ranges[2 * mC[sl] + 1];
          const wMin = ranges[2 * wC[sr]], wMax = ranges[2 * wC[sr] + 1];
          const kMin = ranges[2 * kC[sr]], kMax = ranges[2 * kC[sr] + 1];
          let minKm = BIG;
          for (let m = mMin; m < mMax; ++m)
            for (let k = kMin; k < kMax; ++k) if (SD[m * n + k] < minKm) minKm = f32(SD[m * n + k]);
          for (let u = uMin; u < uMax; ++u) {
            sortSlice(M, n, mMin, mMax, u, mVals, mIdx);
            for (let w = wMin; w < wMax; ++w) {
              sortSlice(M, n, kMin, kMax, w, kVals, kIdx);
              let curMin = BIG;
              for (let mi = 0; mi < mMax - mMin; ++mi) {
                const m = mIdx[mi], Mum = M[u * n + m];
                if (f32(f32(Mum + M[w * n + kIdx[0]]) + minKm) >= curMin) break;
                for (let ki = 0; ki < kMax - kMin; ++ki) {
                  const k = kIdx[ki], Mwk = M[w * n + k];
                  if (f32(f32(Mum + Mwk) + minKm) >= curMin) break;
                  const current = f32(f32(Mum + Mwk) + SD[m * n + k]);
                  if (current < curMin) curMin = current;
                }
              }
              M[u * n + w] = M[w * n + u] = curMin;
              swapStatus[2 * (u * n + w)] = swapStatus[2 * (w * n + u)] = sl;
              swapStatus[2 * (u * n + w) + 1] = swapStatus[2 * (w * n + u) + 1] = sr;
            }
          }
        }
      }
      let curMin = BIG;
      for (let u = ranges[2 * vl]; u < ranges[2 * vl + 1]; ++u)
        for (let w = ranges[2 * vr]; w < ranges[2 * vr + 1]; ++w)
          if (M[u * n + w] < curMin) { curMin = M[u * n + w]; bestU = u; bestW = w; }
      if (vl >= n) mustSwap[vl - n] = swapStatus[2 * (bestU * n + bestW)];
      if (vr >= n) mustSwap[vr - n] = swapStatus[2 * (bestU * n + bestW) + 1];
    }
    // A cluster ends up flipped when an odd number of its ancestors (itself included) must swap.
    const parentOf = new Int32Array(n - 1).fill(-1);
    sZ.forEach(([l, r], i) => { if (l >= n) parentOf[l - n] = i; if (r >= n) parentOf[r - n] = i; });
    return Z.map(([l, r, h, s], j) => {
      let flips = 0;
      for (let a = j; a !== -1; a = parentOf[a]) flips += mustSwap[a];
      return flips % 2 ? [r, l, h, s] : [l, r, h, s];
    });
  }

  function linkage(dists, n, method = "complete", {optimalOrdering = true} = {}) {
    const Z = nnChain(dists, n, method);
    return optimalOrdering ? optimalLeafOrdering(Z, dists, n) : Z;
  }

  // ─────────────────────────────────────────────────────────────── trees (trees.py)

  const blankNode = fields => ({
    leaf: false, visleaf: false, parent: null, treeID: null, expName: null, flatIdx: null, children: [],
    leaf_indices: [], depth: 0, quantity: 0, trip: 0, var: 0, magnitude: 0,
    contributions: [], mag_contribution: 0, var_contribution: 0, ...fields
  });

  // create_tree_from_linkage: sizes are cumulative direction counts per experiment.
  function treeFromLinkage(Z, cumulativeSizes, experimentNames) {
    const rows = Z.length;
    const inner = new Map(Z.map((row, i) => [i + 1 + rows, [row[0], row[1]]]));
    const root = blankNode({name: 2 * rows, timestamp: Date.now() / 1000, origin: rows});
    const stack = [root];
    while (stack.length) {
      const node = stack.pop();
      for (const child of inner.get(node.name) ?? []) {
        let next;
        if (!inner.has(child)) {
          const treeID = cumulativeSizes.findIndex(size => child - size < 0);
          next = blankNode({
            name: child, leaf: true, parent: node.name, treeID, expName: experimentNames[treeID],
            flatIdx: treeID > 0 ? child - cumulativeSizes[treeID - 1] : child
          });
        } else next = blankNode({name: child, parent: node.name});
        node.children.push(next);
        stack.push(next);
      }
    }
    return root;
  }

  function count(node, depth = 0) {
    if (node.leaf) {
      node.quantity = 1;
      node.leaf_indices = [node.name];
      node.depth = depth;
    }
    for (const child of node.children) count(child, depth + 1);
    for (const child of node.children) {
      node.quantity += child.quantity;
      node.leaf_indices.push(...child.leaf_indices);
      node.depth = depth;
    }
    return node;
  }

  function removeLeaves(node) {
    for (const child of node.children) removeLeaves(child);
    node.leaf_indices = [];
    node.quantity = 0;
    node.depth = 0;
    return node;
  }

  // tree_trimming(tree, treemin, mode="size"): stop splitting where a child would be tiny.
  function trim(tree, treemin = 3) {
    for (const child of tree.children) {
      if (!(child.leaf_indices.length >= treemin) || child.leaf) {
        tree.leaf = true;
        tree.visleaf = true;
        return tree;
      }
      trim(child, treemin);
    }
    return tree;
  }

  function walk(node, visit) {
    visit(node);
    for (const child of node.children) walk(child, visit);
  }

  function postOrder(node, out = []) {
    for (const child of node.children) postOrder(child, out);
    out.push(node);
    return out;
  }

  function visleafStats(tree) {
    const mags = [], vars = [];
    for (const node of postOrder(tree)) if (node.visleaf) { mags.push(node.magnitude); vars.push(node.var); }
    return {avgMagnitude: mean(mags), avgStd: mean(vars), magsStd: populationStd(mags), stdsStd: populationStd(vars)};
  }

  // The clusters drawn as leaves (nodes marked leaf by trimming), as lists of direction ids.
  function visibleClusters(tree, key) {
    const out = [];
    const visit = node => {
      if (node.leaf) out.push(node.leaf_indices.length ? node.leaf_indices.map(key) : [key(node.name)]);
      else node.children.forEach(visit);
    };
    visit(tree);
    return out;
  }

  // compute_leaf_weave: how much each new visible cluster scatters across the old ones.
  // legacy=true keys directions by flatIdx like the server, which conflates directions from
  // different experiments that share a flatIdx; legacy=false keys them by global id.
  function weaveScores(oldTree, newTree, flatIdx, {legacy = true} = {}) {
    const key = legacy ? name => flatIdx(name) : name => name;
    const membership = new Map();
    visibleClusters(oldTree, key).forEach((ids, m) => ids.forEach(id => membership.set(id, m)));
    return visibleClusters(newTree, key).map(ids => {
      const mem = ids.map(id => membership.get(id));
      let sum = 0;
      for (let i = 0; i < mem.length; ++i) for (let j = i + 1; j < mem.length; ++j) sum += Math.abs(mem[i] - mem[j]);
      return sum / (mem.length + 0.001);
    });
  }

  function appendWeave(tree, scores) {
    let i = 0;
    const visit = node => {
      if (node.leaf) node.weaveScore = scores[i++];
      else node.children.forEach(visit);
    };
    visit(tree);
  }

  // ─────────────────────────────────────────────────────────────── the data model

  class ConceptLens {
    // bundle: {manifest, features: Uint16Array|Float32Array, magnitude: Float32Array,
    //          directionDistance: {cosine, euclidean}: Float64Array, codeDistance: Float64Array}
    constructor(bundle, {treemin = 3} = {}) {
      const {manifest} = bundle;
      this.manifest = manifest;
      this.nCode = manifest.n_code;
      this.nDirection = manifest.n_direction;
      this.dim = manifest.feature_dim;
      this.experiments = manifest.experiments;
      this.names = manifest.experiments.map(e => e.name);
      this.cumulative = manifest.experiments.map(e => e.offset + e.count);
      this.features = bundle.features;
      this.decode = bundle.features instanceof Uint16Array ? float16Table() : null;
      this.magnitude = bundle.magnitude;
      this.directionDistance = bundle.directionDistance;
      this.codeDistance = bundle.codeDistance;
      this.treemin = treemin;
      this.nodeVarianceCache = new Map();
      let lo = Infinity, hi = -Infinity;
      for (const v of this.magnitude) { if (v < lo) lo = v; if (v > hi) hi = v; }
      this.magmin = lo;
      this.magmax = hi;
    }

    flatIdx = name => {
      const t = this.cumulative.findIndex(size => name - size < 0);
      return t > 0 ? name - this.cumulative[t - 1] : name;
    };

    // Feature row (code, direction) as float32 values.
    row(code, direction, out = new Float32Array(this.dim)) {
      const base = (code * this.nDirection + direction) * this.dim, F = this.features, dec = this.decode;
      if (dec) for (let f = 0; f < this.dim; ++f) out[f] = dec[F[base + f]];
      else for (let f = 0; f < this.dim; ++f) out[f] = F[base + f];
      return out;
    }

    // bf16 mean edit magnitude over codes × directions (trees.py node magnitude).
    nodeMagnitude(codes, directions) {
      const M = this.magnitude, D = this.nDirection;
      let s = 0;
      for (const c of codes) for (const d of directions) s += M[c * D + d];
      return bf16(s / (codes.length * directions.length));
    }

    // Plain mean edit magnitude (the selection summary in views.py).
    meanMagnitude(codes, directions) {
      const M = this.magnitude, D = this.nDirection;
      let s = 0;
      for (const c of codes) for (const d of directions) s += M[c * D + d];
      return f32(s / (codes.length * directions.length));
    }

    // Per code: mean over feature dims of the (unbiased) std across the given directions.
    codeVariances(directions, codes = this.allCodes()) {
      const F = this.dim, n = directions.length, out = new Float64Array(this.nCode).fill(NaN);
      if (n < 2) return out;
      const s1 = new Float64Array(F), s2 = new Float64Array(F), row = new Float32Array(F);
      for (const c of codes) {
        s1.fill(0); s2.fill(0);
        for (const d of directions) {
          this.row(c, d, row);
          for (let f = 0; f < F; ++f) { const v = row[f]; s1[f] += v; s2[f] += v * v; }
        }
        let acc = 0;
        for (let f = 0; f < F; ++f) acc += Math.sqrt(Math.max(0, (s2[f] - (s1[f] * s1[f]) / n) / (n - 1)));
        out[c] = acc / F;
      }
      return out;
    }

    // For every node of a direction tree and every code: that code's variance across the
    // node's directions. Built bottom-up from per-dimension sums, cached per tree shape.
    nodeVariances(tree) {
      const nodes = postOrder(tree);
      const key = nodes.map(n => n.name).join(",");
      if (this.nodeVarianceCache.has(key)) return this.nodeVarianceCache.get(key);
      const F = this.dim, N = nodes.length, index = new Map(nodes.map((n, i) => [n, i]));
      const size = nodes.map(n => (n.children.length ? 0 : 1));
      nodes.forEach((n, i) => n.children.forEach(ch => (size[i] += size[index.get(ch)])));
      const table = nodes.map(() => new Float64Array(this.nCode).fill(NaN));
      const s1 = new Float64Array(N * F), s2 = new Float64Array(N * F), row = new Float32Array(F);
      for (let c = 0; c < this.nCode; ++c) {
        for (let i = 0; i < N; ++i) {
          const node = nodes[i], o = i * F;
          if (!node.children.length) {
            this.row(c, node.name, row);
            for (let f = 0; f < F; ++f) { const v = row[f]; s1[o + f] = v; s2[o + f] = v * v; }
            continue;
          }
          s1.fill(0, o, o + F); s2.fill(0, o, o + F);
          for (const ch of node.children) {
            const p = index.get(ch) * F;
            for (let f = 0; f < F; ++f) { s1[o + f] += s1[p + f]; s2[o + f] += s2[p + f]; }
          }
          const n = size[i];
          let acc = 0;
          for (let f = 0; f < F; ++f) acc += Math.sqrt(Math.max(0, (s2[o + f] - (s1[o + f] * s1[o + f]) / n) / (n - 1)));
          table[i][c] = acc / F;
        }
      }
      const result = new Map(nodes.map((n, i) => [n.name, table[i]]));
      this.nodeVarianceCache.set(key, result);
      return result;
    }

    // compute_direction_coherence over the given codes.
    directionCoherence(tree, codes) {
      const table = this.nodeVariances(tree);
      walk(tree, node => {
        const dirs = ascending(node.leaf_indices);
        node.magnitude = this.nodeMagnitude(codes, dirs);
        const v = table.get(node.name);
        let s = 0;
        for (const c of codes) s += v[c];
        const variance = f32(s / codes.length);
        node.var = Number.isNaN(variance) ? 0 : variance;
      });
    }

    // compute_coherence_code over the given directions.
    codeCoherence(tree, directions) {
      const perCode = this.codeVariances(directions);
      walk(tree, node => {
        const codes = ascending(node.leaf_indices);
        node.magnitude = this.nodeMagnitude(codes, directions);
        let s = 0;
        for (const c of codes) s += perCode[c];
        const variance = f32(s / codes.length);
        node.var = Number.isNaN(variance) ? 0 : variance;
      });
      return perCode;
    }

    allCodes() { return Array.from({length: this.nCode}, (_, i) => i); }
    allDirections() { return Array.from({length: this.nDirection}, (_, i) => i); }

    // views.code_hierarchy_initialization
    codeInitialization({clusteringMethod = "complete"} = {}) {
      const Z = linkage(this.codeDistance, this.nCode, clusteringMethod);
      const tree = count(treeFromLinkage(Z, [this.nCode], this.names));
      this.codeCoherence(tree, this.allDirections());
      const rawtree = structuredClone(tree);
      trim(tree, this.treemin);
      const {avgMagnitude, avgStd} = visleafStats(tree);
      return {tree, rawtree, avgMagnitude, avgStd};
    }

    // views.direction_hierarchy_initialization
    directionInitialization({clusteringMethod = "complete", pairwiseMetric = "cosine"} = {}) {
      const Z = linkage(this.directionDistance[pairwiseMetric], this.nDirection, clusteringMethod);
      return this.#directionTree(Z, this.allCodes());
    }

    #directionTree(Z, coherenceCodes) {
      const tree = count(treeFromLinkage(Z, this.cumulative, this.names));
      this.directionCoherence(tree, coherenceCodes);
      const rawtree = structuredClone(tree);
      trim(tree, this.treemin);
      return {tree, rawtree, ...visleafStats(tree)};
    }

    // Mean pairwise distance between directions, averaged over the given codes, after
    // centering on the mean of those codes' walked features (postprocess_clustering_features).
    directionDistances(codes, metric = "cosine") {
      const D = this.nDirection, F = this.dim;
      const centre = new Float64Array(F), row = new Float32Array(F);
      for (const c of codes) for (let d = 0; d < D; ++d) {
        this.row(c, d, row);
        for (let f = 0; f < F; ++f) centre[f] += row[f];
      }
      for (let f = 0; f < F; ++f) centre[f] /= codes.length * D;
      const out = new Float64Array((D * (D - 1)) / 2);
      const X = new Float64Array(D * F), norm = new Float64Array(D);
      for (const c of codes) {
        for (let d = 0; d < D; ++d) {
          this.row(c, d, row);
          let s = 0;
          for (let f = 0; f < F; ++f) { const v = row[f] - centre[f]; X[d * F + f] = v; s += v * v; }
          norm[d] = Math.sqrt(s);
        }
        let idx = 0;
        for (let i = 0; i < D; ++i) {
          for (let j = i + 1; j < D; ++j) {
            let s = 0;
            if (metric === "cosine") {
              for (let f = 0; f < F; ++f) s += X[i * F + f] * X[j * F + f];
              out[idx++] += 1 - s / (norm[i] * norm[j]);
            } else {
              for (let f = 0; f < F; ++f) { const t = X[i * F + f] - X[j * F + f]; s += t * t; }
              out[idx++] += Math.sqrt(s);
            }
          }
        }
      }
      for (let i = 0; i < out.length; ++i) out[i] /= codes.length;
      return out;
    }

    // views.direction_hierarchy_recluster
    directionRecluster({directionTree, codeIndicesClustering, codeIndicesCoherence, pairwiseMetric = "cosine",
                        clusteringMethod = "complete", legacyWeave = true}) {
      const sortCodes = codes => (codes?.length ? ascending(codes) : this.allCodes());
      const clusteringCodes = sortCodes(codeIndicesClustering);
      const dists = clusteringCodes.length === this.nCode
        ? this.directionDistance[pairwiseMetric]
        : this.directionDistances(clusteringCodes, pairwiseMetric);
      const Z = linkage(dists, this.nDirection, clusteringMethod);
      const result = this.#directionTree(Z, sortCodes(codeIndicesCoherence));
      const scores = weaveScores(directionTree, result.tree, this.flatIdx, {legacy: legacyWeave});
      appendWeave(result.tree, scores);
      return {...result, weaveScores: scores};
    }

    // views.direction_hierarchy_selection: directions were brushed; update the code tree.
    directionSelection({codeTree, codeIndices = [], directionIndices = []}) {
      const tree = structuredClone(codeTree);
      const codes = ascending(codeIndices);
      const dirs = directionIndices.length ? ascending(directionIndices) : this.allDirections();
      const perCode = this.codeCoherence(tree, dirs);
      count(removeLeaves(tree));
      trim(tree, this.treemin);
      return {
        codeTree: tree,
        ...this.#summary(codes.length ? codes : this.allCodes(), dirs, perCode),
        contributions: this.#contributions(codes, dirs, codeIndices)
      };
    }

    // views.code_hierarchy_selection: codes were brushed; update the direction tree.
    codeSelection({directionTree, codeIndices = [], directionIndices = []}) {
      const tree = structuredClone(directionTree);
      const codes = codeIndices.length ? ascending(codeIndices) : this.allCodes();
      const dirs = ascending(directionIndices);
      this.directionCoherence(tree, codes);
      const summaryDirs = dirs.length ? dirs : this.allDirections();
      count(removeLeaves(tree));
      trim(tree, this.treemin);
      return {
        directionTree: tree,
        ...this.#summary(codes, summaryDirs, this.codeVariances(summaryDirs, codes)),
        contributions: this.#contributions(codes, dirs, codeIndices.length ? codeIndices : codes)
      };
    }

    // Mean magnitude and inconsistency of the selected codes × directions.
    #summary(codes, dirs, perCode) {
      let v = 0;
      for (const c of codes) v += perCode[c];
      const magnitude = this.meanMagnitude(codes, dirs);
      return {
        magnitude: Number.isNaN(magnitude) ? 0 : magnitude,
        variance: f32(v / codes.length),
        magmin: this.magmin,
        magmax: this.magmax
      };
    }

    // compute_self_magnitude + the server's reordering by the order codes were passed in.
    #contributions(codes, dirs, unsortedCodes) {
      const rank = new Map(unsortedCodes.map((c, i) => [c, i]));
      const out = [];
      for (const c of codes) for (const d of dirs) {
        const m = this.magnitude[c * this.nDirection + d];
        out.push({code: c, direction: d, mag_contribution: m, var_contribution: m});
      }
      return out.sort((a, b) => (rank.get(a.code) ?? Infinity) - (rank.get(b.code) ?? Infinity));
    }
  }

  // ─────────────────────────────────────────────────────────────── loading a bundle
  // `open(name)` returns something with json(), arrayBuffer() and url() — an Observable
  // FileAttachment, or fromUrl(base)(name) for files on a static host.

  const fromAttachments = files => name => {
    if (!(name in files)) throw new Error(`File not attached to the notebook: ${name}`);
    return files[name];
  };

  const fromUrl = base => name => {
    const href = new URL(name, base.endsWith("/") ? base : base + "/").href;
    const get = async () => {
      const response = await fetch(href);
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${href}`);
      return response;
    };
    return {url: async () => href, json: async () => (await get()).json(), arrayBuffer: async () => (await get()).arrayBuffer()};
  };

  async function loadBundle(open) {
    const manifest = await open("manifest.json").json();
    const typed = {float16: Uint16Array, float32: Float32Array, float64: Float64Array};
    const load = async file => new typed[file.dtype](await open(file.path).arrayBuffer());
    const f = manifest.files;
    const [parts, magnitude, codeDistance] = await Promise.all([
      Promise.all(f.features.parts.map(part => open(part.path).arrayBuffer())),
      load(f.magnitude), load(f.code_distance)
    ]);
    const Features = typed[f.features.dtype];
    const features = new Features(f.features.shape.reduce((a, b) => a * b, 1));
    let offset = 0;
    for (const part of parts) {
      const values = new Features(part);
      features.set(values, offset);
      offset += values.length;
    }
    if (offset !== features.length) throw new Error(`Feature parts hold ${offset} values; expected ${features.length}`);
    const sprite = manifest.images ? code => open(manifest.images.sprite.replace("{code}", code)).url() : null;
    return {manifest, features, magnitude, codeDistance, sprite, open};
  }

  // The bundle restricted to some experiments (methods), laid out exactly as the server would see
  // them with experiment_names = those experiments (in bundle order). `global[d]` maps a local
  // direction back to its index in the full bundle (used for the image sprites).
  async function selectExperiments(bundle, names = bundle.manifest.experiments.map(e => e.name)) {
    const {manifest} = bundle, all = manifest.experiments;
    const chosen = all.filter(e => names.includes(e.name));
    if (!chosen.length) throw new Error("Select at least one method");
    const mask = chosen.reduce((m, e) => m | (1 << all.indexOf(e)), 0);
    const dd = manifest.files.direction_distance;
    if (!dd.masks.includes(mask)) throw new Error(`This combination of methods was not exported: ${chosen.map(e => e.name).join(", ")}`);
    const [cosine, euclidean] = await Promise.all(["cosine", "euclidean"].map(async metric =>
      new Float64Array(await bundle.open(dd.pattern.replace("{metric}", metric).replace("{mask}", mask)).arrayBuffer())));
    const C = manifest.n_code, DA = manifest.n_direction, F = manifest.feature_dim;
    const D = chosen.reduce((s, e) => s + e.count, 0);
    const global = new Int32Array(D);
    let features = bundle.features, magnitude = bundle.magnitude, running = 0;
    const experiments = chosen.map(e => {
      for (let i = 0; i < e.count; ++i) global[running + i] = e.offset + i;
      const local = {...e, offset: running, globalOffset: e.offset};
      running += e.count;
      return local;
    });
    if (D !== DA) {
      features = new features.constructor(C * D * F);
      magnitude = new Float32Array(C * D);
      for (let c = 0; c < C; ++c) {
        for (const e of experiments) {
          const from = c * DA + e.globalOffset, to = c * D + e.offset;
          features.set(bundle.features.subarray(from * F, (from + e.count) * F), to * F);
          magnitude.set(bundle.magnitude.subarray(from, from + e.count), to);
        }
      }
    }
    return {
      ...bundle,
      manifest: {...manifest, n_direction: D, experiments, all_experiments: all.map(e => e.name)},
      features, magnitude, directionDistance: {cosine, euclidean}, global
    };
  }

  return {bf16, float16Table, condensedIndex, leavesList, linkage, treeFromLinkage, count, removeLeaves, trim, walk, weaveScores, ConceptLens, fromAttachments, fromUrl, loadBundle, selectExperiments};
}
