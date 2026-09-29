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
export function bf16(x) {
  if (!Number.isFinite(x)) return x;
  scratchF32[0] = x;
  const u = scratchU32[0];
  scratchU32[0] = ((u + 0x7fff + ((u >>> 16) & 1)) >>> 0) & 0xffff0000;
  return scratchF32[0];
}

// Lookup table from float16 bit patterns to values.
export function float16Table() {
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
export const condensedIndex = (n, i, j) =>
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

export function leavesList(Z, n) {
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

export function linkage(dists, n, method = "complete", {optimalOrdering = true} = {}) {
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
export function treeFromLinkage(Z, cumulativeSizes, experimentNames) {
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

export function count(node, depth = 0) {
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

export function removeLeaves(node) {
  for (const child of node.children) removeLeaves(child);
  node.leaf_indices = [];
  node.quantity = 0;
  node.depth = 0;
  return node;
}

// tree_trimming(tree, treemin, mode="size"): stop splitting where a child would be tiny.
export function trim(tree, treemin = 3) {
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

export function walk(node, visit) {
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
export function weaveScores(oldTree, newTree, flatIdx, {legacy = true} = {}) {
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

export class ConceptLens {
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

export const fromAttachments = files => name => {
  if (!(name in files)) throw new Error(`File not attached to the notebook: ${name}`);
  return files[name];
};

export const fromUrl = base => name => {
  const href = new URL(name, base.endsWith("/") ? base : base + "/").href;
  const get = async () => {
    const response = await fetch(href);
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${href}`);
    return response;
  };
  return {url: async () => href, json: async () => (await get()).json(), arrayBuffer: async () => (await get()).arrayBuffer()};
};

export async function loadBundle(open) {
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
export async function selectExperiments(bundle, names = bundle.manifest.experiments.map(e => e.name)) {
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
