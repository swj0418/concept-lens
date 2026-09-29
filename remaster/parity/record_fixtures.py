"""Record request/response fixtures from the ORIGINAL ConceptLens Django views.

Calls conceptlens.views.* exactly as the React client would (same JSON bodies), so the
JavaScript port (conceptlens-core.js) can be checked against the real server code.
Needs the server's Python environment (torch, scipy, sklearn, django, natsort).

usage:
    python parity/record_fixtures.py --server <repo>/server --data <repo>/resources/experiments \
        --experiments <same comma-separated list, same order, as export_for_observable.py> --out fixtures.json
    node parity/parity.mjs <bundle dir> fixtures.json
"""
import argparse, contextlib, copy, io, json, os, random, sys

p = argparse.ArgumentParser()
p.add_argument("--server", required=True)
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--experiments", default=None, help="comma-separated; default: all folders")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--selections", type=int, default=6)
a = p.parse_args()

sys.path.insert(0, a.server)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "server.settings")
import django
from django.conf import settings
django.setup()
settings.SERVED_DATA_ROOT = a.data
from conceptlens import views  # noqa: E402  (reads SERVED_DATA_ROOT at import)
views.SERVED_DATA_ROOT = a.data

import natsort  # noqa: E402
names = a.experiments.split(",") if a.experiments else natsort.natsorted(os.listdir(a.data))
rnd = random.Random(a.seed)


class Request:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()
        self.method = "POST"


def call(view, payload):
    with contextlib.redirect_stdout(io.StringIO()):
        response = view(Request(payload))
    return json.loads(json.loads(response.content))


def strip(node, top=True):
    """Copy without the duplicated 'leaves' lists (every internal node repeats all its leaf dicts)."""
    if top:
        node = copy.deepcopy(node)
    if isinstance(node, dict):
        node.pop("leaves", None)
        for child in node.get("children", []):
            strip(child, False)
    return node


def leaf_order(node, out):
    if not node["children"]:
        out.append(node["name"])
    for child in node["children"]:
        leaf_order(child, out)
    return out


def nodes(node, out):
    out.append(node)
    for child in node["children"]:
        nodes(child, out)
    return out


fixtures = {"experiments": names, "cases": []}
for method in ["complete", "ward"]:
    code = call(views.code_hierarchy_initialization, {
        "experiment_names": names, "tree": None, "directions": None,
        "pairwise_metric": "cosine", "clustering_method": method, "truncated_tree": True})
    fixtures["cases"].append({"endpoint": "code_initialization", "request": {"clustering_method": method},
                              "response": {k: strip(v) if k in ("tree", "rawtree") else v for k, v in code.items() if k != "timestamp"}})
    for metric in ["cosine", "euclidean"]:
        direction = call(views.direction_hierarchy_initialization, {
            "experiment_names": names, "tree": None, "code_selection_clustering": None,
            "code_selection_coherence": None, "pairwise_metric": metric,
            "clustering_method": method, "truncated_tree": True})
        fixtures["cases"].append({"endpoint": "direction_initialization",
                                  "request": {"clustering_method": method, "pairwise_metric": metric},
                                  "response": {k: strip(v) if k in ("tree", "rawtree") else v for k, v in direction.items() if k != "timestamp"}})
        if not (method == "complete"):
            continue
        # Selections are built like the client builds them: brushed runs of leaves, or a clicked node's leaves.
        d_order = leaf_order(direction["rawtree"], [])
        c_order = leaf_order(code["rawtree"], [])
        d_nodes = [n for n in nodes(direction["rawtree"], []) if 2 <= len(n["leaf_indices"]) < len(d_order)]
        c_nodes = [n for n in nodes(code["rawtree"], []) if 2 <= len(n["leaf_indices"]) < len(c_order)]
        for s in range(a.selections):
            if s % 2 == 0:
                i = rnd.randrange(0, len(d_order) - 2); j = rnd.randrange(i + 2, min(len(d_order), i + 40) + 1)
                dirs = d_order[i:j]
            else:
                dirs = list(rnd.choice(d_nodes)["leaf_indices"])
            codes = [] if s % 3 == 0 else list(rnd.choice(c_nodes)["leaf_indices"])
            if s % 3 == 2:
                i = rnd.randrange(0, len(c_order) - 2); codes = c_order[i:i + rnd.randrange(2, 30)]
            req = {"experiment_names": names, "code_tree": code["rawtree"], "code_indices": codes, "direction_indices": dirs}
            res = call(views.direction_hierarchy_selection, json.loads(json.dumps(req)))
            res.pop("timestamp", None); res["codeTree"] = strip(res["codeTree"])
            fixtures["cases"].append({"endpoint": "direction_selection",
                                      "request": {"metric": metric, "code_indices": codes, "direction_indices": dirs}, "response": res})
            codes2 = codes or list(rnd.choice(c_nodes)["leaf_indices"])
            dirs2 = [] if s % 2 else dirs
            req = {"experiment_names": names, "direction_tree": direction["rawtree"], "code_indices": codes2, "direction_indices": dirs2}
            res = call(views.code_hierarchy_selection, json.loads(json.dumps(req)))
            res.pop("timestamp", None); res["directionTree"] = strip(res["directionTree"])
            fixtures["cases"].append({"endpoint": "code_selection",
                                      "request": {"metric": metric, "code_indices": codes2, "direction_indices": dirs2}, "response": res})
            req = {"experiment_names": names, "direction_tree": direction["tree"], "code_indices_clustering": codes2,
                   "code_indices_coherence": codes2, "pairwise_metric": metric, "clustering_method": method}
            res = call(views.direction_hierarchy_recluster, json.loads(json.dumps(req)))
            res.pop("timestamp", None)
            for k in ("tree", "rawtree"):
                res[k] = strip(res[k])
            fixtures["cases"].append({"endpoint": "direction_recluster",
                                      "request": {"metric": metric, "clustering_method": method, "code_indices": codes2}, "response": res})

with open(a.out, "w") as f:
    json.dump(fixtures, f)
print(f"{len(fixtures['cases'])} cases from {len(names)} experiments -> {a.out}")
