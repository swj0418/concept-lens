import copy
import os
import time
import json

import torch

import numpy as np

from scipy.cluster.hierarchy import linkage
from json import encoder
from .utilities import features, readers, trees

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utilities.features import compute_euclidean_distance

# For debugging purposes
torch.set_warn_always(False)

# GLOBAL
SERVED_DATA_ROOT = 'served_data'
torch.set_printoptions(precision=3, sci_mode=False)
encoder.FLOAT_REPR = lambda o: format(o, '.2f')
node_size_minimum_control = 3
node_size_divisor_global = 40
node_variance_threshold = 1.0  # Go over this variance then do not merge.

# Clustering - End vector and then centering using all codes' mean.
clustering_centering = 'all_code_mean'

# Edit Magnitude - there is no reason to center the difference vector. If you were to center the difference vector,
# You'd have to center the original codes' embedding as well.
mag_feature_derivation = 'diff'
mag_feature_centering = None

# Variance
# centering modes: [None, 'individual_code_mean', 'all_code_mean', 'w_mean', 'global_mean']
var_distance_metric = 'raw'
var_feature_derivation = 'end'
var_feature_centering = None
var_feature_prior_subset = False

use_latent = False
var_normalize = False


def _read_request_form(body):
    data = json.loads(body)

    experiment_names = data['experiment_names']
    previous_tree = data['tree']
    code_selection_clustering = data['code_selection_clustering']
    code_selection_coherence = data['code_selection_coherence']
    pairwise_metric = data['pairwise_metric']
    clustering_method = data['clustering_method']
    truncated_tree = data['truncated_tree']

    if code_selection_clustering:
        code_selection_clustering = sorted(code_selection_clustering)
    if code_selection_coherence:
        code_selection_coherence = sorted(code_selection_coherence)

    return (experiment_names, previous_tree, code_selection_clustering,
            code_selection_coherence, pairwise_metric, clustering_method, truncated_tree)


def _compute_mean_std(node_magnitudes, node_variances):
    # Mean for coherence color encoding domain mean setting.
    mags_mean, stds_mean = np.mean(node_magnitudes), np.mean(node_variances)

    # Standard deviations for coherence color encoding domain min / max setting
    mags_std, stds_std = np.std(node_magnitudes), np.std(node_variances)  # Using node as std

    return mags_mean, stds_mean, mags_std, stds_std


@csrf_exempt
def direction_hierarchy_initialization(request):
    (experiment_names, previous_tree, code_selection_clustering, code_selection_coherence,
     pairwise_metric, clustering_method, truncated_tree) = _read_request_form(request.body)

    # Read features
    if not use_latent:
        walk_features, cumulative_dfm_size = readers.read_walk_features(experiment_names, served_data_root=SERVED_DATA_ROOT)
        original_features = readers.read_original_features(experiment_names, served_data_root=SERVED_DATA_ROOT)
    else:
        walk_features, cumulative_dfm_size = readers.read_walk_latent_codes(experiment_names, served_data_root=SERVED_DATA_ROOT)
        original_features = readers.read_original_latent_codes(experiment_names, served_data_root=SERVED_DATA_ROOT)

    # Process features
    w_mean = torch.mean(original_features, dim=0)
    clustering_features = features.postprocess_clustering_features(walk_features,
                                                                   original_features,
                                                                   code_selection=None,
                                                                   mode='end',
                                                                   prior_subset=True,
                                                                   centering=clustering_centering,
                                                                   w_mean=w_mean)
    clustering_features = features.compute_mean_features_pairwise_distances(clustering_features,
                                                                            dist_metric=pairwise_metric)

    mag_features = features.postprocess_magnitude_features(walk_features,
                                                           original_features,
                                                           code_selection=None,
                                                           direction_selection=None)

    var_features = features.postprocess_variance_features(walk_features,
                                                          original_features,
                                                          mode=var_feature_derivation,
                                                          code_selection=None,
                                                          direction_selection=None,
                                                          centering=var_feature_centering,
                                                          prior_subset=var_feature_prior_subset,
                                                          w_mean=w_mean,
                                                          normalize=var_normalize)

    # Clustering
    z = linkage(clustering_features.numpy(), clustering_method, optimal_ordering=True)

    # Tree-building => convert linkage into tree. (total of 2 traversals)
    tree = trees.create_tree_from_linkage(z, cumulative_dfm_size, experiment_names)
    trees.count(tree)  # Count quantities, depth, and leaves
    trees.compute_direction_coherence(tree,
                                      mag_features,
                                      var_features,
                                      collect_only_leaf=False)

    # Tree-trimming
    rawtree = copy.deepcopy(tree)
    trees.tree_trimming(tree, node_size_minimum_control, mode='size')
    node_magnitudes, node_variances = trees.collect_visleaf_coherence(tree, target_depth=7)
    mags_mean, stds_mean, mags_std, stds_std = _compute_mean_std(node_magnitudes, node_variances)

    resp = json.dumps({
        'tree': tree,
        'rawtree': rawtree,
        'timestamp': time.time(),
        'avgMagnitude': float(mags_mean),
        'avgStd': float(stds_mean),
        'magsStd': float(mags_std),
        'stdsStd': float(stds_std)
    }, indent=2)

    return JsonResponse(resp, safe=False)


@csrf_exempt
def direction_hierarchy_recluster(request):
    data = json.loads(request.body)
    experiment_names = data['experiment_names']
    direction_tree = data['direction_tree']
    code_indices_clustering = data['code_indices_clustering']
    code_indices_coherence = data['code_indices_coherence']
    pairwise_metric = data['pairwise_metric']
    clustering_method = data['clustering_method']

    # Read features
    if not use_latent:
        walk_features, cumulative_dfm_size = readers.read_walk_features(experiment_names, served_data_root=SERVED_DATA_ROOT)
        original_features = readers.read_original_features(experiment_names, served_data_root=SERVED_DATA_ROOT)
    else:
        walk_features, cumulative_dfm_size = readers.read_walk_latent_codes(experiment_names, served_data_root=SERVED_DATA_ROOT)
        original_features = readers.read_original_latent_codes(experiment_names, served_data_root=SERVED_DATA_ROOT)

    w_mean = torch.mean(original_features, dim=0)

    # Post-process features to obtain features for clustering / coherence
    clustering_features = features.postprocess_clustering_features(walk_features,
                                                                   original_features,
                                                                   code_selection=code_indices_clustering,
                                                                   mode='end',
                                                                   prior_subset=True,
                                                                   centering=clustering_centering,
                                                                   w_mean=w_mean)
    clustering_features = features.compute_mean_features_pairwise_distances(clustering_features,
                                                                            dist_metric=pairwise_metric)
    mag_features = features.postprocess_magnitude_features(walk_features,
                                                           original_features,
                                                           code_selection=code_indices_coherence,
                                                           direction_selection=None)
    var_features = features.postprocess_variance_features(walk_features,
                                                          original_features,
                                                          mode=var_feature_derivation,
                                                          code_selection=code_indices_coherence,
                                                          direction_selection=None,
                                                          centering=var_feature_centering,
                                                          prior_subset=var_feature_prior_subset,
                                                          w_mean=w_mean,
                                                          normalize=var_normalize)


    # Clustering
    z = linkage(clustering_features.numpy(), clustering_method, optimal_ordering=True)

    # Tree-building => convert linkage into tree. (total of 2 traversals)
    tree = trees.create_tree_from_linkage(z, cumulative_dfm_size, experiment_names)
    trees.count(tree)  # Count quantities, depth, and leaves

    # Coherence computation
    node_magnitudes, node_variances = trees.compute_direction_coherence(tree,
                                                                        mag_features,
                                                                        var_features,
                                                                        collect_only_leaf=False)

    # Truncate
    rawtree = copy.deepcopy(tree)
    node_size_min, node_variance_min = node_size_minimum_control, 0.5
    node_size_min = walk_features.shape[1] / node_size_divisor_global
    # trees.truncate_tree(tree, node_size_min, mode='size')
    trees.tree_trimming(tree, treemin=node_size_minimum_control, mode='size')
    # trees.tree_trimming(tree, 0.05, mode='variance')

    # Gather leaf nodes' variance and magnitudes
    node_magnitudes, node_variances = trees.collect_visleaf_coherence(tree, target_depth=7)

    # Mean / Std for coherence color encoding domain setting.
    # mags_mean, stds_mean, mags_std, stds_std = _compute_mean_std(mag_features, var_features)
    mags_mean, stds_mean, mags_std, stds_std = _compute_mean_std(node_magnitudes, node_variances)

    # Compute weave score
    weave_scores = trees.compute_leaf_weave(direction_tree, tree)  # Right direction
    trees.append_weave_tree(tree, weave_scores)  # Right direction

    resp = json.dumps({
        'tree': tree,
        'rawtree': rawtree,
        'timestamp': time.time(),
        'avgMagnitude': float(mags_mean),
        'avgStd': float(stds_mean),
        'magsStd': float(mags_std),
        'stdsStd': float(stds_std),
        'weaveScores': weave_scores
    }, indent=2)

    return JsonResponse(resp, safe=False)

def compute_contribution(direction_indices, code_indices, subset_var, magnitude_norms, var_mean, mag_mean):
    # Compute magnitude and variance contribution
    contributions = {}
    var_contributions = []
    for di, diidx in enumerate(direction_indices):
        tmp_var = torch.cat((subset_var[:, :di], subset_var[:, di + 1:]), dim=1)
        var_without = torch.mean(torch.mean(torch.std(tmp_var, dim=1, keepdim=True), dim=-1)).item()
        var_contribution = var_without - var_mean
        var_contributions.append(var_contribution)

    fidx = -1
    for ci, ciidx in enumerate(code_indices):
        for di, diidx in enumerate(direction_indices):
            fidx += 1
            mag_without = torch.mean(magnitude_norms[fidx], dim=0).item()
            # mag_without = torch.mean(torch.cat((magnitude_norms[:fidx], magnitude_norms[fidx + 1:]), dim=0)).item()
            # mag_contribution = mag_without - mag_mean
            mag_contribution = mag_without
            contributions[(ciidx, diidx)] = {
                'mag_contribution': mag_contribution,
                # 'var_contribution': var_contributions[di]
            }
            # print(ciidx, diidx, mag_contribution, var_contributions[di])
    return contributions


def compute_self_magnitude(direction_indices, code_indices, magnitude_norms):
    contributions = {}
    fidx = -1
    for ci, ciidx in enumerate(code_indices):
        for di, diidx in enumerate(direction_indices):
            fidx += 1
            mag_without = torch.mean(magnitude_norms[fidx], dim=0).item()
            mag_contribution = mag_without
            contributions[(ciidx, diidx)] = {
                'mag_contribution': mag_contribution
            }
    return contributions


@csrf_exempt
def direction_hierarchy_selection(request):
    """
    When a selection in the direction hierarchy is made, this code updates code tree's coherence based on the
    selected direction indices

    Args:
        request:

    Returns:

    """
    data = json.loads(request.body)
    experiment_names = data['experiment_names']
    code_tree = data['code_tree']
    unsorted_code_indices = data['code_indices']
    unsorted_direction_indices = data['direction_indices']
    code_indices = sorted(data['code_indices'])  # For computing sub-selection mag and var.
    direction_indices = sorted(data['direction_indices'])
    print("============= Direction Hierarchy Selection Event =============")
    print(f"Code selection made: {code_indices[:10]}...")
    print(f"Direction selection made: {direction_indices[:10]}...")
    print(f"Direction selection made Unsorted: {unsorted_direction_indices[:10]}...")

    # For de-selection error handling. This is not a permanent solution.
    if len(direction_indices) == 0:
        direction_indices = [_ for _ in range(1024)]

    # Read features
    if not use_latent:
        walk_features, cumulative_dfm_size = readers.read_walk_features(experiment_names, served_data_root=SERVED_DATA_ROOT)
        original_features = readers.read_original_features(experiment_names, served_data_root=SERVED_DATA_ROOT)
    else:
        walk_features, cumulative_dfm_size = readers.read_walk_latent_codes(experiment_names, served_data_root=SERVED_DATA_ROOT)
        original_features = readers.read_original_latent_codes(experiment_names, served_data_root=SERVED_DATA_ROOT)

    w_mean = torch.mean(original_features, dim=0)

    """ Coherence """
    mag_features = features.postprocess_magnitude_features(walk_features,
                                                           original_features,
                                                           code_selection=None,
                                                           direction_selection=direction_indices)

    all_mag_features = features.postprocess_magnitude_features(walk_features,
                                                               original_features,
                                                               code_selection=None,
                                                               direction_selection=None)

    var_features = features.postprocess_variance_features(walk_features,
                                                          original_features,
                                                          mode=var_feature_derivation,
                                                          code_selection=None,
                                                          direction_selection=direction_indices,
                                                          centering=var_feature_centering,
                                                          prior_subset=var_feature_prior_subset,
                                                          w_mean=w_mean,
                                                          normalize=var_normalize)

    trees.compute_coherence_code(code_tree, mag_features, var_features, collect_only_leaf=False,
                                 code_indices=code_indices, direction_indices=direction_indices)

    # Compute mag and var for the selected code and directions
    if len(code_indices) == 0:
        subset_mag, subset_var = mag_features, var_features
    else:
        subset_mag, subset_var = mag_features[code_indices,:], var_features[code_indices,:]

    magnitude_norms = torch.norm(subset_mag, dim=-1).flatten(0, 1)

    mag = torch.mean(magnitude_norms).item()
    var = torch.mean(torch.mean(torch.std(subset_var, dim=1, keepdim=True), dim=-1)).item()

    # To compute the range of magnitudes
    all_mag_norms = torch.norm(all_mag_features, dim=-1)
    min_mag, max_mag = torch.min(all_mag_norms), torch.max(all_mag_norms)
    contributions = compute_self_magnitude(direction_indices, code_indices, magnitude_norms)
    # contributions = compute_self_magnitude(unsorted_direction_indices, unsorted_code_indices, magnitude_norms)

    # Tree counting and trimming for ready-visualization data. TODO: These processes can be consolidated later for less tree traversal.
    trees.remove_leaves(code_tree)
    trees.count(code_tree)
    trees.tree_trimming(code_tree, node_size_minimum_control, mode='size')

    # newcontrib = []
    # for k, v in contributions.items():
    #     newcontrib.append({
    #         'code': k[0],
    #         'direction': k[1],
    #         'mag_contribution': v['mag_contribution'],
    #         'var_contribution': v['mag_contribution']
    #     })

    # Create a map for unsorted_code_indices to their original indices
    unsorted_code_index_map = {code: idx for idx, code in enumerate(unsorted_code_indices)}

    # Populate newcontrib
    newcontrib = []
    for k, v in contributions.items():
        newcontrib.append({
            'code': k[0],
            'direction': k[1],
            'mag_contribution': v['mag_contribution'],
            'var_contribution': v['mag_contribution'],
        })

    # Sort newcontrib by the order in unsorted_code_indices
    newcontrib.sort(key=lambda x: unsorted_code_index_map.get(x['code'], float('inf')))

    for v in newcontrib:
        print(v['direction'])

    if np.isnan(mag):
        mag = 0

    resp = json.dumps({
        'codeTree': code_tree,
        'timestamp': time.time(),
        'magnitude': mag,
        'variance': var,
        'contributions': newcontrib,
        'magmin': min_mag.item(),
        'magmax': max_mag.item()
    }, indent=2)

    return JsonResponse(resp, safe=False)


@csrf_exempt
def code_hierarchy_initialization(request):
    """
    Initializing code hierarchy.
    Data needed
        - 'experiment_names': array of experiment names.
        - 'clustering_method': a string indicating a clustering method.

    Previously, I was passing on the following fields as parameters, which I don't always need for initialization.
        - experiment_names
        - tree (previous tree)
        - directions
        - metric
        - pairwise_metric
        - clustering_method
        - truncated_tree

    Args:
        request:

    Returns:

    """
    data = json.loads(request.body)
    experiment_names = data['experiment_names']
    previous_tree = data['tree']
    directions = data['directions']
    # metric = data['metric']
    pairwise_metric = data['pairwise_metric']
    clustering_method = data['clustering_method']
    truncated_tree = data['truncated_tree']

    # Read features
    if not use_latent:
        walk_features, cumulative_dfm_size = readers.read_walk_features(experiment_names, served_data_root=SERVED_DATA_ROOT)
        original_features = readers.read_original_features(experiment_names, served_data_root=SERVED_DATA_ROOT)
    else:
        walk_features, cumulative_dfm_size = readers.read_walk_latent_codes(experiment_names, served_data_root=SERVED_DATA_ROOT)
        original_features = readers.read_original_latent_codes(experiment_names, served_data_root=SERVED_DATA_ROOT)

    w_mean = torch.mean(original_features, dim=0)

    """ Clustering """
    # Post-process features to obtain features for clustering / coherence
    clustering_features = original_features
    z = linkage(clustering_features.numpy(), clustering_method, optimal_ordering=True)

    # Tree-building => convert linkage into tree. (total of 2 traversals)
    tree = trees.create_tree_from_linkage(z, [z.shape[0] + 1], experiment_names)
    trees.count(tree)  # Count quantities, depth, and leaves

    """ Coherence """
    mag_features = features.postprocess_magnitude_features(walk_features,
                                                           original_features,
                                                           code_selection=None,
                                                           direction_selection=None)
    var_features = features.postprocess_variance_features(walk_features,
                                                          original_features,
                                                          mode=var_feature_derivation,
                                                          code_selection=None,
                                                          direction_selection=None,
                                                          centering=var_feature_centering,
                                                          prior_subset=var_feature_prior_subset,
                                                          w_mean=w_mean,
                                                          normalize=var_normalize)

    node_magnitudes, node_variances = trees.compute_coherence_code(tree, mag_features, var_features,
                                                                   collect_only_leaf=False)

    # Truncate
    rawtree = copy.deepcopy(tree)
    node_size_min, node_variance_min = node_size_minimum_control, 0.5
    node_size_min = walk_features.shape[0] / node_size_divisor_global
    # trees.truncate_tree(tree, node_size_min, mode='size')
    trees.tree_trimming(tree, node_size_minimum_control, mode='size')
    # trees.tree_trimming(tree, 0.05, mode='variance')

    # Gather leaf nodes' variance and magnitudes
    node_magnitudes, node_variances = trees.collect_visleaf_coherence(tree, target_depth=7)

    # Mean / Std for coherence color encoding domain setting.
    # mags_mean, stds_mean, mags_std, stds_std = _compute_mean_std(mag_features, var_features)
    mags_mean, stds_mean, mags_std, stds_std = _compute_mean_std(node_magnitudes, node_variances)

    resp = json.dumps({
        'tree': tree,
        'rawtree': rawtree,
        'timestamp': time.time(),
        'avgMagnitude': float(mags_mean),
        'avgStd': float(stds_mean)
    }, indent=2)

    return JsonResponse(resp, safe=False)


@csrf_exempt
def code_hierarchy_selection(request):
    """
    When a selection is made, there is no need to perform clustering. Here, the code updates direction tree's coherence
    based on the selected code indices.

    Args:
        request:

    Returns:

    """
    data = json.loads(request.body)
    experiment_names = data['experiment_names']
    direction_tree = data['direction_tree']
    unsorted_code_indices = data['code_indices']
    unsorted_direction_indices = data['direction_indices']
    code_indices = sorted(data['code_indices'])
    direction_indices = sorted(data['direction_indices'])
    print("============= Code Hierarchy Selection Event =============")
    print(f"Code selection made: {code_indices[:10]}...")
    print(f"Direction selection made: {direction_indices[:10]}...")

    if len(code_indices) == 0:
        code_indices = [_ for _ in range(1024)]

    # Read features
    if not use_latent:
        walk_features, cumulative_dfm_size = readers.read_walk_features(experiment_names, served_data_root=SERVED_DATA_ROOT)
        original_features = readers.read_original_features(experiment_names, served_data_root=SERVED_DATA_ROOT)
    else:
        walk_features, cumulative_dfm_size = readers.read_walk_latent_codes(experiment_names, served_data_root=SERVED_DATA_ROOT)
        original_features = readers.read_original_latent_codes(experiment_names, served_data_root=SERVED_DATA_ROOT)


    w_mean = torch.mean(original_features, dim=0)


    """ ============================================= Original """
    mag_features = features.postprocess_magnitude_features(walk_features,
                                                           original_features,
                                                           code_selection=code_indices,
                                                           direction_selection=None)

    all_mag_features = features.postprocess_magnitude_features(walk_features,
                                                               original_features,
                                                               code_selection=None,
                                                               direction_selection=None)

    var_features = features.postprocess_variance_features(walk_features,
                                                          original_features,
                                                          mode=var_feature_derivation,
                                                          code_selection=code_indices,
                                                          direction_selection=None,
                                                          centering=var_feature_centering,
                                                          prior_subset=var_feature_prior_subset,
                                                          w_mean=w_mean,
                                                          normalize=var_normalize)

    # Coherence computation
    trees.compute_direction_coherence(direction_tree, mag_features, var_features, collect_only_leaf=False,
                                      code_indices=code_indices, direction_indices=direction_indices)

    # Compute mag and var for the selected code and directions
    subset_mag, subset_var = mag_features, var_features
    if len(direction_indices) == 0:
        subset_mag, subset_var = mag_features, var_features
    else:
        subset_mag, subset_var = mag_features[:, direction_indices], var_features[:, direction_indices]

    magnitude_norms = torch.norm(subset_mag, dim=-1).flatten(0, 1)

    mag = torch.mean(magnitude_norms).item()
    var = torch.mean(torch.mean(torch.std(subset_var, dim=1, keepdim=True), dim=-1)).item()

    # To compute the range of magnitudes
    all_mag_norms = torch.norm(all_mag_features, dim=-1)
    min_mag, max_mag = torch.min(all_mag_norms), torch.max(all_mag_norms)
    contributions = compute_self_magnitude(direction_indices, code_indices, magnitude_norms)
    # contributions = compute_self_magnitude(unsorted_direction_indices, unsorted_code_indices, magnitude_norms)

    # Amend "leaves" fields plus more
    trees.remove_leaves(direction_tree)
    trees.count(direction_tree)
    trees.tree_trimming(direction_tree, node_size_minimum_control, mode='size')
    node_magnitudes, node_variances = trees.collect_visleaf_coherence(direction_tree, target_depth=7)

    # newcontrib = []
    # for k, v in contributions.items():
    #     newcontrib.append({
    #         'code': k[0],
    #         'direction': k[1],
    #         'mag_contribution': v['mag_contribution'],
    #         'var_contribution': v['mag_contribution'],
    #     })

    # Create a map for unsorted_code_indices to their original indices
    unsorted_code_index_map = {code: idx for idx, code in enumerate(unsorted_code_indices)}

    # Populate newcontrib
    newcontrib = []
    for k, v in contributions.items():
        newcontrib.append({
            'code': k[0],
            'direction': k[1],
            'mag_contribution': v['mag_contribution'],
            'var_contribution': v['mag_contribution'],
        })

    # Sort newcontrib by the order in unsorted_code_indices
    newcontrib.sort(key=lambda x: unsorted_code_index_map.get(x['code'], float('inf')))

    resp = json.dumps({
        'directionTree': direction_tree,
        'timestamp': time.time(),
        'magnitude': mag,
        'variance': var,
        'contributions': newcontrib,
        'magmin': min_mag.item(),
        'magmax': max_mag.item()
    }, indent=2)

    return JsonResponse(resp, safe=False)
