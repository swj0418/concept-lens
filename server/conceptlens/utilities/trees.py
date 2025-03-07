import math
import time

import torch
import numpy as np


def _compute_subset_magnitude(magnitude_features):
    return torch.mean(torch.norm(magnitude_features, dim=-1), dim=(0, 1), dtype=torch.bfloat16).item()


def _compute_subset_variance(variance_features):
    x = torch.std(variance_features, dim=-2, keepdim=True)  # [code, 1, dimension] std across dimensions
    x = torch.mean(x, dim=-1, keepdim=True)
    var = torch.mean(x).item()
    if math.isnan(var):
        var = 0

    return var

def _build_tree(tree, intermediate_nodes, cumulative_dfm_size, experiment_names):
    """

    Args:
        tree:
        intermediate_nodes: name of nodes that are not leaf nodes.

    Returns:

    """
    parent_name = tree['name']

    # No children for this node
    if parent_name not in intermediate_nodes.keys():
        return

    # Look-ahead building
    for child_name in intermediate_nodes[parent_name]:
        child_name = int(child_name)

        if child_name not in intermediate_nodes.keys():
            # Determine treeID - experimentID
            treeId = _search_tree_idx(child_name, cumulative_dfm_size)
            nodeIdx = child_name  # nodeIdx of children nodes within own experiment.
            if treeId > 0:
                nodeIdx = int(nodeIdx - cumulative_dfm_size[treeId - 1])

            node = {
                'name': child_name,  # Spans all experiment set.
                'leaf': True,
                'visleaf': False,
                'parent': int(parent_name),
                'treeID': treeId,
                'expName': experiment_names[treeId],
                'flatIdx': nodeIdx,  # Spans only within experiment.
                'children': [],
                'leaves': [],
                'leaf_indices': [],
                'depth': 0,
                'quantity': 0,
                'trip': 0,
                'var': 0,
                'magnitude': 0,
                'contributions': [],
                'mag_contribution': 0,
                'var_contribution': 0
            }
        else:  # Intermediate nodes
            node = {
                'name': int(child_name),
                'leaf': False,
                'visleaf': False,
                'parent': int(parent_name),
                'treeID': None,
                'expName': None,
                'flatIdx': None,
                'children': [],
                'leaves': [],
                'leaf_indices': [],
                'depth': 0,
                'quantity': 0,
                'trip': 0,
                'var': 0,
                'magnitude': 0,
                'contributions': [],
                'mag_contribution': 0,
                'var_contribution': 0
            }

        # Add to children field
        tree['children'].append(node)

        # Continue recursion
        _build_tree(node, intermediate_nodes, cumulative_dfm_size, experiment_names)

    return tree


def _search_tree_idx(name, sequence):
    """
    From a sequence of sizes - [n1, n1 + n2, n1 + n2 + n3], find where the "name" sits and return the range that
    it is contained in.

    The sequence is a size, so a size 100 will contain names in [0, 99]. If the next size is 250, then,
    [100, 249].
    Args:
        name:
        sequence:

    Returns:

    """
    for idx, size in enumerate(sequence):
        leftover = name - size
        if leftover < 0:  # Should not include 0.
            return idx


def create_tree_from_linkage(z, cumulative_dfm_size, experiment_names=None):
    """
    Creates tree data structure from linkage matrix.

    Args:
        z: linkage matrix from scpy hierarchy library
        cumulative_dfm_size: cumulative size of dfms in the linkage matrix.
        experiment_names:

    Returns:

    """
    num_rows, num_cols = z.shape

    # Identify intermediate nodes
    intermediate_nodes = {}
    i = 0
    for idx, row in enumerate(z):
        intermediate_nodes[int(idx + 1 + num_rows)] = [row[0], row[1]]
        i += 1

    # Tree initiation
    tree = {
        'name': int(i + num_rows),
        'timestamp': time.time(),
        'origin': int(num_rows),
        'leaf': False,
        'visleaf': False,
        'parent': None,
        'children': [],
        'leaves': [],
        'leaf_indices': [],
        'depth': 0,
        'quantity': 0,
        'trip': 0,
        'var': 0,
        'magnitude': 0,
        'contributions': [],
        'mag_contribution': 0,
        'var_contribution': 0
    }

    # Fill-in and create tree
    _build_tree(tree, intermediate_nodes, cumulative_dfm_size, experiment_names)
    return tree


def count(node, depth=0):
    """
    Perform a post-order traversal of a tree and count the number of nodes at each depth,
    appending a 'quantity' field to each node. Also, append the 'depth' field. Also append the 'leaves' field.

    Args:
        node: The root node of the tree.
        depth: The depth of the current node in the tree.

    Returns:
        None
    """

    def _break_circular(node):
        node['leaves'] = []
        return node

    # Leaf condition
    if node['leaf']:
        node['quantity'] = 1  # Initialize quantity to 1 for the current node - leaf node.
        node['leaves'] = [node]
        node['leaf_indices'] = [node['name']]
        node['depth'] = depth

    # Post-order to reduce computation.
    for child in node['children']:
        count(child, depth + 1)

    for child in node['children']:
        node['quantity'] += child['quantity']  # Update quantity based on child nodes

        child_nodes = []
        for l in child['leaves']:
            n = _break_circular(l)
            child_nodes.append(n)
        node['leaves'].extend(child_nodes)
        node['leaf_indices'].extend(child['leaf_indices'])
        node['depth'] = depth


def count_leaves(node):
    """
    Similar to count() and it performs a post-order traversal of a tree and amends the "leaves" field only. It removes
    previously generated "leaves" field.

    Args:
        node:

    Returns:

    """

    def _break_circular(node):
        node['leaves'] = []
        return node

    # Leaf condition
    if node['leaf'] == 0:
        node['leaves'] = [node]
        node['leaf_indices'] = [node['name']]

    # Post-order to reduce computation.
    for child in node['children']:
        count(child)

    for child in node['children']:
        child_nodes = []
        for l in child['leaves']:
            n = _break_circular(l)
            child_nodes.append(n)
        node['leaves'].extend(child_nodes)
        node['leaf_indices'].extend(child['leaf_indices'])


def remove_leaves(node):
    """
    Empties all "leaves" field for intermediate nodes.

    Args:
        node:

    Returns:

    """
    for child in node['children']:
        remove_leaves(child)

    node['leaves'] = []
    node['leaf_indices'] = []
    node['quantity'] = 0
    node['depth'] = 0


def tree_trimming(tree, treemin, mode):
    """
    New method for consolidating tree.

    At each step, if a subtree size or variance is below "treemin,"
    avoid further splitting and mark it as a leaf.

    Args:
        tree: tree data structure
        treemin: minimum threshold for splitting
        mode: criterion for splitting ('size' or 'variance')
    """

    def _split(quantity):
        # Split condition based on the size or variance of the node
        return quantity >= treemin

    assert mode in ['size', 'variance']
    if not tree:
        return

    children = tree.get('children', [])
    for child in children:
        leaf_count = len(child.get('leaf_indices', []))

        # Decide the quantity based on mode
        quantity = leaf_count if mode == 'size' else child.get('var', 0)

        # Check if current node should be marked as a leaf to avoid small splits
        if not _split(quantity):
            tree['leaf'] = True
            tree['visleaf'] = True
            return  # No further processing needed if we set it as a leaf

        # Recursively apply trimming to child nodes
        if not child.get('leaf', False):
            tree_trimming(child, treemin, mode)
        else:
            # Directly mark single-node trees as leaves to avoid 1-sized nodes
            tree['visleaf'] = True
            tree['leaf'] = True
            return


def truncate_tree(tree, treemin, mode):
    """

    Args:
        tree:
        treemin:
        mode: metric to use for determining consolidation.

    Returns:

    """
    assert mode in ['size', 'variance']
    if not tree:
        return

    def _split_condition(treemin, quantity):
        # Quantity could be either tree size or tree variance. Returns True if the tree **should** be split.
        # Exceptions are when tree suddenly splits into a splint - a single node leaf. This should be prevented.
        if quantity < 4:  # This really can be 2 instead of 4.
            return False
        if mode == 'size':
            return treemin <= quantity
        elif mode == 'variance':
            return treemin <= quantity

    children = tree['children']
    for child in children:
        leaf = child['leaf']
        if mode == 'size':
            quantity = child['quantity']
        elif mode == 'variance':
            quantity = child['var']

        if leaf:
            child['visleaf'] = True  # Leaf node mark just for visualization.
        else:
            if _split_condition(treemin, quantity):
                truncate_tree(child, treemin, mode)
            else:
                child['visleaf'] = True
                child['leaf'] = True  # Compatible with old visualization code.


# =================================== "Compute" Things ===================================
def _get_off_diagonal_elements(M):
    return M[~torch.eye(*M.shape, dtype=torch.bool)].view(M.shape[0], M.shape[1] - 1)


def compute_direction_coherence(tree,
                                mag_features,
                                var_features,
                                collect_only_leaf,
                                code_indices=None,
                                direction_indices=None
                                ):
    all_magnitude, all_variance = [], []

    def _compute_coherence_tree(node):
        if not node:
            return

        children = node['children']
        if len(children) > 0:
            for child in children:
                _compute_coherence_tree(child)

        leaf_indices = sorted(node['leaf_indices'])
        # Mag subset: [code, direction]
        # Var subset: [code, direction, direction]
        mag_subset, var_subset = mag_features[:, leaf_indices], var_features[:, leaf_indices]

        # magnitude - bfloat (range is more important)
        magnitude = torch.mean(torch.norm(mag_subset, dim=-1), dim=(0, 1), dtype=torch.bfloat16).item()
        x = torch.std(var_subset, dim=-2, keepdim=True)  # [code, 1, dimension] std across dimensions
        x = torch.mean(x, dim=-1, keepdim=True)
        var = torch.mean(x).item()
        if math.isnan(var):
            var = 0

        all_magnitude.append(magnitude)
        all_variance.append(var)

        # Collect each node's magnitude and variance.
        if collect_only_leaf:
            if node['leaf']:
                node['magnitude'] = magnitude
                node['var'] = var
        else:
            node['magnitude'] = magnitude
            node['var'] = var

    # In-place compute
    _compute_coherence_tree(tree)
    return all_magnitude, all_variance


def compute_coherence_code(tree,
                           mag_features,
                           var_features,
                           collect_only_leaf,
                           code_indices=None,
                           direction_indices=None
                           ):
    all_magnitude, all_variance = [], []
    def _compute_coherence_tree(node):
        if not node:
            return
        children = node['children']
        if len(children) > 0:
            for child in children:
                _compute_coherence_tree(child)

        leaf_indices = sorted(node['leaf_indices'])  # Code indices
        mag_subset, var_subset = mag_features[leaf_indices], var_features[leaf_indices]

        magnitude = _compute_subset_magnitude(mag_subset)
        var = _compute_subset_variance(var_subset)
        all_magnitude.append(magnitude)
        all_variance.append(var)

        if np.isnan(magnitude):
            magnitude = 0

        # Collect each node's magnitude and variance.
        if collect_only_leaf:
            if node['leaf']:
                node['magnitude'] = magnitude
                node['var'] = var
        else:
            node['magnitude'] = magnitude
            node['var'] = var

    # In-place compute
    _compute_coherence_tree(tree)
    return all_magnitude, all_variance


def collect_visleaf_coherence(tree, target_depth):
    """
    Collect magnitude and coherence at target depth (or leaf)
    Args:
        tree:
        target_depth:

    Returns:

    """
    assert type(target_depth) == int
    all_magnitude, all_variance = [], []

    def _collect_coherence_tree(node):
        if not node:
            return

        if len(node['children']) > 0:
            for child in node['children']:
                _collect_coherence_tree(child)

        # if node['depth'] == target_depth:
        if node['visleaf']:
            all_magnitude.append(node['magnitude'])
            all_variance.append(node['var'])

    # In-place compute
    _collect_coherence_tree(tree)
    return all_magnitude, all_variance


def weave_score_pairwise_distance(membership, membership_pairs):
    """
    After determining how many disparate pairs there are, compute how far each of those disparate pairs are to
    account for the distance.

    For example,
    (0, 0), (0, 0), (0, 1), (0, 1), (0, 2), (0, 2)

    Where 0 and 1 is close in the original leaf list, while 0 and 2 is far in the original leaf list.

        1. The distance in the leaf list need to be computed
            This requires knows the previous membership. This information is already in the pairs

    TODO: At the moment, the metric is not normalized.

    Returns:

    """
    # Subtract membership ID of all disparate pairs
    distances = [abs(a - b) for (a, b) in membership_pairs if a != b]
    return sum(distances) / (len(membership) + .001)


def compute_leaf_weave(t0, t1):
    """
    Computes

    Args:
        t0: counting weave from
        t1: counting weave to

    Returns:

    """
    t0_visleaves, t1_visleaves = [], []
    get_tree_visleaves(t0, t0_visleaves)
    get_tree_visleaves(t1, t1_visleaves)

    # Compute how much change has been made in the two. This is done by building key-value pairs where
    # key == original membership / value == direction_idx
    t0_membership_dict, t0_membership_rev, t1_membership_dict, t1_membership_rev = {}, {}, {}, {}
    for mem_id, direction_indices in enumerate(t0_visleaves):
        for direction_idx in direction_indices:
            t0_membership_dict[direction_idx] = mem_id
    for mem_id, direction_indices in enumerate(t1_visleaves):
        for direction_idx in direction_indices:
            t1_membership_dict[direction_idx] = mem_id

    # Count how many membership changes
    weave_scores = []
    for mem_id, direction_indices in enumerate(t1_visleaves):
        # Build set
        # print(direction_indices)
        # print(sorted(t0_membership_dict.keys()))
        membership = [t0_membership_dict[di] for di in direction_indices]
        membership_pairs = [(a, b) for idx, a in enumerate(membership) for b in membership[idx + 1:]]

        weave_score = weave_score_pairwise_distance(membership, membership_pairs)

        weave_scores.append(weave_score)

    return weave_scores


def append_weave_tree(tree, weave_scores):
    if tree['leaf']:
        tree['weaveScore'] = weave_scores.pop(0)
    else:
        for c in tree['children']:
            append_weave_tree(c, weave_scores)


def get_tree_visleaves(tree, container):
    """
    Retrieve a list of leaves (clusters) that were defined for visualization purposes
    Args:
        tree:

    Returns:
    """

    if tree['leaf']:
        if len(tree['leaves']) == 0:
            container.append([tree['flatIdx']])
        else:
            container.append([t['flatIdx'] for t in tree['leaves']])
    else:
        for child in tree['children']:
            get_tree_visleaves(child, container)


def insert_contribution_data(tree, contribution):
    if len(tree['children']) == 0:  # True leaf nodes
    # if tree['leaf']:  # True leaf nodes
        to_del = []
        for k, v in contribution.items():
            if k[0] == tree['name']:
                # tree['contributions'].append({k: v})
                tree['contributions'].extend([{str(k): v}])
                to_del.append(k)

        # for k in to_del:
        #     del contribution[k]
        return

    else:
        for child in tree['children']:
            insert_contribution_data(child, contribution)


def verify_contribution_data(tree):
    if len(tree['children']) == 0:  # True leaf nodes
        if len(tree['contributions']) != 0:
            print(tree['contributions'])
        return

    else:
        for child in tree['children']:
           verify_contribution_data(child)