import os
import json

import torch
import scipy


"""
Helper Functions

"""


def load_setting(data_root):
    config = None
    with open(os.path.join(data_root, 'setting.json')) as file:
        config = json.load(file)

    experiment_name = config['experiment_name']
    n_code = config['n_code']
    n_methods = config['n_method']
    n_direction_per_method = config['n_direction']

    return experiment_name, n_code, n_methods, n_direction_per_method


def load_feature_data(data_root):
    tensor_root = os.path.join(data_root, 'tensors')

    # [Code, Method, Direction, Step, Feature]
    features = torch.load(os.path.join(tensor_root, 'features.pt'))
    return features


def load_distance_data(data_root):
    tensor_root = os.path.join(data_root, 'tensors')

    # [Code, Method, Direction, Step, Feature]
    features = torch.load(os.path.join(tensor_root, 'features.pt'))


def mute_diagonal(distance_matrix, replacement):
    """
    Modifies diagonals to something smaller - a max value of non-diagonal elements.

    """
    assert replacement in ['min', 'mean', 'max']

    # non-diagonal entries
    nondiag = distance_matrix.masked_select(~torch.eye(distance_matrix.shape[0], dtype=bool))

    if replacement == 'min':
        distance_matrix.fill_diagonal_(torch.min(nondiag))
    elif replacement == 'mean':
        distance_matrix.fill_diagonal_(torch.mean(nondiag))
    elif replacement == 'max':
        distance_matrix.fill_diagonal_(torch.max(nondiag))

    return distance_matrix, nondiag


"""
Data processing helper functions

"""


def compute_feature_similarity(features, metric, dissimilarity):
    """
    1. Compute feature similarity across codes.
    2. Average as provided in the parameter.

    Computing mean first may reduce the variability of feature difference vectors that is a crucial infomation in
    determining if any given two directions are similar.

    Args:
        features: [code, method * direction, feature]
        dissimilarity: True for distance, False for similarity

    Returns:
        feature similarities: [code, method * direction, method * direction]
    """
    assert len(features.shape) == 3  # [code, direction, feature]
    assert metric in ['cosine', 'euclidean']

    similarity = torch.zeros(size=(features.shape[0], features.shape[1], features.shape[1]))
    for idx, code_matrix in enumerate(features):  # Per code
        # Scipy function - Cosine distance
        code_similarity_matrix = scipy.spatial.distance.cdist(code_matrix, code_matrix, metric)
        if not dissimilarity:
            code_similarity_matrix = 1 - code_similarity_matrix
        code_similarity_matrix = torch.from_numpy(code_similarity_matrix)

        similarity[idx] = code_similarity_matrix

    return similarity  # [code, direction, direction]


def postprocess_features(features, code_selection, mode):
    """

    Post process features of form [code, method, direction, step, features]
    to a form                     [code, method * direction, features]

    Args:
        features: tensor of shape and order [code, method, direction, step, features]
        code_selection: Index to select along code dimension. [None] for no selection, [<int>] for selection
        mode: Whether to take a difference [diff], the starting feature [start], or the ending feature [end].

    Returns:
    """
    assert len(features.shape) == 5
    assert type(code_selection) == int or code_selection.__eq__(None)
    assert mode in ['start', 'end', 'diff']

    if type(code_selection) == int:
        if mode == 'start':
            features = features[code_selection, :, :, 0]
        elif mode == 'end':
            features = features[code_selection, :, :, 1]
        elif mode == 'diff':
            features = features[code_selection, :, :, 1] - features[code_selection, :, :, 0]
            features = features / torch.norm(features, dim=-1, keepdim=True)

        features = torch.flatten(features, 0, 1)
        features = torch.unsqueeze(features, dim=0)  # [1, method * direction, features]
    elif code_selection.__eq__(None):
        if mode == 'start':
            features = features[:, :, :, 0]
        elif mode == 'end':
            features = features[:, :, :, 1]
        elif mode == 'diff':
            features = features[:, :, :, 1] - features[:, :, :, 0]
            features = features / torch.norm(features, dim=-1, keepdim=True)

        features = torch.flatten(features, 1, 2)  # [code, method * direction, features]
    else:
        raise Exception("Post processing features failed")

    return features
