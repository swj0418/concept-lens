import scipy
import torch


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


def postprocess_features(features, code_selection, mode, normalize=False):
    """

    Post process features of form [code, method, direction, step, features]
    to a form                     [code, method * direction, features]

    Args:
        features: tensor of shape and order [code, method, direction, step, features]


    Returns:
    """
    assert len(features.shape) == 5 or len(features.shape) == 4
    assert type(code_selection) == list or code_selection.__eq__(None)
    assert mode in ['start', 'end', 'diff']

    if len(features.shape) == 4:  # For backward compatibility.
        features = features[:, None, :]

    if type(code_selection) == list and len(code_selection) > 0:
        if mode == 'start':
            features = features[code_selection, :, :, 0]
        elif mode == 'end':
            features = features[code_selection, :, :, 1]
        elif mode == 'diff':
            features = features[code_selection, :, :, 1] - features[code_selection, :, :, 0]

        if normalize:
            features = features / torch.norm(features, dim=-1, keepdim=True)

        features = torch.flatten(features, 1, 2)
        # features = torch.flatten(features, 0, 1)
        # features = torch.unsqueeze(features, dim=0)  # [1, method * direction, features]
    else:
        if mode == 'start':
            features = features[:, :, :, 0]
        elif mode == 'end':
            features = features[:, :, :, 1]
        elif mode == 'diff':
            features = features[:, :, :, 1] - features[:, :, :, 0]

        if normalize:
            features = features / torch.norm(features, dim=-1, keepdim=True)

        features = torch.flatten(features, 1, 2)  # [code, method * direction, features]

    return features
