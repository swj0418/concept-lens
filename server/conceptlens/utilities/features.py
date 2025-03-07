import scipy
import sklearn.metrics
import torch


def _subsetting(walk_features, original_features, direction_selection, code_selection):
    # Subset feature matrices
    if direction_selection:
        walk_features = walk_features[:, direction_selection]

    if code_selection:
        original_features = original_features[code_selection]
        walk_features = walk_features[code_selection]

    return walk_features, original_features


def _center(wf, of, c_mode, wm):
    if c_mode == 'individual_code_mean':  # Centering individual code.
        per_code_mean = torch.mean(wf, dim=1)
        centering_mean = per_code_mean.unsqueeze(1).repeat((1, wf.shape[1], 1))
        f = wf - centering_mean
    elif c_mode == 'all_code_mean':  # Centering on whole dataset.
        centering_mean = torch.mean(torch.mean(wf, dim=0), dim=0)
        f = wf - centering_mean
    elif c_mode == 'w_mean':
        f = wf - wm
    elif c_mode == 'global_mean':
        # Original + Walk embedding [code, direction, step, feature]
        expanded_of = of.unsqueeze(1).repeat(1, wf.shape[1], 1).unsqueeze(-2)
        all_embs = torch.cat([expanded_of, wf.unsqueeze(2)], dim=2)
        centering_mean = torch.mean(all_embs, dim=(0, 1, 2))
        f = wf - centering_mean
    else:
        f = wf
    return f


def _feature_centering(walk_features, original_features, code_selection, direction_selection, centering, prior_subset=False, w_mean=None):
    assert centering in [None, 'individual_code_mean', 'all_code_mean', 'w_mean', 'global_mean']

    # print(centering, code_selection, direction_selection)
    if prior_subset:
        if direction_selection:
            walk_features = walk_features[:, direction_selection]

        if code_selection:
            original_features = original_features[code_selection]
            walk_features = walk_features[code_selection]
        features = _center(walk_features, original_features, c_mode=centering, wm=w_mean)
    else:
        features = _center(walk_features, original_features, c_mode=centering, wm=w_mean)
        if direction_selection:
            features = features[:, direction_selection]
        if code_selection:
            features = features[code_selection]

    # print("Feature Shape: ", features.shape)
    return features


def _feature_centering_2(walk_features, original_features, centering, w_mean=None):
    assert centering in [None, 'individual_code_mean', 'all_code_mean', 'w_mean', 'global_mean']
    return _center(walk_features, original_features, c_mode=centering, wm=w_mean)


def compute_mean_features(features):
    # This tensor is used to compute hierarchical clustering.
    centered_mean_features = torch.mean(features, dim=0)  # [direction x N, feature_dim] 360, 512

    # [n_code, n_dir, feat_dim]  [n_dir, feat_dim]
    return features, centered_mean_features


def compute_euclidean_distance(features, maintain_diagonal):
    def _get_off_diagonal_elements(M):
        return M[~torch.eye(*M.shape, dtype=torch.bool)].view(M.shape[0], M.shape[1] - 1)

    # features = _feature_centering(features, centering, w_mean)

    if maintain_diagonal:
        feature_cosine_distance = torch.empty(size=(features.size(0), features.size(1), features.size(1)))

        for code_idx in range(features.size(0)):
            dist = sklearn.metrics.pairwise.euclidean_distances(features[code_idx], features[code_idx])
            dist = torch.tensor(dist)
            feature_cosine_distance[code_idx] = dist
    else:
        feature_cosine_distance = torch.empty(size=(features.size(0), features.size(1), features.size(1) - 1))

        for code_idx in range(features.size(0)):
            dist = sklearn.metrics.pairwise.euclidean_distances(features[code_idx], features[code_idx])
            dist = torch.tensor(dist)
            off_diagonal_dist = _get_off_diagonal_elements(dist)
            feature_cosine_distance[code_idx] = off_diagonal_dist

    return feature_cosine_distance


def compute_cosine_distance(features, maintain_diagonal):
    def _get_off_diagonal_elements(M):
        return M[~torch.eye(*M.shape, dtype=torch.bool)].view(M.shape[0], M.shape[1] - 1)

    # features = _feature_centering(features, centering, w_mean)

    if maintain_diagonal:
        feature_cosine_distance = torch.empty(size=(features.size(0), features.size(1), features.size(1)))

        for code_idx in range(features.size(0)):
            dist = sklearn.metrics.pairwise.cosine_distances(features[code_idx], features[code_idx])
            dist = torch.tensor(dist)
            feature_cosine_distance[code_idx] = dist
    else:
        feature_cosine_distance = torch.empty(size=(features.size(0), features.size(1), features.size(1) - 1))

        for code_idx in range(features.size(0)):
            dist = sklearn.metrics.pairwise.cosine_distances(features[code_idx], features[code_idx])
            dist = torch.tensor(dist)
            off_diagonal_dist = _get_off_diagonal_elements(dist)
            feature_cosine_distance[code_idx] = off_diagonal_dist

    print(feature_cosine_distance.shape)
    return feature_cosine_distance


# def postprocess_magnitude_features(walk_features,
#                                   original_features,
#                                   code_selection=None,
#                                   direction_selection=None):
#     if direction_selection:
#         walk_features = walk_features[:, direction_selection]
#
#     if code_selection:
#         walk_features = walk_features[code_selection]
#         original_features = original_features[code_selection]
#     print(walk_features.shape, original_features.shape)
#
#     # features = walk_features - original_features.unsqueeze(1).repeat(1, walk_features.shape[1], 1)
#     features = walk_features - original_features.unsqueeze(1).repeat(1, walk_features.shape[0], 1)
#     print(features.shape)
#     return features

def postprocess_magnitude_features(walk_features, original_features, code_selection=None, direction_selection=None):
    """
    Postprocesses magnitude features by computing differences between walk and original features.

    Parameters:
        walk_features (torch.Tensor): The features derived from walks.
        original_features (torch.Tensor): The original features to compare against.
        code_selection (slice or list, optional): Specifies indices for selecting subsets of data.
        direction_selection (slice or list, optional): Specifies indices for selecting specific directions.

    Returns:
        torch.Tensor: The feature differences.
    """
    if direction_selection is not None:
        walk_features = walk_features[:, direction_selection]

    if code_selection is not None:
        walk_features = walk_features[code_selection]
        original_features = original_features[code_selection]

    # print(f"walk_features shape: {walk_features.shape}, original_features shape: {original_features.shape}")

    # Compute the difference tensor
    if len(walk_features.shape) == 2:
        n_code = original_features.shape[0]
        walk_features = walk_features.reshape(shape=(n_code, int(walk_features.shape[0] / n_code), walk_features.shape[1]))
        features = walk_features - original_features.unsqueeze(1).expand_as(walk_features)
        # features = features.flatten(start_dim=0, end_dim=1)
    else:
        features = walk_features - original_features.unsqueeze(1).expand_as(walk_features)

    # print(f"Computed features shape: {features.shape}")
    return features


def postprocess_clustering_features(walk_features,
                                    original_features,
                                    code_selection,
                                    mode,
                                    prior_subset,
                                    centering,
                                    w_mean):
    if prior_subset:
        walk_features, original_features = _subsetting(walk_features, original_features, None, code_selection)

    # Center
    walk_features = _feature_centering_2(walk_features, original_features, centering, w_mean)

    if not prior_subset:
        walk_features, _ = _subsetting(walk_features, original_features, None, code_selection)

    # Subtraction
    if mode == 'end':
        pass
    elif mode == 'diff':
        walk_features = walk_features - original_features.unsqueeze(1).repeat(1, walk_features.shape[1], 1)

    return walk_features


def postprocess_variance_features(walk_features,
                                  original_features,
                                  mode,
                                  code_selection=None,
                                  direction_selection=None,
                                  centering=None,
                                  prior_subset=False,
                                  w_mean=None,
                                  normalize=False):
    if prior_subset:
        walk_features, original_features = _subsetting(walk_features, original_features, direction_selection, code_selection)

    # Center
    walk_features = _feature_centering_2(walk_features, original_features, centering, w_mean)

    if not prior_subset:
        walk_features, _ = _subsetting(walk_features, original_features, direction_selection, code_selection)

    # Subtraction
    if mode == 'end':
        pass
    elif mode == 'diff':
        # print("Ori: ", original_features.shape, "   Walked: ", walk_features.shape)
        postprocessed_original_features = original_features.unsqueeze(1).repeat(1, walk_features.shape[1], 1)
        if code_selection:
            postprocessed_original_features = postprocessed_original_features[code_selection, :, :]
            walk_features = walk_features - postprocessed_original_features
        else:
            walk_features = walk_features - postprocessed_original_features

    if normalize:
        walk_features = walk_features / torch.norm(walk_features, dim=-1, keepdim=True)

    return walk_features


def compute_mean_features_pairwise_distances(features: torch.Tensor, dist_metric):
    """

    Args:
        features: [code, direction, feature]
        dist_metric: distance metric between feature vectors
        centering: center feature vectors.

    Returns:

    """
    # Compute pairwise distance of directions for each code.
    pairwise_distance = []
    for feat in features:
        d = scipy.spatial.distance.pdist(feat.numpy(), metric=dist_metric)  # I need a condensed form
        pairwise_distance.append(torch.tensor(d))
    pairwise_distance = torch.stack(pairwise_distance)

    mean_pairwise_distance = torch.mean(pairwise_distance, dim=0)  # Code average
    return mean_pairwise_distance
