import os

import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform

from helpers import load_setting, load_feature_data, postprocess_features


# def precompute_pairwise_distance():
#     experiment_name = 's2cliplarge'
#     feature_processing = 'diff'
#     metric = 'euclidean'
#
#     # Read Data
#     root = os.path.join('served_data', experiment_name)
#
#     # Load: [Code, Method, Direction, Step, Feature]
#     features = load_feature_data(data_root=root)
#
#     # Postprocess: [Code, Method * Direction => Direction, Feature]
#     features = postprocess_features(features, None, feature_processing)
#
#     # Pairwise distance
#     pairwise = np.zeros(shape=(features.shape[0], features.shape[1], features.shape[1]))
#     for i, code in enumerate(features):
#         tmp = scipy.spatial.distance.cdist(code, code, metric=metric)
#         print(i)
#         pairwise[i] = tmp
#
#     savepath = os.path.join('served_data', experiment_name, f'pairwise-{feature_processing}-{metric}')
#     torch.save(pairwise, savepath)


def precompute_pairwise_distance(tensor, distance_metric):
    num_codes, num_directions, _ = tensor.shape
    distance_matrix = np.zeros((num_codes, num_directions, num_directions))

    for code in range(num_codes):
        distance_matrix[code] = squareform(pdist(tensor[code], metric=distance_metric))

    return distance_matrix


if __name__ == '__main__':
    experiment_names = [
        's2ffhq400late'
    ]
    re_center = ['raw', 'rec']
    feature_processing = ['end', 'diff']
    metric = ['cosine', 'euclidean']

    # Read Data
    for experiment_name in experiment_names:
        root = os.path.join('served_data', experiment_name)

        for fp in feature_processing:
            for m in metric:
                for re in re_center:
                    savepath = os.path.join('served_data', experiment_name, f'pairwise-{fp}-{m}-{re}')

                    # Load: [Code, Method, Direction, Step, Feature]
                    features = load_feature_data(data_root=root)

                    # Postprocess: [Code, Method * Direction => Direction, Feature]
                    features = postprocess_features(features, None, fp)

                    # Feature re-centering
                    if re == 'rec':
                        mean = torch.mean(features, dim=0)
                        mean = torch.mean(mean, dim=0)

                        # compute feature vectors from the new origin
                        recentered_features = features - mean

                    # Pairwise
                    distance_matrix = precompute_pairwise_distance(features, m)

                    # Difference
                    # edit_diffs = (features - source_acts[a].unsqueeze(0).view(1, n_channels, -1)).abs().mean(
                    #     dim=-1).mean(dim=-1)

                    torch.save(distance_matrix, savepath, pickle_protocol=5)
                    # print(distance_matrix.shape)  # Should be (4, 5, 5)
