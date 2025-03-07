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
    feature_processing = ['originaldiff']
    metric = ['cosine', 'euclidean']

    # Read Data
    for experiment_name in experiment_names:
        root = os.path.join('served_data', experiment_name)

        for fp in feature_processing:
            for m in metric:
                for re in re_center:
                    savepath = os.path.join('served_data', experiment_name, f'dist-{fp}-{m}-{re}')

                    # Load: [Code, Method, Direction, Step, Feature]
                    features = load_feature_data(data_root=root)

                    # Postprocess: [Code, Direction, Feature]
                    features_start = postprocess_features(features, None, 'start')
                    features_end = postprocess_features(features, None, 'end')
                    # feature_diff = postprocess_features(features, None, 'diff')

                    # Feature re-centering
                    if re == 'rec':
                        mean = torch.mean(features, dim=0, keepdim=True)
                        mean = torch.mean(mean, dim=0, keepdim=True)

                        # compute feature vectors from the new origin
                        recentered_features = features - mean

                    # code, direction
                    # 1, direction
                    # mean(1, direction)


                    # Opposite for code

                    # Difference distance [C, D, F] => [C, D]
                    edit_diffs = (features_end - features_start).abs().mean(dim=-1)
                    # edit_diffs = feature_diff.abs().mean(dim=-1)

                    torch.save(edit_diffs, savepath, pickle_protocol=5)
                    # print(distance_matrix.shape)  # Should be (4, 5, 5)
