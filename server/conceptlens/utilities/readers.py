import json
import os
from typing import Tuple

import torch
from torch import Tensor


def load_setting(data_root):
    config = None
    with open(os.path.join(data_root, 'setting.json')) as file:
        config = json.load(file)

    experiment_name = config['experiment_name']
    n_code = config['n_code']
    n_direction_per_method = config['n_direction']
    uid = config['uid']

    # Think again for dependencies before changing this function.
    return experiment_name, n_code, n_direction_per_method, uid


def load_uid(data_root):
    _, _, _, uid = load_setting(data_root)
    return uid


def load_walk_features(data_root):
    """
    loads walked images' feature data.

    Args:
        data_root:

    Returns:

    """
    tensor_root = os.path.join(data_root, 'tensors')
    walks = torch.load(os.path.join(tensor_root, 'features.pt'), map_location='cpu', weights_only=False)
    if len(walks.shape) == 2:
        walks = walks.reshape(shape=(200, int(walks.shape[0] / 200), walks.shape[1]))

    # [Code, Direction, Feature]
    return walks


def load_original_features(data_root) -> torch.Tensor:
    """
    loads original images' feature data.

    Args:
        data_root:

    Returns:

    """
    tensor_root = os.path.join(data_root, 'tensors')
    # [Code, Feature]
    return torch.load(os.path.join(tensor_root, 'features_original.pt'), map_location='cpu', weights_only=False)


def load_difference_features(data_root):
    """
    loads walked - original difference vector data.

    Args:
        data_root:

    Returns:

    """
    tensor_root = os.path.join(data_root, 'tensors')

    # [Code, Feature]
    return torch.load(os.path.join(tensor_root, 'features_diff.pt'), map_location='cpu', weight_only=False)


def load_original_latent(data_root) -> torch.Tensor:
    return torch.load(os.path.join(data_root, 'codes.pt'), map_location='cpu', weights_only=False)[:, 0, 0, :]


def load_walk_latent(data_root) -> torch.Tensor:
    codes = load_original_latent(data_root)
    directions = torch.load(os.path.join(data_root, 'directions.pt'), map_location='cpu', weights_only=False)
    return codes.unsqueeze(1).repeat(1, directions.shape[0], 1) + directions


def _read_experiment_uids(experiment_names, served_data_root) -> list:
    """
    Function to retrieve features of all listed experiment.

    Args:
        experiment_names: list of experiment names
        served_data_root: root directory containing experiment folders.

    Returns:

    """
    # Read data
    data_roots = [os.path.join(served_data_root, exp_name) for exp_name in experiment_names]
    uids = [load_uid(root) for root in data_roots]

    return uids


def read_walk_features(experiment_names, served_data_root='served_data') -> tuple[Tensor, Tensor]:
    """
    Function to retrieve features of all listed experiment as a concatenated tensor.

    Args:
        experiment_names: list of experiment names
        served_data_root:

    Returns: [code, direction, feature], cumulative direction-finding-method sizes.

    """
    # Read data
    data_roots = [os.path.join(served_data_root, exp_name) for exp_name in experiment_names]
    walk_features = [load_walk_features(root) for root in data_roots]
    cumulative_dfm_size = _compute_cumulative_dfm_size(walk_features)

    walk_features = torch.cat(walk_features, dim=1)
    return walk_features, cumulative_dfm_size


def read_original_features(experiment_names, served_data_root='served_data') -> torch.Tensor:
    """
    Function to retrieve features of original experiments as a concatenated tensor.
    If the domain of experiments are different, an error is thrown.

    Args:
        experiment_names:
        served_data_root:

    Returns: [code, feature]

    """
    # Determine features domain match
    domains = set([exp_name.split('-')[0] for exp_name in experiment_names])
    # if len(domains) > 1:
    #     raise Exception("Multiple domains given. Cannot concatenate original features of multiple domains.")

    # Read data
    data_root = os.path.join(served_data_root, experiment_names[0])
    return load_original_features(data_root)


def read_original_latent_codes(experiment_names, served_data_root='served_data') -> torch.Tensor:
    domains = set([exp_name.split('-')[0] for exp_name in experiment_names])
    if len(domains) > 1:
        raise Exception("Multiple domains given. Cannot concatenate original features of multiple domains.")

    data_root = os.path.join(served_data_root, experiment_names[0])
    return load_original_latent(data_root)


def read_walk_latent_codes(experiment_names, served_data_root='served_data') -> tuple[Tensor, Tensor]:
    domains = set([exp_name.split('-')[0] for exp_name in experiment_names])
    if len(domains) > 1:
        raise Exception("Multiple domains given. Cannot concatenate original features of multiple domains.")

    data_roots = [os.path.join(served_data_root, exp_name) for exp_name in experiment_names]
    walk_features = [load_walk_latent(root) for root in data_roots]
    cumulative_dfm_size = _compute_cumulative_dfm_size(walk_features)

    walk_features = torch.cat(walk_features, dim=1)
    print(f"Walk Latent Shape: {walk_features.shape}")
    return walk_features, cumulative_dfm_size


def _compute_cumulative_dfm_size(features: list):
    return torch.cumsum(torch.tensor([x.shape[1] for x in features]), 0)
