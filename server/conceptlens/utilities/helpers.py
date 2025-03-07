import torch

"""
Helper Functions

"""


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
