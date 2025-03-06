import os
import torch
import numpy as np
from sklearn.cluster import KMeans


def _compute_sefa_directions(style_weights, layers: list, top_dir: int = None,
                             skip_first: bool = True, drop_cut: int = None):
    """
    Computes SVD-based directions for the specified layers.

    Args:
        style_weights (dict): Dictionary mapping layer indices to weight tensors.
        layers (list): List of layer indices to include.
        top_dir (int, optional): Number of top singular directions to select.
        skip_first (bool, optional): Whether to skip the first SVD component.
        drop_cut (int, optional): Number of random rows to sample from the concatenated weights.

    Returns:
        torch.Tensor: Selected eigenvectors (directions), shape [dim, directions].
    """
    # Concatenate weights from specified layers.
    weights = torch.cat([layer_weight for i, layer_weight in enumerate(style_weights.values())
                         if i in layers])
    # Randomly select a subset of rows.
    drop_indices = torch.tensor(np.random.choice(weights.shape[0], drop_cut, replace=False),
                                dtype=torch.long)
    weights = weights[drop_indices]

    # Compute SVD.
    U, S, Vh = torch.linalg.svd(weights, full_matrices=False)
    offset = 2 if skip_first else 1
    if top_dir is not None:
        eigvec = Vh[offset - 1: top_dir + (offset - 1), :].T
    else:
        eigvec = Vh.T
    return eigvec  # [dim, directions]


class DirectionSampler:
    """Handles methods for sampling latent directions using different strategies."""

    def __init__(self, seed: int = 1, n_directions: int = 192):
        self.seed = seed
        self.n_directions = n_directions

    def get_centroid_directions(self, w_codes: torch.Tensor) -> torch.Tensor:
        """
        Clusters W-space codes and computes normalized centroid pair directions.

        Args:
            w_codes (torch.Tensor): Tensor containing W-space latent codes.

        Returns:
            torch.Tensor: Normalized direction vectors.
        """
        # Cluster into 20 centroids.
        kmeans = KMeans(n_clusters=20, random_state=self.seed).fit(w_codes.cpu())
        centroids = kmeans.cluster_centers_

        torch.manual_seed(self.seed)
        pairs = []
        # Sample unique pairs until reaching the desired number of directions.
        while len(pairs) < self.n_directions:
            sample = torch.randint(low=0, high=len(centroids), size=(2,))
            if sample[0] != sample[1]:
                pairs.append(sample)

        # Compute difference vectors between centroid pairs.
        directions = [centroids[p[0]] - centroids[p[1]] for p in pairs]
        directions = torch.tensor(directions, dtype=torch.float32)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        return directions

    def get_sefa_directions(self,
                            generator,
                            layer_range: list,
                            sampling_rate: int,
                            top_direction_per_sample: int,
                            skip_first_svd: bool,
                            device) -> torch.Tensor:
        """
        Computes SeFA-based directions by sampling SVD-based directions from a generator's style weights,
        then clustering and selecting the closest ones.

        Args:
            generator: Object that provides style weights via get_style_weights().
            layer_range (list): List of layer indices to include.
            sampling_rate (int): Number of samples to generate.
            top_direction_per_sample (int): Number of top SVD directions per sample.
            skip_first_svd (bool): Whether to skip the first SVD component.
            device: Torch device to place the final tensor on.

        Returns:
            torch.Tensor: Normalized and clustered direction vectors.
        """
        style_weights = generator.get_style_weights()
        # Concatenate weights for specified layers.
        weight = torch.cat([layer_weight for i, layer_weight in enumerate(style_weights.values())
                            if i in layer_range])
        directions = []
        # Generate candidate directions using varying drop_cut values.
        for _ in range(sampling_rate):
            drop_max = min(512, weight.shape[0])
            drop_min = max(64, drop_max - 1)
            drop_cut = np.random.randint(low=drop_min, high=drop_max)
            tmp = _compute_sefa_directions(style_weights,
                                           layers=layer_range,
                                           top_dir=top_direction_per_sample,
                                           skip_first=skip_first_svd,
                                           drop_cut=drop_cut).T
            directions.append(tmp)
        directions = torch.cat(directions)
        # Cluster the candidate directions.
        cluster = KMeans(n_clusters=self.n_directions, n_init="auto").fit(directions)
        closest_directions = []
        # For each cluster, select the candidate direction that is closest to the cluster center.
        for cluster_center in torch.tensor(cluster.cluster_centers_):
            normalized_center = cluster_center / torch.norm(cluster_center)
            sims = torch.cosine_similarity(normalized_center.unsqueeze(0), directions)
            closest_idx = torch.argmax(sims)
            closest_directions.append(directions[closest_idx])
        directions = torch.stack(closest_directions).to(device)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        return directions
