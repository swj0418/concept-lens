from abc import ABC

import torch
import numpy as np

from data_generation.utils.walker_connector import GeneratorConnector


def _compute_sefa_directions(style_weights,
                             layers: list,
                             top_dir: int = None,
                             skip_first=True,
                             drop_cut: int = None):
    # Organize weights
    # weight = []
    # for i, layer_weight in enumerate(list(style_weights.values())):
    #     if i in layers:
    #         to_append = layer_weight.clone()
    #         if drop_cut:
    #             if drop_cut >= layer_weight.shape[0]:  # Cutting output layers. Is this right?
    #                 drop_cut = layer_weight.shape[0]
    #             choice = torch.tensor(np.random.choice(layer_weight.shape[0], drop_cut, replace=False)).to(
    #                 torch.long)
    #             to_append = layer_weight[choice]
    #         weight.append(to_append)
    # # Concatenate
    # weight = torch.concat(weight, axis=0).cpu()  # Long matrix -- [output (layer * k), input]

    # Different method
    weight = []
    for i, layer_weight in enumerate(list(style_weights.values())):
        if i in layers:
            weight.append(layer_weight)
    weight = torch.cat(weight)
    drop_choice = torch.tensor(np.random.choice(weight.shape[0], drop_cut, replace=False)).to(torch.long)
    weight = weight[drop_choice]

    # SVD
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    if top_dir is not None:
        offset = 2 if skip_first else 1
        eigvec = Vh[offset - 1: top_dir + (offset - 1), :].T

    # Eig decomp
    # weight_cov = weight / torch.linalg.norm(weight, axis=1, keepdims=True)  # Row normalize
    # weight_cov = weight_cov.T @ weight_cov
    # eigval, eigvec = np.linalg.eigh(weight_cov)
    # eigvec = torch.tensor(eigvec)
    # if top_dir is not None:
    #     offset = 2 if skip_first else 1
    #     eigvec = eigvec[:, -top_dir - offset:-offset]

    return eigvec  # [dim, dir]


def _compute_sefa_directions_filter(style_weights,
                                     layers: list,
                                     top_dir: int = None,
                                     skip_first=True,
                                     drop_cut: int = None):
    # Organize weights
    weight = []
    for i, layer_weight in enumerate(list(style_weights.values())):
        if i in layers:
            to_append = layer_weight.clone()
            if drop_cut:
                if drop_cut >= layer_weight.shape[0]:  # Cutting output layers. Is this right?
                    drop_cut = layer_weight.shape[0]
                choice = torch.tensor(np.random.choice(layer_weight.shape[0], drop_cut, replace=False)).to(
                    torch.long)
                to_append = layer_weight[choice]
            weight.append(to_append)

    # Concatenate
    weight = torch.concat(weight, axis=0).cpu()  # Long matrix -- [output (layer * k), input]

    # Dimension per row.
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
    if top_dir is not None:
        offset = 2 if skip_first else 1
        eigvec = Vh[offset - 1: top_dir + (offset - 1), :].T

    return eigvec  # [dim, dir]


def walk_ws(w, direction, walk_distance, walked_layers: list=None):
    """
    Walk ws of shape [batch, n_layer, latent_dim]. Since the ws latent vectors have uniform latent dimension,
    the directions can be applied easily.

    Strictly creates walk of size 1

    Args:
        w: w code of shape [batchsize, n_layer, latent_dim]
        direction: direction vector of shape [n_layer, latent_dim] or [latent_dim]
        walk_distance: integer
        step_size: integer

    Returns:

    """
    assert len(direction.shape) <= 2

    # Convert direction to torch tensor
    # direction = direction.clone()

    # Prepare for batch operation
    if len(direction.shape) == 1:
        direction = direction.unsqueeze(0).unsqueeze(0).repeat(w.shape[0], w.shape[1], 1)
    elif len(direction.shape) == 2:
        direction = direction.unsqueeze(0).repeat(w.shape[0], w.shape[1], 1)

    # Apply direction to w vector - [batch, n_layer, latent_dim]
    if walked_layers:
        # Walk only the layers that are selected
        layer_negation = [_ for _ in range(w.shape[1]) if _ not in walked_layers]
        direction[:, layer_negation, :] = torch.zeros(size=(w.shape[2], )).to(direction.device)
        edit_w = w + (walk_distance * direction)
    else:
        edit_w = w + (walk_distance * direction)
    return edit_w


def walk(code, direction, walk_length, step_size, one_sided, neutralize=False, device=None):
    """
    A generic function for generating a walk of codes.

    Args:
        code: A code to walk.
        direction: Vector of walk.
        walk_length: Total length of a walk (in unit direction, from one end to another).
        step_size: Number of codes in a walk.
        one_sided: If True, walk only towards a positive direction.
        device:

    Returns:

    """
    assert len(code.shape) == 1
    assert len(direction.shape) == 1

    direction = direction.to(device)
    code = code.to(device)

    # Assert unit vector
    direction = direction / torch.norm(direction)

    if one_sided:
        multipliers = torch.linspace(0, walk_length, step_size)
        multipliers = torch.clamp(multipliers, min=0, max=walk_length)
    else:
        wl = walk_length / 2
        multipliers = torch.linspace(-wl, wl, step_size)
        multipliers = torch.clamp(multipliers, min=-wl, max=wl)

    multipliers = multipliers.to(device)
    scale = multipliers[:, None] * direction

    if neutralize:
        neutral = code - (direction * (code.dot(direction)))
        walk_v = neutral + scale
    else:
        walk_v = code + scale

    return walk_v


class Walker(ABC):
    def __init__(self, g_connector: GeneratorConnector, device):
        self.method_name = None
        self.g_connector: GeneratorConnector = g_connector
        self.device = device

    def __repr__(self):
        return self.method_name

    def compute_directions(self, **kwargs):
        pass

    def _generate_in_w(self):
        pass

    def _generate_in_ws(self):
        pass

    def _generate_in_stylespace(self):
        pass

    def generate_walk_images(self,
                             z: torch.Tensor, direction,
                             step_size, walk_length,
                             one_sided, neutralize,
                             layers=None, **kwargs):
        """
        given a latent code z and a single direction, generate codes in a walk.

        TODO: Some workflow in this function is still tailored to StyleGAN - to fix this, make general function in
         GConnector, for both latent space transforming (such as z->w) and latent code tranforming (such as w->image).
         For example, non-stylegan generators will have an empty latent space transformation.

        Args:
            z:
            direction:
            step_size:
            walk_length:
            kwargs: This argument includes image generation hyper-parameters necessary for different architectures.
                    StyleGAN requires {truncation_psi, truncation_cutoff}.

        Returns:

        """
        # Only one code & direction.
        assert len(z.shape) == 1
        assert len(direction.shape) == 1

        # Assert unit vector
        direction = direction / torch.norm(direction)

        z = z.to(self.device)
        if layers:
            w = self.g_connector.forward_z_w(z.unsqueeze(0))
            ws = self.g_connector.forward_w_ws(w, update_emas=False)  # [16, 512]
            ws = ws.repeat([step_size, 1, 1])
            walk_v = walk(w[0], direction, walk_length, step_size, one_sided, neutralize, self.device)  # [2, 512]

            for layer in layers:
                ws[:, layer] = walk_v

            # print(layers, ws.shape, walk_v.shape)
            walk_images = self.g_connector.forward_ws_img(ws, **kwargs)
        else:
            w = self.g_connector.forward_z_w(z.unsqueeze(0))[0]

            walk_v = walk(w, direction, walk_length, step_size, one_sided, neutralize, self.device)
            walk_images = self.g_connector.forward_w_img(walk_v, **kwargs)

        return walk_v, walk_images


class PCAWalker(Walker):
    def __init__(self, g_connector: GeneratorConnector, device):
        Walker.__init__(self, g_connector, device)
        self.method_name = 'pca'

    def get_w_collection(self, sample_size: int, z_dim=512, seed=0):
        w_collection = torch.zeros(size=(sample_size, z_dim)).to(self.device)
        for i in range(sample_size):
            z = torch.randn(size=(1, z_dim)).to(self.device)
            w = self.g_connector.forward_z_w(z)
            w_collection[i] = w

        return w_collection

    def compute_directions(self, sample_size: int, top_dir=None, skip_first=True, permute_cut=None, z_dim=512, seed: int=0):
        """
        PCA Walker requires to sample some codes in the Z-space to W-space.
        Args:
            sample_size: sample size for obtaining a collection of w latent codes in W-space.

        Returns: Basis V for W.
        """
        w_collection = torch.zeros(size=(sample_size, z_dim)).to(self.device)
        for i in range(sample_size):
            z = torch.randn(size=(1, z_dim)).to(self.device)
            w = self.g_connector.forward_z_w(z)
            w_collection[i] = w

        if permute_cut:
            choice = torch.tensor(np.random.choice(z_dim, permute_cut, replace=False))
            w_collection = w_collection[choice]

        # PCA
        mean_adj_w = w_collection - torch.mean(w_collection, dim=0)  # [code, dim]
        w_cov = mean_adj_w.T@mean_adj_w
        eigval, eigvec = torch.linalg.eigh(w_cov)

        if top_dir is not None:
            offset = 2 if skip_first else 1
            eigvec = eigvec[:, -top_dir-offset:-offset]

        return eigvec

    def compute_directions_pre(self, sample_size: int, top_dir=None, skip_first=True, permute_cut=256, z_dim=512):
        """
        PCA Walker requires to sample some codes in the Z-space to W-space.
        Args:
            sample_size: sample size for obtaining a collection of w latent codes in W-space.

        Returns: Basis V for W.
        """
        w_collection = torch.zeros(size=(sample_size, z_dim)).to(self.device)
        for i in range(sample_size):
            z = torch.randn(size=(1, z_dim)).to(self.device)
            w = self.g_connector.forward_z_w(z)
            w_collection[i] = w

        # PCA
        mean_adj_w = w_collection - torch.mean(w_collection, dim=0)  # [code, dim]
        w_cov = mean_adj_w.T@mean_adj_w
        eigval, eigvec = torch.linalg.eigh(w_cov)

        if top_dir is not None:
            offset = 2 if skip_first else 1
            eigvec = eigvec[:, -top_dir-offset:-offset]

        return eigvec

    def obtain_h(self, z: torch.Tensor, layers=None, **kwargs):
        # Only one code & direction.
        assert len(z.shape) == 1

        z = z.to(self.device)
        if layers:
            w = self.g_connector.forward_z_w(z.unsqueeze(0))
            ws = self.g_connector.forward_w_ws(w, update_emas=False)  # [16, 512]
            wb = self.g_connector.forward_to_beyond(ws, **kwargs)
            print(wb.shape)
        else:
            raise Exception("Configure layers!")

        return wb


    def generate_walk_images(self,
                             z: torch.Tensor, direction,
                             step_size, walk_length,
                             one_sided, neutralize,
                             layers=None, **kwargs):
        # Only one code & direction.
        assert len(z.shape) == 1
        assert len(direction.shape) == 1

        # Assert unit vector
        direction = direction / torch.norm(direction)

        z = z.to(self.device)
        if layers:
            w = self.g_connector.forward_z_w(z.unsqueeze(0))
            ws = self.g_connector.forward_w_ws(w, update_emas=False)  # [16, 512]
            ws = ws.repeat([step_size, 1, 1])
            walk_v = walk(w[0], direction, walk_length, step_size, one_sided, neutralize, self.device)  # [2, 512]

            for layer in layers:
                ws[:, layer] = walk_v

            # print(layers, ws.shape, walk_v.shape)
            walk_images = self.g_connector.forward_ws_img(ws, **kwargs)
        else:
            raise Exception("Configure layers!")

        return walk_v, walk_images


class SeFAWalker(Walker):
    def __init__(self, g_connector: GeneratorConnector, device):
        Walker.__init__(self, g_connector, device)
        self.method_name = 'sefa'

    def _compute_directions(self,
                            style_weights,
                            layers: list,
                            top_dir: int = None,
                            skip_first=True,
                            drop_cut: int = None):
        # Organize weights
        weight = []
        for i, layer_weight in enumerate(list(style_weights.values())):
            if i in layers:
                to_append = layer_weight.clone()
                if drop_cut:
                    if drop_cut >= layer_weight.shape[0]:  # Cutting output layers. Is this right?
                        drop_cut = layer_weight.shape[0]
                    choice = torch.tensor(np.random.choice(layer_weight.shape[0], drop_cut, replace=False)).to(
                        torch.long)
                    to_append = layer_weight[choice]
                weight.append(to_append)

        # Concatenate
        weight = torch.concat(weight, axis=0).cpu()  # Long matrix -- [output (layer * k), input]

        weight_cov = weight / torch.linalg.norm(weight, axis=1, keepdims=True)  # Row normalize
        weight_cov = weight_cov.T @ weight_cov
        # eigval, eigvec = torch.linalg.eigh(weight_cov)
        eigval, eigvec = np.linalg.eigh(weight_cov)
        eigvec = torch.tensor(eigvec)
        eigvec = eigvec.to(self.device)

        if top_dir is not None:
            offset = 2 if skip_first else 1
            eigvec = eigvec[:, -top_dir - offset:-offset]

        return eigvec  # [dim, dir]

    @DeprecationWarning
    def compute_directions(self,
                           layers: list,
                           top_dir: int = None,
                           skip_first=True,
                           drop_cut: int = None):
        """
        Compute SeFA directions for given style layers.

        Normalization ==> SeFA Github treats weight matrix as [output, input]. I am doing [input, output].
                          At the end, I am doing the equivalent by dividing by norm along the output axis.

        Returns:
        """
        weights = self.g_connector.get_style_weights()
        return self._compute_directions(weights, layers, top_dir, skip_first, drop_cut)

    def generate_walk_images(self,
                             z: torch.Tensor, direction,
                             step_size, walk_length,
                             one_sided, neutralize,
                             layers=None, **kwargs):

        # Only one code & direction.
        assert len(z.shape) == 1
        assert len(direction.shape) == 1

        # Assert unit vector
        direction = direction / torch.norm(direction)

        z = z.to(self.device)
        w = self.g_connector.forward_z_w(z.unsqueeze(0))
        walk_v = walk(w[0], direction, walk_length, step_size, one_sided, neutralize, self.device)  # [2, 512]

        if layers is None:
            walk_images = self.g_connector.forward_w_img(walk_v, **kwargs)
        else:
            ws = self.g_connector.forward_w_ws(w, update_emas=False)  # [16, 512]
            ws = ws.repeat([step_size, 1, 1])
            for layer in layers:
                ws[:, layer] = walk_v
            walk_images = self.g_connector.forward_ws_img(ws, **kwargs)

        return walk_v, walk_images


class StyleSpaceWalker(Walker):
    def __init__(self, g_connector: GeneratorConnector, device):
        Walker.__init__(self, g_connector, device)
        self.method_name = 'stylespace'

    def compute_directions(self):
        return None

    def generate_walk_images(self,
                             z: torch.Tensor,
                             direction,
                             step_size,
                             walk_length,
                             one_sided,
                             neutralize,
                             layers=None, **kwargs):

        # Only one code & direction.
        assert len(z.shape) == 1
        assert len(direction.shape) == 1

        # Assert unit vector
        direction = direction / torch.norm(direction)

        z = z.to(self.device)
        w = self.g_connector.forward_z_w(z.unsqueeze(0))
        walk_v = walk(w[0], direction, walk_length, step_size, one_sided, neutralize, self.device)  # [2, 512]

        if layers is None:
            walk_images = self.g_connector.forward_w_img(walk_v, **kwargs)
        else:
            ws = self.g_connector.forward_w_ws(w, update_emas=False)  # [16, 512]
            ws = ws.repeat([step_size, 1, 1])
            for layer in layers:
                ws[:, layer] = walk_v
            walk_images = self.g_connector.forward_ws_img(ws, **kwargs)

        return walk_v, walk_images

