import torch
from torch.utils.data import Dataset


class LatentCodeDataset(Dataset):
    def __init__(self, latent_codes):
        assert type(latent_codes) == torch.Tensor
        print(f"Creating PyTorch dataset of latent codes of size {latent_codes.shape}")
        self.latent_codes = latent_codes

    def __len__(self):
        return self.latent_codes.shape[0]  # [n_code, n_dim]

    def __getitem__(self, item):
        return self.latent_codes[item]


class WalkedCodeDataset(Dataset):
    def __init__(self, latent_codes):
        assert type(latent_codes) == torch.Tensor  # [code, direction, layer, dim]
        print(f"Creating PyTorch Walked Code dataset of latent codes of size {latent_codes.shape}")
        self.num_code, self.num_direction = latent_codes.shape[0], latent_codes.shape[1]
        self.latent_codes = latent_codes.reshape(latent_codes.shape[0] * latent_codes.shape[1],
                                                 latent_codes.shape[2],
                                                 latent_codes.shape[3])

    def __len__(self):
        return self.latent_codes.shape[0]  # [n_code, n_dim]

    def __getitem__(self, item):
        code_idx = item // self.num_direction
        direction_idx = item - (code_idx * self.num_direction)
        return self.latent_codes[item], code_idx, direction_idx
