import os
import pickle
import ast

import torch

from data_generation._model_utils import copy_params_and_buffers
from data_generation._stylegan2 import Generator as G2
from data_generation._stylegan3 import Generator as G3

from pathlib import Path


def parse_generator_fp(domain):
    # Use the parent directory, then the "resources/checkpoints" folder as the base.
    base_dir = os.path.join(os.path.dirname(Path(__file__).parent), 'resources', 'checkpoints')

    # Mapping domains to file names and generator_size.
    domain_map = {
        # NVIDIA official StyleGAN 2
        's2_ffhq256':      {'filename': 'stylegan2-ffhq-256x256.pkl', 'generator_size': 13},
        's2_ffhq1024':     {'filename': 'stylegan2-ffhq-1024x1024.pkl', 'generator_size': 15},
        's2_celeba256':    {'filename': 'stylegan2-celebahq-256x256.pkl', 'generator_size': 13},
        's2_metface1024': {'filename': 'stylegan2-metfaces-1024x1024.pkl', 'generator_size': 15},
        's2_cat512': {'filename': 'stylegan2-afhqcat-512x512.pkl', 'generator_size': 15},
        's2_wild512': {'filename': 'stylegan2-afhqwild-512x512.pkl', 'generator_size': 15},

        # NVIDIA official StyleGAN 3
        's3t_ffhq1024':    {'filename': 'stylegan3-t-ffhq-1024x1024.pkl', 'generator_size': 15},
        's3t_wild512':     {'filename': 'stylegan3-t-afhqv2-512x512.pkl', 'generator_size': 15},
        's3t_metfaces1024':{'filename': 'stylegan3-t-metfaces-1024x1024.pkl', 'generator_size': 15},
        's3r_ffhq1024':    {'filename': 'stylegan3-r-ffhq-1024x1024.pkl', 'generator_size': 15},
        's3r_wild512':     {'filename': 'stylegan3-r-afhqv2-512x512.pkl', 'generator_size': 15},
        's3r_metfaces1024':{'filename': 'stylegan3-r-metfaces-1024x1024.pkl', 'generator_size': 15},

        # Unofficial StyleGAN 2
        's2_beach256': {'filename': 'stylegan2-beach-256x256.pkl', 'generator_size': 13},
        's2_map512':       {'filename': 'stylegan2-map-512x512.pkl', 'generator_size': 15},
        's2_micro512':     {'filename': 'stylegan2-micro-512x512.pkl', 'generator_size': 15},

        # Unofficial StyleGAN 3
        's3_landscape256': {'filename': 'stylegan3-landscape-256x256.pkl', 'generator_size': 15},
    }

    if domain not in domain_map:
        raise ValueError(f"Unknown domain: {domain}")

    config = domain_map[domain]
    model_path = os.path.join(base_dir, config['filename'])
    generator_size = config['generator_size']

    return model_path, generator_size



def parse_layer_configuration(layer_config, generator_size):
    # Custom layer configuration
    if isinstance(layer_config, str) and layer_config.strip().startswith('[') and layer_config.strip().endswith(']'):
        try:
            custom_config = ast.literal_eval(layer_config)
            if isinstance(custom_config, list) and all(isinstance(x, int) for x in custom_config):
                return custom_config
            else:
                raise ValueError("Custom layer configuration must be a list of integers.")
        except Exception as e:
            raise ValueError(f"Invalid custom layer configuration: {layer_config}. Error: {e}")

    # Otherwise, use pre-defined configurations based on generator_size.
    if generator_size == 13:
        predefined = {
            'early': [0, 1, 2, 3],
            'early_0': [0, 1],
            'early_1': [2, 3],
            'middle': [4, 5, 6, 7],
            'middle_0': [4, 5],
            'middle_1': [6, 7],
            'late': [8, 9, 10, 11],
            'late_0': [8, 9],
            'late_1': [10, 11],
            'all': list(range(14)),
        }
    elif generator_size == 15:
        predefined = {
            'early': [0, 1, 2, 3, 4],
            'early_0': [0, 1],
            'early_1': [2, 3],
            'middle': [5, 6, 7, 8, 9],
            'middle_0': [4, 5],
            'middle_1': [6, 7],
            'middle_2': [8, 9],
            'late': [10, 11, 12, 13, 14],
            'late_0': [10, 11],
            'late_1': [12, 13],
            'late_2': [14, 15],
            'all': list(range(16)),
        }
    else:
        raise ValueError("Unsupported generator size. Supported sizes are 13 and 15.")

    if layer_config in predefined:
        return predefined[layer_config]
    else:
        raise ValueError(f"Unknown layer configuration: {layer_config}")


def prepare_model(domain, model_path, device):
    # Prepare model
    with open(model_path, 'rb') as f:
        old_g = pickle.load(f)['G_ema']
        if domain.startswith('s3'):
            generator = G3(*old_g.init_args, **old_g.init_kwargs)
        else:
            generator = G2(*old_g.init_args, **old_g.init_kwargs)
        with torch.no_grad():
            copy_params_and_buffers(old_g, generator, require_all=True)
    generator = generator.to(device)
    return generator
from pathlib import Path
