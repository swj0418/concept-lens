import os
import pickle

import torch

from data_generation._model_utils import copy_params_and_buffers
from data_generation._stylegan2 import Generator as G2
from data_generation._stylegan3 import Generator as G3


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

def parse_generator_fp(domain):
    # Settings
    model_path = None
    generator_size = 13 if domain.endswith("256") else 15
    if domain == 's2_ffhq256':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan2-ffhq-256x256.pkl')
    elif domain == 's2_ffhq1024':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan2-ffhq-1024x1024.pkl')
    elif domain == 's2_celeba256':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan2-celebahq-256x256.pkl')
    elif domain == 's2_beach256':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan2-beach-256x256.pkl')
    elif domain == 's3t_ffhq1024':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan3-t-ffhq-1024x1024.pkl')
    elif domain == 's2_metface1024':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan2-metfaces-1024x1024.pkl')
    elif domain == 's2_cat512':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan2-afhqcat-512x512.pkl')
    elif domain == 's2_wild512':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan2-afhqwild-512x512.pkl')
    elif domain == 's3t_wild512':
        model_path = os.path.join(os.path.dirname(Path(__file__).parent), 'resources',
                                  'checkpoints/stylegan3-t-afhqv2-512x512.pkl')
        # model_path = os.path.join('resources', 'checkpoints/stylegan3-t-afhqv2-512x512.pkl')
    elif domain == 's3r_wild512':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan3-r-afhqv2-512x512.pkl')
    elif domain == 's2_map512':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan2-map-512x512.pkl')
    elif domain == 's2_micro512':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan2-micro-512x512.pkl')
    elif domain == 's3_landscape256':
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'checkpoints/stylegan3-landscape-256x256.pkl')
        generator_size = 15

    return model_path, generator_size


class ConceptLensDataset():
    def __init__(self, domain, application, layer, df_method, exp_name=None, semantic=None):
        self.domain = domain
        self.application = application
        self.layer = layer
        self.df_method = df_method

        # Optional
        self.exp_name = exp_name
        self.semantic = semantic

        self._make_folders()

    def _make_folders(self):
        # Directories
        if self.exp_name:
            self.dataset_root = os.path.join('output', self.exp_name)
        else:
            if not self.semantic:
                self.dataset_root = f'output/{self.domain}-{self.df_method}-{self.application}-{self.layer}/'
            else:
                self.dataset_root = f'output/{self.domain}-{self.df_method}_{self.semantic}-{self.application}-{self.layer}/'

        self.code_output_root = os.path.join(self.dataset_root, 'codes')
        self.walked_output_root = os.path.join(self.dataset_root, 'walked')

        os.makedirs(self.dataset_root, exist_ok=True)
        os.makedirs(self.code_output_root, exist_ok=True)
        os.makedirs(self.walked_output_root, exist_ok=True)

    def get_dataset_root(self):
        return self.dataset_root

    def get_code_output_root(self):
        return self.code_output_root

    def get_walked_output_root(self):
        return self.walked_output_root


def parse_layer_configuration(layer_name, generator_size):
    layer_range = None
    if generator_size == 13:
        if layer_name == 'early':
            layer_range = [0, 1, 2, 3]
        elif layer_name == 'early_0':
            layer_range = [0, 1]
        elif layer_name == 'early_1':
            layer_range = [2, 3]
        elif layer_name == 'middle':
            layer_range = [4, 5, 6, 7]
        elif layer_name == 'middle_0':
            layer_range = [4, 5]
        elif layer_name == 'middle_1':
            layer_range = [6, 7]
        elif layer_name == 'late':
            layer_range = [8, 9, 10, 11]
        elif layer_name == 'late_0':
            layer_range = [8, 9]
        elif layer_name == 'late_1':
            layer_range = [10, 11]
        elif layer_name == 'all':
            layer_range = [_ for _ in range(14)]
    elif generator_size == 15:
        if layer_name == 'early':
            layer_range = [0, 1, 2, 3, 4]
        elif layer_name == 'early_0':
            layer_range = [0, 1]
        elif layer_name == 'early_1':
            layer_range = [2, 3]
        elif layer_name == 'middle':
            layer_range = [5, 6, 7, 8, 9]
        elif layer_name == 'middle_0':
            layer_range = [4, 5]
        elif layer_name == 'middle_1':
            layer_range = [6, 7]
        elif layer_name == 'middle_2':
            layer_range = [8, 9]
        elif layer_name == 'late':
            layer_range = [10, 11, 12, 13, 14]
        elif layer_name == 'late_0':
            layer_range = [10, 11]
        elif layer_name == 'late_1':
            layer_range = [12, 13]
        elif layer_name == 'late_2':
            layer_range = [14, 15]
        elif layer_name == 'all':
            layer_range = [_ for _ in range(16)]

    return layer_range
