import os
import torch
from torch.utils.data import DataLoader

from argparse import ArgumentParser

from common import LatentCodeGenerator, DatasetManager, ImageGenerator
from common_sampler import DirectionSampler

from utils.latent_code_dataset import LatentCodeDataset

from _helpers import parse_layer_configuration, parse_generator_fp, prepare_model


# Command-line arguments
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--domain", type=str, default="s3t_wild512")
    parser.add_argument("--layer_name", type=str, default="early")
    parser.add_argument("--method", type=str, default="sefa")
    parser.add_argument("--n_codes", type=int, default=10)
    parser.add_argument("--edit_dist", type=int, default=5)
    parser.add_argument("--n_directions", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1004)
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator_fp, n_layers = parse_generator_fp(args.domain)
    generator = prepare_model(args.domain, generator_fp, device)
    layer_range = parse_layer_configuration(args.layer_name, n_layers)

    latent_gen = LatentCodeGenerator(args.domain, generator)
    z_codes, w_codes = latent_gen.sample_latent_codes(args.n_codes, device)

    direction_sampler = DirectionSampler(seed=args.seed, n_directions=args.n_directions)
    directions = direction_sampler.get_sefa_directions(
        generator=generator,
        layer_range=layer_range,
        sampling_rate=24,
        top_direction_per_sample=8,
        skip_first_svd=True,
        device=device
    )
    print(directions.shape)

    dataset_manager = DatasetManager(output_root="output")
    dataset_manager.save_tensor(directions, "directions.pt")

    dataloader = DataLoader(LatentCodeDataset(z_codes), batch_size=1, shuffle=False)

    image_gen = ImageGenerator(generator, device)
    image_gen.apply_directions(dataloader, directions, args.edit_dist, dataset_manager.walked_output)