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
    parser.add_argument("--method", type=str, default="va")
    parser.add_argument("--layer_name", type=str, default="early")
    parser.add_argument("--application", type=str, default="global")

    parser.add_argument("--n_codes", type=int, default=20)
    parser.add_argument("--n_code_samples", type=int, default=100000)
    parser.add_argument("--edit_dist", type=int, default=5)
    parser.add_argument("--n_directions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1004)

    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Generating Dataset for Vector Arithmetic direction-finding method.")
    print("="*80)

    generator_fp, n_layers = parse_generator_fp(args.domain)
    generator = prepare_model(args.domain, generator_fp, device)
    layer_range = parse_layer_configuration(args.layer_name, n_layers)

    latent_gen = LatentCodeGenerator(args.domain, generator)
    _, w_codes = latent_gen.sample_latent_codes(args.n_code_samples, device)

    direction_sampler = DirectionSampler(seed=args.seed, n_directions=args.n_directions)
    directions = direction_sampler.get_centroid_directions(
        w_codes,
        device=device
    )
    print(f"{directions.shape[0]} Directions sampled from {w_codes.shape[0]} latent codes.")

    output_fp = f'{args.domain}_{args.method}_{args.layer_name}_{args.application}'
    dataset_manager = DatasetManager(output_root=output_fp)
    dataset_manager.save_tensor(directions, "directions.pt")

    z_codes, w_codes = latent_gen.sample_latent_codes(args.n_codes, device)
    dataloader = DataLoader(LatentCodeDataset(z_codes), batch_size=args.batch_size, shuffle=False)

    image_gen = ImageGenerator(generator, device)
    image_gen.generate_images(w_codes, dataset_manager.code_output)
    image_gen.apply_directions(dataloader, directions, args.edit_dist, dataset_manager.walked_output)