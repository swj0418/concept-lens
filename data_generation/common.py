import os
import sys
import torch
import torchvision

from pathlib import Path
from PIL import Image
import tqdm
from utils.image_utils import convert_images_to_uint8


class LatentCodeGenerator:
    def __init__(self, domain, generator, truncation_psi=0.7, truncation_cutoff=8):
        self.domain = domain
        self.generator = generator
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

    def sample_latent_codes(self, num_samples, device):
        """Samples Z-space latent codes and maps to W-space."""
        z_codes = torch.randn(size=(num_samples, 512)).to(device)
        w_codes = self.generator.map_ws(z_codes, None, self.truncation_psi, self.truncation_cutoff)[:, 0, :]
        return z_codes, w_codes

    def sample_full_ws(self, num_samples, device):
        """Samples Z-space latent codes and maps to W-space."""
        z_codes = torch.randn(size=(num_samples, 512)).to(device)
        w_codes = self.generator.map_ws(z_codes, None, self.truncation_psi, self.truncation_cutoff)
        return z_codes, w_codes


class ImageGenerator:
    """Handles StyleGAN-based forward passes and image generation."""

    def __init__(self, generator, device):
        self.generator = generator
        self.device = device
        self.psi = 0.7

    def generate_images(self, latent_codes, save_dir):
        """Generates images from latent codes and saves them."""
        os.makedirs(save_dir, exist_ok=True)
        images = []
        for i, z in enumerate(latent_codes):
            with torch.no_grad():
                img, ws = self.generator.forward(z.unsqueeze(0), None, self.psi, 8, noise_mode='const')
            images.append(ws)
            img = convert_images_to_uint8(img.cpu().numpy(), nchw_to_nhwc=True)[0]
            Image.fromarray(img).save(os.path.join(save_dir, f"{i}.jpg"))
        return images

    # def apply_directions(self, dataloader: torch.utils.data.DataLoader, directions, edit_dist, output_dir):
    #     """Applies directions to latent codes and saves manipulated images."""
    #     os.makedirs(output_dir, exist_ok=True)
    #     pbar = tqdm.tqdm(dataloader)
    #
    #     sample = []
    #     processed_idx = 0
    #     for batch_idx, code_batch in enumerate(pbar):
    #         for di, direction in enumerate(directions):
    #             with torch.no_grad():
    #                 walk_image, _ = self.generator.forward_walking(code_batch, None, direction, edit_dist, None, self.psi, 8,
    #                                                         noise_mode='const')
    #             # walk_image = convert_images_to_uint8(img.cpu().numpy(), nchw_to_nhwc=True)
    #             for img_idx, img in enumerate(walk_image):
    #                 image_index = processed_idx + img_idx
    #                 sample.append(img.detach().cpu())
    #
    #                 img = convert_images_to_uint8(img.unsqueeze(0).cpu().numpy(), nchw_to_nhwc=True)
    #                 Image.fromarray(img[0]).save(os.path.join(output_dir, f"{image_index}-{di}.jpg"))
    #         processed_idx += len(code_batch)
    #
    #     # Save summary grid image
    #     samples = torch.stack(sample)[:dataloader.batch_size*len(directions)]
    #     print(samples.shape)
    #     samples = samples.view(dataloader.batch_size, len(directions), samples.shape[1], samples.shape[2], samples.shape[3],)
    #     samples = samples[:4]  # 4 codes
    #     samples = samples.swapdims(0, 1)
    #     print(samples.shape)
    #     samples = samples.flatten(start_dim=0, end_dim=1)
    #     print(samples.shape)
    #     torchvision.utils.save_image(samples, os.path.join(output_dir, f"00-00.jpg"),
    #                                  nrow=len(directions),
    #                                  normalize=True)
    def apply_directions(self, dataloader: torch.utils.data.DataLoader, directions, edit_dist, output_dir):
        """Applies directions to latent codes and saves manipulated images."""
        os.makedirs(output_dir, exist_ok=True)
        pbar = tqdm.tqdm(dataloader)

        sample = []
        processed_idx = 0
        for batch_idx, code_batch in enumerate(pbar):
            for di, direction in enumerate(directions):
                with torch.no_grad():
                    walk_image, _ = self.generator.forward_walking(
                        code_batch, None, direction, edit_dist, None, self.psi, 8, noise_mode='const')
                for img_idx, img in enumerate(walk_image):
                    image_index = processed_idx + img_idx
                    sample.append(img.detach().cpu())

                    img_to_save = convert_images_to_uint8(img.unsqueeze(0).cpu().numpy(), nchw_to_nhwc=True)
                    Image.fromarray(img_to_save[0]).save(os.path.join(output_dir, f"{image_index}-{di}.jpg"))
            processed_idx += len(code_batch)

        # Save summary grid image
        samples = torch.stack(sample)[:dataloader.batch_size * len(directions)]
        # Reshape using the correct ordering.
        # Our flat list is ordered as (direction, code) so first view as (len(directions), batch_size, C, H, W)
        samples = samples.view(len(directions), dataloader.batch_size, samples.shape[1], samples.shape[2], samples.shape[3])
        # Transpose so that rows correspond to codes and columns to directions.
        samples = samples.transpose(0, 1)  # now shape: (batch_size, len(directions), C, H, W)
        # Optionally, take only the first 4 codes
        samples = samples[:4]
        # Flatten to shape (4 * len(directions), C, H, W) for saving.
        samples = samples.flatten(0, 1)
        torchvision.utils.save_image(samples, os.path.join(Path(output_dir).parent, "samples.jpg"),
                                     nrow=len(directions),
                                     normalize=True)


class DatasetManager:
    """Handles dataset organization and saving/loading operations."""

    def __init__(self, output_root):
        self.proj_root = os.path.join(os.path.dirname(Path(__file__).parent), 'resources', 'experiments')
        self.output_root = os.path.join(self.proj_root, output_root)
        self.code_output = os.path.join(self.output_root, "codes")
        self.walked_output = os.path.join(self.output_root, "walked")
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.code_output, exist_ok=True)
        os.makedirs(self.walked_output, exist_ok=True)

    def save_tensor(self, tensor, filename):
        torch.save(tensor.cpu(), os.path.join(self.output_root, filename))
