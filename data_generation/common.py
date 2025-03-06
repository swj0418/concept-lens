import os
import torch

from PIL import Image
import tqdm
from utils.image_utils import convert_images_to_uint8


class LatentCodeGenerator:
    """Handles latent code sampling and mapping."""

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

    def apply_directions(self, dataloader, directions, edit_dist, output_dir):
        """Applies directions to latent codes and saves manipulated images."""
        os.makedirs(output_dir, exist_ok=True)
        pbar = tqdm.tqdm(dataloader)
        for batch_idx, code_batch in enumerate(pbar):
            for di, direction in enumerate(directions):
                with torch.no_grad():
                    img, _ = self.generator.forward_walking(code_batch, None, direction, edit_dist, None, self.psi, 8,
                                                            noise_mode='const')
                walk_image = convert_images_to_uint8(img.cpu().numpy(), nchw_to_nhwc=True)
                for img_idx, img in enumerate(walk_image):
                    image_index = batch_idx * len(code_batch) + img_idx
                    Image.fromarray(img).save(os.path.join(output_dir, f"{image_index}-{di}.jpg"))


class DatasetManager:
    """Handles dataset organization and saving/loading operations."""

    def __init__(self, output_root):
        self.output_root = output_root
        self.code_output = os.path.join(output_root, "codes")
        self.walked_output = os.path.join(output_root, "walked")
        os.makedirs(self.output_root, exist_ok=True)
        os.makedirs(self.code_output, exist_ok=True)
        os.makedirs(self.walked_output, exist_ok=True)

    def save_tensor(self, tensor, filename):
        torch.save(tensor.cpu(), os.path.join(self.output_root, filename))
