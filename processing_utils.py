from abc import abstractmethod

import PIL.PngImagePlugin
from PIL import Image

import open_clip
import numpy as np
import torch
import torchvision.transforms


class FeatureTransforms:
    def __init__(self, device):
        self.device = device

    def postprocess_gan_output(self, output):
        """
        output of the synthesis model is not capped between [-1, 1].
        We want to assume a range between [-1, 1], so you cut out all values that are outside the range.


        Args:
            output:

        Returns:

        """
        img = (output * 127.5)
        img = img + 128
        img = img.clamp(0, 255).to(torch.uint8)
        return img

    @abstractmethod
    def transform_image(self, image):
        pass

    def transform_walk(self, walk):
        assert len(walk.shape) == 4  # Assert sequence of images

        transformed = torch.zeros_like(walk)
        for step_idx, image in enumerate(walk):
            image_feature = self.transform_image(image)
            transformed[step_idx] = image_feature

        return transformed

    @abstractmethod
    def transform_bulk(self, walk_images):
        pass


class EmptyTransform(FeatureTransforms):
    def transform_image(self, image):
        return image

    def transform_walk(self, walk):
        return walk

    def transform_bulk(self, walk_images):
        return walk_images


class CLIPTransform(FeatureTransforms):
    def __init__(self, device, clip_model='ViT-B/32'):
        FeatureTransforms.__init__(self, device)

        # PIL
        self.toPIL = torchvision.transforms.ToPILImage()

        # Load clip model
        # Preprocess (448, 448)
        # mean=(0.48145466, 0.4578275, 0.40821073),
        # std=(0.26862954, 0.26130258, 0.27577711))
        # self.model, self.preprocess = clip.load(clip_model, device=self.device)
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name='ViT-H-14-378-quickgelu', pretrained='dfn5b')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name='ViT-H-14-quickgelu', pretrained='dfn5b')
        self.model = self.model.cuda()
        self.output_dim = self.model.visual.output_dim

    def preprocess_image(self, image):
        # assert len(image.shape) == 3  # assert single image at a time.
        if type(image) == torch.Tensor:
            t_image = Image.fromarray(image.numpy())
        elif type(image) == np.ndarray:
            t_image = Image.fromarray(image)
        elif type(image) == Image.Image:
            t_image = image
        elif type(image) == PIL.PngImagePlugin.PngImageFile:
            t_image = image
        elif type(image) == PIL.JpegImagePlugin.JpegImageFile:
            t_image = image
        else:
            raise Exception("Wrong image type in image transform")

        t_image = self.preprocess(t_image)
        if len(t_image.shape) == 3:
            t_image = t_image.unsqueeze(0)

        return t_image

    def transform_preprocessed_images(self, images):
        assert len(images.shape) == 4

        t_image = images.to(self.device)
        with torch.no_grad():
            feature = self.model.encode_image(t_image).cpu()

        return feature

    def transform_image(self, image):
        # assert len(image.shape) == 3  # assert single image at a time.
        if type(image) == torch.Tensor:
            t_image = Image.fromarray(image.numpy())
        elif type(image) == np.ndarray:
            t_image = Image.fromarray(image)
        elif type(image) == Image.Image:
            t_image = image
        elif type(image) == PIL.PngImagePlugin.PngImageFile:
            t_image = image
        elif type(image) == PIL.JpegImagePlugin.JpegImageFile:
            t_image = image
        else:
            raise Exception("Wrong image type in image transform")

        t_image = self.preprocess(t_image)
        if len(t_image.shape) == 3:
            t_image = t_image.unsqueeze(0)

        t_image = t_image.to(self.device)
        with torch.no_grad():
            feature = self.model.encode_image(t_image).cpu()

        return feature

    def transform_bulk(self, walk_images):
        """
        Check walk image shapes and reshape
        Args:
            walk_images:

        Returns:

        """
        # [code, direction(s), step, channel, width, height]

        transformed = np.empty(shape=[
            walk_images.shape[0],
            walk_images.shape[1],
            walk_images.shape[2],
            self.output_dim
        ])
        for code_idx, code in enumerate(walk_images):
            for dir_idx, direction in enumerate(code):
                walk_features = self.transform_walk(direction).numpy()
                transformed[code_idx, dir_idx] = walk_features

        return transformed

