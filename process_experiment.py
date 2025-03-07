import json
import os
import uuid

import natsort
import torch
import tqdm
import shutil

from PIL import Image
from argparse import ArgumentParser
from processing_utils import CLIPTransform


def infer_dataset_size(filenames, old=False):
    """
    Given a list of file names, infer number of codes and directions.

    Equal number of directions must be applied to every code.

    Args:
        filenames:

    Returns:

    """
    try:
        filenames.remove(".DS_Store")
    except:
        pass

    code_ids = natsort.natsorted(list(set([fn.split("-")[0] for fn in filenames])))
    directions = natsort.natsorted([fn for fn in filenames if fn.split("-")[0] == code_ids[0]])

    if old:
        return len(code_ids), (len(directions)/2)
    else:
        return len(code_ids), len(directions)


class ImageSetProcessorV3:
    """
    Processes slimmed down dataset folder that does not use excessive space.

    Further slimmed down version of V2. [code, direction, feature] - I don't need the step dimension.
    """
    def __init__(self, dataset_root, feature_extraction_method, device, older=False, cache_size=100):
        self.device = device
        self.older = older  # For conversion of an older dataset
        self.cache_size = cache_size  # Number of items to cache before writing to disk
        self.cache_counter = 0  # Counter to track how many items are in cache
        self.cache_dir = os.path.join(dataset_root, 'cache')  # Directory to store cached tensors

        # Folders
        self.dataset_root = dataset_root
        self.code_fp = os.path.join(dataset_root, 'codes')
        self.walk_fp = os.path.join(dataset_root, 'walked')
        self.tensor_fp = os.path.join(dataset_root, 'tensors')
        self.settings_fp = os.path.join(dataset_root, 'setting.json')
        os.makedirs(self.code_fp, exist_ok=True)
        os.makedirs(self.walk_fp, exist_ok=True)
        os.makedirs(self.tensor_fp, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Size variable
        self.n_code, self.n_direction = infer_dataset_size(sorted(os.listdir(self.walk_fp)))
        print(self.n_code, self.n_direction)

        self.feature_extraction_method, self.feature_extractor = feature_extraction_method, None
        if feature_extraction_method == 'clip':
            # self.feature_extractor = CLIPTransform(self.device, clip_model='ViT-L/14')
            self.feature_extractor = CLIPTransform(self.device, clip_model='ViT-H-14-378-quickgelu')
        else:
            raise Exception(f"Feature Extraction Method {feature_extraction_method} is not implemented.")

        # Data containers
        self.code_features = torch.empty(size=(self.n_code, self.feature_extractor.output_dim))
        self.features = []  # Use a list to store features temporarily to avoid large memory usage

    def _process_codes(self):
        """
        Process codes before processing walked version of them.

        Returns:

        """
        # Read codes
        code_fps = natsort.natsorted(os.listdir(self.code_fp))
        code_fps = [os.path.join(self.code_fp, c) for c in code_fps]
        # print("Code filenames: ", code_fps)
        for i, code_fp in enumerate(code_fps):
            image = Image.open(code_fp)
            feature = self.feature_extractor.transform_image(image)[0]  # Just one image at-a-time here.
            self.code_features[i] = feature

    def _process_walks(self):
        """
        Main process.

        Returns:

        """
        pbar = tqdm.tqdm(total=self.n_code * self.n_direction)

        for code_idx in range(self.n_code):
            for direction_idx in range(self.n_direction):
                pbar.update(1)
                fp_b = os.path.join(self.walk_fp, f'{code_idx}-{direction_idx}.jpg')  # Walked image
                feature = self.feature_extractor.transform_image(Image.open(fp_b))[0]
                self.features.append(feature)

                # Cache features if cache size is reached
                if len(self.features) >= self.cache_size:
                    self._cache_features()

        # Cache any remaining features
        if len(self.features) > 0:
            self._cache_features()

        # Consolidate cached features
        self._consolidate_cache()

        # Save original code features
        torch.save(self.code_features, os.path.join(self.tensor_fp, 'features_original.pt'))

    def _cache_features(self):
        """
        Cache the current features to disk and reset the feature list.
        """
        cache_fp = os.path.join(self.cache_dir, f'features_cache_{self.cache_counter}.pt')
        torch.save(torch.stack(self.features), cache_fp)
        self.features = []
        self.cache_counter += 1

    def _consolidate_cache(self):
        """
        Consolidate all cached feature files into a single tensor file.
        """
        cached_tensors = []
        for cache_file in natsort.natsorted(os.listdir(self.cache_dir)):
            if cache_file.startswith('features_cache_') and cache_file.endswith('.pt'):
                cache_fp = os.path.join(self.cache_dir, cache_file)
                cached_tensors.append(torch.load(cache_fp))

        consolidated_features = torch.cat(cached_tensors, dim=0)

        # Reshape
        consolidated_features = consolidated_features.reshape(shape=(self.n_code, self.n_direction, -1))
        print(consolidated_features.shape)

        torch.save(consolidated_features, os.path.join(self.tensor_fp, 'features.pt'))

        # Clean up cache files
        shutil.rmtree(self.cache_dir)

    def run(self):
        print("Processing: ", self.dataset_root)
        self._process_codes()
        self._process_walks()

    def conversion(self):
        """
        Converting an older dataset.

        1. Convert .png to .jpg
        2. Remove 'images' folder
            What are these needed for? Only in case my starting images are different for all walks. Do I have a usecase
            for that right now? I do not have that usecase in the foreseeable future.

        Returns:

        """
        walk_fns = os.listdir(self.walk_fp)
        walk_fps = [os.path.join(self.walk_fp, w) for w in walk_fns if w.endswith('.png')]

        bar = tqdm.tqdm(walk_fps)
        for wfp in bar:
            code_idx, direction_idx = wfp.split('/')[-1].split('-')[0], wfp.split('/')[-1].split('-')[1].split('.')[0]
            new_wfp = os.path.join(self.walk_fp, f'{code_idx}-{direction_idx}.jpg')

            img = Image.open(wfp).save(new_wfp)
            os.remove(wfp)

        # Remove 'images' folder
        images_fp = os.path.join(self.dataset_root, 'images')
        shutil.rmtree(images_fp)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--root', '-r', type=str)
    parser.add_argument('--feature_extraction_method', '-f', default='clip', help='Model for feature extraction')
    parser.add_argument('--conversion', '-c', type=bool, default=False)
    return parser.parse_args()


if __name__ == '__main__':
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    args = parse_args()
    dataset_processor = ImageSetProcessorV3(args.root, args.feature_extraction_method, device, cache_size=5000)

    if args.conversion == False:
        dataset_processor.run()
    else:
        dataset_processor.conversion()
