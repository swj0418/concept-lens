import os
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import open_clip
from PIL import Image
import torchvision.transforms as transforms

def direction_lstsq(direction, affine_weight):
    # [M, N] / [M, K] => [concat(output), input] / [concat(output), n direction]
    V, res, rank, S = np.linalg.lstsq(affine_weight.detach().cpu().numpy(),
                                      direction.detach().cpu().numpy().T, rcond=None)
    return V, res, rank, S  # Columns are corresponding directions (or codes)

def pca_direction(code_collection, top_dir, skip_first=True):
    # PCA
    mean_adj_w = code_collection - torch.mean(code_collection, dim=0, keepdim=True)  # [code, dim]
    w_cov = mean_adj_w.T @ mean_adj_w
    eigval, eigvec = np.linalg.eigh(w_cov.detach().cpu().numpy())
    eigvec = torch.tensor(eigvec).T
    eigvec = torch.flip(eigvec, dims=[0])
    if top_dir is not None:
        eigvec = eigvec[:top_dir, :]
    return torch.tensor(eigvec, dtype=torch.float)

def _compute_sefa_directions(style_weights, layers: list, top_dir: int = None,
                             skip_first: bool = True, drop_cut: int = None):
    """
    Computes SVD-based directions for the specified layers.
    """
    # Concatenate weights from specified layers.
    weights = torch.cat([layer_weight for i, layer_weight in enumerate(style_weights.values())
                         if i in layers])
    # Randomly select a subset of rows.
    drop_indices = torch.tensor(np.random.choice(weights.shape[0], drop_cut, replace=False),
                                dtype=torch.long)
    weights = weights[drop_indices]
    # Compute SVD.
    U, S, Vh = torch.linalg.svd(weights, full_matrices=False)
    offset = 2 if skip_first else 1
    if top_dir is not None:
        eigvec = Vh[offset - 1: top_dir + (offset - 1), :].T
    else:
        eigvec = Vh.T
    return eigvec  # [dim, directions]

def load_clip_model(device):
    """
    Loads a pre-trained OpenCLIP model and the corresponding preprocessing transform.
    """
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.to(device)
    model.eval()
    return model, preprocess

def get_clip_text_embedding(concept, clip_model, device):
    """
    Returns the normalized CLIP text embedding for a given concept.
    """
    # Tokenize the input text.
    text = open_clip.tokenize([concept]).to(device)
    with torch.no_grad():
        text_embedding = clip_model.encode_text(text)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    return text_embedding

def get_clip_image_embedding(image, clip_model, preprocess, device):
    """
    Returns the normalized CLIP image embedding for a given image.
    If the image is a tensor, it is first converted to a PIL image.
    """
    if isinstance(image, torch.Tensor):
        # Convert a tensor image (assumed to be in [C, H, W] format) to a PIL image.
        to_pil = transforms.ToPILImage()
        image = to_pil(image[0])
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = clip_model.encode_image(image_input)
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
    return image_embedding

class DirectionSampler:
    """Handles methods for sampling latent directions using different strategies."""

    def __init__(self, seed: int = 1, n_directions: int = 192):
        self.seed = seed
        self.n_directions = n_directions

    def get_supervised_directions(self, generator, ws_codes: torch.Tensor, concepts, device) -> torch.Tensor:
        """
        For each concept:
          1. Generates an image from each ws code.
          2. Computes CLIP image embeddings.
          3. Computes the CLIP text embedding for the concept.
          4. Uses cosine similarity to label the top-K images as positive.
          5. Trains a linear SVM on ws codes using these binary labels.
          6. Uses the SVM weight (normal to the decision boundary) as the latent direction.
        Returns a tensor of latent directions (one per concept).
        """
        # Generate images from ws_codes.
        images = []
        for ws in ws_codes:  # each ws is assumed to be a latent code of shape [16, 512] or similar.
            image = generator.forward_ws(ws.unsqueeze(0))
            images.append(image.detach().cpu())

        # Load CLIP model and preprocessing transform.
        clip_model, preprocess = load_clip_model(device)

        # Compute CLIP image embeddings.
        image_embeddings = []
        for img in images:
            emb = get_clip_image_embedding(img, clip_model, preprocess, device)
            image_embeddings.append(emb.squeeze(0))
        image_embeddings = torch.stack(image_embeddings)  # shape: [num_images, embed_dim]

        # Compute CLIP text embeddings for each concept.
        text_embeddings = []
        for concept in concepts:
            text_emb = get_clip_text_embedding(concept, clip_model, device)
            text_embeddings.append(text_emb.squeeze(0))
        text_embeddings = torch.stack(text_embeddings)  # shape: [num_concepts, embed_dim]

        directions = []
        # Define top K as 30% of the images (at least 1).
        K = max(1, int(0.3 * len(images)))
        ws_codes_np = ws_codes.detach().cpu()[:, 0, :].squeeze().numpy()  # shape: [num_images, latent_dim]

        # For each concept, compute similarities and train an SVM.
        for i, text_emb in enumerate(text_embeddings):
            # Compute cosine similarities between text embedding and each image embedding.
            sims = torch.nn.functional.cosine_similarity(text_emb.unsqueeze(0), image_embeddings)
            topk_indices = torch.topk(sims, k=K).indices  # indices of top K images

            # Create binary labels: 1 for top K images, 0 otherwise.
            labels = np.zeros(len(images))
            labels[topk_indices.cpu().numpy()] = 1

            # Train a linear SVM on the ws codes.
            clf = LinearSVC(random_state=self.seed, max_iter=10000)
            clf.fit(ws_codes_np, labels)

            # The normal vector to the decision boundary is given by clf.coef_.
            direction = clf.coef_.flatten()
            # Normalize the direction vector.
            direction = direction / np.linalg.norm(direction)
            directions.append(torch.tensor(direction, dtype=torch.float32))

        # Stack all directions and move to the specified device.
        directions = torch.stack(directions).to(device)
        return directions

    def get_centroid_directions(self, w_codes: torch.Tensor, device) -> torch.Tensor:
        """
        Clusters W-space codes and computes normalized centroid pair directions.
        """
        # Cluster into 20 centroids.
        kmeans = KMeans(n_clusters=20, random_state=self.seed).fit(w_codes.cpu().detach().numpy())
        centroids = kmeans.cluster_centers_

        torch.manual_seed(self.seed)
        pairs = []
        # Sample unique pairs until reaching the desired number of directions.
        while len(pairs) < self.n_directions:
            sample = torch.randint(low=0, high=len(centroids), size=(2,))
            if sample[0] != sample[1]:
                pairs.append(sample)

        # Compute difference vectors between centroid pairs.
        directions = [centroids[p[0]] - centroids[p[1]] for p in pairs]
        directions = torch.tensor(directions, dtype=torch.float32)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        return directions.to(device)

    def get_sefa_directions(self,
                            generator,
                            layer_range: list,
                            sampling_rate: int,
                            top_direction_per_sample: int,
                            skip_first_svd: bool,
                            device) -> torch.Tensor:
        """
        Computes SeFA-based directions by sampling SVD-based directions from a generator's style weights,
        then clustering and selecting the closest ones.
        """
        style_weights = generator.get_style_weights()
        # Concatenate weights for specified layers.
        weight = torch.cat([layer_weight for i, layer_weight in enumerate(style_weights.values())
                            if i in layer_range])
        directions = []
        # Generate candidate directions using varying drop_cut values.
        for _ in range(sampling_rate):
            drop_max = min(512, weight.shape[0])
            drop_min = max(64, drop_max - 1)
            drop_cut = np.random.randint(low=drop_min, high=drop_max)
            tmp = _compute_sefa_directions(style_weights,
                                           layers=layer_range,
                                           top_dir=top_direction_per_sample,
                                           skip_first=skip_first_svd,
                                           drop_cut=drop_cut).T
            directions.append(tmp)
        directions = torch.cat(directions)
        # Cluster the candidate directions.
        cluster = KMeans(n_clusters=self.n_directions, n_init="auto").fit(directions)
        closest_directions = []
        # For each cluster, select the candidate direction that is closest to the cluster center.
        for cluster_center in torch.tensor(cluster.cluster_centers_):
            normalized_center = cluster_center / torch.norm(cluster_center)
            sims = torch.cosine_similarity(normalized_center.unsqueeze(0), directions)
            closest_idx = torch.argmax(sims)
            closest_directions.append(directions[closest_idx])
        directions = torch.stack(closest_directions).to(device)
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)
        return directions

    def get_ganspace_directions(self, generator, w_codes, layer_range: list, device,
                                sampling_rate: int = 16, top_direction_per_sample: int = 6,
                                sub_minimum: int = 32, sub_maximum: int = 1024) -> torch.Tensor:
        # Get the style weights and biases from the generator.
        affine_weights, affine_biases = generator.get_style_weights_and_biases()
        affine_weights, affine_biases = ([list(affine_weights.values())[v].to(device) for v in range(len(affine_weights.values())) if v in layer_range],
                                         [list(affine_biases.values())[v].to(device) for v in range(len(affine_biases.values())) if v in layer_range])

        styles = torch.empty(size=(w_codes.shape[0], len(layer_range), w_codes.shape[-1]), device=device)
        for i, code in enumerate(w_codes):
            for j, (W, B) in enumerate(zip(affine_weights, affine_biases)):
                style = code @ W.T + B
                styles[i, j] = style
        styles = styles.flatten(start_dim=1)
        print("Generated styles: ", styles.shape)

        directions = []
        for t in range(sampling_rate):
            sample_indices = torch.randint(0, len(styles), size=(torch.randint(sub_minimum, sub_maximum, size=(1,)), 1))
            target = styles[sample_indices].squeeze()
            sample = pca_direction(target, top_dir=top_direction_per_sample)
            directions.append(sample)
        directions = torch.cat(directions, dim=0)

        # Mapping back to W space
        weights = torch.cat(affine_weights, dim=0)
        print("Affine weights shape: ", weights.shape)

        V, res, rank, S = direction_lstsq(directions, weights)
        directions = torch.tensor(V.T)

        # Normalize
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        return directions.to(device)
