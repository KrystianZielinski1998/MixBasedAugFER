import os
from PIL import Image
import imagehash
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import shutil
import logging
import networkx as nx
import numpy as np
from utils.logging_config import setup_logging
from pathlib import Path

class VectorizedDuplicateFinder:
    """ Finds and groups duplicate images using hashing method. """

    def __init__(self,
                 dataset_name: str,
                 nearest_neighbors: int,
                 hamming_threshold: int,
                 hash_method: str,
                 hash_size: int):
        """
        Args:
            dataset_name: Name of the dataset directory
            nearest_neighbors: How many neighbors to check for each image.
            hamming_threshold: Max Hamming distance to consider duplicate.
            hash_method: "phash", "whash", or "dhash".
            hash_size: Hash size.
        """  

        self.input_dir = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name}"
        self.output_dir = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name} duplicates"
        self.nearest_neighbors = nearest_neighbors
        self.hamming_threshold = hamming_threshold
        self.hash_method = hash_method
        self.hash_size = hash_size

        self.logger = logging.getLogger(__name__)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_hash(self, img_path):
        """
        Compute a hash vector for an image using the selected hash method.

        Returns:
            hash_vector (np.ndarray of bool): flattened hash, shape (hash_size**2,)
        """

        img = Image.open(img_path).convert("L")

        match self.hash_method:
            case "phash":
                h = imagehash.phash(img, hash_size=self.hash_size)
            case "whash":
                h = imagehash.whash(img, hash_size=self.hash_size)
            case "dhash":
                h = imagehash.dhash(img, hash_size=self.hash_size)
            case _:
                raise ValueError(f"Unknown hash method: {self.hash_method}")

        hash_vector = h.hash.flatten() 

        return hash_vector

    def gather_images(self):
        """ Gathers all image paths, classes, and computes hashes. """
        file_paths = []
        classes = []
        hashes = []

        self.logger.info(f"Computing hashes for all images using {self.hash_method}...")

        # Iterate over class directories
        for cls_dir in self.input_dir.iterdir():
            if not cls_dir.is_dir():
                continue

            cls = cls_dir.name
            files = list(cls_dir.iterdir())

            for img_path in tqdm(files, desc=f"Computing hash for images in class: {cls}"):
                if not img_path.is_file():
                    continue

                try:
                    h_vector = self.compute_hash(img_path)
                except Exception as e:
                    self.logger.warning(f"Could not open {img_path}: {e}")
                    continue

                file_paths.append(img_path)
                classes.append(cls)
                hashes.append(h_vector)

        return file_paths, classes, np.array(hashes)

    def find_duplicates(self):
        """ Finds duplicates and group them into unique sets. """

        self.logger.info("Starting duplicate search...")

        file_paths, classes, hashes = self.gather_images()
        path_to_class = dict(zip(file_paths, classes))

        n_images = len(hashes)
        self.logger.info(f"Total images processed: {n_images}")

        if n_images == 0:
            self.logger.warning("No images found to process!")
            return [], [], {}

        # Nearest neighbor search with Hamming distance
        self.logger.info("Performing nearest neighbor search...")
        nn = NearestNeighbors(n_neighbors=self.nearest_neighbors + 1, metric='hamming')
        nn.fit(hashes)
        distances, indices = nn.kneighbors(hashes)

        n_bits = self.hash_size ** 2
        distances_bits = distances * n_bits

        # Build a graph where edges connect duplicates
        G = nx.Graph()
        G.add_nodes_from(file_paths)

        for i in range(n_images):
            for j in range(1, self.nearest_neighbors + 1):  # skip self
                if distances_bits[i, j] <= self.hamming_threshold:
                    idx = indices[i, j]
                    G.add_edge(file_paths[i], file_paths[idx])

        # Each connected component is one duplicate set
        duplicate_groups = [
            [
                {"path": fp, "class": path_to_class[fp]}
                for fp in comp
            ]
            for comp in nx.connected_components(G)
            if len(comp) > 1
        ]

        self.logger.info(f"Found {len(duplicate_groups)} duplicate groups.")

        return duplicate_groups

    def copy_duplicates(self, duplicate_groups):
        """ Copies duplicates into subfolders for each original image. """

        self.logger.info(f"Copying duplicates to output directory: {self.output_dir}")

        for group in tqdm(duplicate_groups, desc="Copying duplicates"):
            original = group[0]
            orig_path = Path(original["path"])
            orig_class = original["class"]

            orig_name = orig_path.stem
            duplicate_dir = self.output_dir / orig_name
            duplicate_dir.mkdir(parents=True, exist_ok=True)

            for item in group:
                img_path = Path(item["path"])
                img_class = item["class"]

                filename = f"{img_class}_{img_path.name}"
                dest_path = duplicate_dir / filename

                if not dest_path.exists():
                    shutil.copy(img_path, dest_path)

        self.logger.info("Copying duplicates finished.")

    def __call__(self):
        
        duplicate_groups = self.find_duplicates()
        self.copy_duplicates(duplicate_groups)
  
if __name__ == "__main__":
    setup_logging()
    
    vectorized_duplicate_finder = VectorizedDuplicateFinder(
        dataset_name="FER+",
        nearest_neighbors=20,
        hamming_threshold=3,
        hash_method="phash", 
        hash_size=8
    )

    vectorized_duplicate_finder()
    