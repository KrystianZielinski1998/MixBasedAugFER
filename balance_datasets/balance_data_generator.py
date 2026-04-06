import json
import random
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
import logging
import argparse
from utils.logging_config import setup_logging


class BalanceDataGenerator:
    """Generates JSON files for balancing datasets by removing or augmenting images."""

    def __init__(self, 
                 dataset_name: str, 
                 count_target: int, 
                 seed: int = 42):
        """
        Args:
            dataset_name: Name of the dataset folder
            count_target: Target number of images per class
            seed: Random seed for reproducibility
        """
        # Path to the folder with k-fold CV data
        self.input_dir = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name} 5-fold CV" 
        
        self.count_target = count_target  
        self.seed = seed  

        # Logger setup
        self.logger = logging.getLogger(__name__) 

        # Ensure the dataset directory exists
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.input_dir}")


    def __call__(self):
        """
        Creates two JSON files:
        - `to_remove_*.json` for majority classes to remove extra images
        - `to_augment_*.json` for minority classes to augment images
        """
        to_remove = {} 
        to_augment = {}  
        rng = random.Random(self.seed)  

        # Get all fold directories sorted
        folds = sorted([p for p in self.input_dir.iterdir() if p.is_dir()])  # list of fold directories
        num_folds = len(folds)  

        # Outer progress bar for folds
        pbar = tqdm(folds, total=num_folds)
        for fold_idx, fold in enumerate(pbar, start=1):
            pbar.set_description(f"Fold {fold_idx}/{num_folds}")  # dynamic fold description

            fold_name = fold.name  # e.g., 'fold_1'
            train_path = fold / "train"  # path to train subfolder
            class_dirs = [p for p in train_path.iterdir() if p.is_dir()]  # list of class directories in train

            to_remove[fold_name] = {}  # initialize dictionary for this fold
            to_augment[fold_name] = {}  # initialize dictionary for this fold

            # Inner progress bar for classes in the fold
            for class_dir in class_dirs:
                class_name = class_dir.name  # class label
                images = sorted([p for p in class_dir.iterdir() if p.is_file()])  # list of image paths
                num_images = len(images)  # number of images in class

                # Majority classes: remove extra images
                if num_images > self.count_target:
                    n_remove = num_images - self.count_target  # number of images to remove
                    img_to_remove = rng.sample(images, n_remove)  # randomly select images to remove
                    to_remove[fold_name][class_name] = [
                        str(img.relative_to(self.input_dir)) for img in img_to_remove  # relative paths for JSON
                    ]

                # Minority classes: create balanced pairs for augmentation
                elif num_images < self.count_target:
                    n_needed = self.count_target - num_images  # how many extra images needed
                    pairs = []

                    if n_needed >= num_images:
                        repeats = n_needed // num_images
                        remainder = n_needed % num_images
                        image_pool = []
                        for idx, img in enumerate(images):
                            times = repeats + (1 if idx < remainder else 0)
                            image_pool.extend([img] * times)
                        rng.shuffle(image_pool)
                    else:
                        # n_needed < num_images: pick n_needed images without repeating
                        image_pool = rng.sample(images, n_needed)

                    # Generate pairs by pairing each image with a random other image
                    for img in image_pool:
                        partner = rng.choice([p for p in images if p != img])
                        pairs.append((img, partner))

                    to_augment[fold_name][class_name] = [
                        [str(p.relative_to(self.input_dir)) for p in pair] for pair in pairs
                    ]

        # Save JSON files
        augment_json_path = Path(f"to_augment_{self.input_dir.name}.json")  # JSON for augmentation
        remove_json_path = Path(f"to_remove_{self.input_dir.name}.json")  # JSON for removal

        with augment_json_path.open("w") as f:
            json.dump(to_augment, f, indent=2)

        with remove_json_path.open("w") as f:
            json.dump(to_remove, f, indent=2)

        self.logger.info(f"Saved augmentation JSON to {augment_json_path}")  
        self.logger.info(f"Saved removal JSON to {remove_json_path}") 


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate JSON data with image pair paths to augment minority classes"
    )
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Directory name of dataset")
    parser.add_argument("-c", "--count_target", type=int, required=True, help="Number of images to keep per class")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()  

    args = parse_args()

    balance_data_generator = BalanceDataGenerator(
        dataset_name=args.dataset_name,
        count_target=args.count_target,
        seed=args.seed,
    )
    
    balance_data_generator()





    



            
    
    
    