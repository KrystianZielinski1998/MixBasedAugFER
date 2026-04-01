import logging
from pathlib import Path
import json
from tqdm import tqdm
from utils.logging_config import setup_logging

class RandomRemover:
    """Removes images from majority classes based on a pre-generated JSON file."""

    def __init__(self, dataset_name: str):
        """
        Args:
            dataset_name: Name of the dataset directory
        """
        
        # Input dataset directory (expects "dataset_name 5-fold CV")
        self.input_dir = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name} 5-fold CV"

        # JSON file containing list of images to remove for each CV fold
        self.json_balance_data = Path(f"balance_datasets/to_remove_{self.input_dir.name}.json")

        # Logger setup
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RandomRemover initialized for dataset: {dataset_name}")

    def __call__(self):
        """Remove images listed in the JSON file from the dataset."""

        # Check if JSON exists
        if not self.json_balance_data.exists():
            self.logger.error(f"JSON file not found: {self.json_balance_data}")
            raise FileNotFoundError(f"JSON file not found: {self.json_balance_data}")

        # Load JSON file
        with open(self.json_balance_data, "r") as f:
            img_to_remove = json.load(f)

        # Convert dict items to a list to use len() for fold count
        folds = list(img_to_remove.items())
        num_folds = len(folds)

        # Fold-level progress bar
        pbar = tqdm(folds, desc="Folds", unit="fold")
        self.logger.info("Removing images...")

        # Iterate over folds with dynamic description
        for fold_idx, (fold_name, fold_data) in enumerate(pbar, start=1):
            pbar.set_description(f"Fold {fold_idx}/{num_folds}: {fold_name}")
            self.logger.info(f"Processing fold {fold_idx}/{num_folds}: {fold_name}")

            # Iterate over classes in the fold
            for class_name, img_rel_paths in fold_data.items():
                self.logger.info(f"Removing images from class: {class_name}")

                # Progress bar for each class
                for img_rel_path in tqdm(img_rel_paths, desc=f"{fold_name}/{class_name}", leave=False):
                    img_path = self.input_dir / img_rel_path
                    if img_path.exists():
                        img_path.unlink()
                    else:
                        self.logger.warning(f"File not found, cannot remove: {img_path}")

        self.logger.info("Image removal completed.")

            
def args_parser():
    parser = argparse.ArgumentParser(description="Remove images based on the balance JSON file")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Directory name of dataset")
    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()

    args = args_parser()

    remover = RandomRemover(dataset_name=args.dataset_name)

    remover()






    



            
    
    
    