import shutil
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from utils.logging_config import setup_logging

class CVFoldGenerator:
    """
    Generates k-fold cross-validation splits and copies images 
    into fold-specific train, validation, and test subfolders.
    """

    def __init__(self,
                dataset_name: str, 
                num_folds: int,
                random_state: int,
                ):
         """
        Args:
            dataset_name: Name of the dataset folder
            num_folds: Number of folds for k-fold CV
            random_state: Random seed for reproducibility
        """

        self.input_dir = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name}"
        self.output_dir = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name} {num_folds}-fold CV"

        self.num_folds = num_folds
        self.random_state = random_state

        self.logger = logging.getLogger(__name__)
    
    def load_data(self):
        """ Loads image paths and its labels. """
        self.image_paths = [Path(p) for p in glob(str(self.input_dir / '*' / '*'))]
        self.labels = [p.parent.name for p in self.image_paths]

    def create_folds(self):
        """ Creates folds for k-fold CV. """

        self.folds = []

        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_state)

        for _, (train_idx, val_test_idx) in enumerate(skf.split(self.image_paths, self.labels), start=1):

            # Split into train and test_val
            train_paths = np.array(self.image_paths)[train_idx]
            train_labels = np.array(self.labels)[train_idx]
            val_test_paths = np.array(self.image_paths)[val_test_idx]
            val_test_labels = np.array(self.labels)[val_test_idx]

            # Further split val_test into validation and test
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                val_test_paths, val_test_labels,
                test_size=0.50,
                stratify=val_test_labels,
                random_state=self.random_state
            )

            # Append the fold data
            self.folds.append({
                'train': (train_paths, train_labels),
                'val': (val_paths, val_labels),
                'test': (test_paths, test_labels)
            })

    def _copy_images(self, image_paths, labels, target_dir):
        """ Copies images to train/val/test subfolders for each iteration of CV. """ 

        target_dir = Path(target_dir)

        for path, label in tqdm(zip(image_paths, labels), total=len(image_paths), desc="Copying images"):
            target_label_dir = target_dir / label
            target_label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, target_label_dir)

    def __call__(self):
        self.logger.info("Generating subfolders for each iteration of k-fold CV.")

        self.load_data()
        self.create_folds()

        pbar = tqdm(self.folds)

        self.logger.info("Copying images...")

        for fold_idx, fold in enumerate(pbar, start=1):
            pbar.set_description(f"Fold {fold_idx}/{self.num_folds}")

            # Define path to current fold directory
            fold_dir = self.output_dir / f'fold_{fold_idx}'

            # Create train/val/test directories
            (fold_dir / "train").mkdir(parents=True, exist_ok=True)
            (fold_dir / "validation").mkdir(parents=True, exist_ok=True)
            (fold_dir / "test").mkdir(parents=True, exist_ok=True)

            # Copy images
            self._copy_images(fold['train'][0], fold['train'][1], fold_dir / "train")
            self._copy_images(fold['val'][0], fold['val'][1], fold_dir / "validation")
            self._copy_images(fold['test'][0], fold['test'][1], fold_dir / "test")

        self.logger.info("Copying finished!")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate k-fold cross-validation splits.")

    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Directory name of dataset")
    parser.add_argument("-k", "--num_folds", type=int, default=5, help="Number of folds")
    parser.add_argument("-r", "--random_state", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    setup_logging()

    args = parse_args()

    cv_fold_generator = CVFoldGenerator(
        dataset_name=args.dataset_name,
        num_folds=args.num_folds,
        random_state=args.random_state,
    )
    
    cv_fold_generator()
    