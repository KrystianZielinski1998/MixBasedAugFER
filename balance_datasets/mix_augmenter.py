from pathlib import Path
import json
from tqdm import tqdm
import argparse
import cv2
from dataclasses import dataclass
from typing import Callable
import multiprocessing as mp
import os
import logging

from augmentations.rowmix import RowMix
#from augmentations.puzzlemix import PuzzleMix
from utils.logging_config import setup_logging

@dataclass
class Task:
    """ Dataclass for task passed to worker for multiprocessing. """
    img1_path: str
    img2_path: str
    output_path: str
    augment_method_cls: Callable

def worker(task: Task):
    """ Worker function for generating new image sample. """
    img1 = cv2.imread(task.img1_path)
    img2 = cv2.imread(task.img2_path)

    output_image = task.augment_method_cls(img1, img2)

    cv2.imwrite(task.output_path, output_image)

class MixAugmenter:
    """
    Generates new image samples using three Mix-Based augmentations: ColumnMix, RowMix, PuzzleMix.
    """

    def __init__(self, 
                dataset_name: str, 
                augment_method: str, 
                num_workers: int):
                
        self.input_dir = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name} 5-fold CV"
        self.json_balance_data = Path(f"balance_datasets/to_augment_{self.input_dir.name}.json")

        self.augment_method = augment_method
        self.num_workers = num_workers or os.cpu_count()
        self.logger = logging.getLogger(__name__)

        # Select augmentation method
        match self.augment_method:
            case "rowmix":
                self.augment_method_cls = RowMix()
            case "puzzlemix":
                self.augment_method_cls = PuzzleMix()
            case _:
                raise ValueError(f"Unknown augment method: {self.augment_method}")

        self.logger.info(f"Using augment method: {self.augment_method}")
        self.logger.info(f"Number of workers: {self.num_workers}")

    def __call__(self):
        """Run augmentation pipeline."""

        if not self.json_balance_data.exists():
            raise FileNotFoundError(f"JSON not found: {self.json_balance_data}")

        with open(self.json_balance_data, "r") as f:
            img_to_augment = json.load(f)

        folds = list(img_to_augment.items())
        pbar = tqdm(folds, desc="Folds", unit="fold")

        for fold_idx, (fold_name, fold_data) in enumerate(pbar, start=1):
            pbar.set_description(f"Fold {fold_idx}/{len(folds)}")

            for class_name, pairs in fold_data.items():
                self.logger.debug(f"Class: {class_name} | Pairs: {len(pairs)}")

                tasks = self.create_tasks(pairs)
                self.process_parallel(tasks)

        self.logger.info("Augmentation completed.")

    def create_tasks(self, pairs):
        """Create multiprocessing tasks from image pairs."""
        tasks = []

        for img1_rel_path, img2_rel_path in pairs:
            img1_path = self.input_dir / img1_rel_path
            img2_path = self.input_dir / img2_rel_path

            img1_name = Path(img1_rel_path).stem
            img2_name = Path(img2_rel_path).stem

            output_dir = img1_path.parent
            filename = f"{self.augment_method}_{img1_name}_{img2_name}.png"
            output_path = output_dir / filename

            tasks.append(
                Task(
                    img1_path=str(img1_path),
                    img2_path=str(img2_path),
                    output_path=str(output_path),
                    augment_method_cls=self.augment_method_cls,
                )
            )

        return tasks

    def process_parallel(self, tasks):
        """Process tasks in parallel using multiprocessing."""

        with mp.Pool(self.num_workers) as pool:
            for _ in pool.imap_unordered(worker, tasks):
                pass


def args_parser():
    parser = argparse.ArgumentParser(description="Generate augmented images using mix-based methods")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "-a",
        "--augment_method",
        type=str,
        choices=["rowmix", "columnmix", "puzzlemix"],
        required=True,
        help="Augmentation method",
    )
    parser.add_argument("-w", "--num_workers", type=int, help="Number of parallel workers")

    return parser.parse_args()


if __name__ == "__main__":
    setup_logging()

    args = args_parser()

    augmenter = MixAugmenter(
        dataset_name=args.dataset_name,
        augment_method=args.augment_method,
        num_workers=args.num_workers,
    )

    augmenter()





    



            
    
    
    