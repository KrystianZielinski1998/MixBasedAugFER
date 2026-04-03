import os
import shutil
import logging
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from utils.logging_config import setup_logging
from pathlib import Path

class DuplicateRemover:
    """ Removes duplicates from original dataset (FER+) and resolves class conflicts using voting for duplicate images. """

    def __init__(
        self,
        dataset_name: str,
        csv_name: str
    ):
        """
        Args:
            dataset_name: Root dataset folder.
            csv_name: Path to fer2013new.csv.
        """

        self.dataset_dir = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name}"
        self.duplicates_dir = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name} duplicates"
        self.csv_path = Path(__file__).resolve().parent.parent / "preprocessing" / f"{csv_name}"
        
        # Define emotion classes
        self.EMOTIONS = [
            "neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"
        ]

        self.df = pd.read_csv(csv_path, usecols=self.EMOTIONS)
        self.votes_array = self.df.values

        self.logger = logging.getLogger(__name__)

    def _parse_filename(self, filename):
        """ Extract class and index from filename. """
        path = Path(filename)
        cls, idx = path.stem.split("_")
        return cls, int(idx)

    def _resolve_class_by_voting(self, indices):
        """Resolve final class using summed votes from CSV."""
        vote_sum = defaultdict(int)

        for idx in indices:
            if idx >= len(self.votes_array):
                continue

            row = self.votes_array[idx]

            for i, class_name in enumerate(self.EMOTIONS):
                vote_sum[class_name] += int(row[i])

        return max(vote_sum, key=vote_sum.get)

    def _process_duplicate_group(self, folder_path):
        """Process a single duplicate group folder."""
        files = [p.name for p in folder_path.iterdir() if p.is_file()]
        if not files:
            return

        parsed = []
        classes_in_group = set()

        for f in files:
            try:
                cls, idx = self._parse_filename(f)
                parsed.append((f, cls, idx))
                classes_in_group.add(cls)
            except Exception as e:
                self.logger.warning(f"Skipping malformed file {f}: {e}")

        if not parsed:
            return

        if len(classes_in_group) == 1:
            final_class = next(iter(classes_in_group))
        else:
            indices = [idx for _, _, idx in parsed]
            self.logger.info("Votes per image in conflict group:")

            for filename, cls, idx in parsed:
                if idx >= len(self.votes_array):
                    self.logger.warning(f"{filename}: index out of range")
                    continue

                votes = self.votes_array[idx].tolist()
                self.logger.info(f"{filename} : {votes}")

            final_class = self._resolve_class_by_voting(indices)
            self.logger.info(f"Resolved {classes_in_group} → {final_class}")

        rep_file, rep_class, rep_idx = parsed[0]

        rep_path = self.dataset_dir / rep_class / f"{rep_idx}.png"
        if not rep_path.exists():
            self.logger.warning(f"Representative image not found, skipping: {rep_path}")
            return

        if rep_class != final_class:
            new_path = self.dataset_dir / final_class / f"{rep_idx}.png"
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(rep_path, new_path)
            self.logger.info(f"Moved {rep_idx}: {rep_class} → {final_class}")

        for _, cls, idx in parsed[1:]:
            dup_path = self.dataset_dir / cls / f"{idx}.png"

            if dup_path.exists():
                dup_path.unlink()

    def __call__(self):
        folders = [p for p in self.duplicates_dir.iterdir() if p.is_dir()]

        self.logger.info(f"Found {len(folders)} duplicate groups.")

        for folder_path in tqdm(folders, desc="Processing duplicates"):
            self._process_duplicate_group(folder_path)

        self.logger.info("Duplicate removal complete.")

if __name__ == "__main__":
    setup_logging()
   
    duplicate_remover = DuplicateRemover(
        dataset_name="FER+",
        csv_name="fer2013new.csv"
    )

    duplicate_remover()
    