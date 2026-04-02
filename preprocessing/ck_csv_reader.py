import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import logging
from pathlib import Path
from utils.logging_config import setup_logging


class CKcsvReader:
    """ 
    Converts pixel string read from CSV file into images and assigns them to corresponding class directory for CK+ dataset.
    """

    def __init__(self, ck_csv_name: str):
        """
        Args:
            ck_csv_name: Name of the .csv file with CK+ dataset  
        """

        self.ck_csv_path = Path(__file__).resolve().parent.parent / "preprocessing" / ck_csv_name
        self.output_dir = Path(__file__).resolve().parent.parent / "datasets" / "CK+"
        self.logger = logging.getLogger(__name__)

        self.CK_CLASSES = {
            0: "anger",
            1: "disgust",
            2: "fear",
            3: "happiness",
            4: "sadness",
            5: "surprise",
            6: "neutral",
            7: "contempt",
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _str_to_image(image_blob: str) -> Image.Image:
        """Converts string of pixels from CSV to 48x48 PIL image."""
        data = np.fromiter(map(int, image_blob.split()), dtype=np.uint8).reshape(48, 48)
        return Image.fromarray(data)

    def _create_class_dirs(self):
        """Create subdirectories for each emotion class."""
        for emotion in self.CK_CLASSES.values():
            (self.output_dir / emotion).mkdir(parents=True, exist_ok=True)

    def __call__(self):
        self.logger.info("Processing CK+ .csv file")

        self._create_class_dirs()

        # Count rows for tqdm
        with self.ck_csv_path.open() as f:
            total_rows = sum(1 for _ in f)

        # Read CSV
        with self.ck_csv_path.open() as f:
            reader = csv.reader(f)
            for idx, row in enumerate(tqdm(reader, total=total_rows, desc="Processing .csv rows")):
                try:
                    label = int(row[0])
                    pixels = row[1]
                except Exception as e:
                    self.logger.warning(f"Skipping row {idx}: {e}")
                    continue

                if label not in self.CK_CLASSES:
                    continue

                emotion = self.CK_CLASSES[label]

                try:
                    image = self._str_to_image(pixels)
                except Exception:
                    self.logger.warning(f"Bad image at index {idx}")
                    continue

                save_path = self.output_dir / emotion / f"{idx}.png"
                image.save(save_path, compress_level=0)

        self.logger.info("Processing finished.")


if __name__ == "__main__":
    setup_logging()

    ck_csv_reader = CKcsvReader(ck_csv_name="ckextended.csv")
    ck_csv_reader()
  
    
    
    