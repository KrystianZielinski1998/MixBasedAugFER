import csv
import numpy as np
from PIL import Image
from itertools import islice
from tqdm import tqdm
import logging
from pathlib import Path
from utils.logging_config import setup_logging

class FERcsvReader:
    """ 
    Converts pixel strings read from CSV files into images and assigns them to corresponding class directory for FER+ dataset.
    """

    def __init__(self,
            fer2013_csv_name: str,
            ferplus_csv_name: str,
        ):
        """
        Args:
            fer2013_csv_name: Name of the .csv file with CK+ dataset.  
            ferplus_csv_name: Name of the .csv file with CK+ dataset.  
        """

        self.fer2013_csv_path = Path(__file__).resolve().parent.parent / "preprocessing" / fer2013_csv_name
        self.ferplus_csv_path = Path(__file__).resolve().parent.parent / "preprocessing" / ferplus_csv_name
        self.output_dir = Path(__file__).resolve().parent.parent / "datasets" / "FER+"

        self.logger = logging.getLogger(__name__)

        # Define emotion classes in same order as in CSV files
        self.EMOTIONS = [
            "neutral", "happiness", "surprise", "sadness",
            "anger", "disgust", "fear", "contempt"
        ]

    @staticmethod
    def _str_to_image(image_blob):
        """ Changes string of pixels read from .csv file into images. """
        data = np.fromstring(image_blob, dtype=np.uint8, sep=' ').reshape(48, 48)
        return Image.fromarray(data)
        
    @staticmethod
    def _get_label_from_votes(votes):
        """ Changes voting into class label """
        votes = votes[:-2]
        if np.sum(votes) == 0:
            return None
        return np.argmax(votes)

    def _create_class_dirs(self):
        """ Creates class dirs in specified dataset directory. """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for emotion in self.EMOTIONS:
            path = self.output_dir / emotion
            path.mkdir(parents=True, exist_ok=True)
 
    def __call__(self):
        self.logger.info("Processing FER+ .csv files")

        # Create dirs for each class label
        self._create_class_dirs()

        # Get number of rows from .csv file for tqdm iterator
        with open(self.fer2013_csv_path) as f:
            total_rows = sum(1 for _ in f) - 1

        with open(self.fer2013_csv_path) as f1, open(self.ferplus_csv_path) as f2:
            r1, r2 = csv.reader(f1), csv.reader(f2)
            next(r1)
            next(r2)

            for idx, (fer2013_row, ferplus_row) in enumerate(
                tqdm(zip(r1, r2), total=total_rows, desc="Processing .csv rows...")
            ):
                # Get image pixels from pixel str read from FER2013 .csv file
                image = self._str_to_image(fer2013_row[1])

                # Get votes for that image from FERPlus .csv file
                votes = np.asarray(ferplus_row[2:], dtype=float)

                # Get class label based on the voting
                label_idx = self._get_label_from_votes(votes)

                if label_idx is None:
                    continue

                # Get emotion name from the emotions list
                emotion = self.EMOTIONS[label_idx]

                # Create path to save the image in proper class dir
                save_path = self.output_dir / emotion / f"{idx}.png"
                
                # Save the image
                image.save(save_path, compress_level=0)

        logger.info("Processing finished.")

if __name__== "__main__":
    setup_logging()

    ferplus_csv_reader = FERcsvReader(
        fer2013_csv_name="fer_2013.csv",
        ferplus_csv_name="fer_2013new.csv",
    )

    ferplus_csv_reader()
  
    
    
    