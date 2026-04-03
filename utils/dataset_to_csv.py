import pandas as pd
from pathlib import Path
import numpy as np
from PIL import Image
import argparse

class DatasetToCSV:
    def __init__(self, dataset_name):
        self.input_dir = Path(__file__).resolve().parent.parent / "datasets" / f"{dataset_name}"
        self.output_csv = f"{dataset_name}.csv"

    def _get_pixels(self, path):
        img = Image.open(path).convert("L")
        return " ".join(map(str, np.array(img).flatten()))

    def __call__(self):
        data = {}

        fold_dirs = sorted(self.input_dir.glob("fold_*"))

        for fold_dir in fold_dirs:
            fold_name = fold_dir.name  # np. fold_1

            for split in ["train", "validation", "test"]:
                split_dir = fold_dir / split

                if not split_dir.exists():
                    continue

                for class_dir in split_dir.iterdir():
                    if not class_dir.is_dir():
                        continue

                    label = class_dir.name

                    for img_path in class_dir.glob("*"):
                        if not img_path.is_file():
                            continue

                        filename = img_path.name

                        if filename not in data:
                            data[filename] = {
                                "filename": filename,
                                "class": label,
                                "pixels": self._get_pixels(img_path),
                            }

                        data[filename][fold_name] = split

        df = pd.DataFrame(data.values())

        fold_cols = sorted([f.name for f in fold_dirs])
        for col in fold_cols:
            if col not in df.columns:
                df[col] = None

        df = df[["filename", "class"] + fold_cols + ["pixels"]]

        df.to_csv(self.output_csv, index=False)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate csv file with images and splits.")

    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Directory name of dataset")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    dataset_to_csv = DatasetToCSV(args.dataset_name)

    dataset_to_csv()

