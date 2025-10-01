import os
import pandas as pd
from PIL import Image


BASE_PATH = "food_cls"


def fullpath(path):
    return os.path.join(BASE_PATH, path)


def check_csv_sanity(csv_path):
    print(f"Checking CSV file '{csv_path}'...", end="")
    df = pd.read_csv(csv_path)
    for path in df["path"].to_list():
        Image.open(fullpath(path)).close()
    print("done!")


def main():
    csv_paths = [
        "train.csv",
        "val.csv",
        "imbalanced.csv",
        "augmented.csv"
    ]
    for path in csv_paths:
        check_csv_sanity(fullpath(path))
    print("All clear!")


if __name__ == "__main__":
    main()