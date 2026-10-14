"""Download the Kaggle dataset into data/raw/.

Usage:
    python download_data.py

Downloads the Global Fashion Retail Stores Dataset from Kaggle's public
API endpoint, extracts the six CSV files into data/raw/, and cleans up
the temporary zip. Total download is around 235 MB; extracted files
occupy roughly 1 GB.
"""

import os
import urllib.request
import zipfile

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "ricgomes/global-fashion-retail-stores-dataset"
)

EXPECTED_FILES = [
    "transactions.csv",
    "customers.csv",
    "products.csv",
    "stores.csv",
    "employees.csv",
    "discounts.csv",
]


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(base, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    if all(os.path.exists(os.path.join(raw_dir, f)) for f in EXPECTED_FILES):
        print("All six CSV files already exist in data/raw/. Nothing to do.")
        return

    zip_path = os.path.join(raw_dir, "dataset.zip")

    print("Downloading dataset from Kaggle...")
    req = urllib.request.Request(DATASET_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=600) as response:
        with open(zip_path, "wb") as f:
            f.write(response.read())
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Downloaded {size_mb:.1f} MB")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(raw_dir)

    os.remove(zip_path)

    print("Extracted files:")
    for name in EXPECTED_FILES:
        path = os.path.join(raw_dir, name)
        if os.path.exists(path):
            size = os.path.getsize(path) / (1024 * 1024)
            print(f"  {name} ({size:.1f} MB)")
        else:
            print(f"  {name} MISSING - extraction may have failed")

    print("\nDone. Run 'python main_pipeline.py' next.")


if __name__ == "__main__":
    main()
