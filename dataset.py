import torch
import pandas as pd
from torch.utils.data import Dataset
import os
import zipfile
import subprocess

class SudokuDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        quizzes, solutions = self.df.iloc[idx]
        x = torch.tensor([int(ch) for ch in quizzes], dtype=torch.long)
        y = torch.tensor([int(ch) for ch in solutions], dtype=torch.long)
        return x, y

def ensure_sudoku_csv(csv_path: str = "sudoku.csv"):
    """
    Ensure sudoku.csv exists. If not, download from Kaggle using the Kaggle API.
    """
    if os.path.exists(csv_path):
        return
    print(f"{csv_path} not found. Attempting to download from Kaggle...")
    # Check if kaggle CLI is installed
    if subprocess.call(["which", "kaggle"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) != 0:
        raise RuntimeError("Kaggle CLI not found. Please install it with: pip install kaggle")
    # Download the dataset
    rc = subprocess.call([
        "kaggle", "datasets", "download", "-d", "bryanpark/sudoku", "-p", "."
    ])
    if rc != 0:
        raise RuntimeError("Failed to download dataset from Kaggle. Make sure your Kaggle API credentials are set up.")
    # Unzip
    with zipfile.ZipFile("sudoku.zip", "r") as zip_ref:
        zip_ref.extractall(".")
    os.remove("sudoku.zip")
    print(f"Downloaded and extracted {csv_path}.")
