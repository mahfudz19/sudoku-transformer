import pandas as pd
import torch

from train import train, predict, train_autoregressive
from util import print_side_by_side

sudokuDaatrain = "/Users/m/Documents/Github/AI/sudoku.csv"

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(sudokuDaatrain)
    df = df.dropna().reset_index(drop=True)  # bersihkan data dari baris yang kosong
    print(f"Loaded {len(df)} sudoku samples.")

    df.columns = df.columns.str.strip()

    # Use a small subset to train fast
    train_idx = int(len(df) * 0.0008)
    val_idx = int(len(df) * 0.0009)
    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx : val_idx + 1]
    print(
        f"train_df = df[:{train_idx}]\nval_df = df[{train_idx}:{val_idx}]\ntest_df = df[{val_idx}:{val_idx+1}]"
    )

    # Set your desired architecture here
    model = train_autoregressive(
        train_df,
        val_df,
        d_model=128,
        nhead=8,
        num_layers=4,
        batch_size=1800,
        epochs=10,
    )

    # === Inference ===
    print("\n=== Inference ===")
    model.eval()

    with torch.no_grad():
        for i in range(len(test_df)):
            test_sample = test_df.iloc[i]
            input_puzzle = [int(ch) for ch in test_sample["quizzes"]]
            solution = [int(ch) for ch in test_sample["solutions"]]
            pred = predict(model, input_puzzle)
            print_side_by_side(input_puzzle, pred, solution)
