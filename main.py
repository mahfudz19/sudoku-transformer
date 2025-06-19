import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd


# grid sudoku
def format_sudoku(s: list):
    rows = []
    for i in range(9):
        row = s[i * 9 : (i + 1) * 9]
        # Convert all elements to string for joining
        row = ["." if ch == 0 else str(ch) for ch in row]
        formatted_row = " | ".join(" ".join(row[j : j + 3]) for j in range(0, 9, 3))
        rows.append(formatted_row)
        if i % 3 == 2 and i != 8:
            rows.append("-" * 21)
    return rows


import re


def strip_ansi_codes(text):
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


def print_side_by_side(puzzle, predicted, ground_truth):
    puzzle_lines = format_sudoku(puzzle)
    truth_lines = format_sudoku(ground_truth)

    # Prepare predicted lines with color for wrong digits
    predicted_lines = []
    for i in range(9):
        row_pred = predicted[i * 9 : (i + 1) * 9]
        row_truth = ground_truth[i * 9 : (i + 1) * 9]
        row_str = ""
        for j in range(9):
            ch = str(row_pred[j])
            if row_pred[j] != row_truth[j]:
                # ANSI escape code for red color
                ch = f"\033[91m{ch}\033[0m"
            row_str += ch + " "
            if (j + 1) % 3 == 0 and j != 8:
                row_str += "| "
        predicted_lines.append(row_str)
        if (i + 1) % 3 == 0 and i != 8:
            predicted_lines.append("-" * 21)

    # Pad predicted lines based on visible length (excluding ANSI codes)
    padded_predicted_lines = []
    for line in predicted_lines:
        visible_len = len(strip_ansi_codes(line))
        padding = 25 - visible_len
        if padding > 0:
            line += " " * padding
        padded_predicted_lines.append(line)

    print(f"\n{'Puzzle':<25} {'Predicted':<25} {'Ground Truth':<25}")
    print("=" * 75)
    for p, pr, gt in zip(puzzle_lines, padded_predicted_lines, truth_lines):
        print(f"{p:<25} {pr} {gt:<25}")


# Set seed
torch.manual_seed(42)


# Dataset
class SudokuDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        quizzes, solutions = self.df.iloc[idx]
        x = torch.tensor([int(ch) for ch in quizzes], dtype=torch.long)
        y = torch.tensor([int(ch) for ch in solutions], dtype=torch.long)

        return x, y


class SudokuTransformer(nn.Module):
    def __init__(self, vocab_size=10, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(
            torch.rand(81, d_model)
        )  # positional encoding for 81 tokens

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc = nn.Linear(d_model, 9)  # output is digit 1-9

    def forward(self, x):
        # x shape: (batch, 81)
        emb = self.embedding(x) + self.pos_encoding  # (batch, 81, d_model)
        emb = emb.permute(1, 0, 2)  # (81, batch, d_model)
        out = self.transformer(emb)  # (81, batch, d_model)
        out = self.fc(out)  # (81, batch, 9)
        return out.permute(1, 0, 2)  # (batch, 81, 9)


def is_valid_sudoku(pred):
    grid = pred.reshape(9, 9)
    # Check rows and columns
    for i in range(9):
        row = grid[i, :]
        col = grid[:, i]
        if len(set(row.tolist())) != 9 or len(set(col.tolist())) != 9:
            return False
    # Check 3x3 blocks
    for i in range(3):
        for j in range(3):
            block = grid[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3].flatten()
            if len(set(block.tolist())) != 9:
                return False
    return True


def sudoku_loss(pred, target, criterion, penalty_weight=0.2):
    ce_loss = criterion(pred.reshape(-1, 9), target.view(-1))
    pred_labels = torch.argmax(pred, dim=2)
    batch_size = pred_labels.size(0)
    penalty = 0.0
    for i in range(batch_size):
        if not is_valid_sudoku(pred_labels[i].cpu().numpy()):
            penalty += 1.0
    penalty = penalty_weight * penalty / batch_size
    return ce_loss + penalty


def train(train_df: torch.utils.data.Dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("\n\n", device)
    model = SudokuTransformer().to(device)
    dataset = SudokuDataset(train_df)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device) - 1  # make target in range 0–8
            out = model(x)

            loss = sudoku_loss(out, y, criterion)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss / len(dataloader):.4f}")

    return model


def predict(model, puzzle):
    """
    Predict with constraint-aware decoding to enforce Sudoku rules.
    """
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor(puzzle, dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)  # (batch=1, 81, 9)
        logits = out.squeeze(0)  # (81, 9)
        preds = torch.zeros(81, dtype=torch.long, device=device)

        # Initialize grid with puzzle clues (non-zero cells)
        grid = torch.tensor(puzzle, dtype=torch.long, device=device).reshape(9, 9)

        # Mask for cells to fill (0 means empty)
        mask = grid == 0

        # For each empty cell, select the highest logit digit that does not violate Sudoku constraints
        def is_valid(grid, row, col, val):
            # Check row
            if val in grid[row, :]:
                return False
            # Check column
            if val in grid[:, col]:
                return False
            # Check 3x3 block
            start_row, start_col = 3 * (row // 3), 3 * (col // 3)
            if val in grid[start_row : start_row + 3, start_col : start_col + 3]:
                return False
            return True

        # Fill known cells first
        for i in range(81):
            r, c = divmod(i, 9)
            if not mask[r, c]:
                preds[i] = grid[r, c]
            else:
                preds[i] = 0

        # For empty cells, try digits in descending order of logits until valid found
        for i in range(81):
            r, c = divmod(i, 9)
            if mask[r, c]:
                logits_i = logits[i]
                # Sort digits by descending logit score
                sorted_digits = (
                    torch.argsort(logits_i, descending=True) + 1
                )  # digits 1-9
                for digit in sorted_digits:
                    if is_valid(grid, r, c, digit.item()):
                        preds[i] = digit
                        grid[r, c] = digit
                        break
                else:
                    # If no valid digit found, assign highest logit digit anyway (fallback)
                    preds[i] = sorted_digits[0]
                    grid[r, c] = sorted_digits[0]

    return preds.cpu().numpy()


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("sudoku.csv")
    df = df.dropna().reset_index(drop=True)  # bersihkan data dari baris yang kosong
    print(f"Loaded {len(df)} sudoku samples.")

    df.columns = df.columns.str.strip()

    # Use a small subset to train fast
    train_df = df[:50000]
    test_df = df[50000 : 50000 + 10]

    model = train(train_df)

    # === Inference ===
    print("\n=== Inference ===")
    model.eval()

    with torch.no_grad():
        for i in range(len(test_df)):
            test_sample = test_df.iloc[i]
            input_puzzle = [int(ch) for ch in test_sample["quizzes"]]
            solution = [int(ch) for ch in test_sample["solutions"]]
            pred = predict(model, input_puzzle)
            print_side_by_side(input_puzzle, pred.tolist(), solution)
