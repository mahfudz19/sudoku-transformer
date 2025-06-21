from model import SudokuTransformer
from dataset import SudokuDataset, ensure_sudoku_csv
from utils import format_sudoku, strip_ansi_codes, print_side_by_side, load_checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import argparse


# Set seed
torch.manual_seed(42)


def is_valid_sudoku_torch(grid):
    # grid: (9, 9) torch tensor, values 1-9, device can be cuda or cpu
    # Check rows and columns
    for i in range(9):
        if torch.unique(grid[i, :]).numel() != 9:
            return False
        if torch.unique(grid[:, i]).numel() != 9:
            return False
    # Check 3x3 blocks
    for i in range(3):
        for j in range(3):
            block = grid[i*3:(i+1)*3, j*3:(j+1)*3].reshape(-1)
            if torch.unique(block).numel() != 9:
                return False
    return True


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
        if not is_valid_sudoku_torch(pred_labels[i].reshape(9, 9)):
            penalty += 1.0
    penalty = penalty_weight * penalty / batch_size
    return ce_loss + penalty


def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-5):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Linear decay after warmup
        return max(
            min_lr / optimizer.defaults['lr'],
            float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, train_df: pd.DataFrame, val_df: pd.DataFrame = None, patience: int = 10, batch_size: int = 512, epochs: int = 50, start_epoch: int = 0, best_val_loss: float = float('inf')) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    dataset = SudokuDataset(train_df)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if val_df is not None:
        val_dataset = SudokuDataset(val_df)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    epochs_no_improve = 0
    best_model_state = None
    global_step = 0

    os.makedirs('checkpoints', exist_ok=True)  # Ensure checkpoints directory exists

    for epoch in range(start_epoch, epochs):
        total_loss = 0
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device) - 1
            out = model(x)
            loss = sudoku_loss(out, y, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | lr={scheduler.get_last_lr()[0]:.6f} | loss={loss.item():.4f}")
        avg_train_loss = total_loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}", end="")

        # Validation
        if val_df is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(device), y.to(device) - 1
                    out = model(x)
                    loss = sudoku_loss(out, y, criterion)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            print(f" | Val Loss: {avg_val_loss:.4f}")

            # Save only the best model (overwrite if improved)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                # Save model, optimizer, scheduler, epoch, and best_val_loss
                checkpoint_prefix = f"best_checkpoint_{model.embedding.embedding_dim}_{model.transformer.layers[0].self_attn.num_heads}_{len(model.transformer.layers)}"
                best_model_path = f'checkpoints/{checkpoint_prefix}.pt'
                best_model_epoch_path = f'checkpoints/{checkpoint_prefix}.txt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, best_model_path)
                with open(best_model_epoch_path, 'w') as f:
                    f.write(str(epoch))
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break
        else:
            print()

    writer.close()
    return model


def predict(model, puzzle):
    """
    Deprecated: Use model.predict(puzzle) instead.
    """
    return model.predict(puzzle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=lambda x: (str(x).lower() == 'true'), default=True, help='Set False to skip training and only run test')
    args = parser.parse_args()

    # Ensure dataset is present or download it
    ensure_sudoku_csv("/workspace/sudoku-transformer/sudoku.csv")

    # Load data
    df = pd.read_csv("/workspace/sudoku-transformer/sudoku.csv")
    df = df.dropna().reset_index(drop=True)
    print(f"Loaded {len(df)} sudoku samples.")
    df.columns = df.columns.str.strip()
    train_idx = int(len(df) * 0.8)
    val_idx = int(len(df) * 0.9)
    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx:]

    d_model = 128
    nhead = 8
    num_layers = 4
    batch_size = 1800
    epoch = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SudokuTransformer(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    total_steps = epoch * (len(train_df) // batch_size + 1)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    checkpoint_prefix = f"best_checkpoint_{d_model}_{nhead}_{num_layers}"
    best_model_path = f'checkpoints/{checkpoint_prefix}.pt'
    best_model_epoch_path = f'checkpoints/{checkpoint_prefix}.txt'
    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, best_model_path, best_model_epoch_path, device)

    if args.training:
        model = train(
            model, optimizer, scheduler,
            train_df, val_df,
            batch_size=batch_size,
            epochs=epoch,
            start_epoch=start_epoch,
            best_val_loss=best_val_loss
        )

    # === Inference & Evaluation ===
    print("\n=== Inference ===")
    model.eval()
    correct_exact = 0
    correct_valid = 0
    total = 0
    pbar = tqdm(enumerate(test_df.iterrows()), total=len(test_df), desc="Testing")
    
    with torch.no_grad():
        for i, (_, test_sample) in pbar:
            input_puzzle = [int(ch) for ch in test_sample["quizzes"]]
            solution = [int(ch) for ch in test_sample["solutions"]]
            pred = predict(model, input_puzzle)
            # 1. Exact match
            if pred.tolist() == solution:
                correct_exact += 1
            # 2. Valid sudoku
            if is_valid_sudoku_torch(torch.tensor(pred.reshape(9, 9))):
                correct_valid += 1
            total += 1
            pbar.set_postfix({
                "Exact Match": f"{correct_exact}/{total} ({correct_exact/total:.4f})",
                "Valid Sudoku": f"{correct_valid}/{total} ({correct_valid/total:.4f})"
            })

    print(f"Test Accuracy (Exact match): {correct_exact}/{total} = {correct_exact/total:.4f}")
    print(f"Test Accuracy (Valid Sudoku): {correct_valid}/{total} = {correct_valid/total:.4f}")
