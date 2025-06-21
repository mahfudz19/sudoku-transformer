import re
from typing import List

def format_sudoku(s: list) -> List[str]:
    rows = []
    for i in range(9):
        row = s[i * 9 : (i + 1) * 9]
        row = ["." if ch == 0 else str(ch) for ch in row]
        formatted_row = " | ".join(" ".join(row[j : j + 3]) for j in range(0, 9, 3))
        rows.append(formatted_row)
        if i % 3 == 2 and i != 8:
            rows.append("-" * 21)
    return rows

def strip_ansi_codes(text: str) -> str:
    ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)

def print_side_by_side(puzzle: list, predicted: list, ground_truth: list) -> None:
    puzzle_lines = format_sudoku(puzzle)
    truth_lines = format_sudoku(ground_truth)
    predicted_lines = []
    for i in range(9):
        row_pred = predicted[i * 9 : (i + 1) * 9]
        row_truth = ground_truth[i * 9 : (i + 1) * 9]
        row_str = ""
        for j in range(9):
            ch = str(row_pred[j])
            if row_pred[j] != row_truth[j]:
                ch = f"\033[91m{ch}\033[0m"
            row_str += ch + " "
            if (j + 1) % 3 == 0 and j != 8:
                row_str += "| "
        predicted_lines.append(row_str)
        if (i + 1) % 3 == 0 and i != 8:
            predicted_lines.append("-" * 21)
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

def load_checkpoint(model, optimizer, scheduler, checkpoint_path: str, epoch_path: str, device: str):
    import os
    import torch
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(f"Resuming from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint)
            if os.path.exists(epoch_path):
                with open(epoch_path, 'r') as f:
                    start_epoch = int(f.read().strip()) + 1
                print(f"Resuming from epoch {start_epoch}")
            else:
                print("No epoch info found, resuming from epoch 0")
    return start_epoch, best_val_loss
