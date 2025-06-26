import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import DecoderOnly, EncoderOnly, SudokuDataset
from tqdm import tqdm


def get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps, min_lr=1e-5):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Linear decay after warmup
        return max(
            min_lr / optimizer.defaults["lr"],
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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
            block = grid[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3].reshape(-1)
            if torch.unique(block).numel() != 9:
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


def train(
    train_df: torch.utils.data.Dataset,
    val_df: torch.utils.data.Dataset = None,
    patience: int = 10,
    d_model=8,
    nhead=2,
    num_layers=2,
    batch_size=512,
    epochs=50,
    warmup_ratio=0.1,
):
    # 1. Siapkan DataFrame df (quizzes, solutions)
    dataset = SudokuDataset(train_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Inisialisasi model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = DecoderOnly(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    writer = SummaryWriter()
    total_steps = epochs * len(dataloader)
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # 2.1 Optional do validation
    if val_df is not None:
        val_dataset = SudokuDataset(val_df)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2.2 Save optimizer and scheduler state_dicts in checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_prefix = f"best_checkpoint_{d_model}_{nhead}_{num_layers}"
    best_model_path = f"checkpoints/{checkpoint_prefix}.pt"
    best_model_epoch_path = f"checkpoints/{checkpoint_prefix}.txt"
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    # Resume if best model exists
    if os.path.exists(best_model_path):
        print(f"Resuming from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"Resuming from epoch {start_epoch}")
        else:
            # Fallback for old checkpoints (model only)
            model.load_state_dict(checkpoint)
            if os.path.exists(best_model_epoch_path):
                with open(best_model_epoch_path, "r") as f:
                    start_epoch = int(f.read().strip()) + 1
                print(f"Resuming from epoch {start_epoch}")
            else:
                print("No epoch info found, resuming from epoch 0")

    # 3. Training loop
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            y = y - 1
            output = model(x)
            loss = sudoku_loss(output, y, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_description(f"Epoch {epoch+1}/{epochs} | | loss={loss.item():.4f}")
        avg_train_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
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
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            print(f" | Val Loss: {avg_val_loss:.4f}")

            # Save only the best model (overwrite if improved)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                # Save model, optimizer, scheduler, epoch, and best_val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    },
                    best_model_path,
                )
                with open(best_model_epoch_path, "w") as f:
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


def train_autoregressive(
    train_df: torch.utils.data.Dataset,
    val_df: torch.utils.data.Dataset = None,
    patience: int = 10,
    d_model=8,
    nhead=2,
    num_layers=2,
    batch_size=512,
    epochs=50,
    warmup_ratio=0.1,
):
    """
    True Autoregressive Training untuk DecoderOnly
    - Training step-by-step, satu posisi per kali
    - Lebih lambat tapi lebih sesuai dengan nature autoregressive
    """
    # 1. Siapkan DataFrame df (quizzes, solutions)
    dataset = SudokuDataset(train_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Inisialisasi model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("Training Mode: TRUE AUTOREGRESSIVE (step-by-step)")
    model = DecoderOnly(d_model=d_model, nhead=nhead, num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    writer = SummaryWriter()
    total_steps = epochs * len(dataloader) * 81  # 81x more steps because step-by-step
    warmup_steps = int(warmup_ratio * total_steps)
    scheduler = get_linear_warmup_scheduler(optimizer, warmup_steps, total_steps)

    # 2.1 Optional do validation
    if val_df is not None:
        val_dataset = SudokuDataset(val_df)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 2.2 Save optimizer and scheduler state_dicts in checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_prefix = f"best_checkpoint_autoregressive_{d_model}_{nhead}_{num_layers}"
    best_model_path = f"checkpoints/{checkpoint_prefix}.pt"
    best_model_epoch_path = f"checkpoints/{checkpoint_prefix}.txt"
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None
    start_epoch = 0

    # Resume if best model exists
    if os.path.exists(best_model_path):
        print(f"Resuming from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0) + 1
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"Resuming from epoch {start_epoch}")
        else:
            # Fallback for old checkpoints (model only)
            model.load_state_dict(checkpoint)
            if os.path.exists(best_model_epoch_path):
                with open(best_model_epoch_path, "r") as f:
                    start_epoch = int(f.read().strip()) + 1
                print(f"Resuming from epoch {start_epoch}")
            else:
                print("No epoch info found, resuming from epoch 0")

    # 3. Training loop - TRUE AUTOREGRESSIVE
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        step_count = 0
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)  # (batch, 81)
            y = y - 1  # Convert to 0-8
            batch_size = x.size(0)

            # Autoregressive training: step by step
            for t in range(1, 81):  # Start from position 1 (predict based on 0)
                # Input: partial sequence up to position t-1
                input_seq = x[:, :t]  # (batch, t)
                target_pos = y[:, t - 1]  # Target for position t-1 (0-indexed)

                # Forward pass
                output = model(input_seq)  # (batch, t, 9)
                pred_at_t = output[:, -1, :]  # Last position prediction (batch, 9)

                # Loss only for the current position
                loss = criterion(pred_at_t, target_pos)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()
                step_count += 1

            pbar.set_description(f"Epoch {epoch+1}/{epochs} | loss={loss.item():.4f}")

        avg_train_loss = total_loss / step_count
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}", end="")

        # Validation - use full sequence for consistency
        if val_df is not None:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for x, y in val_dataloader:
                    x, y = x.to(device), y.to(device) - 1
                    out = model(x)  # Full sequence for validation
                    loss = sudoku_loss(out, y, criterion)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            print(f" | Val Loss: {avg_val_loss:.4f}")

            # Save only the best model (overwrite if improved)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                # Save model, optimizer, scheduler, epoch, and best_val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                    },
                    best_model_path,
                )
                with open(best_model_epoch_path, "w") as f:
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


def predict(model, input_puzzle):
    """
    input_puzzle: list of 81 int (0-9), 0 untuk sel kosong
    return: list of 81 int (hasil prediksi)
    """
    import torch

    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        # Check if model is DecoderOnly or EncoderOnly
        if isinstance(model, DecoderOnly):
            # For DecoderOnly: True Autoregressive generation
            output = []

            # Option 1: Start with original puzzle (hybrid approach)
            # x = torch.tensor(input_puzzle, dtype=torch.long, device=device).unsqueeze(0)  # (1, 81)

            # Option 2: Pure autoregressive - start empty and build sequence
            x = torch.zeros(
                1, 81, dtype=torch.long, device=device
            )  # Start with all zeros

            for t in range(81):
                # For hybrid approach: use original input for known cells
                if input_puzzle[t] != 0:
                    # If cell is not empty in original puzzle, use that value
                    pred = input_puzzle[t] - 1  # Convert to 0-8 range for model
                    x[0, t] = pred
                    output.append(input_puzzle[t])  # Keep original value
                else:
                    # If cell is empty, generate prediction
                    input_seq = x[:, : t + 1]  # (1, t+1)
                    out = model(input_seq)  # (1, t+1, 9)
                    pred = out[0, -1].argmax(-1).item()  # Last position prediction
                    x[0, t] = pred  # Update sequence with prediction
                    output.append(pred + 1)  # Convert back to 1-9 range

            return output
        else:
            # For EncoderOnly: Single forward pass
            x = torch.tensor(input_puzzle, dtype=torch.long, device=device).unsqueeze(
                0
            )  # (1, 81)
            out = model(x)  # (1, 81, 9)
            pred = out.argmax(-1).squeeze(0)  # (81,)
            return (pred + 1).tolist()  # Convert back to 1-9 range
