# Step 6: Training Loop untuk Autoregressive Transformer
# Comprehensive training dengan early stopping dan model saving

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple
import time


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: str,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    patience: int = 10,
    save_path: str = "best_model.pth",
) -> Dict:
    print(f"\n🚀 STEP 6: TRAINING AUTOREGRESSIVE TRANSFORMER")
    print("=" * 60)

    # Setup optimizer dan loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Training history tracking
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    print(f"\n📊 Starting Training...")
    print(f"Epoch | Train Loss | Val Loss   | Best Val | Patience | Time")
    print(f"------|------------|------------|----------|----------|------")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # ============== TRAINING PHASE ==============
        model.train()  # Set ke training mode
        total_train_loss = 0.0
        num_train_batches = 0

        for sequences, targets in train_loader:
            # Move data ke device (CPU/GPU)
            sequences = sequences.to(device)  # Shape: (batch_size, seq_len)
            targets = targets.to(device)  # Shape: (batch_size,)

            # Add channel dimension untuk input
            sequences = sequences.unsqueeze(-1)  # Shape: (batch_size, seq_len, 1)
            targets = targets.unsqueeze(-1)  # Shape: (batch_size, 1)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(sequences)  # Shape: (batch_size, 1)

            # Calculate loss
            loss = criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Track loss
            total_train_loss += loss.item()
            num_train_batches += 1

        # Average training loss untuk epoch ini
        avg_train_loss = total_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        # ============== VALIDATION PHASE ==============
        model.eval()  # Set ke evaluation mode
        total_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():  # Tidak perlu gradients untuk validation
            for sequences, targets in val_loader:
                # Move data ke device
                sequences = sequences.to(device)
                targets = targets.to(device)

                # Add channel dimension
                sequences = sequences.unsqueeze(-1)
                targets = targets.unsqueeze(-1)

                # Forward pass
                predictions = model(sequences)

                # Calculate loss
                loss = criterion(predictions, targets)

                # Track loss
                total_val_loss += loss.item()
                num_val_batches += 1

        # Average validation loss
        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)

        # ============== EARLY STOPPING & MODEL SAVING ==============
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "best_val_loss": best_val_loss,
                },
                save_path,
            )

            status = "💾 SAVED"
        else:
            patience_counter += 1
            status = f"⏳ {patience_counter}/{patience}"

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print epoch results
        print(
            f"{epoch+1:5d} | {avg_train_loss:.8f} | {avg_val_loss:.8f} | {best_val_loss:.8f} | {status:8s} | {epoch_time:.1f}s"
        )

        # Early stopping check
        if patience_counter >= patience:
            print(f"\n⏰ EARLY STOPPING!")
            print(f"   No improvement for {patience} epochs")
            print(
                f"   Best validation loss: {best_val_loss:.8f} at epoch {best_epoch+1}"
            )
            break

    # ============== TRAINING COMPLETE ==============
    print(f"\n🎉 TRAINING COMPLETE!")

    # Return training history
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs": epoch + 1,
        "model_path": save_path,
    }


def plot_training_history(history: Dict, save_plot: bool = True):
    """
    Plot training dan validation loss curves

    📊 Untuk JavaScript developer:
    Seperti membuat line chart dengan Chart.js:

    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: epochs,
            datasets: [
                { label: 'Training Loss', data: trainLosses, color: 'blue' },
                { label: 'Validation Loss', data: valLosses, color: 'red' }
            ]
        }
    });

    Parameters:
    - history: hasil dari train_model()
    - save_plot: apakah menyimpan plot ke file
    """
    print(f"\n📊 PLOTTING TRAINING HISTORY")
    print("=" * 40)

    plt.figure(figsize=(12, 5))

    # Plot 1: Loss curves
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history["train_losses"]) + 1)

    plt.plot(epochs, history["train_losses"], "b-", label="Training Loss", linewidth=2)
    plt.plot(epochs, history["val_losses"], "r-", label="Validation Loss", linewidth=2)

    # Mark best epoch
    best_epoch = history["best_epoch"] + 1
    best_val_loss = history["best_val_loss"]
    plt.plot(
        best_epoch,
        best_val_loss,
        "go",
        markersize=10,
        label=f"Best (Epoch {best_epoch})",
    )

    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Loss difference (overfitting check)
    plt.subplot(1, 2, 2)
    loss_diff = np.array(history["val_losses"]) - np.array(history["train_losses"])
    plt.plot(epochs, loss_diff, "purple", linewidth=2)
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.5)

    plt.title("Validation - Training Loss\n(Overfitting Check)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Difference")
    plt.grid(True, alpha=0.3)

    # Add annotation
    final_diff = loss_diff[-1]
    if final_diff > 0.01:
        plt.text(
            0.5,
            0.95,
            "Possible Overfitting",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
        )
    else:
        plt.text(
            0.5,
            0.95,
            "Good Generalization",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.3),
        )

    plt.tight_layout()

    if save_plot:
        plot_path = "training_history.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"💾 Plot saved to: {plot_path}")

    plt.show()

    # Print training summary
    print(f"\n📈 TRAINING SUMMARY:")
    print(f"   🎯 Best validation loss: {history['best_val_loss']:.8f}")
    print(f"   📍 Best epoch: {history['best_epoch'] + 1}")
    print(f"   🔄 Total epochs: {history['total_epochs']}")
    print(f"   📊 Final train loss: {history['train_losses'][-1]:.8f}")
    print(f"   📊 Final val loss: {history['val_losses'][-1]:.8f}")

    # Overfitting check
    final_diff = history["val_losses"][-1] - history["train_losses"][-1]
    if final_diff > 0.01:
        print(f"   ⚠️  Warning: Possible overfitting (diff: {final_diff:.6f})")
    else:
        print(f"   ✅ Good generalization (diff: {final_diff:.6f})")


def load_best_model(model: nn.Module, model_path: str, device: str):
    """
    Load best model dari checkpoint

    💾 Untuk JavaScript developer:
    Seperti loading saved state:
    const savedModel = localStorage.getItem('bestModel');
    model.loadState(JSON.parse(savedModel));

    Parameters:
    - model: PyTorch model untuk load weights
    - model_path: path ke saved model
    - device: device untuk model

    Returns:
    - model: model dengan loaded weights
    - checkpoint info
    """
    print(f"\n💾 LOADING BEST MODEL")
    print("=" * 30)

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return model, None

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded successfully!")
    print(f"   📍 Epoch: {checkpoint['epoch'] + 1}")
    print(f"   📈 Train loss: {checkpoint['train_loss']:.8f}")
    print(f"   📉 Val loss: {checkpoint['val_loss']:.8f}")
    print(f"   🎯 Best val loss: {checkpoint['best_val_loss']:.8f}")

    return model, checkpoint


# Test function untuk training
def test_training_setup():
    """Test function untuk memastikan training setup bekerja"""
    print("🧪 Testing training setup...")

    try:
        from model import create_model
        from data_loader import create_dataloaders
        import numpy as np

        # Create dummy data
        dummy_train = (np.random.rand(100, 5), np.random.rand(100))
        dummy_val = (np.random.rand(20, 5), np.random.rand(20))
        dummy_test = (np.random.rand(20, 5), np.random.rand(20))

        # Create dataloaders
        dataloader_components = create_dataloaders(
            dummy_train, dummy_val, dummy_test, batch_size=8
        )

        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(device=device)

        # Test short training (1 epoch)
        history = train_model(
            model=model,
            train_loader=dataloader_components["train_loader"],
            val_loader=dataloader_components["val_loader"],
            device=device,
            num_epochs=2,  # Just 2 epochs for test
            patience=5,
        )

        print(f"✅ Training test passed!")
        print(f"   Train losses: {len(history['train_losses'])} epochs")
        print(f"   Val losses: {len(history['val_losses'])} epochs")

        return True

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False


if __name__ == "__main__":
    # Test training setup
    test_training_setup()
