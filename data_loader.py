# Data Loader untuk Stock Prediction
# Handle dataset creation dan DataLoader setup

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.typing import NDArray


class StockDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_dataloaders(
    train_data: tuple[NDArray, NDArray],
    val_data: tuple[NDArray, NDArray],
    test_data: tuple[NDArray, NDArray],
    batch_size=16,
):
    print(f"\n📦 STEP 5a: CREATING DATA LOADERS)")
    print("=" * 40)

    # Create datasets
    train_dataset = StockDataset(train_data[0], train_data[1])
    val_dataset = StockDataset(val_data[0], val_data[1])
    test_dataset = StockDataset(test_data[0], test_data[1])

    # Create data loaders (untuk batch processing)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print(f"📊 Done Creating DataLoaders")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }


# Test function
def test_dataloader_creation():
    """Test function untuk memastikan DataLoader bisa dibuat dengan benar"""
    print("🧪 Testing DataLoader creation...")

    # Dummy data untuk test
    dummy_train = (np.random.rand(50, 5), np.random.rand(50))
    dummy_val = (np.random.rand(10, 5), np.random.rand(10))
    dummy_test = (np.random.rand(10, 5), np.random.rand(10))

    try:
        result = create_dataloaders(dummy_train, dummy_val, dummy_test, batch_size=8)

        train_loader = result["train_loader"]

        # Test batch iteration
        for sequences, targets in train_loader:
            print(f"✅ Batch iteration successful!")
            print(f"   Batch sequences shape: {sequences.shape}")
            print(f"   Batch targets shape: {targets.shape}")
            break

        print(f"✅ DataLoader test passed!")
        return True

    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        return False


if __name__ == "__main__":
    # Test DataLoader creation
    test_dataloader_creation()
