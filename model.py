import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils.data import Dataset, DataLoader
from numpy.typing import NDArray

class PositionalEncoding(nn.Module):
    """Positional encoding untuk Transformer"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerStockPredictor(nn.Module):
    """
    Transformer-based model untuk prediksi saham
    
    📈 State-of-the-art architecture:
    - Self-attention mechanism
    - Parallel processing
    - Dapat capture complex patterns
    """
    
    def __init__(self, d_model=64, nhead=8, num_layers=2, seq_len=5, dropout=0.1):
        super(TransformerStockPredictor, self).__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size, seq_len = x.shape
        
        # Add feature dimension and project to d_model
        x = x.unsqueeze(-1)  # (batch_size, seq_len, 1)
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Scale by sqrt(d_model) as in original transformer
        x = x * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Transformer encoding
        transformer_out = self.transformer(x)  # (batch_size, seq_len, d_model)
        
        # Global average pooling atau ambil last token
        # Disini kita ambil last token untuk prediction
        last_output = transformer_out[:, -1, :]  # (batch_size, d_model)
        
        # Final prediction layers
        x = self.dropout(last_output)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output.squeeze(-1)



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
    # Create datasets
    train_dataset = StockDataset(train_data[0], train_data[1])
    val_dataset = StockDataset(val_data[0], val_data[1])
    test_dataset = StockDataset(test_data[0], test_data[1])

    # Create data loaders (untuk batch processing)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }
