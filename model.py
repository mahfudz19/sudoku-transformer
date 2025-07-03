# Autoregressive Transformer Model untuk Stock Prediction
# Pure model definition tanpa data handling

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Buat matrix positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # Gunakan sin untuk even indices, cos untuk odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register sebagai buffer (tidak akan di-update saat training)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[: x.size(1), :].transpose(0, 1)
        return self.dropout(x)


class AutoregressiveTransformer(nn.Module):
    def __init__(
        self,
        input_size=1,
        d_model=64,
        nhead=8,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.1,
    ):
        super(AutoregressiveTransformer, self).__init__()

        self.d_model = d_model

        # 1. Input projection: 1 nilai price → d_model dimensi
        # Seperti mengubah satu angka jadi vector dengan banyak "features"
        self.input_projection = nn.Linear(input_size, d_model)

        # 2. Positional encoding untuk urutan
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # 3. Transformer Decoder layers
        # Ini adalah "otak" dari model - bisa "memperhatikan" data penting
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, feature)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers
        )

        # 4. Output projection: d_model → 1 prediction
        # Convert kembali dari vector ke satu nilai prediksi
        self.output_projection = nn.Linear(d_model, 1)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask untuk autoregressive behavior

        🚫 Untuk JavaScript developer:
        Seperti "blindfold" - model tidak boleh "melihat" data masa depan:

        const mask = [
            [0, -∞, -∞, -∞, -∞],  // day1 hanya lihat day1
            [0,  0, -∞, -∞, -∞],  // day2 lihat day1-2
            [0,  0,  0, -∞, -∞],  // day3 lihat day1-3
            [0,  0,  0,  0, -∞],  // day4 lihat day1-4
            [0,  0,  0,  0,  0]   // day5 lihat day1-5
        ];
        """
        mask = torch.triu(torch.ones(sz, sz) * float("-inf"), diagonal=1)
        return mask

    def forward(self, src):
        """
        Forward pass - jalankan prediksi

        Parameters:
        - src: Input tensor shape (batch_size, seq_len, 1)
               Contoh: batch 16 sequences, masing-masing 5 hari, 1 price value

        Returns:
        - output: Predicted next value shape (batch_size, 1)
        """
        batch_size, seq_len, _ = src.shape

        # 1. Project input dari 1 dimensi ke d_model dimensi
        # [170.5] → [0.2, -0.5, 0.8, ..., 0.1] (64 nilai)
        src = self.input_projection(src)

        # 2. Add positional encoding
        # Kasih tahu model urutan: day1, day2, day3, day4, day5
        src = self.positional_encoding(src)

        # 3. Create causal mask (autoregressive)
        # Model tidak boleh "nyontek" dari masa depan
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(src.device)

        # 4. Transformer processing
        # Ini bagian "magic" - attention mechanism bekerja
        output = self.transformer_decoder(
            tgt=src,  # Target sequence
            memory=src,  # Memory (sama dengan target untuk self-attention)
            tgt_mask=tgt_mask,  # Causal mask
        )

        # 5. Project ke output dan ambil timestep terakhir
        # Ambil hasil dari day5 (timestep terakhir) untuk prediksi day6
        output = self.output_projection(output[:, -1, :])  # Shape: (batch_size, 1)

        return output


def create_model(model_config=None, device="cpu"):
    print(f"\n🧠STEP 5b: CREATING AUTOREGRESSIVE TRANSFORMER")
    print("=" * 50)

    # Default config jika tidak ada
    if model_config is None:
        model_config = {
            "input_size": 1,
            "d_model": 64,
            "nhead": 8,
            "num_layers": 3,
            "dim_feedforward": 256,
            "dropout": 0.1,
        }

    # Create model
    model = AutoregressiveTransformer(**model_config)
    model = model.to(device)

    print(f"\n✅ Model ready for training!")

    return model


# Test function
def test_model_creation():
    print("🧪 Testing model creation...")

    try:
        # Test dengan config default
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(device=device)

        # Test forward pass dengan dummy data
        batch_size = 4
        seq_len = 5
        input_size = 1

        # Create dummy input
        dummy_input = torch.randn(batch_size, seq_len, input_size).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(
            f"   Output range: {output.min().item():.4f} to {output.max().item():.4f}"
        )

        print(f"✅ Model test passed!")
        return True

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


if __name__ == "__main__":
    # Test model creation
    test_model_creation()
