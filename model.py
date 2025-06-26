import torch
import torch.nn as nn


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


class DecoderOnly(nn.Module):
    def __init__(self, vocab_size=10, d_model=16, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(81, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 9)

    def forward(self, x):
        # x: (batch_size, t) t bisa < 81 (autoregressive)
        emb = self.embedding(x) + self.pos_encoding[: x.size(1)]  # (batch, t, d_model)
        emb = emb.permute(1, 0, 2)  # (t, batch, d_model)

        # Generate mask untuk causal attention
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(
            x.device
        )

        # Decoder secara autoregressive
        out = self.decoder(emb, emb, tgt_mask=tgt_mask)
        out = self.fc(out)
        return out.permute(1, 0, 2)  # (batch, t, 9)


class EncoderOnly(nn.Module):
    def __init__(self, vocab_size=10, d_model=8, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # vocab_size=10, d_model=16
        self.pos_encoding = nn.Parameter(torch.randn(81, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 9)

    def forward(self, x):
        # x: (batch_size, 81)
        emb = self.embedding(x) + self.pos_encoding  # (batch_size, 81, d_model)
        emb = emb.permute(1, 0, 2)  # (81, batch, d_model)
        out = self.encoder(emb)  # (81, batch, d_model)
        out = self.fc(out)  # (81, batch, 9)
        return out.permute(1, 0, 2)  # (batch, 81, 9)
