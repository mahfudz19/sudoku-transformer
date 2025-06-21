import torch
import torch.nn as nn

class SudokuTransformer(nn.Module):
    def __init__(self, vocab_size: int = 10, d_model: int = 8, nhead: int = 2, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.rand(81, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x) + self.pos_encoding
        emb = emb.permute(1, 0, 2)
        out = self.transformer(emb)
        out = self.fc(out)
        return out.permute(1, 0, 2)

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from torchinfo import summary
    device = torch.device("cpu")  # Use CPU for visualization
    model = SudokuTransformer().to(device)
    dummy_input = torch.randint(0, 10, (1, 81), dtype=torch.long, device=device)
    print("\n===== torchinfo.summary() =====\n")
    summary(model, input_size=(1, 81), dtypes=[torch.long], col_names=["input_size", "output_size", "num_params", "params_percent"], depth=4, device=device)
    writer = SummaryWriter("runs/model_graph")
    writer.add_graph(model, dummy_input)
    writer.close()