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

    def predict(self, puzzle):
        """
        Predict with constraint-aware decoding to enforce Sudoku rules.
        """
        self.eval()
        device = next(self.parameters()).device
        x = torch.tensor(puzzle, dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            out = self(x)  # (batch=1, 81, 9)
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