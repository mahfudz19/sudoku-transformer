# Sudoku Transformer Project

## Dataset Download

This project uses the [Sudoku dataset from Kaggle](https://www.kaggle.com/datasets/bryanpark/sudoku). The dataset file `sudoku.csv` is not included in the repository due to its size.

When you run the project, the data loader will automatically download `sudoku.csv` from Kaggle if it is not present. You must:

1. Install the Kaggle CLI:
   ```bash
   pip install kaggle
   ```
2. Set up your Kaggle API credentials. See [Kaggle API documentation](https://github.com/Kaggle/kaggle-api#api-credentials) for instructions.

The first run will download and extract `sudoku.csv` into the project directory.

## Note
- `sudoku.csv` is listed in `.gitignore` and will not be tracked by git.
