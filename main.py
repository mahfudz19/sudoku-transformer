import sys
import os
import torch

# Import dari file terpisah
from model import TransformerStockPredictor, create_dataloaders
from get_data_utils import create_output_directories, fetch_stock_data, normalize_data, create_sequences, split_data
from learning import train_model, evaluate_model, predict_next_price
from visualization import plot_training_loss, plot_prediction_vs_actual

# Test fungsi yang sudah dibuat
if __name__ == "__main__":
    # Setup output directories first
    create_output_directories()

    # Step 1: Ambil data saham
    df = fetch_stock_data("AAPL")
    print(f"📊 Data AAPL berhasil diambil: {len(df)} baris")

    if df is None or len(df) <= 10:
        print("❌ Gagal mengambil data. Coba lagi nanti.")
        sys.exit(1)

    # Step 2: Normalisasi
    scaler, normalized_data = normalize_data(df, "close")

    # Step 3: Create sequences
    x, y = create_sequences(normalized_data, window_size=5)

    # Step 4: Split data
    train_data, val_data, test_data = split_data(x, y)

    # Step 5a: Setup data loaders
    dataloader_components = create_dataloaders(train_data, val_data, test_data, batch_size=16)

    # Step 5b: Create model
    model = TransformerStockPredictor(d_model=64, nhead=8, num_layers=2)
    print(f"🤖 Model created: {model.__class__.__name__}")

    # Otomatis load best model jika file ada
    best_model_path = "file/best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print(f"✅ Best model loaded from {best_model_path}")

    # Step 6: Training
    trained_model, history = train_model(
        model=model,
        train_loader=dataloader_components['train_loader'],
        val_loader=dataloader_components['val_loader'],
        num_epochs=250,
        lr=0.001
    )

    # Step 7: Evaluation
    results = evaluate_model(
        model=trained_model,
        test_loader=dataloader_components['test_loader'],
        scaler=scaler
    )

    # Step 8: Prediksi harga berikutnya
    last_sequence = x[-1]  # Ambil sequence terakhir
    predicted_price = predict_next_price(
        model=trained_model,
        last_sequence=last_sequence,
        scaler=scaler
    )

    # Step 9: VISUALISASI 📊 (Hanya 2 chart yang dibutuhkan)
    print(f"\n🎨 GENERATING VISUALIZATIONS...")

    # Chart 1: Training vs Validation Loss
    plot_training_loss(history)

    # Chart 2: Prediction Accuracy
    plot_prediction_vs_actual(results)

    print(f"\n🎉 SEMUA TAHAP SELESAI!")
    print(f"📈 Prediksi harga AAPL berikutnya: ${predicted_price:.2f}")
    print(f"📊 Charts disimpan di folder 'plt/'")

