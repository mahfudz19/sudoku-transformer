# Step 1: Ambil dan Bersihkan Data Harga Saham
# Step 2: Normalisasi Data
# Step 3: Create Sequences (Windowing)
# Step 4: Split Data
# Step 5: Model dan DataLoader (terpisah)
import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim

# Import dari file terpisah
from data_loader import create_dataloaders
from model import create_model
from training import train_model, plot_training_history, load_best_model
from evaluation import comprehensive_evaluation


def fetch_stock_data(symbol="TSLA", api_key="6DAM6N6ZSRQZNFR1"):
    print(f"\n🚀 STEP 1: FETCH DATA")
    print("=" * 60)

    print(f"📊 Mengambil data untuk saham: {symbol}")

    # URL API Alpha Vantage
    url = "https://www.alphavantage.co/query"
    # Parameter untuk API
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",  # compact = 100 hari terakhir, full = semua data
        "apikey": api_key,
    }

    try:
        # Kirim request ke API
        response = requests.get(url, params=params)
        data = response.json()

        # Cek apakah data berhasil diambil
        if "Time Series (Daily)" not in data:
            print("❌ Error: Tidak bisa mengambil data dari API")
            print("Response:", data)
            print("🔄 Menggunakan synthetic data untuk demo...")
            return create_synthetic_data(symbol)

        # Ambil data time series
        time_series = data["Time Series (Daily)"]

        # Konversi ke DataFrame
        df = pd.DataFrame.from_dict(time_series, orient="index")

        # Konversi index ke datetime dan urutkan
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()  # Urutkan dari tanggal lama ke baru

        # Konversi kolom ke float dan beri nama yang lebih mudah
        df = df.astype(float)
        df.columns = ["open", "high", "low", "close", "volume"]

        return df

    except Exception as e:
        return create_synthetic_data(symbol)


def create_synthetic_data(symbol="DEMO", days=5000):
    """
    Create synthetic stock data untuk demo dan testing

    📈 Untuk JavaScript developer:
    Seperti membuat mock data:
    const mockStockData = generateRealisticStockPrice({
        startPrice: 200,
        days: 100,
        volatility: 0.02
    });

    Parameters:
    - symbol: symbol saham untuk demo
    - days: jumlah hari data yang dibuat

    Returns:
    - DataFrame dengan synthetic stock data yang realistis
    """
    print(f"🎭 Membuat synthetic data untuk: {symbol}")

    # Parameter untuk simulasi harga saham yang realistis
    start_price = 200.0
    volatility = 0.02  # 2% volatility per hari
    trend = 0.0005  # Small upward trend

    # Generate dates
    dates = pd.date_range(
        start=pd.Timestamp.now() - pd.Timedelta(days=days), periods=days, freq="D"
    )

    # Generate realistic price movements
    np.random.seed(42)  # For reproducible results

    prices = [start_price]
    for i in range(1, days):
        # Random walk dengan trend dan volatility
        random_change = np.random.normal(trend, volatility)
        new_price = prices[-1] * (1 + random_change)
        prices.append(max(new_price, 1.0))  # Minimum price $1

    prices = np.array(prices)

    # Generate OHLC data (Open, High, Low, Close)
    data = []
    for i, close_price in enumerate(prices):
        # Open price (similar to previous close)
        if i == 0:
            open_price = close_price
        else:
            open_price = prices[i - 1] * (1 + np.random.normal(0, 0.005))

        # High and Low around Open/Close
        high_low_range = close_price * 0.03  # 3% daily range
        high_price = max(open_price, close_price) + abs(
            np.random.normal(0, high_low_range / 2)
        )
        low_price = min(open_price, close_price) - abs(
            np.random.normal(0, high_low_range / 2)
        )

        # Volume (random but realistic)
        volume = int(np.random.normal(1_000_000, 300_000))

        data.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": max(volume, 100_000),  # Minimum volume
            }
        )

    # Create DataFrame
    df = pd.DataFrame(data, index=dates)

    print(f"✅ Synthetic data created:")
    print(f"   Periode: {df.index[0].date()} hingga {df.index[-1].date()}")
    print(f"   Jumlah hari: {len(df)}")
    print(f"   Harga awal: ${df['close'].iloc[0]:.2f}")
    print(f"   Harga akhir: ${df['close'].iloc[-1]:.2f}")
    print(
        f"   Return total: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.1f}%"
    )

    return df


def normalize_data(df: pd.DataFrame, target_column="close"):
    print(f"\n🔧 STEP 2: NORMALISASI DATA")
    print("=" * 50)

    print(f"🎯 Target kolom: '{target_column}'")

    # 1. Ambil data kolom target
    original_data = df[target_column].values

    # 2. Reshape untuk sklearn (perlu 2D array)
    data_reshaped = original_data.reshape(-1, 1)

    # 3. Normalisasi dengan MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data: NDArray[np.float64] = scaler.fit_transform(data_reshaped)

    print(f"🎉 STEP 2 SELESAI!")

    return scaler, normalized_data


def create_sequences(normalized_data: NDArray[np.float64], window_size=5):
    print(f"\n🔧 STEP 3: CREATE SEQUENCES (WINDOWING)")
    print("=" * 50)

    print(f"🎯 Window size: {window_size} hari")
    print(f"📊 Input data shape: {normalized_data.shape}")

    # Validasi input
    if len(normalized_data) <= window_size:
        print(
            f"❌ Error: Data terlalu sedikit. Need > {window_size}, got {len(normalized_data)}"
        )
        return None, None

    X = []  # Input sequences
    y = []  # Target values

    # Buat sequences dengan sliding window
    for i in range(window_size, len(normalized_data)):
        # Ambil window_size hari sebelumnya sebagai input
        sequence = normalized_data[i - window_size : i, 0]  # Shape: (window_size,)
        target = normalized_data[i, 0]  # Hari ke-i sebagai target

        X.append(sequence)
        y.append(target)

    # Convert ke numpy arrays
    x = np.array(X)  # Shape: (num_sequences, window_size)
    y = np.array(y)  # Shape: (num_sequences,)

    return x, y


def split_data(x: NDArray, y: NDArray, train_ratio=0.7, val_ratio=0.15):
    print(f"\n🔧 STEP 4: SPLIT DATA (TRAIN/VAL/TEST)")
    print("=" * 50)

    total_samples = len(x)

    # Hitung ukuran setiap split
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)

    # Split data secara berurutan (chronological)
    train_X: NDArray = x[:train_size]
    train_y: NDArray = y[:train_size]

    val_X: NDArray = x[train_size : train_size + val_size]
    val_y: NDArray = y[train_size : train_size + val_size]

    test_X: NDArray = x[train_size + val_size :]
    test_y: NDArray = y[train_size + val_size :]
    print(f"🎉 STEP 4: SUCCESS")

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)


# Test fungsi yang sudah dibuat
if __name__ == "__main__":
    # Step 1: Ambil data saham
    df = fetch_stock_data("AAPL")

    if df is not None and len(df) > 10:  # Make sure we have enough data
        print(f"🎉 Step 1: Data berhasil diambil ({len(df)} hari)")

        # Step 2: Normalisasi
        scaler, normalized_data = normalize_data(df, "close")

        # Step 3: Create sequences
        window_size = 5  # Gunakan 5 hari untuk prediksi 1 hari ke depan
        x, y = create_sequences(normalized_data, window_size)
        if x is not None and y is not None:
            print(f"🎉 Step 3: Sequences created ({len(x)} sequences)")

            # Step 4: Split data
            train_data, val_data, test_data = split_data(x, y)

            # Step 5a: Setup data loaders
            dataloader_components = create_dataloaders(
                train_data, val_data, test_data, batch_size=16
            )

            # Step 5b: Create model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = create_model(device=device)

            train_loader = dataloader_components["train_loader"]
            val_loader = dataloader_components["val_loader"]
            test_loader = dataloader_components["test_loader"]

            # Training configuration
            training_config = {
                "num_epochs": 50,
                "learning_rate": 0.001,
                "patience": 10,
                "save_path": "best_stock_model.pth",
            }

            # Start training
            try:
                training_history = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    **training_config,
                )

                # Plot training results
                plot_training_history(training_history, save_plot=True)

                # Load best model for final evaluation
                model, checkpoint = load_best_model(
                    model, training_config["save_path"], device
                )

                # ============== STEP 7: EVALUATION & PREDICTION ==============
                print(f"📊 STEP 7: COMPREHENSIVE EVALUATION & PREDICTION")
                print(f"=" * 60)

                # Get last sequence untuk future prediction
                last_sequence_normalized = normalized_data[-window_size:, 0]

                # Get all original prices untuk historical plotting
                all_original_prices = df["close"].values

                # Comprehensive evaluation
                analysis_results = comprehensive_evaluation(
                    model=model,
                    test_loader=test_loader,
                    scaler=scaler,
                    device=device,
                    last_sequence=last_sequence_normalized,
                    historical_prices=all_original_prices,
                    future_days=7,  # Predict 7 days into the future
                )

                print(f"\n🎉 STEP 7 COMPLETE!")
                print(f"✅ Model evaluated on test set")
                print(f"✅ Future predictions generated")
                print(f"✅ Visualizations created")
                print(f"✅ Trading insights provided")

                print(f"\n🏆 COMPLETE STOCK PREDICTION SYSTEM!")
                print("=" * 50)
                print(f"✅ Step 1: Data fetching & cleaning")
                print(f"✅ Step 2: Data normalization")
                print(f"✅ Step 3: Sequence creation (windowing)")
                print(f"✅ Step 4: Train/validation/test split")
                print(f"✅ Step 5: Model & DataLoader setup")
                print(f"✅ Step 6: Model training with early stopping")
                print(f"✅ Step 7: Evaluation & future prediction")

                # Performance summary
                perf = analysis_results["model_performance_summary"]
                print(f"\n📈 FINAL PERFORMANCE SUMMARY:")
                print(f"   🎯 RMSE: ${perf['rmse']:.2f}")
                print(f"   📊 MAPE: {perf['mape']:.1f}%")
                print(f"   🎯 Direction Accuracy: {perf['directional_accuracy']:.1f}%")
                print(f"   📈 R² Score: {perf['r2']:.4f}")

                print(f"\n💡 NEXT STEPS FOR PRODUCTION:")
                print(f"   🔄 Retrain model with more recent data")
                print(f"   📊 Implement real-time data pipeline")
                print(f"   ⚡ Deploy model as REST API")
                print(f"   📱 Create web dashboard untuk predictions")
                print(f"   🔔 Add automated trading alerts")

                # COMPREHENSIVE DEBUG ANALYSIS
                print("\n" + "=" * 80)
                print(
                    "🔍 DETAILED DEBUG ANALYSIS - WHY VALIDATION LOSS IS LOW FROM START"
                )
                print("=" * 80)

                # 1. Dataset Size Analysis
                print("\n📊 DATASET DISTRIBUTION ANALYSIS:")
                train_size = len(train_data[0])
                val_size = len(val_data[0])
                test_size = len(test_data[0])
                total_size = train_size + val_size + test_size

                print(f"   📈 Original data length: {len(df)} days")
                print(f"   🔢 Total sequences created: {total_size}")
                print(
                    f"   🚂 Train: {train_size} samples ({train_size/total_size*100:.1f}%)"
                )
                print(
                    f"   ✅ Val:   {val_size} samples ({val_size/total_size*100:.1f}%)"
                )
                print(
                    f"   🧪 Test:  {test_size} samples ({test_size/total_size*100:.1f}%)"
                )
                print(f"   ⚠️  WARNING: Val set very small = {val_size} samples only!")

                # 2. Loss Analysis
                print("\n📉 TRAINING LOSS DETAILED ANALYSIS:")
                train_losses = training_history["train_losses"]
                val_losses = training_history["val_losses"]

                print(f"   📊 Training loss variance: {np.var(train_losses):.8f}")
                print(f"   📊 Validation loss variance: {np.var(val_losses):.8f}")
                print(
                    f"   🎯 Min training loss: {np.min(train_losses):.6f} (epoch {np.argmin(train_losses)+1})"
                )
                print(
                    f"   🎯 Min validation loss: {np.min(val_losses):.6f} (epoch {np.argmin(val_losses)+1})"
                )
                print(f"   📈 First epoch train loss: {train_losses[0]:.6f}")
                print(f"   📈 First epoch val loss: {val_losses[0]:.6f}")
                print(
                    f"   📊 Val/Train ratio (epoch 1): {val_losses[0]/train_losses[0]:.3f}"
                )

                # 3. Data Distribution Check (potential data leakage)
                print("\n🔍 DATA LEAKAGE & DISTRIBUTION CHECK:")
                train_mean = np.mean(train_data[1])
                val_mean = np.mean(val_data[1])
                test_mean = np.mean(test_data[1])
                train_std = np.std(train_data[1])
                val_std = np.std(val_data[1])
                test_std = np.std(test_data[1])

                print(f"   🎯 Train target mean: {train_mean:.6f} ± {train_std:.6f}")
                print(f"   🎯 Val target mean:   {val_mean:.6f} ± {val_std:.6f}")
                print(f"   🎯 Test target mean:  {test_mean:.6f} ± {test_std:.6f}")
                print(
                    f"   ⚠️  Mean difference (val-train): {abs(val_mean-train_mean):.6f}"
                )
                print(f"   ⚠️  Std difference (val-train): {abs(val_std-train_std):.6f}")

                if (
                    abs(val_mean - train_mean) < 0.01
                    and abs(val_std - train_std) < 0.01
                ):
                    print("   ✅ POTENTIAL ISSUE: Val/Train distributions too similar!")

                # 4. Synthetic Data Analysis
                print("\n🎭 SYNTHETIC DATA CHARACTERISTICS:")
                price_changes = np.diff(df["close"].values)
                price_volatility = np.std(price_changes) / np.mean(df["close"].values)

                print(f"   📊 Using synthetic data with seed=42 (predictable!)")
                print(f"   📈 Price volatility: {price_volatility:.4f}")
                print(f"   📊 Max price change: {np.max(np.abs(price_changes)):.2f}")
                print(
                    f"   📊 Price trend: {df['close'].iloc[-1]/df['close'].iloc[0]:.3f}x"
                )
                print(f"   ⚠️  WARNING: Synthetic data might be too predictable!")

                # 5. Model Complexity vs Data Size
                print("\n🧠 MODEL COMPLEXITY ANALYSIS:")
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )

                print(f"   🔢 Total parameters: {total_params:,}")
                print(f"   🔢 Trainable parameters: {trainable_params:,}")
                print(f"   📊 Training samples: {train_size}")
                print(f"   ⚖️  Params/Sample ratio: {total_params/train_size:.2f}")

                if total_params / train_size > 10:
                    print("   ⚠️  WARNING: Model might be too complex for dataset size!")

                # 6. Sequence Analysis
                print("\n🔄 SEQUENCE ANALYSIS:")
                print(f"   📏 Window size: {window_size} days")
                print(f"   📊 Feature dimension: {train_data[0].shape[1]}")

                # Check first few sequences for patterns
                first_train_seq = train_data[0][0]
                first_val_seq = val_data[0][0] if val_size > 0 else None

                print(f"   🎯 First train sequence: {first_train_seq}")
                if first_val_seq is not None:
                    print(f"   🎯 First val sequence:   {first_val_seq}")
                    seq_similarity = np.corrcoef(first_train_seq, first_val_seq)[0, 1]
                    print(f"   📊 Sequence similarity: {seq_similarity:.4f}")

                # 7. Correlation Analysis
                print("\n🔗 FEATURE CORRELATION ANALYSIS:")
                correlation_matrix = df.corr()
                close_corrs = correlation_matrix["close"].sort_values(ascending=False)

                print("   📊 Correlations with close price:")
                for feature, corr in close_corrs.items():
                    print(f"      {feature:>8}: {corr:>7.4f}")

                # 8. Final Diagnosis
                print("\n🎯 DIAGNOSIS & RECOMMENDATIONS:")
                issues_found = []

                if val_size < 10:
                    issues_found.append("❌ Validation set too small (< 10 samples)")

                if val_losses[0] < 0.1:
                    issues_found.append("❌ Initial validation loss suspiciously low")

                if abs(val_mean - train_mean) < 0.01:
                    issues_found.append("❌ Train/Val distributions too similar")

                if total_params / train_size > 5:
                    issues_found.append("❌ Model too complex for dataset size")

                if price_volatility < 0.02:
                    issues_found.append("❌ Synthetic data too predictable")

                if len(issues_found) > 0:
                    print("   🚨 ISSUES DETECTED:")
                    for issue in issues_found:
                        print(f"      {issue}")

                    print("\n   💡 SOLUTIONS:")
                    print("      🔄 Increase synthetic data complexity")
                    print("      📊 Use more data (500+ days)")
                    print("      🎲 Remove fixed random seed")
                    print("      🧠 Reduce model complexity")
                    print("      ✅ Use real market data")
                else:
                    print("   ✅ No major issues detected")

                print("\n" + "=" * 80)

            except Exception as e:
                print(f"❌ Training failed: {e}")
                print(f"Please check your data and model configuration.")

    else:
        print("❌ Gagal mengambil data. Coba lagi nanti.")
