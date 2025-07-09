import os
import requests
import pandas as pd
import numpy as np

# Import dari file terpisah
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler


def create_output_directories():
    print("🗂️  Setting up output directories...")
    directories = ["file", "plt"]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"📁 ✅ Created/verified directory: {directory}/")
        except PermissionError:
            print(f"❌ Permission denied creating: {directory}/")
        except Exception as e:
            print(f"❌ Error creating {directory}/: {e}")


def fetch_stock_data(symbol="TSLA", api_key="6DAM6N6ZSRQZNFR1"):
    # URL API Alpha Vantage
    url = "https://www.alphavantage.co/query"
    # Parameter untuk API
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": api_key,
    }

    try:
        # Kirim request ke API
        response = requests.get(url, params=params)
        data = response.json()

        # Cek apakah data berhasil diambil
        if "Time Series (Daily)" not in data:
            return create_synthetic_data()

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
        return create_synthetic_data()


def create_synthetic_data(days=5000):
    # Parameter untuk simulasi harga saham yang realistis
    start_price = 200.0
    volatility = 0.02  # 2% volatility per hari
    trend = 0.0005  # Small upward trend

    # Generate dates
    dates = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=days), periods=days, freq="D")

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
        high_price = max(open_price, close_price) + abs(np.random.normal(0, high_low_range / 2))
        low_price = min(open_price, close_price) - abs(np.random.normal(0, high_low_range / 2))

        # Volume (random but realistic)
        volume = int(np.random.normal(1_000_000, 300_000))

        data.append(
            {
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": max(volume, 100_000),
            }
        )

    # Create DataFrame
    df = pd.DataFrame(data, index=dates)

    return df


def normalize_data(df: pd.DataFrame, target_column="close"):
    # 1. Ambil data kolom target
    original_data = df[target_column].values

    # 2. Reshape untuk sklearn (perlu 2D array)
    data_reshaped = original_data.reshape(-1, 1)

    # 3. Normalisasi dengan MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data: NDArray[np.float64] = scaler.fit_transform(data_reshaped)

    return scaler, normalized_data


def create_sequences(normalized_data: NDArray[np.float64], window_size=5):
    # Validasi input
    if len(normalized_data) <= window_size:
        print(f"❌ Error: Data terlalu sedikit. Need > {window_size}, got {len(normalized_data)}")
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

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)

