# Fixed Implementation - No Data Leakage
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict


def normalize_data_properly(
    df: pd.DataFrame, target_column="close", train_ratio=0.7, val_ratio=0.15
):
    """
    Proper normalization WITHOUT data leakage
    """
    print(f"\n🔧 STEP 2: PROPER NORMALIZATION (NO LEAKAGE)")
    print("=" * 60)

    # 1. Get raw data
    data = df[target_column].values

    # 2. FIRST split, THEN normalize
    total_len = len(data)
    train_size = int(train_ratio * total_len)
    val_size = int(val_ratio * total_len)

    # Chronological split
    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    print(f"📊 Data split sizes:")
    print(f"   Train: {len(train_data)} samples")
    print(f"   Val:   {len(val_data)} samples")
    print(f"   Test:  {len(test_data)} samples")

    # 3. Fit scaler ONLY on training data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()

    # 4. Transform val/test using training statistics
    val_scaled = scaler.transform(val_data.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

    # 5. Check distributions
    print(f"\n📈 NORMALIZED DISTRIBUTIONS:")
    print(f"   Train: {np.mean(train_scaled):.6f} ± {np.std(train_scaled):.6f}")
    print(f"   Val:   {np.mean(val_scaled):.6f} ± {np.std(val_scaled):.6f}")
    print(f"   Test:  {np.mean(test_scaled):.6f} ± {np.std(test_scaled):.6f}")

    # Check for suspicious differences
    val_train_mean_diff = abs(np.mean(val_scaled) - np.mean(train_scaled))
    if val_train_mean_diff > 0.5:
        print(f"⚠️  WARNING: Large mean difference: {val_train_mean_diff:.3f}")
    else:
        print(f"✅ Normal mean difference: {val_train_mean_diff:.3f}")

    return {
        "train_data": train_scaled,
        "val_data": val_scaled,
        "test_data": test_scaled,
        "scaler": scaler,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
    }


def create_sequences_no_leakage(train_data, val_data, test_data, window_size=20):
    """
    Create sequences with proper separation (no leakage)
    """
    print(f"\n🔧 STEP 3: CREATE SEQUENCES (NO LEAKAGE)")
    print("=" * 60)

    def make_sequences(data, name):
        if len(data) <= window_size:
            print(f"❌ {name}: Not enough data ({len(data)} <= {window_size})")
            return np.array([]), np.array([])

        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size])

        return np.array(X), np.array(y)

    # Create sequences for each split
    train_X, train_y = make_sequences(train_data, "Train")
    val_X, val_y = make_sequences(val_data, "Val")
    test_X, test_y = make_sequences(test_data, "Test")

    print(f"📊 Sequences created:")
    print(f"   Train: {len(train_X)} sequences")
    print(f"   Val:   {len(val_X)} sequences")
    print(f"   Test:  {len(test_X)} sequences")

    # Verify no overlap
    if len(train_X) > 0 and len(val_X) > 0:
        last_train_seq = train_X[-1]
        first_val_seq = val_X[0]
        similarity = np.corrcoef(last_train_seq, first_val_seq)[0, 1]
        print(f"📊 Train-Val sequence correlation: {similarity:.4f}")

        if similarity > 0.9:
            print("⚠️  WARNING: High correlation between train and val!")
        else:
            print("✅ Good separation between train and val")

    return (train_X, train_y), (val_X, val_y), (test_X, test_y)


def analyze_fixed_distributions(train_data, val_data, test_data):
    """Analyze if the fix worked"""
    print(f"\n📊 FIXED DISTRIBUTION ANALYSIS")
    print("=" * 50)

    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_y = test_data

    print(f"📈 TARGET DISTRIBUTIONS:")
    print(f"   Train targets: {np.mean(train_y):.6f} ± {np.std(train_y):.6f}")
    print(f"   Val targets:   {np.mean(val_y):.6f} ± {np.std(val_y):.6f}")
    print(f"   Test targets:  {np.mean(test_y):.6f} ± {np.std(test_y):.6f}")

    # Expected: All should be similar now (no more 509% difference!)
    val_train_diff = abs(np.mean(val_y) - np.mean(train_y))
    val_train_ratio = val_train_diff / abs(np.mean(train_y)) * 100

    print(f"\n🎯 VALIDATION:")
    print(f"   Mean difference: {val_train_diff:.6f}")
    print(f"   Percentage diff: {val_train_ratio:.1f}%")

    if val_train_ratio < 50:  # Much better than 509%!
        print("✅ GOOD: Normal distribution difference")
    else:
        print("⚠️  Still high difference - check data generation")

    return val_train_ratio


# Test the fix
if __name__ == "__main__":
    # Create test data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")

    # Simulate price data with trend
    prices = []
    price = 200.0
    for i in range(1000):
        change = np.random.normal(0.001, 0.02)  # Small upward trend with volatility
        price *= 1 + change
        prices.append(price)

    df = pd.DataFrame({"date": dates, "close": prices})

    print("🧪 TESTING FIX FOR DATA LEAKAGE")
    print("=" * 50)

    # Apply the fix
    normalized_result = normalize_data_properly(df)

    sequences_result = create_sequences_no_leakage(
        normalized_result["train_data"],
        normalized_result["val_data"],
        normalized_result["test_data"],
    )

    analyze_fixed_distributions(*sequences_result)
