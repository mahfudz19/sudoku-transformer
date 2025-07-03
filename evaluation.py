# Step 7: Evaluation & Prediction untuk Stock Price Prediction
# Comprehensive model evaluation dan future prediction

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")


def evaluate_model(
    model: nn.Module, test_loader, scaler, device: str, verbose: bool = True
) -> Dict:
    if verbose:
        print(f"\n📊 STEP 7: MODEL EVALUATION")
        print("=" * 50)

    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in test_loader:
            # Move to device
            sequences = sequences.to(device)
            targets = targets.to(device)

            # Add channel dimension
            sequences = sequences.unsqueeze(-1)  # (batch, seq_len, 1)

            # Get predictions
            predictions = model(sequences)  # (batch, 1)

            # Convert to numpy and store
            predictions_np = predictions.cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()

            all_predictions.extend(predictions_np)
            all_targets.extend(targets_np)

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)

    # Inverse transform ke harga asli
    predictions_original = scaler.inverse_transform(
        predictions.reshape(-1, 1)
    ).flatten()
    targets_original = scaler.inverse_transform(targets.reshape(-1, 1)).flatten()

    # Calculate metrics
    mse = mean_squared_error(targets_original, predictions_original)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets_original, predictions_original)

    # MAPE (Mean Absolute Percentage Error)
    mape = (
        np.mean(np.abs((targets_original - predictions_original) / targets_original))
        * 100
    )

    # R² Score (coefficient of determination)
    ss_res = np.sum((targets_original - predictions_original) ** 2)
    ss_tot = np.sum((targets_original - np.mean(targets_original)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Directional Accuracy (apakah prediksi arah naik/turun benar)
    actual_direction = np.diff(targets_original) > 0
    predicted_direction = np.diff(predictions_original) > 0
    directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

    if verbose:
        print(f"🎯 EVALUATION RESULTS:")
        print(f"   📈 Test samples: {len(targets_original)}")
        print(
            f"   💰 Price range: ${targets_original.min():.2f} - ${targets_original.max():.2f}"
        )
        print(f"\n📊 REGRESSION METRICS:")
        print(f"   MSE:  {mse:.6f}")
        print(f"   RMSE: ${rmse:.2f}")
        print(f"   MAE:  ${mae:.2f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   R²:   {r2:.4f}")
        print(f"\n🎯 TRADING METRICS:")
        print(f"   Directional Accuracy: {directional_accuracy:.1f}%")

        # Performance interpretation
        print(f"\n💡 INTERPRETASI UNTUK JAVASCRIPT DEVELOPER:")
        if rmse < targets_original.mean() * 0.05:  # RMSE < 5% of mean price
            print(f"   ✅ Excellent: RMSE sangat rendah (${rmse:.2f})")
        elif rmse < targets_original.mean() * 0.10:  # RMSE < 10% of mean price
            print(f"   👍 Good: RMSE acceptable (${rmse:.2f})")
        else:
            print(f"   ⚠️  Warning: RMSE tinggi (${rmse:.2f})")

        if mape < 5:
            print(f"   ✅ Excellent: MAPE sangat rendah ({mape:.1f}%)")
        elif mape < 10:
            print(f"   👍 Good: MAPE acceptable ({mape:.1f}%)")
        else:
            print(f"   ⚠️  Warning: MAPE tinggi ({mape:.1f}%)")

        if directional_accuracy > 60:
            print(f"   ✅ Good direction prediction ({directional_accuracy:.1f}%)")
        else:
            print(f"   ⚠️ Poor direction prediction ({directional_accuracy:.1f}%)")

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
        "predictions_normalized": predictions,
        "targets_normalized": targets,
        "predictions_original": predictions_original,
        "targets_original": targets_original,
        "num_samples": len(targets_original),
    }


def plot_predictions(evaluation_results: Dict, save_plot: bool = True):
    """
    Plot prediksi vs actual prices dengan berbagai visualisasi

    📊 Untuk JavaScript developer:
    Seperti membuat multiple charts dengan Chart.js:

    const charts = [
        createTimeSeriesChart(predictions, actual),
        createScatterPlot(predictions, actual),
        createErrorDistribution(errors)
    ];

    Parameters:
    - evaluation_results: hasil dari evaluate_model()
    - save_plot: save plots to files
    """
    print(f"\n📊 PLOTTING PREDICTIONS")
    print("=" * 40)

    predictions = evaluation_results["predictions_original"]
    targets = evaluation_results["targets_original"]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Stock Price Prediction Results", fontsize=16, fontweight="bold")

    # Plot 1: Time series comparison
    ax1 = axes[0, 0]
    x_axis = range(len(predictions))
    ax1.plot(x_axis, targets, label="Actual Prices", color="blue", linewidth=2)
    ax1.plot(
        x_axis,
        predictions,
        label="Predicted Prices",
        color="red",
        linewidth=2,
        alpha=0.8,
    )
    ax1.set_title("Predictions vs Actual Prices")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot (Perfect prediction = diagonal line)
    ax2 = axes[0, 1]
    ax2.scatter(targets, predictions, alpha=0.6, color="purple")
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax2.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect Prediction",
    )
    ax2.set_title("Prediction Accuracy Scatter Plot")
    ax2.set_xlabel("Actual Prices ($)")
    ax2.set_ylabel("Predicted Prices ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction errors
    ax3 = axes[1, 0]
    errors = predictions - targets
    ax3.plot(x_axis, errors, color="red", alpha=0.7)
    ax3.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax3.set_title("Prediction Errors Over Time")
    ax3.set_xlabel("Time Steps")
    ax3.set_ylabel("Error ($)")
    ax3.grid(True, alpha=0.3)

    # Add error statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax3.axhline(
        y=mean_error,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Mean Error: ${mean_error:.2f}",
    )
    ax3.axhline(y=mean_error + std_error, color="orange", linestyle="--", alpha=0.7)
    ax3.axhline(
        y=mean_error - std_error,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"±1 STD: ${std_error:.2f}",
    )
    ax3.legend()

    # Plot 4: Error distribution histogram
    ax4 = axes[1, 1]
    ax4.hist(errors, bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax4.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero Error")
    ax4.axvline(
        x=mean_error,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean: ${mean_error:.2f}",
    )
    ax4.set_title("Error Distribution")
    ax4.set_xlabel("Prediction Error ($)")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        plot_path = "plt/prediction_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"💾 Prediction plots saved to: {plot_path}")

    plt.show()

    # Print plot summary
    print(f"\n📈 PLOT SUMMARY:")
    print(f"   🎯 Mean Error: ${mean_error:.2f}")
    print(f"   📊 Error Std Dev: ${std_error:.2f}")
    print(f"   📈 Max Error: ${errors.max():.2f}")
    print(f"   📉 Min Error: ${errors.min():.2f}")


def predict_future_prices(
    model: nn.Module,
    last_sequence: np.ndarray,
    scaler,
    device: str,
    num_days: int = 5,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict future stock prices menggunakan autoregressive approach

    🔮 Untuk JavaScript developer:
    Seperti recursive prediction:

    async function predictFuture(model, lastPrices, days) {
        let predictions = [];
        let currentSequence = [...lastPrices];

        for (let i = 0; i < days; i++) {
            const nextPrice = await model.predict(currentSequence);
            predictions.push(nextPrice);

            // Update sequence: remove first, add prediction
            currentSequence.shift();
            currentSequence.push(nextPrice);
        }
        return predictions;
    }

    Parameters:
    - model: trained model
    - last_sequence: sequence terakhir dari data (window_size,)
    - scaler: untuk inverse transform
    - device: 'cpu' atau 'cuda'
    - num_days: berapa hari ke depan diprediksi
    - verbose: print details

    Returns:
    - (normalized_predictions, original_predictions)
    """
    if verbose:
        print(f"\n🔮 FUTURE PRICE PREDICTION")
        print("=" * 40)
        print(f"🎯 Predicting {num_days} days into the future")

    model.eval()
    predictions_normalized = []
    predictions_original = []

    # Copy sequence untuk manipulation
    current_sequence = last_sequence.copy()

    with torch.no_grad():
        for day in range(num_days):
            # Convert to tensor
            sequence_tensor = (
                torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1)
            )  # (1, seq_len, 1)
            sequence_tensor = sequence_tensor.to(device)

            # Get prediction
            prediction = model(sequence_tensor)  # (1, 1)
            pred_value = prediction.cpu().numpy().flatten()[0]

            # Store predictions
            predictions_normalized.append(pred_value)

            # Inverse transform untuk harga asli
            pred_original = scaler.inverse_transform([[pred_value]])[0][0]
            predictions_original.append(pred_original)

            # Update sequence untuk next prediction (sliding window)
            # Remove first element, add new prediction
            current_sequence = np.append(current_sequence[1:], pred_value)

            if verbose:
                print(f"   Day {day+1}: ${pred_original:.2f}")

    predictions_normalized = np.array(predictions_normalized)
    predictions_original = np.array(predictions_original)

    if verbose:
        print(f"\n📊 FUTURE PREDICTION SUMMARY:")
        print(f"   🎯 Predicted days: {num_days}")
        print(f"   📈 Highest price: ${predictions_original.max():.2f}")
        print(f"   📉 Lowest price: ${predictions_original.min():.2f}")
        print(f"   📊 Average price: ${predictions_original.mean():.2f}")

        # Calculate trend
        total_change = predictions_original[-1] - predictions_original[0]
        percent_change = (total_change / predictions_original[0]) * 100

        if total_change > 0:
            print(f"   📈 Trend: BULLISH (+{percent_change:.1f}%)")
        else:
            print(f"   📉 Trend: BEARISH ({percent_change:.1f}%)")

    return predictions_normalized, predictions_original


def plot_future_predictions(
    historical_prices: np.ndarray,
    future_predictions: np.ndarray,
    num_historical_days: int = 30,
    save_plot: bool = True,
):
    """
    Plot historical prices dengan future predictions

    📊 Untuk JavaScript developer:
    Seperti kombinasi historical + forecasted line chart:

    const chartData = {
        labels: [...historicalDates, ...futureDates],
        datasets: [
            { label: 'Historical', data: historicalPrices },
            { label: 'Predicted', data: futurePredictions }
        ]
    };

    Parameters:
    - historical_prices: harga historical (original scale)
    - future_predictions: prediksi masa depan (original scale)
    - num_historical_days: berapa hari historical yang ditampilkan
    - save_plot: save plot to file
    """
    print(f"\n📊 PLOTTING FUTURE PREDICTIONS")
    print("=" * 40)

    # Ambil historical data terakhir
    recent_historical = historical_prices[-num_historical_days:]

    # Create time axis
    historical_x = range(len(recent_historical))
    future_x = range(
        len(recent_historical), len(recent_historical) + len(future_predictions)
    )

    # Create plot
    plt.figure(figsize=(12, 8))

    # Plot historical prices
    plt.plot(
        historical_x,
        recent_historical,
        label=f"Historical Prices ({num_historical_days} days)",
        color="blue",
        linewidth=2,
        marker="o",
        markersize=4,
    )

    # Plot future predictions
    plt.plot(
        future_x,
        future_predictions,
        label=f"Future Predictions ({len(future_predictions)} days)",
        color="red",
        linewidth=2,
        marker="s",
        markersize=4,
        linestyle="--",
    )

    # Connect last historical to first prediction
    plt.plot(
        [historical_x[-1], future_x[0]],
        [recent_historical[-1], future_predictions[0]],
        color="green",
        linewidth=2,
        alpha=0.7,
    )

    # Add vertical line to separate historical and future
    plt.axvline(
        x=len(recent_historical) - 0.5,
        color="gray",
        linestyle=":",
        alpha=0.7,
        linewidth=2,
    )
    plt.text(
        len(recent_historical) - 0.5,
        plt.ylim()[1] * 0.95,
        "Prediction Start",
        rotation=90,
        verticalalignment="top",
        fontsize=10,
    )

    # Formatting
    plt.title(
        "Stock Price: Historical vs Future Predictions", fontsize=14, fontweight="bold"
    )
    plt.xlabel("Time Steps")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add price annotations
    plt.annotate(
        f"Last: ${recent_historical[-1]:.2f}",
        xy=(len(recent_historical) - 1, recent_historical[-1]),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="blue", alpha=0.3),
    )

    plt.annotate(
        f"Predicted: ${future_predictions[-1]:.2f}",
        xy=(future_x[-1], future_predictions[-1]),
        xytext=(10, 10),
        textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3),
    )

    plt.tight_layout()

    if save_plot:
        plot_path = "plt/future_predictions.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"💾 Future prediction plot saved to: {plot_path}")

    plt.show()


def comprehensive_evaluation(
    model: nn.Module,
    test_loader,
    scaler,
    device: str,
    last_sequence: np.ndarray,
    historical_prices: np.ndarray,
    future_days: int = 5,
) -> Dict:
    # 1. Evaluate on test set
    test_results = evaluate_model(model, test_loader, scaler, device, verbose=True)

    # 2. Plot test predictions
    plot_predictions(test_results, save_plot=True)

    # 3. Future predictions
    future_norm, future_orig = predict_future_prices(
        model, last_sequence, scaler, device, future_days, verbose=True
    )

    # 4. Plot future predictions
    plot_future_predictions(historical_prices, future_orig, save_plot=True)

    # 5. Summary report
    print(f"\n📋 FINAL ANALYSIS REPORT")
    print("=" * 50)
    print(f"🧪 TEST SET PERFORMANCE:")
    print(f"   📊 Samples evaluated: {test_results['num_samples']}")
    print(f"   📈 RMSE: ${test_results['rmse']:.2f}")
    print(f"   📊 MAPE: {test_results['mape']:.1f}%")
    print(f"   🎯 Directional Accuracy: {test_results['directional_accuracy']:.1f}%")
    print(f"   📈 R² Score: {test_results['r2']:.4f}")

    print(f"\n🔮 FUTURE PREDICTIONS:")
    print(f"   📅 Days predicted: {future_days}")
    print(
        f"   📈 Expected trend: {'+' if future_orig[-1] > future_orig[0] else ''}{((future_orig[-1]/future_orig[0]-1)*100):.1f}%"
    )
    print(f"   💰 Price range: ${future_orig.min():.2f} - ${future_orig.max():.2f}")

    # Trading recommendation
    print(f"\n💡 TRADING INSIGHTS:")
    if test_results["directional_accuracy"] > 60 and test_results["mape"] < 10:
        print(f"   ✅ Model shows good predictive capability")
        if future_orig[-1] > future_orig[0]:
            print(f"   📈 Consider LONG position (bullish trend predicted)")
        else:
            print(f"   📉 Consider SHORT position (bearish trend predicted)")
    else:
        print(f"   ⚠️  Model performance needs improvement")
        print(f"   🔄 Consider retraining with more data or tuning hyperparameters")

    return {
        "test_results": test_results,
        "future_predictions_normalized": future_norm,
        "future_predictions_original": future_orig,
        "model_performance_summary": {
            "rmse": test_results["rmse"],
            "mape": test_results["mape"],
            "directional_accuracy": test_results["directional_accuracy"],
            "r2": test_results["r2"],
        },
    }


# Test function
def test_evaluation():
    """Test evaluation functions dengan dummy data"""
    print("🧪 Testing evaluation functions...")

    try:
        from model import create_model
        from data_loader import create_dataloaders
        from sklearn.preprocessing import MinMaxScaler

        # Create dummy data
        dummy_train = (np.random.rand(50, 5), np.random.rand(50))
        dummy_val = (np.random.rand(10, 5), np.random.rand(10))
        dummy_test = (np.random.rand(20, 5), np.random.rand(20))

        # Create dataloaders
        dataloader_components = create_dataloaders(
            dummy_train, dummy_val, dummy_test, batch_size=8
        )

        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_model(device=device)

        # Create dummy scaler
        scaler = MinMaxScaler()
        dummy_data = np.random.rand(100, 1) * 100 + 100  # $100-$200
        scaler.fit(dummy_data)

        # Test evaluation
        test_results = evaluate_model(
            model, dataloader_components["test_loader"], scaler, device, verbose=True
        )

        # Test future prediction
        last_seq = np.random.rand(5)
        future_norm, future_orig = predict_future_prices(
            model, last_seq, scaler, device, num_days=3, verbose=True
        )

        print(f"✅ Evaluation test passed!")
        return True

    except Exception as e:
        print(f"❌ Evaluation test failed: {e}")
        return False


if __name__ == "__main__":
    # Test evaluation functions
    test_evaluation()
