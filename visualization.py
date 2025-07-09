import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, Any
import os

def plot_training_history(history: Dict[str, list], save_path: str = "plt/training_history.png"):
    """
    Plot training dan validation loss curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss Curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Tambahkan annotations untuk min validation loss
    min_val_idx = np.argmin(history['val_loss'])
    min_val_loss = history['val_loss'][min_val_idx]
    ax1.annotate(f'Best Val Loss: {min_val_loss:.6f}\nEpoch: {min_val_idx + 1}',
                xy=(min_val_idx + 1, min_val_loss),
                xytext=(min_val_idx + 1 + len(epochs) * 0.1, min_val_loss + max(history['val_loss']) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    # Plot 2: Loss Difference (Overfitting Detection)
    loss_diff = np.array(history['train_loss']) - np.array(history['val_loss'])
    ax2.plot(epochs, loss_diff, 'g-', label='Train Loss - Val Loss', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_title('Overfitting Detection', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss Difference')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Color zones for overfitting detection
    ax2.fill_between(epochs, loss_diff, 0, where=(loss_diff > 0), 
                     color='red', alpha=0.2, label='Potential Overfitting')
    ax2.fill_between(epochs, loss_diff, 0, where=(loss_diff <= 0), 
                     color='green', alpha=0.2, label='Good Generalization')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Training history plot saved to: {save_path}")

def plot_prediction_accuracy(results: Dict[str, Any], save_path: str = "plt/prediction_accuracy.png"):
    """
    Plot prediksi vs actual untuk menilai akurasi model
    """
    predictions = results['predictions']
    actuals = results['actuals']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Scatter Plot (Predictions vs Actuals)
    ax1.scatter(actuals, predictions, alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual Prices ($)')
    ax1.set_ylabel('Predicted Prices ($)')
    ax1.set_title('Predictions vs Actuals', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(actuals, predictions)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    # Plot 2: Time Series Comparison
    sample_size = min(100, len(predictions))  # Show last 100 predictions
    indices = range(sample_size)
    
    ax2.plot(indices, actuals[:sample_size], 'b-', label='Actual', linewidth=2)
    ax2.plot(indices, predictions[:sample_size], 'r-', label='Predicted', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Price ($)')
    ax2.set_title(f'Time Series Comparison (Last {sample_size} samples)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error Distribution
    errors = predictions - actuals
    ax3.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(errors):.2f}')
    ax3.axvline(0, color='green', linestyle='-', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Prediction Error ($)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Metrics Summary
    metrics = {
        'MAE': results['mae'],
        'RMSE': results['rmse'],
        'MAPE': results['mape'],
        'Test Loss': results['test_loss']
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax4.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    ax4.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Value')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Prediction accuracy plot saved to: {save_path}")

def plot_comprehensive_analysis(history: Dict[str, list], results: Dict[str, Any], 
                              save_path: str = "plt/comprehensive_analysis.png"):
    """
    Comprehensive analysis plot combining training and evaluation
    """
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Training Loss (Top Left)
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Predictions vs Actuals (Top Right)
    ax2 = fig.add_subplot(gs[0, 2:])
    predictions = results['predictions']
    actuals = results['actuals']
    ax2.scatter(actuals, predictions, alpha=0.6, s=20)
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    ax2.set_xlabel('Actual Prices ($)')
    ax2.set_ylabel('Predicted Prices ($)')
    ax2.set_title('Prediction Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Time Series (Middle, Full Width)
    ax3 = fig.add_subplot(gs[1, :])
    sample_size = min(200, len(predictions))
    indices = range(sample_size)
    ax3.plot(indices, actuals[:sample_size], 'b-', label='Actual', linewidth=2)
    ax3.plot(indices, predictions[:sample_size], 'r-', label='Predicted', linewidth=2, alpha=0.8)
    ax3.fill_between(indices, actuals[:sample_size], predictions[:sample_size], 
                     alpha=0.2, color='gray', label='Prediction Error')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Price ($)')
    ax3.set_title(f'Time Series Comparison (Last {sample_size} samples)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Distribution (Bottom Left)
    ax4 = fig.add_subplot(gs[2, :2])
    errors = predictions - actuals
    ax4.hist(errors, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2)
    ax4.axvline(0, color='green', linestyle='-', linewidth=2)
    ax4.set_xlabel('Prediction Error ($)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Metrics Summary (Bottom Right)
    ax5 = fig.add_subplot(gs[2, 2:])
    metrics = {
        'MAE ($)': results['mae'],
        'RMSE ($)': results['rmse'],
        'MAPE (%)': results['mape'],
        'Correlation': np.corrcoef(actuals, predictions)[0, 1]
    }
    
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    bars = ax5.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    ax5.set_title('Performance Summary', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Value')
    
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('🤖 Stock Price Prediction Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Comprehensive analysis saved to: {save_path}")

def generate_model_report(history: Dict[str, list], results: Dict[str, Any], 
                         model_name: str = "TransformerStockPredictor"):
    """
    Generate a text report of model performance
    """
    print("\n" + "="*80)
    print(f"📊 MODEL PERFORMANCE REPORT: {model_name}")
    print("="*80)
    
    # Training Summary
    print(f"\n🏋️ TRAINING SUMMARY:")
    print(f"   └─ Total Epochs: {len(history['train_loss'])}")
    print(f"   └─ Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"   └─ Final Val Loss: {history['val_loss'][-1]:.6f}")
    print(f"   └─ Best Val Loss: {min(history['val_loss']):.6f} (Epoch {np.argmin(history['val_loss']) + 1})")
    
    # Overfitting Analysis
    final_diff = history['train_loss'][-1] - history['val_loss'][-1]
    if final_diff > 0.01:
        overfitting_status = "⚠️ Potential Overfitting"
    elif final_diff > 0.005:
        overfitting_status = "⚡ Slight Overfitting"
    else:
        overfitting_status = "✅ Good Generalization"
    
    print(f"   └─ Overfitting Status: {overfitting_status}")
    
    # Evaluation Metrics
    print(f"\n📈 EVALUATION METRICS:")
    print(f"   └─ MAE (Mean Absolute Error): ${results['mae']:.2f}")
    print(f"   └─ RMSE (Root Mean Square Error): ${results['rmse']:.2f}")
    print(f"   └─ MAPE (Mean Absolute Percentage Error): {results['mape']:.2f}%")
    print(f"   └─ Test Loss (MSE): {results['test_loss']:.6f}")
    
    # Performance Assessment
    correlation = np.corrcoef(results['actuals'], results['predictions'])[0, 1]
    print(f"   └─ Correlation: {correlation:.3f}")
    
    if results['mape'] < 10:
        performance = "🏆 Excellent"
    elif results['mape'] < 20:
        performance = "👍 Good"
    elif results['mape'] < 30:
        performance = "⚠️ Fair"
    else:
        performance = "❌ Poor"
    
    print(f"   └─ Overall Performance: {performance}")
    
    print("\n" + "="*80)

def plot_training_loss(history, save_path="plt/training_loss.png"):
    """
    Chart 1: Perbandingan Training Loss vs Validation Loss
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highlight best validation loss
    min_val_idx = np.argmin(history['val_loss'])
    min_val_loss = history['val_loss'][min_val_idx]
    plt.annotate(f'Best: {min_val_loss:.6f}',
                xy=(min_val_idx + 1, min_val_loss),
                xytext=(min_val_idx + 1 + len(epochs) * 0.1, min_val_loss + max(history['val_loss']) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Training loss chart saved to: {save_path}")

def plot_prediction_vs_actual(results, save_path="plt/prediction_accuracy.png"):
    """
    Chart 2: Akurasi Prediksi vs Data Actual
    """
    predictions = results['predictions']
    actuals = results['actuals']
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Time series comparison
    plt.subplot(1, 2, 1)
    sample_size = min(100, len(predictions))  # Show last 100 predictions
    indices = range(sample_size)
    
    plt.plot(indices, actuals[:sample_size], 'b-', label='Actual', linewidth=2)
    plt.plot(indices, predictions[:sample_size], 'r--', label='Predicted', linewidth=2)
    
    plt.title('Predicted vs Actual Prices', fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Scatter plot (perfect prediction line)
    plt.subplot(1, 2, 2)
    plt.scatter(actuals, predictions, alpha=0.6, s=20)
    
    # Perfect prediction line (diagonal)
    min_val = min(min(actuals), min(predictions))
    max_val = max(max(actuals), max(predictions))
    plt.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Prediction Accuracy', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add metrics text
    mae = results['mae']
    mape = results['mape']
    correlation = np.corrcoef(actuals, predictions)[0, 1]
    
    plt.text(0.05, 0.95, f'MAE: ${mae:.2f}\nMAPE: {mape:.1f}%\nCorr: {correlation:.3f}', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Prediction accuracy chart saved to: {save_path}")