import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import TransformerStockPredictor

def train_model(model: TransformerStockPredictor, train_loader, val_loader, num_epochs=50, lr=0.001):
    # Setup training components
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)  # Add weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # History untuk tracking loss
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"🚀 MEMULAI TRAINING untuk {num_epochs} epochs...")
    print(f"📋 Learning Rate: {lr}, Weight Decay: 1e-5")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 15
    
    for epoch in range(num_epochs):
        # ===== TRAINING PHASE =====
        model.train()
        total_train_loss = 0
        num_train_batches = 0
        
        for batch_x, batch_y in train_loader:
            # Convert ke tensor jika belum
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            
            # Forward pass
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping untuk stabilitas training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            num_train_batches += 1
        
        # Hitung rata-rata training loss
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        
        # ===== VALIDATION PHASE =====
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.float()
                batch_y = batch_y.float()
                
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                
                total_val_loss += loss.item()
                num_val_batches += 1
        
        # Hitung rata-rata validation loss
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
        
        # Update learning rate scheduler
        # Tambahkan manual logging untuk LR changes
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if old_lr != new_lr:
            print(f"📉 Learning rate reduced: {old_lr:.1e} → {new_lr:.1e}")
        
        # Simpan ke history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'file/best_model.pth')
            improvement_indicator = "📈"
        else:
            patience_counter += 1
            improvement_indicator = "📉" if patience_counter > 5 else "➖"
        
        # Print progress dengan informasi lebih detail
        current_lr = optimizer.param_groups[0]['lr']
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:3d}/{num_epochs} {improvement_indicator} | "
                  f"Train: {avg_train_loss:.6f} | "
                  f"Val: {avg_val_loss:.6f} | "
                  f"LR: {current_lr:.1e} | "
                  f"Patience: {patience_counter}/{early_stopping_patience}")
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
            print(f"   Best validation loss: {best_val_loss:.6f}")
            break
    
    print("=" * 60)
    print(f"✅ TRAINING SELESAI!")
    print(f"📊 Final Metrics:")
    print(f"   └─ Best Val Loss: {best_val_loss:.6f}")
    print(f"   └─ Total Epochs: {len(history['train_loss'])}")
    print(f"   └─ Final LR: {optimizer.param_groups[0]['lr']:.1e}")
    
    # Load best model jika ada
    try:
        model.load_state_dict(torch.load('best_model.pth'))
        print(f"🔄 Loaded best model from checkpoint")
    except FileNotFoundError:
        print(f"⚠️  Best model checkpoint not found, using current model")
    
    return model, history


def evaluate_model(model: TransformerStockPredictor, test_loader, scaler):
    model.eval()
    criterion = nn.MSELoss()
    
    all_predictions = []
    all_targets = []
    total_test_loss = 0
    num_test_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float()
            batch_y = batch_y.float()
            
            # Prediksi
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            
            # Simpan untuk analisis
            all_predictions.extend(predictions.numpy())
            all_targets.extend(batch_y.numpy())
            
            total_test_loss += loss.item()
            num_test_batches += 1
    
    # Hitung metrics
    avg_test_loss = total_test_loss / num_test_batches
    
    # Convert ke numpy arrays
    predictions_array = np.array(all_predictions).reshape(-1, 1)
    targets_array = np.array(all_targets).reshape(-1, 1)
    
    # Denormalisasi untuk mendapat nilai harga asli
    pred_prices = scaler.inverse_transform(predictions_array).flatten()
    actual_prices = scaler.inverse_transform(targets_array).flatten()
    
    # Hitung error dalam harga asli
    mae = np.mean(np.abs(pred_prices - actual_prices))
    rmse = np.sqrt(np.mean((pred_prices - actual_prices) ** 2))
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
    
    # Hasil evaluasi
    results = {
        'test_loss': avg_test_loss,
        'mae': mae,  # Mean Absolute Error
        'rmse': rmse,  # Root Mean Square Error
        'mape': mape,  # Mean Absolute Percentage Error
        'predictions': pred_prices,
        'actuals': actual_prices
    }
    
    print(f"📊 HASIL EVALUASI:")
    print(f"   Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"   MAE: ${mae:.2f}")
    print(f"   RMSE: ${rmse:.2f}")
    print(f"   MAPE: {mape:.2f}%")
    print(f"   Sample predictions: {pred_prices[:5]}")
    print(f"   Sample actuals: {actual_prices[:5]}")
    
    return results


def predict_next_price(model: TransformerStockPredictor, last_sequence, scaler):
    """
    🔮 Function untuk prediksi harga berikutnya
    
    Args:
        model: Model yang sudah di-train
        last_sequence: Sequence terakhir (5 hari terakhir)
        scaler: Scaler untuk denormalisasi
    
    Returns:
        predicted_price: Harga prediksi dalam nilai asli
    """
    print(f"\n🔮 PREDIKSI HARGA BERIKUTNYA")
    print("=" * 50)
    
    model.eval()
    
    with torch.no_grad():
        # Convert ke tensor
        sequence_tensor = torch.tensor(last_sequence).float().unsqueeze(0)  # Add batch dimension
        
        # Prediksi
        prediction = model(sequence_tensor)
        
        # Denormalisasi ke harga asli
        prediction_reshaped = prediction.numpy().reshape(-1, 1)
        predicted_price = scaler.inverse_transform(prediction_reshaped)[0, 0]
        
        print(f"🎯 Prediksi harga berikutnya: ${predicted_price:.2f}")
        
        return predicted_price
