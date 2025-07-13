import os
import datetime
import torch

def get_experiment_name(prefix="exp"):
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{now}"

def get_best_model_path(experiment_name):
    return f"file/best_model_{experiment_name}.pth"

def get_val_loss_log_path(experiment_name):
    return f"file/best_model_{experiment_name}_valloss.txt"

def save_best_model(model, path, val_loss, epoch, log_path=None):
    torch.save(model.state_dict(), path)
    if log_path:
        with open(log_path, "w") as f:
            f.write(f"Best val loss: {val_loss:.6f} at epoch {epoch}\n")
    print(f"✅ Best model saved to: {path}")
    if log_path:
        print(f"📝 Best val loss logged to: {log_path}")

def load_best_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"✅ Best model loaded from {path}")
        return True
    else:
        print(f"⚠️  Best model checkpoint not found at {path}")
        return False
