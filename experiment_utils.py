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

def load_best_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"✅ Best model loaded from {path}")
        return True
    else:
        print(f"⚠️  Best model checkpoint not found at {path}")
        return False

def find_best_model_in_folder(folder="file"):
    """
    Cari best model dengan val loss terendah di folder.
    Return path model dan nama eksperimen.
    """
    best_loss = float("inf")
    best_model_path = None
    best_experiment_name = None

    # Scan semua file *_valloss.txt
    for fname in os.listdir(folder):
        if fname.endswith("_valloss.txt"):
            valloss_path = os.path.join(folder, fname)
            with open(valloss_path, "r") as f:
                line = f.readline()
                # Format: Best val loss: 0.000309 at epoch 79
                try:
                    loss_str = line.split(":")[1].split("at")[0].strip()
                    val_loss = float(loss_str)
                    if val_loss < best_loss:
                        best_loss = val_loss
                        # Ambil nama eksperimen dari filename
                        experiment_name = fname.replace("best_model_", "").replace("_valloss.txt", "")
                        best_experiment_name = experiment_name
                        best_model_path = os.path.join(folder, f"best_model_{experiment_name}.pth")
                except Exception:
                    continue

    return best_model_path, best_experiment_name, best_loss
