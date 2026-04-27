"""
utils.py – Shared utilities for PRNN Assignment 2
=================================================
All dataset classes, model architectures, training/evaluation loops, and
helper functions used across the six assignment notebooks live here.

Each notebook imports only the symbols it needs, e.g.:
    from utils import TimeSeriesDataset, train_loop, OwnVanillaRNN, get_device
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_image


# ──────────────────────────────────────────────────────────────────────────────
# Device helper
# ──────────────────────────────────────────────────────────────────────────────

def get_device():
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Miscellaneous helpers
# ──────────────────────────────────────────────────────────────────────────────

def is_it_nan(val):
    """Utility to check if a value is NaN, since NaN != NaN."""
    return val != val


def build_sequences(data, window=72, horizon=24, pm25_idx=5):
    """Build (X, Y) where X[i] = flattened window of all features,
       Y[i] = pm2_5 value at t+horizon.  (Notebook 1 / numpy-based.)"""
    X, Y = [], []
    for i in range(window, len(data) - horizon):
        X.append(data[i - window: i, :].flatten())   # (window * N_FEATURES,)
        Y.append(data[i + horizon, pm25_idx])          # scalar
    return (np.array(X, dtype=np.float32),
            np.array(Y, dtype=np.float32).reshape(-1, 1))


def count_params(model, trainable_only=False):
    """Count parameters of *model*.

    If *trainable_only* is False (default) returns a single int – total
    parameter count (state_dict style, used in NB4).
    If *trainable_only* is True returns (total, trainable) tuple (NB5/NB6 style).
    """
    if trainable_only:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable
    # NB4 style: count via state_dict
    total = 0
    for name, param in model.state_dict().items():
        total += param.numel()
    return total


def compute_accuracy(dataloader, model, device):
    """Compute classification accuracy (NB5 style)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += len(y)
    return correct / total


def evaluate_accuracy(model, loader, device):
    """Compute classification accuracy (NB2 style)."""
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def evaluate_mse(dataloader, model, loss_fn, device):
    """Evaluate MSE on *dataloader* (NB5 style).  Prints and returns the MSE."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.float().to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
    mse = total_loss / len(dataloader)
    print(f"Test MSE: {mse:.4f}")
    return mse


def evaluate_model(test_dl, model, criterion, device):
    """Run inference on the test set (NB6 style).
    Returns (all_logits, all_labels)."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item()
            all_logits.extend(logits.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    total_loss /= len(test_dl)
    print(f"Test Loss ({criterion.__class__.__name__}): {total_loss:.4f}")
    return all_logits, all_labels


# ──────────────────────────────────────────────────────────────────────────────
# AQI / Time-series Dataset classes
# ──────────────────────────────────────────────────────────────────────────────

class AQISeqDataset(Dataset):
    """Wraps pre-built numpy arrays (X, Y) into a PyTorch Dataset.
    Used by Notebook 1 (MLP baseline)."""

    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class TimeSeriesDataset(Dataset):
    """Sliding-window dataset over the Delhi AQI CSV.

    Supports both the *full* multi-feature variant (Notebook 3) and
    the *simple* single-feature variant (Notebook 4) via the *all_features*
    flag.

    Parameters
    ----------
    file_path : str
        Path to delhi_aqi.csv.
    seq_len : int
        Length of the input sequence (hours).
    hop : int
        Gap between the end of the sequence and the target step.
    col_name : str
        Name of the target column.
    classify : bool
        If True, binarise the target at threshold 200 µg/m³ (NB3).
    all_features : bool
        If True, use all non-date columns as input (NB3).
        If False, use only *col_name* as a 1-D series (NB4).
    train_frac : float
        Fraction of data used for computing normalisation statistics.
    """

    def __init__(self, file_path, seq_len, hop, col_name="pm2_5",
                 classify=False, all_features=False, train_frac=0.7):
        df = pd.read_csv(file_path)

        if all_features:
            feature_cols = [c for c in df.columns if c != "date"]
            data = torch.tensor(df[feature_cols].values, dtype=torch.float32)
            pm2_5_idx = feature_cols.index(col_name)
            self.input_dim = seq_len * len(feature_cols)
            self.num_features = len(feature_cols)
        else:
            data = torch.tensor(df[col_name].values, dtype=torch.float32)
            pm2_5_idx = None
            self.input_dim = seq_len
            self.num_features = 1

        # Normalise using training portion statistics only (no leakage)
        n_train = int(train_frac * len(data))
        self.X_mean = data[:n_train].mean(dim=0)
        self.X_std = data[:n_train].std(dim=0).clamp(min=1e-8)
        # Expose as data_mean / data_std for single-feature compat (NB4)
        self.data_mean = self.X_mean
        self.data_std = self.X_std
        data = (data - self.X_mean) / self.X_std

        self.y_mean = self.X_mean[pm2_5_idx] if all_features else self.X_mean
        self.y_std = self.X_std[pm2_5_idx] if all_features else self.X_std

        self.X = []
        self.y = []

        for i in range(len(data) - seq_len - hop):
            x_seq = data[i: i + seq_len]
            y_target = (data[i + seq_len + hop, pm2_5_idx]
                        if all_features else data[i + seq_len + hop])

            self.X.append(x_seq)
            if classify:
                self.y.append(
                    y_target > (200 - self.y_mean) / self.y_std
                )
            else:
                self.y.append(y_target)

        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EncoderDecoderDataset(Dataset):
    """Encoder-Decoder (sequence-to-sequence) variant of the AQI dataset.
    Used by Notebook 4."""

    def __init__(self, file_path, seq_len, output_seq_len,
                 col_name="pm2_5", train_frac=0.7):
        df = pd.read_csv(file_path)
        data = torch.tensor(df[col_name].values, dtype=torch.float32)
        self.input_dim = seq_len

        n_train = int(train_frac * len(data))
        self.data_mean = data[:n_train].mean()
        self.data_std = data[:n_train].std().clamp(min=1e-8)
        data = (data - self.data_mean) / self.data_std

        self.X = []
        self.y = []
        for i in range(len(data) - seq_len - output_seq_len):
            self.X.append(data[i: i + seq_len])
            self.y.append(data[i + seq_len: i + seq_len + output_seq_len])

        self.X = torch.stack(self.X)
        self.y = torch.stack(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class AQITimeSeriesDataset(Dataset):
    """Sliding-window dataset over Delhi AQI (NB5 / Transformer notebook).
    Returns X of shape (seq_len,), y is a scalar."""

    def __init__(self, csv_path, seq_len=72, horizon=24,
                 col="pm2_5", train_frac=0.7):
        df = pd.read_csv(csv_path)
        data = torch.tensor(df[col].values, dtype=torch.float32)

        n_train = int(train_frac * len(data))
        self.mean = data[:n_train].mean()
        self.std = data[:n_train].std().clamp(min=1e-8)
        data = (data - self.mean) / self.std

        self.seq_len = seq_len
        self.input_dim = seq_len  # for MLP compatibility

        xs, ys = [], []
        for i in range(len(data) - seq_len - horizon):
            xs.append(data[i: i + seq_len])
            ys.append(data[i + seq_len + horizon])

        self.X = torch.stack(xs)   # (N, seq_len)
        self.y = torch.stack(ys)   # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Image Dataset classes
# ──────────────────────────────────────────────────────────────────────────────

class PlantVillageDataset(Dataset):
    """Loads PlantVillage images and integer class labels (NB5 plain-image)."""

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        class_names = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        for cls in class_names:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx])[:3].float() / 255.0
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


class PatchedPlantVillageDataset(Dataset):
    """PlantVillage dataset that returns non-overlapping 16×16 patches (NB5 ViT)."""

    def __init__(self, root_dir, transform=None, patch_size=16):
        self.transform = transform
        self.patch_size = patch_size
        self.image_paths = []
        self.labels = []

        class_names = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        for cls in class_names:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = read_image(self.image_paths[idx])[:3].float() / 255.0
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if self.transform:
            img = self.transform(img)

        p = self.patch_size
        H, W = img.shape[1], img.shape[2]
        patches = []
        for row in range(0, H, p):
            for col in range(0, W, p):
                patch = img[:, row: row + p, col: col + p]
                if patch.shape[1] == p and patch.shape[2] == p:
                    patches.append(patch.flatten())   # (3*p*p,)

        patches = torch.stack(patches)               # (num_patches, 3*p*p)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return patches, label


class SeverityDataset(Dataset):
    """Wraps a PlantVillage Subset and assigns a synthetic severity score (NB2)."""

    def __init__(self, subset, full_dataset, rng_seed=0):
        self.subset = subset
        healthy_classes = {
            i for i, name in enumerate(full_dataset.classes)
            if "healthy" in name.lower()
        }
        rng = np.random.default_rng(rng_seed)
        labels = [full_dataset.targets[idx] for idx in subset.indices]
        severity = []
        for lbl in labels:
            if lbl in healthy_classes:
                severity.append(0.0)
            else:
                severity.append(float(rng.uniform(0.2, 0.9)))
        self.severity = torch.tensor(severity, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, _ = self.subset[idx]
        return img, self.severity[idx]


class TransformSubset(Dataset):
    """PlantVillage subset with a custom transform applied at load time (NB2)."""

    def __init__(self, base_dataset, indices, transform):
        self.base = base_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        from PIL import Image
        orig_idx = self.indices[idx]
        path, label = self.base.samples[orig_idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


class RetinalFundusDataset(Dataset):
    """PyTorch Dataset for the APTOS 2019 Blindness Detection task (NB6)."""

    def __init__(self, csv_path, img_dir, transform=None):
        self.transform = transform
        self.img_paths = []
        self.labels = []

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            full_path = os.path.join(img_dir, row["id_code"] + ".png")
            if os.path.exists(full_path):
                self.img_paths.append(full_path)
                self.labels.append(row["diagnosis"])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = read_image(self.img_paths[idx])[:3].float() / 255.0
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
# Training / evaluation loops
# ──────────────────────────────────────────────────────────────────────────────

def train_regression(model, train_loader, val_loader, device,
                     epochs=100, lr=1e-3, patience=10,
                     save_path="best_model_q1_1.pt"):
    """Train regression MLP with early stopping (NB1).
    Returns (model, train_losses, val_losses)."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        running_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, Y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(X_batch)
        train_losses.append(running_loss / len(train_loader.dataset))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch in val_loader:
                X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, Y_batch).item() * len(X_batch)
        val_losses.append(val_loss / len(val_loader.dataset))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train MSE: {train_losses[-1]:.2f} | "
                  f"Val MSE: {val_losses[-1]:.2f}")

    model.load_state_dict(torch.load(save_path))
    return model, train_losses, val_losses


def train_loop(train_dataloader, val_dataloader, model, loss_fn, optimizer,
               device, max_iter=100, patience=10):
    """Training loop with early stopping (NB3 / NB4 style)."""
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for iter in range(max_iter):
        train_loss = 0
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.float().to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss

        train_loss /= len(train_dataloader)

        test_loss = 0
        model.eval()
        with torch.no_grad():
            for X, y in val_dataloader:
                X, y = X.to(device), y.float().to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= len(val_dataloader)

        if iter % 10 == 0:
            print(f"Epoch {iter}: Training Loss: {train_loss:.4f}, "
                  f"Validation Loss: {test_loss:.4f}")

        if test_loss < best_val_loss:
            best_val_loss = test_loss
            epochs_no_improve = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {iter}. "
                      f"Best Validation Loss: {best_val_loss:.4f}")
                model.load_state_dict(best_model_state)
                return None

    return None


def test_loop(dataloader, model, loss_fn, device):
    """Evaluation loop returning (pred_list, y_list) (NB3 / NB4 style)."""
    model.eval()
    test_loss = 0
    pred_list = []
    y_list = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.float().to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            pred_list.extend(pred.tolist())
            y_list.extend(y.tolist())

    test_loss /= len(dataloader)
    print(f"Test Loss (MSE): {test_loss:.4f}")
    return pred_list, y_list


def train_loop_rnn(train_dataloader, model, loss_fn, optimizer, device):
    """Training loop for the exploding-gradients demo (NB3 Problem 3.9).
    Runs until NaN loss is detected; returns largest singular value of Whh."""
    loss = torch.tensor(0.0)
    prev_state = {k: v.clone() for k, v in model.state_dict().items()}
    i = 0

    while not is_it_nan(loss.item()):
        prev_state = {k: v.clone() for k, v in model.state_dict().items()}
        train_loss = 0
        model.train()
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.float().to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss

        train_loss /= len(train_dataloader)
        i += 1
        print(f"Epoch Number:{i}")

    model.load_state_dict(prev_state)
    svd = torch.linalg.svd(model.linearh.weight.detach().cpu())
    return svd.S[0].item()


def train_model(train_loader, val_loader, model, loss_fn, optimizer, device,
                num_epochs=100, patience=10):
    """Training loop with early stopping (NB5 / Transformer style)."""
    best_val_loss = float("inf")
    no_improve_count = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.float().to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.float().to(device)
                pred = model(X)
                val_loss_sum += loss_fn(pred, y).item()
        val_loss = val_loss_sum / len(val_loader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch}.  "
                      f"Best val_loss={best_val_loss:.4f}")
                model.load_state_dict(best_state)
                return


def train_classifier(train_loader, val_loader, model, loss_fn, optimizer, device,
                     num_epochs=20, patience=10, save_path=None):
    """Classification training loop with optional checkpoint saving (NB5 ViT)."""
    best_val_loss = float("inf")
    no_improve_count = 0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_loss_sum += loss_fn(model(X), y).item()
        val_loss = val_loss_sum / len(val_loader)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save(best_state, save_path)
                print(f"  ✔ Saved best model to {save_path}  "
                      f"(val_loss={best_val_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch}.  "
                      f"Best val_loss={best_val_loss:.4f}")
                model.load_state_dict(best_state)
                return


def run_training(train_dl, val_dl, model, criterion, optimizer, device,
                 num_epochs=10, patience=5):
    """Standard training loop with early stopping (NB6 style).
    Returns per-epoch validation loss history."""
    best_val = float("inf")
    no_improv = 0
    best_state = None
    val_history = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                val_loss += criterion(model(X), y).item()
        val_loss /= len(val_dl)
        val_history.append(val_loss)

        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}  |  "
              f"Val Loss = {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            no_improv = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improv += 1
            if no_improv >= patience:
                print(f"Early stopping triggered at epoch {epoch}. "
                      f"Best val loss: {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return val_history


# ──────────────────────────────────────────────────────────────────────────────
# Model classes
# ──────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """3-layer MLP for PM2.5 regression (NB1)."""

    def __init__(self, input_dim, hidden_dim=256, activation=nn.ReLU):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.act = activation()

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.layer2(x))
        x = self.layer3(x)
        return x


class OwnVanillaRNN(nn.Module):
    """Custom vanilla RNN with BPTT decay analysis (NB3)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linearx = nn.Linear(input_dim, hidden_dim)
        self.linearh = nn.Linear(hidden_dim, hidden_dim)
        self.lineary = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
        batch_size, seq_length, _ = x.shape

        h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        for t in range(seq_length):
            x_t = x[:, t, :]
            h_t = torch.tanh(self.linearx(x_t) + self.linearh(h_t))

        y = self.lineary(h_t)
        return y.squeeze(1)

    def bptt_decay(self, seq_len=100, input_dim=1):
        """Compute gradient magnitudes at t=0, t=50, t=100 (runs on CPU)."""
        cpu = torch.device("cpu")
        self.to(cpu)
        hidden_dim = self.hidden_dim
        x = torch.randn(1, seq_len, input_dim, device=cpu)
        h_t = torch.zeros(1, hidden_dim, device=cpu)
        hidden_states = {}

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = torch.tanh(self.linearx(x_t) + self.linearh(h_t))
            h_t.retain_grad()
            hidden_states[t] = h_t

        y = self.lineary(h_t)
        target = torch.zeros(1, device=cpu)
        loss = nn.MSELoss()(y.squeeze(1), target)
        loss.backward()

        print(f"||dL/dh|| at t=100: {hidden_states[99].grad.norm().item():.2e}")
        print(f"||dL/dh|| at t=50:  {hidden_states[49].grad.norm().item():.2e}")
        print(f"||dL/dh|| at t=0:   {hidden_states[0].grad.norm().item():.2e}")

        # Move back to original device (caller must handle this)
        return hidden_states


class LSTMRegressor(nn.Module):
    """LSTM-based regression model (NB4)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm_layer(x)
        out = self.output_layer(lstm_out[:, -1, :])
        return out.squeeze(1)

    def bptt_decay(self, seq_len=100, input_dim=1):
        """Gradient magnitudes at key timesteps; runs on CPU."""
        cpu = torch.device("cpu")
        self.to(cpu)
        hidden_dim = self.lstm_layer.hidden_size
        rand_input = torch.randn(1, seq_len, input_dim, device=cpu)
        h_state = torch.zeros(1, 1, hidden_dim, device=cpu)
        c_state = torch.zeros(1, 1, hidden_dim, device=cpu)
        recorded_hidden = {}

        for t in range(seq_len):
            x_t = rand_input[:, t: t + 1, :]
            _, (h_state, c_state) = self.lstm_layer(x_t, (h_state, c_state))
            h_state.retain_grad()
            recorded_hidden[t] = h_state

        final_out = self.output_layer(h_state.squeeze(0))
        dummy_target = torch.zeros(1, 1, device=cpu)
        loss = nn.MSELoss()(final_out, dummy_target)
        loss.backward()

        print(f"||dL/dh|| at t=100: {recorded_hidden[99].grad.norm().item():.2e}")
        print(f"||dL/dh|| at t=50:  {recorded_hidden[49].grad.norm().item():.2e}")
        print(f"||dL/dh|| at t=0:   {recorded_hidden[0].grad.norm().item():.2e}")

        return recorded_hidden


class GRURegressor(nn.Module):
    """GRU-based regression model (NB4)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.gru_layer = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        gru_out, _ = self.gru_layer(x)
        out = self.output_layer(gru_out[:, -1, :])
        return out.squeeze(1)


class EncoderDecoderLSTM(nn.Module):
    """Encoder-Decoder LSTM for multi-step sequence forecasting (NB4)."""

    def __init__(self, input_dim, hidden_dim, output_dim, forecast_len):
        super().__init__()
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder_cell = nn.LSTMCell(output_dim, hidden_dim)
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.forecast_len = forecast_len

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        _, (h_n, c_n) = self.encoder_lstm(x)
        h_dec = h_n.squeeze(0)
        c_dec = c_n.squeeze(0)
        dec_input = x[:, -1, :]

        forecasts = []
        for _ in range(self.forecast_len):
            h_dec, c_dec = self.decoder_cell(dec_input, (h_dec, c_dec))
            step_pred = self.projection(h_dec)
            forecasts.append(step_pred)
            dec_input = step_pred

        return torch.stack(forecasts, dim=1).squeeze(-1)


class PlantCNN(nn.Module):
    """3-block CNN for PlantVillage classification (NB2 / NB5)."""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PlantCNN_Regression(nn.Module):
    """Same CNN backbone as PlantCNN but with a regression head (NB2)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.head = nn.Sequential(
            nn.Linear(32 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        return self.head(x)


class ScaledDotProductAttentionModel(nn.Module):
    """Single-head Transformer Encoder for time-series regression / ViT (NB5)."""

    def __init__(self, input_dim, model_dim, output_dim):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(model_dim, model_dim)
        self.value_proj = nn.Linear(model_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim)
        self.scale_factor = model_dim ** 0.5

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h = self.input_proj(x)
        Q = self.query_proj(h)
        K = self.key_proj(h)
        V = self.value_proj(h)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        out = self.output_proj(context[:, -1, :])
        return out.squeeze(1)

    @torch.no_grad()
    def get_attention_weights(self, x):
        """Return the (T, T) attention weight matrix for a single sample."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h = self.input_proj(x)
        Q = self.query_proj(h)
        K = self.key_proj(h)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        weights = torch.softmax(scores, dim=-1)
        return weights.cpu().numpy()


class TransformerWithPosEncoding(nn.Module):
    """Single-head Transformer with sinusoidal positional encoding (NB5)."""

    def __init__(self, input_dim, model_dim, output_dim, max_seq_len=72):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.query_proj = nn.Linear(model_dim, model_dim)
        self.key_proj = nn.Linear(model_dim, model_dim)
        self.value_proj = nn.Linear(model_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, output_dim)
        self.scale_factor = model_dim ** 0.5

        pe = torch.zeros(max_seq_len, model_dim)
        positions = torch.arange(0, max_seq_len).unsqueeze(1).float()
        freq_div = torch.exp(
            torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim)
        )
        pe[:, 0::2] = torch.sin(positions * freq_div)
        pe[:, 1::2] = torch.cos(positions * freq_div)
        self.register_buffer("pos_encoding", pe.unsqueeze(0))

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        h = self.input_proj(x)
        T = h.size(1)
        h = h + self.pos_encoding[:, :T, :]
        Q = self.query_proj(h)
        K = self.key_proj(h)
        V = self.value_proj(h)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        out = self.output_proj(context[:, -1, :])
        return out.squeeze(1)


class FocalLoss(nn.Module):
    """Focal Loss for multi-class classification (NB6).

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    gamma=0 reduces to standard cross-entropy.
    """

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_per_sample = nn.CrossEntropyLoss(reduction="none")(logits, targets)
        p_t = torch.exp(-ce_per_sample)
        focal_weight = (1.0 - p_t) ** self.gamma
        return (focal_weight * ce_per_sample).mean()
