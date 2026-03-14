# benchmark/train_lstm.py
"""Train the LSTM sequence detector on benchmark data."""
import json
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

sys.path.insert(0, "src")

from mlshield.detectors.models.lstm_detector import TrajectoryLSTM, EventFeaturizer
from mlshield.detectors.models.isolation import GPUIsolationForest


class TrajectoryDataset(Dataset):
    """PyTorch dataset for trajectory sequences."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def prepare_data(benchmark_path: str = "benchmark/data/mlshield_benchmark_v1.json"):
    """Load benchmark and featurize all trajectories."""
    print("Loading benchmark data...")
    with open(benchmark_path) as f:
        dataset = json.load(f)

    featurizer = EventFeaturizer()
    X = []
    y = []
    gpu_events_normal = []

    for traj in dataset:
        label = 0 if traj["label"] == "benign" else 1
        feature_matrix = featurizer.featurize_trajectory(traj["events"], max_len=50)
        X.append(feature_matrix)
        y.append(label)

        # Collect normal GPU events for Isolation Forest
        if label == 0:
            for event in traj["events"]:
                details = event.get("details", {})
                if "DCGM_FI_DEV_GPU_UTIL" in details:
                    gpu_events_normal.append(details)

    X = np.array(X)
    y = np.array(y)

    print(f"Prepared {len(X)} trajectories: {sum(y == 0)} benign, {sum(y == 1)} malicious")
    print(f"Feature shape: {X.shape}")
    print(f"Normal GPU events for Isolation Forest: {len(gpu_events_normal)}")

    return X, y, gpu_events_normal


def train_lstm(
    X_train, y_train, X_val, y_val,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.001,
):
    """Train the LSTM model."""
    train_ds = TrajectoryDataset(X_train, y_train)
    val_ds = TrajectoryDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = TrajectoryLSTM(input_dim=32, hidden_dim=64, num_layers=2, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Class weights for imbalanced data
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # We use BCEWithLogitsLoss, so we need to modify forward to return logits for training
    # But our model has Sigmoid built in, so we use BCE loss
    criterion = nn.BCELoss()

    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                pred = model(batch_X)
                loss = criterion(pred, batch_y)
                val_loss += loss.item()
                val_preds.extend(pred.squeeze().tolist())
                val_labels.extend(batch_y.squeeze().tolist())

        val_loss /= len(val_loader)

        # Handle single-element batches
        if isinstance(val_preds, float):
            val_preds = [val_preds]
        if isinstance(val_labels, float):
            val_labels = [val_labels]

        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)

        try:
            auc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            auc = 0.0

        binary_preds = (val_preds > 0.5).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            val_labels, binary_preds, average="binary", zero_division=0,
        )

        if auc > best_auc:
            best_auc = auc
            best_state = model.state_dict().copy()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"AUC: {auc:.4f} | "
                f"P: {prec:.3f} R: {rec:.3f} F1: {f1:.3f}"
            )

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    return model, best_auc


def train_isolation_forest(gpu_events: list):
    """Train Isolation Forest on normal GPU events."""
    print(f"\nTraining Isolation Forest on {len(gpu_events)} normal GPU events...")
    iso_forest = GPUIsolationForest(contamination=0.05)
    iso_forest.fit(gpu_events)
    print(f"  Isolation Forest fitted: {iso_forest.is_fitted}")
    return iso_forest


def evaluate_final(model, X_test, y_test, iso_forest, benchmark_path):
    """Run final evaluation on held-out test set."""
    print("\n" + "=" * 60)
    print("  Final Evaluation on Test Set")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        preds = model(X_tensor).squeeze().numpy()

    auc = roc_auc_score(y_test, preds)
    binary_preds = (preds > 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, binary_preds, average="binary", zero_division=0,
    )

    print(f"\n  LSTM Results:")
    print(f"    AUC:       {auc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1 Score:  {f1:.4f}")

    # Test Isolation Forest on some examples
    with open(benchmark_path) as f:
        dataset = json.load(f)

    iso_tp, iso_fp, iso_fn, iso_tn = 0, 0, 0, 0
    for traj in dataset[:500]:
        label = traj["label"] != "benign"
        for event in traj["events"]:
            details = event.get("details", {})
            if "DCGM_FI_DEV_GPU_UTIL" in details:
                is_anomaly = iso_forest.predict(details)
                is_malicious = event.get("is_malicious", False)
                if is_anomaly and is_malicious:
                    iso_tp += 1
                elif is_anomaly and not is_malicious:
                    iso_fp += 1
                elif not is_anomaly and is_malicious:
                    iso_fn += 1
                else:
                    iso_tn += 1

    total = iso_tp + iso_fp + iso_fn + iso_tn
    if total > 0:
        print(f"\n  Isolation Forest Results (on GPU events):")
        print(f"    TP: {iso_tp}, FP: {iso_fp}, FN: {iso_fn}, TN: {iso_tn}")
        if iso_tp + iso_fp > 0:
            print(f"    Precision: {iso_tp / (iso_tp + iso_fp):.4f}")
        if iso_tp + iso_fn > 0:
            print(f"    Recall:    {iso_tp / (iso_tp + iso_fn):.4f}")

    print("=" * 60)
    return auc


def main():
    benchmark_path = "benchmark/data/mlshield_benchmark_v1.json"
    model_dir = "benchmark/data/models"
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Prepare data
    X, y, gpu_events_normal = prepare_data(benchmark_path)

    # Split: 70% train, 15% val, 15% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.176, random_state=42, stratify=y_trainval,
    )

    print(f"\nSplit: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    print(f"  Train: {sum(y_train == 0)} benign, {sum(y_train == 1)} malicious")
    print(f"  Val:   {sum(y_val == 0)} benign, {sum(y_val == 1)} malicious")
    print(f"  Test:  {sum(y_test == 0)} benign, {sum(y_test == 1)} malicious")

    # Train LSTM
    print("\nTraining LSTM...")
    model, best_auc = train_lstm(X_train, y_train, X_val, y_val, epochs=30, batch_size=32)
    print(f"\nBest validation AUC: {best_auc:.4f}")

    # Train Isolation Forest
    iso_forest = train_isolation_forest(gpu_events_normal)

    # Final evaluation
    test_auc = evaluate_final(model, X_test, y_test, iso_forest, benchmark_path)

    # Save models
    lstm_path = f"{model_dir}/lstm_detector.pt"
    iso_path = f"{model_dir}/isolation_forest.pkl"
    torch.save(model.state_dict(), lstm_path)
    iso_forest.save(iso_path)
    print(f"\nModels saved:")
    print(f"  LSTM:             {lstm_path}")
    print(f"  Isolation Forest: {iso_path}")

    # Success check
    if test_auc >= 0.85:
        print(f"\n  MILESTONE ACHIEVED: Test AUC {test_auc:.4f} >= 0.85")
    else:
        print(f"\n  WARNING: Test AUC {test_auc:.4f} < 0.85 target")

    return test_auc


if __name__ == "__main__":
    main()
