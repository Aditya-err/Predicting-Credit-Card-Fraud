import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class TabularMLP(nn.Module):
    """
    3-Layer High-Performance Tabular MLP Architecture.
    Utilizes BatchNorm, SiLU activation, and dropout regularization.
    """

    def __init__(self, in_features: int, hidden_dims: List[int] = [128, 64, 32], dropout: float = 0.2):
        super().__init__()
        layers = []
        curr_dim = in_features

        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.SiLU())
            d_rate = dropout * (0.8 ** i)  # Progressive dropout tapering
            if d_rate > 0.05:
                layers.append(nn.Dropout(d_rate))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def train_tabular_mlp(
    X_train_num: np.ndarray,
    y_train: np.ndarray,
    X_val_num: np.ndarray,
    y_val: np.ndarray,
    X_test_num: Optional[np.ndarray] = None,
    epochs: int = 8,
    batch_size: int = 4096,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
    """
    Trains a TabularMLP on the current fold and generates validation + test probability predictions.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TabularDataset(X_train_num, y_train)
    val_ds = TabularDataset(X_val_num, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    model = TabularMLP(in_features=X_train_num.shape[1], hidden_dims=[128, 64, 32], dropout=0.2).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_auc = 0.0
    best_val_preds = np.zeros(len(y_val), dtype=np.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        scheduler.step()

        # Validation evaluation
        model.eval()
        val_preds_list = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                probs = torch.sigmoid(model(batch_x)).cpu().numpy()
                val_preds_list.append(probs)

        current_val_preds = np.concatenate(val_preds_list)
        epoch_val_auc = float(roc_auc_score(y_val, current_val_preds))

        if epoch_val_auc > best_val_auc:
            best_val_auc = epoch_val_auc
            best_val_preds = current_val_preds

    # Predict test if provided
    test_preds = None
    if X_test_num is not None:
        model.eval()
        test_ds = TabularDataset(X_test_num)
        test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)
        test_preds_list = []
        with torch.no_grad():
            for batch_x in test_loader:
                batch_x = batch_x.to(device)
                probs = torch.sigmoid(model(batch_x)).cpu().numpy()
                test_preds_list.append(probs)
        test_preds = np.concatenate(test_preds_list)

    return best_val_preds, test_preds, best_val_auc
