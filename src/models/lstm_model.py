"""PyTorch LSTM signal classifier.

Sequence of LSTM_SEQ_LEN bars → P(up-move).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.config import (
    LSTM_SEQ_LEN,
    LSTM_HIDDEN,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_LR,
    LSTM_RANDOM_SEED,
)


class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden: list, dropout: float):
        super().__init__()
        layers = []
        in_size = input_size
        for i, h in enumerate(hidden):
            layers.append(nn.LSTM(in_size, h, batch_first=True))
            in_size = h
        self.lstms = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for lstm in self.lstms:
            out, _ = lstm(out)
            out = self.dropout(out)
        last = out[:, -1, :]   # take final timestep
        return self.sigmoid(self.head(last)).squeeze(1)


class LSTMSignalModel:
    """Sequence model: takes last LSTM_SEQ_LEN bars of features → P(up-move)."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.net: _LSTMNet | None = None
        self.feature_names: list = []
        self.seq_len = LSTM_SEQ_LEN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(LSTM_RANDOM_SEED)

    def _build_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple:
        """Convert (T, F) array → (T-seq_len, seq_len, F) sequences."""
        n = len(X)
        seqs, targets = [], []
        for i in range(self.seq_len, n):
            seqs.append(X[i - self.seq_len : i])
            if y is not None:
                targets.append(y[i])
        Xs = np.array(seqs, dtype=np.float32)
        ys = np.array(targets, dtype=np.float32) if y is not None else None
        return Xs, ys

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMSignalModel":
        self.feature_names = list(X.columns)
        X_arr = self.scaler.fit_transform(X.values)
        y_arr = y.values.astype(np.float32)

        Xs, ys = self._build_sequences(X_arr, y_arr)
        if len(Xs) == 0:
            return self

        dataset = TensorDataset(
            torch.from_numpy(Xs), torch.from_numpy(ys)
        )
        loader = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)

        self.net = _LSTMNet(Xs.shape[2], LSTM_HIDDEN, LSTM_DROPOUT).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=LSTM_LR)
        loss_fn = nn.BCELoss()

        self.net.train()
        for _ in range(LSTM_EPOCHS):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(up-move) for each row (aligned to original index)."""
        if self.net is None:
            raise RuntimeError("Model not fitted.")

        X_arr = self.scaler.transform(X[self.feature_names].values)
        Xs, _ = self._build_sequences(X_arr)

        if len(Xs) == 0:
            return np.full(len(X), 0.5)

        self.net.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(Xs).to(self.device)
            preds = self.net(tensor).cpu().numpy()

        # Pad first seq_len rows with 0.5 (no prediction available)
        full = np.full(len(X), 0.5, dtype=np.float32)
        full[self.seq_len :] = preds
        return full
