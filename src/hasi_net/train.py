"""Training, evaluation and device handling for HASI-Net."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .config import Config
from .losses import count_loss
from .model import HASINet


def select_device(pref: str = "auto") -> torch.device:
    if pref == "mps":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WindowDataset(Dataset):
    """Sliding spatiotemporal windows from a [T, N, C] counts tensor."""

    def __init__(self, counts: np.ndarray, lookback: int, horizon: int,
                 indices: List[int]):
        self.counts = torch.from_numpy(counts).float()
        self.lookback = lookback
        self.horizon = horizon
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        t = self.indices[i]
        x = self.counts[t:t + self.lookback]
        y = self.counts[t + self.lookback:t + self.lookback + self.horizon]
        return x, y


def make_splits(T: int, lookback: int, horizon: int,
                val_ratio: float = 0.15, test_ratio: float = 0.2) -> Tuple[
        List[int], List[int], List[int]]:
    last = T - lookback - horizon + 1
    idx = list(range(last))
    n_test = max(1, int(last * test_ratio))
    n_val = max(1, int(last * val_ratio))
    test = idx[-n_test:]
    val = idx[-(n_test + n_val):-n_test]
    train = idx[:-(n_test + n_val)] if (n_test + n_val) < last else idx[:1]
    return train, val, test


@dataclass
class Metrics:
    mae: float
    rmse: float
    rmsle: float   # root mean squared log error (handles zero counts)
    wape: float    # weighted absolute percentage error = sum|err|/sum|true|
    r2: float
    csi: float     # critical success index for hotspot detection

    def as_dict(self) -> Dict[str, float]:
        return {"MAE": self.mae, "RMSE": self.rmse, "RMSLE": self.rmsle,
                "WAPE": self.wape, "R2": self.r2, "CSI": self.csi}


def _csi(pred: np.ndarray, true: np.ndarray, q: float = 0.9) -> float:
    """Critical Success Index for hotspot (top-10%) detection."""
    thr = np.quantile(true, q)
    p_hot = pred >= thr
    t_hot = true >= thr
    tp = float((p_hot & t_hot).sum())
    fp = float((p_hot & ~t_hot).sum())
    fn = float((~p_hot & t_hot).sum())
    if tp + fp + fn == 0:
        return 0.0
    return tp / (tp + fp + fn)


def evaluate(model: HASINet, loader: DataLoader, device: torch.device,
             horizon: int) -> Metrics:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            mu = model.predict_mean(x, horizon).cpu().numpy()
            preds.append(mu)
            trues.append(y.numpy())
    pred = np.concatenate(preds, axis=0).reshape(-1)
    true = np.concatenate(trues, axis=0).reshape(-1)
    pred = np.clip(pred, 0.0, None)
    eps = 1e-6
    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmsle = float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2)))
    denom = float(np.sum(np.abs(true))) + eps
    wape = float(np.sum(np.abs(pred - true)) / denom * 100.0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2)) + eps
    r2 = 1.0 - ss_res / ss_tot
    csi = _csi(pred, true)
    return Metrics(mae, rmse, rmsle, wape, r2, csi)


def train_one(model: HASINet, cfg: Config, train_loader: DataLoader,
              val_loader: DataLoader, device: torch.device,
              verbose: bool = True) -> Dict:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5)
    best_val = float("inf")
    best_state = None
    bad = 0
    history = {"train": [], "val": []}

    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x, cfg.horizon)
            loss = count_loss(out, y, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.item()) * x.size(0)
        train_loss = total / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device, cfg.horizon)
        val_loss = val_metrics.mae
        sched.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if verbose and (epoch % 5 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:3d} | train {train_loss:.4f} | "
                  f"val MAE {val_loss:.4f} | RMSE {val_metrics.rmse:.4f}")
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_val_mae": best_val}


def build_model(cfg: Config, panel, device: torch.device) -> HASINet:
    import torch as _t
    A_geo = _t.from_numpy(panel["a_geo"]).float().to(device)
    A_socio = _t.from_numpy(panel["a_socio"]).float().to(device)
    n_nodes = panel["counts"].shape[1]
    n_crime = panel["counts"].shape[2]
    nf_dim = panel["node_feats"].shape[1]
    model = HASINet(cfg, n_nodes, n_crime, nf_dim, A_geo, A_socio).to(device)
    model.set_node_features(_t.from_numpy(panel["node_feats"]).float().to(device))
    return model