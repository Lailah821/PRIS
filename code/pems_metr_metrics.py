"""PEMS-BAY / METR-LA evaluation metrics.

DCRNN/GWN community convention: mask out zero-speed entries.
Computed in original (un-normalized) units.
"""
from __future__ import annotations

import numpy as np


def _zero_mask(y_true: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    return (np.abs(y_true) > eps).astype(np.float32)


def masked_mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    mask = _zero_mask(y_true)
    err = np.abs(y_pred - y_true) * mask
    denom = mask.sum()
    if denom < 1e-6:
        return float("nan")
    return float(err.sum() / denom)


def masked_rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    mask = _zero_mask(y_true)
    sq = ((y_pred - y_true) ** 2) * mask
    denom = mask.sum()
    if denom < 1e-6:
        return float("nan")
    return float(np.sqrt(sq.sum() / denom))


def masked_mape(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    mask = _zero_mask(y_true)
    safe_denom = np.where(mask > 0, np.abs(y_true), 1.0)
    pct = np.abs(y_pred - y_true) / safe_denom * mask
    denom = mask.sum()
    if denom < 1e-6:
        return float("nan")
    return float(pct.sum() / denom * 100.0)


def per_horizon_metrics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    horizons=(3, 6, 12, 24),
) -> dict:
    """y_pred, y_true: either (N_samples, H, N_nodes) or (N_samples, H).

    Returns {h: {'mae': ..., 'rmse': ..., 'mape': ...}, ...}.
    Horizon index is 1-based: h=3 means index 2 (15min @ 5min freq).
    """
    out = {}
    H = y_pred.shape[1]
    is_3d = y_pred.ndim == 3
    for h in horizons:
        if h > H:
            continue
        if is_3d:
            yp = y_pred[:, h - 1, :]
            yt = y_true[:, h - 1, :]
        else:
            yp = y_pred[:, h - 1]
            yt = y_true[:, h - 1]
        out[h] = {
            "mae": masked_mae(yp, yt),
            "rmse": masked_rmse(yp, yt),
            "mape": masked_mape(yp, yt),
        }
    return out


def per_horizon_metrics_seen_unseen(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    sensor_ids: np.ndarray,
    pool_tag: np.ndarray,
    horizons=(3, 6, 12, 24),
) -> dict:
    """Split per-horizon metrics by sensor pool (seen vs unseen).

    Parameters
    ----------
    y_pred, y_true : (N_samples, H) — shadowed setup outputs.
    sensor_ids     : (N_samples,) — sensor index per sample (test loader emits).
    pool_tag       : (N_nodes,) — 0=train(seen), 1=val(unseen), 2=test(unseen).

    Returns
    -------
    dict with keys 'all', 'seen', 'unseen', 'val_unseen', 'test_unseen', each a
    per-horizon metric dict (same shape as per_horizon_metrics). Empty splits
    are omitted.
    """
    sensor_ids = np.asarray(sensor_ids)
    tags = pool_tag[sensor_ids]  # (N_samples,)
    splits = {
        "all":         np.ones_like(tags, dtype=bool),
        "seen":        tags == 0,
        "unseen":      tags > 0,
        "val_unseen":  tags == 1,
        "test_unseen": tags == 2,
    }
    out: dict = {}
    for name, mask in splits.items():
        n = int(mask.sum())
        if n == 0:
            continue
        out[name] = {
            "n_samples": n,
            "per_h": per_horizon_metrics(y_pred[mask], y_true[mask], horizons=horizons),
        }
    return out
