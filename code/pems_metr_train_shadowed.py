"""Shadowed setup train/eval loop.

Dataloader returns 4-tuple: (x, y_norm, y_raw, sensor_id).
Model maps x -> (B, T_out) target sensor prediction in normalized space.
Loss: masked MAE in original units. Selection: val MAE mean across horizons.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pems_metr_metrics import per_horizon_metrics, per_horizon_metrics_seen_unseen


def masked_mae_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-3
) -> torch.Tensor:
    mask = (torch.abs(y_true) > eps).float()
    err = torch.abs(y_pred - y_true) * mask
    denom = mask.sum().clamp_min(1.0)
    return err.sum() / denom


def evaluate(model: nn.Module, loader, mean: float, std: float, device: str):
    """Return (y_pred_raw, y_true_raw, sensor_ids) numpy arrays.

    Shapes: y_pred_raw, y_true_raw -> (N_samples, T_out); sensor_ids -> (N_samples,).
    """
    model.eval()
    preds = []
    trues = []
    sids = []
    with torch.no_grad():
        for x, _y_norm, y_raw, sid in loader:
            x = x.to(device)
            sid_dev = sid.to(device) if torch.is_tensor(sid) else torch.as_tensor(sid, device=device)
            pred_norm = model(x, sid_dev)  # (B, T_out)
            pred_raw = pred_norm.detach().cpu().numpy() * std + mean
            preds.append(pred_raw)
            trues.append(y_raw.numpy())
            sids.append(np.asarray(sid, dtype=np.int64))
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(trues, axis=0),
        np.concatenate(sids, axis=0),
    )


def train_eval_shadowed(
    model: nn.Module,
    loaders: dict,
    *,
    mean: float,
    std: float,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip: float = 5.0,
    horizons=(3, 6, 12, 24),
    log_every: int = 1,
    save_path: str | None = None,
    early_stop_patience: int | None = 10,
):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    test_loader = loaders["test_loader"]
    pool_tag = loaders.get("pool_tag")  # (N,) seen/unseen split tag

    best_val_mae = float("inf")
    best_state = None
    best_epoch = -1
    bad_epochs = 0
    history = []

    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_b = 0
        n_skip = 0
        for x, _y_norm, y_raw, sid in train_loader:
            x = x.to(device)
            y_raw = y_raw.to(device)
            sid_dev = sid.to(device) if torch.is_tensor(sid) else torch.as_tensor(sid, device=device)
            pred_norm = model(x, sid_dev)
            pred_raw = pred_norm * std + mean
            loss = masked_mae_loss(pred_raw, y_raw)
            if not torch.isfinite(loss):
                n_skip += 1
                continue
            opt.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
            ep_loss += float(loss.item())
            n_b += 1
        train_loss = ep_loss / max(n_b, 1)

        y_pred_v, y_true_v, _sids_v = evaluate(model, val_loader, mean, std, device)
        val_m = per_horizon_metrics(y_pred_v, y_true_v, horizons=horizons)
        val_mae_mean = float(np.mean([val_m[h]["mae"] for h in val_m]))

        improved = val_mae_mean < best_val_mae - 1e-5
        if improved:
            best_val_mae = val_mae_mean
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append({
            "epoch": ep,
            "train_mae": train_loss,
            "val_mae_mean": val_mae_mean,
            "val_per_h": val_m,
        })

        if n_skip > 0:
            print(f"  ep {ep:3d} warning: skipped {n_skip} non-finite-loss batches", flush=True)

        if ep % log_every == 0 or ep == 1 or ep == epochs:
            v_sum = " ".join(f"h{h}={val_m[h]['mae']:.2f}" for h in val_m)
            tag = " *" if improved else ""
            print(
                f"  ep {ep:3d}/{epochs}  train={train_loss:.3f}  val[{v_sum}] mean={val_mae_mean:.3f}{tag}",
                flush=True,
            )

        if early_stop_patience is not None and bad_epochs >= early_stop_patience:
            print(f"  early stop at ep {ep} (no improvement for {early_stop_patience} epochs)", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  best val MAE={best_val_mae:.4f} at epoch {best_epoch}", flush=True)

    y_pred, y_true, sids_t = evaluate(model, test_loader, mean, std, device)
    test_m = per_horizon_metrics(y_pred, y_true, horizons=horizons)
    print(f"  test metrics - all (best ckpt @ ep {best_epoch}):", flush=True)
    for h, d in test_m.items():
        print(
            f"    h={h:2d}  MAE={d['mae']:.3f}  RMSE={d['rmse']:.3f}  MAPE={d['mape']:.2f}%",
            flush=True,
        )

    test_m_split: dict = {}
    if pool_tag is not None:
        test_m_split = per_horizon_metrics_seen_unseen(
            y_pred, y_true, sids_t, pool_tag, horizons=horizons
        )
        for name in ("seen", "unseen", "val_unseen", "test_unseen"):
            if name not in test_m_split:
                continue
            d_split = test_m_split[name]
            print(
                f"  test metrics - {name}  (n={d_split['n_samples']:,}):",
                flush=True,
            )
            for h, d in d_split["per_h"].items():
                print(
                    f"    h={h:2d}  MAE={d['mae']:.3f}  RMSE={d['rmse']:.3f}  MAPE={d['mape']:.2f}%",
                    flush=True,
                )
    print(f"  total elapsed {time.time() - t_start:.1f}s", flush=True)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
            "test_metrics": test_m,
            "mean": mean,
            "std": std,
        }, save_path)
        print(f"  saved -> {save_path}", flush=True)

    return {
        "history": history,
        "best_val_mae": best_val_mae,
        "best_epoch": best_epoch,
        "test_metrics": test_m,
        "test_metrics_split": test_m_split,
    }
