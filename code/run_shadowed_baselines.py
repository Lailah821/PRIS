"""Launcher: shadowed baselines on PEMS-BAY / METR-LA × C0 / Cstd.

Setup: target self-mask + Gaussian adj top-8 + stride 12 + 40% random eval mask.

Models (paper-strict, body unchanged; only hidden capacity normalized):
  lstm     : MultinodeLSTM       (slot×channel flatten + nn.LSTM)
  dlinear  : MultinodeDLinear    (slot×channel flatten + 2 channel-indep Linear)
  stgcn    : MultinodeSTGCN      (ChebConv + temporal GLU, paper Yu 2018)
  agcrn    : MultinodeAGCRN      (adaptive adj + GRU, Bai 2020)
  gwn      : MultinodeGWNet      (gated TCN + adaptive adj, Wu 2019)
  dcrnn    : MultinodeDCRNN      (diffusion conv GRU, Li 2018)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

from pems_metr_dataloader_v2 import build_loaders_shadowed
from pems_metr_train_shadowed import train_eval_shadowed

LOG_ROOT = Path(__file__).parent / "logs" / "shadowed_baselines"
CKPT_ROOT = Path(__file__).parent / "ckpts" / "shadowed_baselines"


def build_model(model_name: str, pkg: dict, *, hidden_size: int, num_layers: int,
                input_len: int, output_len: int):
    n_slots = 1 + pkg["top_n"]
    in_ch = pkg["in_channels"]
    if model_name == "lstm":
        from pems_metr_lstm_multinode import MultinodeLSTM
        return MultinodeLSTM(
            in_channels=in_ch, n_slots=n_slots, out_len=output_len,
            hidden_size=hidden_size, num_layers=num_layers,
        )
    if model_name == "dlinear":
        from pems_metr_dlinear_multinode import MultinodeDLinear
        return MultinodeDLinear(
            in_channels=in_ch, n_slots=n_slots,
            seq_len=input_len, pred_len=output_len,
            kernel_size=25, individual=False,
        )
    if model_name == "stgcn":
        from pems_metr_stgcn_shadowed import STGCNShadowed
        adj_block = torch.from_numpy(pkg["adj_block"]).float()
        return STGCNShadowed(
            adj_block=adj_block, top_n=pkg["top_n"], in_channels=in_ch,
            hidden_dim=hidden_size, out_len=output_len, input_len=input_len,
        )
    if model_name == "agcrn":
        from pems_metr_agcrn_shadowed import AGCRNShadowed
        return AGCRNShadowed(
            top_n=pkg["top_n"], in_channels=in_ch,
            hidden_dim=hidden_size, num_layers=num_layers, out_len=output_len,
        )
    if model_name == "gwn":
        from pems_metr_gwn_shadowed import GWNetShadowed
        adj_block = torch.from_numpy(pkg["adj_block"]).float()
        return GWNetShadowed(
            adj_block=adj_block, top_n=pkg["top_n"], in_channels=in_ch,
            hidden_dim=hidden_size, out_len=output_len, input_len=input_len,
        )
    if model_name == "dcrnn":
        from pems_metr_dcrnn_shadowed import DCRNNShadowed
        adj_block = torch.from_numpy(pkg["adj_block"]).float()
        return DCRNNShadowed(
            adj_block=adj_block, top_n=pkg["top_n"], in_channels=in_ch,
            hidden_dim=hidden_size, num_layers=num_layers, out_len=output_len,
        )
    raise ValueError(f"unknown model {model_name}")


def run_one(
    model_name: str,
    dataset: str,
    cell: str,
    *,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    lr: float,
    input_len: int,
    output_len: int,
    top_n: int,
    train_stride: int,
    mask_ratio_eval: float,
    device: str,
    log_every: int,
    early_stop_patience: int,
):
    tag = f"{dataset}_{cell}"
    print(f"\n========== shadowed {model_name.upper()} {tag} ==========", flush=True)
    print(
        f"  epochs={epochs} batch={batch_size} hidden={hidden_size} "
        f"layers={num_layers} lr={lr} top_n={top_n} "
        f"train_stride={train_stride} mask_ratio={mask_ratio_eval} device={device}",
        flush=True,
    )

    pkg = build_loaders_shadowed(
        dataset=dataset, cell=cell, input_len=input_len, output_len=output_len,
        top_n=top_n, train_stride=train_stride, mask_ratio_eval=mask_ratio_eval,
        batch_size=batch_size,
    )
    print(
        f"  nodes={pkg['num_nodes']}  in_ch={pkg['in_channels']}  "
        f"mean={pkg['mean']:.3f}  std={pkg['std']:.3f}",
        flush=True,
    )
    print(
        f"  train={pkg['train_samples']:,}  val={pkg['val_samples']:,}  test={pkg['test_samples']:,}",
        flush=True,
    )

    model = build_model(model_name, pkg, hidden_size=hidden_size, num_layers=num_layers,
                        input_len=input_len, output_len=output_len)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  model params: {n_params:,}", flush=True)

    save_path = CKPT_ROOT / f"shadowed_{model_name}_{tag}.pt"
    result = train_eval_shadowed(
        model, pkg, mean=pkg["mean"], std=pkg["std"], device=device,
        epochs=epochs, lr=lr, log_every=log_every, save_path=str(save_path),
        early_stop_patience=early_stop_patience,
    )

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"shadowed_{model_name}_{tag}.json"
    log_payload = {
        "model": model_name,
        "dataset": dataset,
        "cell": cell,
        "setup": "shadowed",
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "lr": lr,
        "input_len": input_len,
        "output_len": output_len,
        "top_n": top_n,
        "train_stride": train_stride,
        "mask_ratio_eval": mask_ratio_eval,
        "num_nodes": pkg["num_nodes"],
        "in_channels": pkg["in_channels"],
        "mean": pkg["mean"],
        "std": pkg["std"],
        "n_params": n_params,
        "train_samples": pkg["train_samples"],
        "val_samples": pkg["val_samples"],
        "test_samples": pkg["test_samples"],
        "best_val_mae": result["best_val_mae"],
        "best_epoch": result["best_epoch"],
        "test_metrics": result["test_metrics"],
        "test_metrics_split": result.get("test_metrics_split", {}),
        "train_v_count": int(len(pkg["train_ids"])),
        "val_v_count": int(len(pkg["val_ids"])),
        "test_v_count": int(len(pkg["test_ids"])),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_payload, f, indent=2)
    print(f"  log -> {log_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["lstm"],
                    choices=["lstm", "dlinear", "stgcn", "agcrn", "gwn", "dcrnn"])
    ap.add_argument("--datasets", nargs="+", default=["metr_la"], choices=["metr_la", "pems_bay"])
    ap.add_argument("--cells", nargs="+", default=["C0"], choices=["C0", "Cstd"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--input_len", type=int, default=12)
    ap.add_argument("--output_len", type=int, default=24)
    ap.add_argument("--top_n", type=int, default=8)
    ap.add_argument("--train_stride", type=int, default=12)
    ap.add_argument("--mask_ratio_eval", type=float, default=0.40)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--early_stop_patience", type=int, default=8)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in args.models:
        for ds in args.datasets:
            for cell in args.cells:
                run_one(
                    model_name, ds, cell,
                    epochs=args.epochs, batch_size=args.batch_size,
                    hidden_size=args.hidden_size, num_layers=args.num_layers,
                    lr=args.lr, input_len=args.input_len, output_len=args.output_len,
                    top_n=args.top_n, train_stride=args.train_stride,
                    mask_ratio_eval=args.mask_ratio_eval, device=device,
                    log_every=args.log_every, early_stop_patience=args.early_stop_patience,
                )

    print("\nDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
