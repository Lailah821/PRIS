"""Launcher: shadowed PGTFT on PEMS-BAY / METR-LA x C0 / Cstd.

Setup: target self-mask + Gaussian adj top-8 + stride 12 + 40% random eval mask.

Variants (3 lite + 11 mar 77 ablations, total 14):

Adj-fusion lite (GRN bypass, VSN, soft_gcn + dist_adj):
  a       : DCRNN-style
  b       : AGCRN-style (adaptive_adj + NAPL)
  c       : Hybrid (NAPL only)

mar 77 PGTFT ablations (from `1) for_paper_work/C0_C3_matrix_fillnan.md`):
  paper                : full PGTFT body (SoftGCN + VSN + attn_grn + peak_grn + gate_block)
  r1_1_hskip           : paper + horizon-skip (w_max=0.3, tau=60)
  r2_a_l6a             : paper - peak_grn - dist_adj
  r2_b_l6b             : r2_a + bypass VSN / attn_grn / gate_block
  r2_c1                : r2_b + TCN (replace SoftGCN)
  r2_c3                : r2_b + SoftGCN + TCN (additive)
  r2_c4                : r2_b + target-node-select (slot 1)
  r3_delta             : r2_c1 + TCN dilations [1,1,2,2]
  r3_lstgf             : r2_b + LSTGF γ-fusion
  r3_delta_gelu        : r3_delta + GELU activation
  r3_delta_gelu_peak   : r3_delta_gelu + per-layer peak gate

Shadowed-NA mar 77 variants (skipped):
  R1-A swap / R1-D lite — require peak feature swap, no peak in shadowed.
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
from pems_metr_pgtft_shadowed import PGTFTShadowed
from pems_metr_train_shadowed import train_eval_shadowed

LOG_ROOT = Path(__file__).parent / "logs" / "shadowed_pgtft"
CKPT_ROOT = Path(__file__).parent / "ckpts" / "shadowed_pgtft"


VARIANT_FLAGS = {
    # ── adj-fusion lite (GRN bypass, VSN active, soft_gcn + dist_adj) ──
    "a": dict(
        use_adaptive_adj=False, use_napl=False,
        use_attn_grn=False, use_peak_grn=False, use_final_gate_grn=False,
    ),
    "b": dict(
        use_adaptive_adj=True, use_napl=True,
        use_attn_grn=False, use_peak_grn=False, use_final_gate_grn=False,
    ),
    "c": dict(
        use_adaptive_adj=False, use_napl=True,
        use_attn_grn=False, use_peak_grn=False, use_final_gate_grn=False,
    ),
    # ── mar 77 ablations (DCRNN-style adj fusion) ──
    "paper": dict(),  # all defaults: full PGTFT body
    "r1_1_hskip": dict(
        use_horizon_skip=True, hskip_w_max=0.3, hskip_tau=60.0,
    ),
    "r2_a_l6a": dict(
        use_peak_grn=False, use_dist_adj=False,
    ),
    "r2_b_l6b": dict(
        use_peak_grn=False, use_dist_adj=False,
        use_vsn=False, use_attn_grn=False, use_final_gate_grn=False,
    ),
    "r2_c1": dict(
        use_peak_grn=False, use_dist_adj=False,
        use_vsn=False, use_attn_grn=False, use_final_gate_grn=False,
        use_soft_gcn=False, use_tcn=True,
    ),
    "r2_c3": dict(
        use_peak_grn=False, use_dist_adj=False,
        use_vsn=False, use_attn_grn=False, use_final_gate_grn=False,
        use_soft_gcn=True, use_tcn=True,
    ),
    "r2_c4": dict(
        use_peak_grn=False, use_dist_adj=False,
        use_vsn=False, use_attn_grn=False, use_final_gate_grn=False,
        use_target_node_select=True,
    ),
    "r3_delta": dict(
        use_peak_grn=False, use_dist_adj=False,
        use_vsn=False, use_attn_grn=False, use_final_gate_grn=False,
        use_soft_gcn=False, use_tcn=True, tcn_dilations=[1, 1, 2, 2],
    ),
    "r3_lstgf": dict(
        use_peak_grn=False, use_dist_adj=False,
        use_vsn=False, use_attn_grn=False, use_final_gate_grn=False,
        use_soft_gcn=False, use_lstgf=True,
    ),
    "r3_delta_gelu": dict(
        use_peak_grn=False, use_dist_adj=False,
        use_vsn=False, use_attn_grn=False, use_final_gate_grn=False,
        use_soft_gcn=False, use_tcn=True,
        tcn_dilations=[1, 1, 2, 2], tcn_activation="gelu",
    ),
    "r3_delta_gelu_peak": dict(
        use_peak_grn=False, use_dist_adj=False,
        use_vsn=False, use_attn_grn=False, use_final_gate_grn=False,
        use_soft_gcn=False, use_tcn=True,
        tcn_dilations=[1, 1, 2, 2], tcn_activation="gelu",
        tcn_use_peak_gate=True, tcn_peak_alpha=0.3,
    ),
}


def _short_flag_str(flags: dict) -> str:
    """Short one-liner of non-default flags for logs."""
    if not flags:
        return "(all defaults = paper)"
    return " ".join(f"{k}={v}" for k, v in flags.items())


def run_one(
    dataset: str,
    cell: str,
    variant: str,
    *,
    epochs: int,
    batch_size: int,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
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
    tag = f"{dataset}_{cell}_v{variant}"
    flags = VARIANT_FLAGS[variant]
    print(f"\n========== shadowed PGTFT({variant}) {dataset}_{cell} ==========", flush=True)
    print(f"  variant flags: {_short_flag_str(flags)}", flush=True)
    print(
        f"  epochs={epochs} batch={batch_size} hidden={hidden_size} "
        f"layers={num_layers} heads={num_heads} lr={lr} top_n={top_n} "
        f"train_stride={train_stride} mask_ratio={mask_ratio_eval} device={device}",
        flush=True,
    )

    pkg = build_loaders_shadowed(
        dataset=dataset,
        cell=cell,
        input_len=input_len,
        output_len=output_len,
        top_n=top_n,
        train_stride=train_stride,
        mask_ratio_eval=mask_ratio_eval,
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
    print(
        f"  spatial split: TRAIN_V={len(pkg['train_ids'])}  "
        f"VAL_V={len(pkg['val_ids'])}  TEST_V={len(pkg['test_ids'])}  "
        f"(excluded={pkg['n_excluded']} zero-neighbor sensors)",
        flush=True,
    )

    adj_block = torch.from_numpy(pkg["adj_block"]).float()
    model = PGTFTShadowed(
        adj_block=adj_block,
        top_n=pkg["top_n"],
        in_channels=pkg["in_channels"],
        hidden_dim=hidden_size,
        num_layers=num_layers,
        out_len=output_len,
        num_heads=num_heads,
        **flags,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  model params: {n_params:,}  (top_n={pkg['top_n']}, hidden={hidden_size}, heads={num_heads})",
        flush=True,
    )

    save_path = CKPT_ROOT / f"shadowed_pgtft_{tag}.pt"
    result = train_eval_shadowed(
        model,
        pkg,
        mean=pkg["mean"],
        std=pkg["std"],
        device=device,
        epochs=epochs,
        lr=lr,
        log_every=log_every,
        save_path=str(save_path),
        early_stop_patience=early_stop_patience,
    )

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"shadowed_pgtft_{tag}.json"
    log_payload = {
        "dataset": dataset,
        "cell": cell,
        "variant": variant,
        "setup": "shadowed",
        "epochs": epochs,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "lr": lr,
        "input_len": input_len,
        "output_len": output_len,
        "top_n": top_n,
        "train_stride": train_stride,
        "mask_ratio_eval": mask_ratio_eval,
        "variant_flags": {k: (v if not isinstance(v, list) else list(v)) for k, v in flags.items()},
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
    ap.add_argument("--datasets", nargs="+", default=["metr_la"], choices=["metr_la", "pems_bay"])
    ap.add_argument("--cells", nargs="+", default=["C0"], choices=["C0", "Cstd"])
    ap.add_argument("--variants", nargs="+", default=["paper"], choices=list(VARIANT_FLAGS.keys()))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=4)
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
    for ds in args.datasets:
        for cell in args.cells:
            for variant in args.variants:
                run_one(
                    ds,
                    cell,
                    variant,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    hidden_size=args.hidden_size,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    lr=args.lr,
                    input_len=args.input_len,
                    output_len=args.output_len,
                    top_n=args.top_n,
                    train_stride=args.train_stride,
                    mask_ratio_eval=args.mask_ratio_eval,
                    device=device,
                    log_every=args.log_every,
                    early_stop_patience=args.early_stop_patience,
                )

    print("\nDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
