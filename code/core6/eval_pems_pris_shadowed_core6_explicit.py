"""PRIS Explicit-policy launcher on PEMS-BAY / METR-LA (frozen — do not modify logic).

This is the PRIS **Explicit-policy** variant from the SIGSPATIAL 2026 paper. The
Explicit-policy supplies aggregate specialist-effectiveness hints to the prompt:
original TOOL EXPERTISE with dataset-anchored hints, spread few-shot examples
(weights 0.35-0.42), and a `tool_prior` field. (There is NO "weighting policy" block.)
Its companion is the Open-policy launcher, which withholds these hints.

Frozen for reproducibility of the published Explicit-policy numbers. Internal
identifiers and output-JSON keys retain the legacy `plan_c`/`core7` names; the logic is
unchanged. Do not edit the logic; use this only to reproduce the Explicit-policy result.

Reproduction command (Explicit-policy, 6 Specialists):
  python code/core6/eval_pems_pris_shadowed_core6_explicit_no_knn.py \\
      --dataset pems_bay --eval-cell Cstd \\
      --n-sensors 30 --windows-per-sensor 20 \\
      --sensor-offset 0 --window-offset 0 \\
      --agent-model qwen3:14b \\
      --output-json results_interp_a/pems_main_repro_30s20w.json

Default protocol:
  - dataset=pems_bay, eval_cell=Cstd
  - tools: KNN-L1(k=6), PGTFT-b C0/Cstd, GRU C0, GraphWaveNet Cstd, AGCRN Cstd, DCRNN Cstd
  - pgtft variant: b (paper-strict adj-fusion lite)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

CORE7_DIR = Path(__file__).resolve().parent
PROJECT_2 = CORE7_DIR.parent  # core7/ → project_2/
if str(PROJECT_2) not in sys.path:
    sys.path.insert(0, str(PROJECT_2))

from phase_6_llm_client import call_agent  # noqa: E402
from pems_metr_dataloader_v2 import build_loaders_shadowed, load_adj  # noqa: E402
from pems_metr_gru_multinode import MultinodeGRU  # noqa: E402
from pems_metr_gwn_shadowed import GWNetShadowed  # noqa: E402
from pems_metr_pgtft_shadowed import PGTFTShadowed  # noqa: E402
from pems_metr_metrics import per_horizon_metrics  # noqa: E402
from run_shadowed_baselines import build_model as build_standard_model  # noqa: E402
from run_shadowed_pgtft import VARIANT_FLAGS as PGTFT_VARIANT_FLAGS  # noqa: E402


TOOLS_CORE5 = [
    "knn_l1_k6",
    "pgtft_c0",
    "pgtft_cstd",
    "gru_c0",
    "graph_wavenet_cstd",
]
TOOLS_CORE6_DCRNN = TOOLS_CORE5 + ["dcrnn_cstd"]
TOOLS_CORE7 = TOOLS_CORE5 + ["agcrn_cstd", "dcrnn_cstd"]
TOOLS = list(TOOLS_CORE7)  # core7 default: 7 tools (AGCRN + DCRNN added)


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unsupported checkpoint format: {path}")


def _build_pkg(dataset: str, cell: str, batch_size: int) -> dict[str, Any]:
    return build_loaders_shadowed(
        dataset=dataset,
        cell=cell,
        input_len=12,
        output_len=24,
        top_n=8,
        train_stride=12,
        val_stride=12,
        test_stride=12,
        mask_ratio_eval=0.40,
        batch_size=batch_size,
    )


def _first_indices_per_sensor(test_ds, limit: int | None = None) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for idx in range(len(test_ds)):
        _wi, ki = test_ds.flat[idx]
        sid = int(test_ds.mask_sets[int(_wi), int(ki)])
        if sid in seen:
            continue
        seen.add(sid)
        out.append(idx)
        if limit is not None and len(out) >= limit:
            break
    return out


def _indices_by_sensor_windows(
    test_ds,
    *,
    n_sensors: int | None,
    windows_per_sensor: int,
    sensor_offset: int = 0,
    window_offset: int = 0,
) -> list[int]:
    """Select up to K windows for each of the first N distinct sensors."""
    if sensor_offset < 0 or window_offset < 0:
        raise ValueError("sensor_offset and window_offset must be non-negative")
    if windows_per_sensor <= 1:
        if sensor_offset or window_offset:
            raise ValueError("offset sampling requires windows_per_sensor > 1")
        return _first_indices_per_sensor(test_ds, limit=n_sensors)

    sensor_order: list[int] = []
    selected: dict[int, list[int]] = {}
    seen_windows: dict[int, set[int]] = {}
    skipped_windows: dict[int, set[int]] = {}
    for idx in range(len(test_ds)):
        wi, ki = test_ds.flat[idx]
        sid = int(test_ds.mask_sets[int(wi), int(ki)])
        if sid not in selected:
            if len(sensor_order) < sensor_offset:
                sensor_order.append(sid)
                continue
            if n_sensors is not None and len(sensor_order) >= sensor_offset + n_sensors:
                continue
            sensor_order.append(sid)
            selected[sid] = []
            seen_windows[sid] = set()
            skipped_windows[sid] = set()
        if sid not in selected:
            continue
        if int(wi) in seen_windows[sid] or int(wi) in skipped_windows[sid]:
            continue
        if len(skipped_windows[sid]) < window_offset:
            skipped_windows[sid].add(int(wi))
            continue
        if len(selected[sid]) >= windows_per_sensor:
            continue
        selected[sid].append(idx)
        seen_windows[sid].add(int(wi))
        if (
            n_sensors is not None
            and len(sensor_order) >= sensor_offset + n_sensors
            and all(len(selected[s]) >= windows_per_sensor for s in sensor_order[sensor_offset:])
        ):
            break

    out: list[int] = []
    active_sensor_order = sensor_order[sensor_offset:] if n_sensors is None else sensor_order[sensor_offset:sensor_offset + n_sensors]
    for sid in active_sensor_order:
        out.extend(selected.get(sid, []))
    return out


def _batch_from_indices(ds, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    xs = []
    ys = []
    sids = []
    for idx in indices:
        x, _y_norm, y_raw, sid = ds[idx]
        xs.append(x)
        ys.append(y_raw)
        sids.append(int(sid))
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), np.asarray(sids, dtype=np.int64)


def _load_gru(dataset: str, cell: str, pkg: dict[str, Any], device: str) -> torch.nn.Module:
    model = MultinodeGRU(
        in_channels=pkg["in_channels"],
        n_slots=1 + pkg["top_n"],
        out_len=24,
        hidden_size=64,
        num_layers=2,
    )
    path = PROJECT_2 / "ckpts" / "shadowed_gru" / f"shadowed_gru_{dataset}_{cell}.pt"
    model.load_state_dict(_load_state(path))
    return model.to(device).eval()


def _load_gwn(dataset: str, cell: str, pkg: dict[str, Any], device: str) -> torch.nn.Module:
    model = GWNetShadowed(
        adj_block=torch.from_numpy(pkg["adj_block"]).float(),
        top_n=pkg["top_n"],
        in_channels=pkg["in_channels"],
        hidden_dim=64,
        out_len=24,
        input_len=12,
    )
    path = PROJECT_2 / "ckpts" / "shadowed_baselines" / f"shadowed_gwn_{dataset}_{cell}.pt"
    model.load_state_dict(_load_state(path))
    return model.to(device).eval()


def _load_standard_baseline(
    name: str,
    dataset: str,
    cell: str,
    pkg: dict[str, Any],
    device: str,
) -> torch.nn.Module:
    model = build_standard_model(
        name,
        pkg,
        hidden_size=64,
        num_layers=2,
        input_len=12,
        output_len=24,
    )
    path = PROJECT_2 / "ckpts" / "shadowed_baselines" / f"shadowed_{name}_{dataset}_{cell}.pt"
    model.load_state_dict(_load_state(path))
    return model.to(device).eval()


def _load_pgtft(
    dataset: str,
    cell: str,
    variant: str,
    pkg: dict[str, Any],
    device: str,
) -> torch.nn.Module:
    flags = PGTFT_VARIANT_FLAGS[variant]
    model = PGTFTShadowed(
        adj_block=torch.from_numpy(pkg["adj_block"]).float(),
        top_n=pkg["top_n"],
        in_channels=pkg["in_channels"],
        hidden_dim=64,
        num_layers=2,
        out_len=24,
        num_heads=4,
        **flags,
    )
    path = PROJECT_2 / "ckpts" / "shadowed_pgtft" / f"shadowed_pgtft_{dataset}_{cell}_v{variant}.pt"
    model.load_state_dict(_load_state(path))
    return model.to(device).eval()


def _predict_raw(model, x: torch.Tensor, sids: np.ndarray, mean: float, std: float, device: str) -> np.ndarray:
    with torch.no_grad():
        sid_t = torch.as_tensor(sids, dtype=torch.long, device=device)
        pred_norm = model(x.to(device), sid_t)
    return pred_norm.detach().cpu().numpy() * float(std) + float(mean)


def _knn_l1_k6_raw(x_cstd: torch.Tensor, mean: float, std: float, out_len: int = 24) -> np.ndarray:
    # Use last input step neighbor speeds, ignore zero-masked neighbor slots.
    last = x_cstd[:, -1, 1:, 0].numpy()
    preds = []
    for row in last:
        nz = row[np.abs(row) > 1e-8]
        if nz.size == 0:
            val = 0.0
        else:
            # Gaussian top-neighbor order is already strongest-first; k=6 anchor.
            val = float(np.mean(nz[:6]))
        preds.append(np.full(out_len, val * float(std) + float(mean), dtype=np.float32))
    return np.stack(preds, axis=0)


def _metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    mape = float(np.mean(np.abs(err) / np.maximum(np.abs(y_true), 1e-2)) * 100.0)
    mask_gt5 = np.abs(y_true) > 5.0
    mape_gt5 = (
        float(np.mean(np.abs(err[mask_gt5]) / np.abs(y_true[mask_gt5])) * 100.0)
        if np.any(mask_gt5)
        else float("nan")
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mape_gt5mph": mape_gt5,
        "mape_gt5mph_fraction": float(np.mean(mask_gt5)),
    }


def _bucket_mape_gt5(y_pred: np.ndarray, y_true: np.ndarray) -> dict[str, dict[str, float | int]]:
    """Element-level bucket MAPE using the paper metric mask y_true > 5 mph."""
    buckets = {
        "congestion_0_30": (0.0, 30.0),
        "regular_30_55": (30.0, 55.0),
        "freeflow_55_100": (55.0, 100.0),
    }
    out: dict[str, dict[str, float | int]] = {}
    for name, (lo, hi) in buckets.items():
        mask = (y_true >= lo) & (y_true < hi) & (y_true > 5.0)
        count = int(mask.sum())
        if count:
            value = float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / np.abs(y_true[mask])) * 100.0)
        else:
            value = float("nan")
        out[name] = {"count": count, "mape_gt5mph": value}
    return out


def _load_sensor_geo(dataset: str) -> dict[str, Any]:
    """Load benchmark sensor ids and coordinates as a synthetic road layer.

    PEMS/METR do not ship MAR-style road_metadata/SVI/road_master files. For
    the second-dataset probe, the sensor graph itself is the spatial layer:
    external sensor id, latitude/longitude, Gaussian adjacency, and distance
    CSV when available.
    """
    root = PROJECT_2.parent / "project_1" / "input_storage" / "benchmark_data" / dataset
    if dataset == "pems_bay":
        adj_path = root / "adj_mx_bay.pkl"
        loc_path = root / "graph_sensor_locations_bay.csv"
    else:
        adj_path = root / "adj_mx.pkl"
        loc_path = root / "graph_sensor_locations.csv"

    with adj_path.open("rb") as f:
        adj_obj = pickle.load(f, encoding="latin1")
    sensor_ids = [str(x) for x in adj_obj[0]] if isinstance(adj_obj, (list, tuple)) else []

    loc_by_external: dict[str, dict[str, float]] = {}
    with loc_path.open(newline="", encoding="utf-8") as f:
        sample = f.read(256)
        f.seek(0)
        has_header = "latitude" in sample.lower() or "sensor_id" in sample.lower()
        if has_header:
            for row in csv.DictReader(f):
                sid = str(row.get("sensor_id") or row.get("sensor") or row.get("id"))
                lat = row.get("latitude") or row.get("lat")
                lon = row.get("longitude") or row.get("lon")
                if sid and lat and lon:
                    loc_by_external[sid] = {"lat": float(lat), "lon": float(lon)}
        else:
            for row in csv.reader(f):
                if len(row) >= 3:
                    loc_by_external[str(row[0])] = {"lat": float(row[1]), "lon": float(row[2])}

    loc_by_index = {
        i: {
            "external_sensor_id": sid,
            "lat": loc_by_external.get(sid, {}).get("lat"),
            "lon": loc_by_external.get(sid, {}).get("lon"),
        }
        for i, sid in enumerate(sensor_ids)
    }
    return {
        "source": "benchmark_sensor_locations_and_gaussian_adjacency",
        "sensor_ids": sensor_ids,
        "loc_by_index": loc_by_index,
    }


def _haversine_km(lat1: float | None, lon1: float | None, lat2: float | None, lon2: float | None) -> float | None:
    if None in (lat1, lon1, lat2, lon2):
        return None
    r = 6371.0
    p1 = math.radians(float(lat1))
    p2 = math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if 0 <= start < end:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}")


def _unwrap_a2_decision(obj: dict[str, Any]) -> dict[str, Any]:
    if isinstance(obj.get("a2_decision"), dict):
        return obj["a2_decision"]
    if "blend_weights" in obj or "forecast_tool" in obj:
        return obj
    return {}


def _normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    vals: dict[str, float] = {}
    for tool in TOOLS:
        try:
            v = float(weights.get(tool, 0.0))
        except Exception:
            v = 0.0
        vals[tool] = max(0.0, v)
    s = sum(vals.values())
    if not math.isfinite(s) or s <= 1e-8:
        return {tool: 1.0 / len(TOOLS) for tool in TOOLS}
    return {tool: vals[tool] / s for tool in TOOLS}


def _a0_graph_similarity_context(
    *,
    adj: np.ndarray,
    pkg: dict[str, Any],
    sensor_id: int,
    x_cstd_one: torch.Tensor,
    sensor_geo: dict[str, Any],
) -> dict[str, Any]:
    nbr_idx = pkg["neighbor_idx"][sensor_id]
    valid = pkg["neighbor_valid"][sensor_id].astype(bool)
    weights = adj[sensor_id, nbr_idx][valid].astype(float)
    loc_by_index = sensor_geo.get("loc_by_index", {})
    target_loc = loc_by_index.get(int(sensor_id), {})
    valid_neighbor_indices = [int(x) for x in nbr_idx[valid][:6]]
    neighbor_rows = []
    distances = []
    for nidx, w in zip(valid_neighbor_indices, weights[:6]):
        nloc = loc_by_index.get(int(nidx), {})
        dist_km = _haversine_km(target_loc.get("lat"), target_loc.get("lon"), nloc.get("lat"), nloc.get("lon"))
        if dist_km is not None:
            distances.append(float(dist_km))
        neighbor_rows.append({
            "sensor_index": int(nidx),
            "external_sensor_id": nloc.get("external_sensor_id"),
            "adj_weight": float(w),
            "distance_km": dist_km,
        })
    last_neighbor_speed = x_cstd_one[-1, 1:, 0].numpy()
    last_valid_speed = last_neighbor_speed[valid]
    last_nonzero_speed = last_valid_speed[np.abs(last_valid_speed) > 1e-8]
    anchor_mean_z = float(np.mean(last_nonzero_speed)) if last_nonzero_speed.size else 0.0
    anchor_std_z = float(np.std(last_nonzero_speed)) if last_nonzero_speed.size else 0.0
    anchor_mean_mph = anchor_mean_z * float(pkg["std"]) + float(pkg["mean"])
    anchor_std_mph = anchor_std_z * float(pkg["std"])
    return {
        "available": True,
        "source": "pems_sensor_graph_coordinates_no_svi",
        "target_sensor_index": int(sensor_id),
        "target_external_sensor_id": target_loc.get("external_sensor_id"),
        "target_lat": target_loc.get("lat"),
        "target_lon": target_loc.get("lon"),
        "valid_neighbor_count": int(valid.sum()),
        "top_neighbor_sensor_indices": valid_neighbor_indices,
        "top_neighbor_sensors": neighbor_rows,
        "neighbor_distance_km_mean_top6": float(np.mean(distances)) if distances else None,
        "neighbor_distance_km_min_top6": float(np.min(distances)) if distances else None,
        "target_neighbor_weight_max": float(np.max(weights)) if weights.size else 0.0,
        "target_neighbor_weight_mean_top6": float(np.mean(weights[:6])) if weights.size else 0.0,
        "target_neighbor_weight_sum_top6": float(np.sum(weights[:6])) if weights.size else 0.0,
        "last_step_observed_neighbor_count": int(last_nonzero_speed.size),
        "last_step_neighbor_speed_mean_z": anchor_mean_z,
        "last_step_neighbor_speed_std_z": anchor_std_z,
        "last_step_neighbor_speed_mean_mph": float(anchor_mean_mph),
        "last_step_neighbor_speed_std_mph": float(anchor_std_mph),
    }


def _a1_graph_sequence_candidates(
    *,
    adj: np.ndarray,
    pkg: dict[str, Any],
    sensor_id: int,
    sensor_geo: dict[str, Any],
    max_sequences: int = 4,
) -> list[dict[str, Any]]:
    """Build PEMS A1 candidates with a graph-only K-score.

    This mirrors the MAR A1 idea (choose among K-score-ranked candidate
    sequences) while replacing road/SVI distance terms with sensor graph terms.
    K = 0.4*(1-U) + 0.3*C + 0.3*P
      U: uncertainty from valid neighbor support count.
      C: graph connectivity strength of the target->support->support chain.
      P: progress/proximity toward the target, represented by first-hop graph
         strength because every PEMS candidate is a target-neighbor sequence.
    """
    nbr_idx = pkg["neighbor_idx"][sensor_id]
    valid = pkg["neighbor_valid"][sensor_id].astype(bool)
    first_hop = nbr_idx[valid]
    first_weights = adj[sensor_id, first_hop].astype(float)
    loc_by_index = sensor_geo.get("loc_by_index", {})
    target_loc = loc_by_index.get(int(sensor_id), {})
    max_first = float(np.max(first_weights)) if first_weights.size else 1.0
    candidates: list[dict[str, Any]] = []
    for pos in range(len(first_hop)):
        n1 = int(first_hop[pos])
        n1_weight = float(first_weights[pos])
        n1_nbr_idx = pkg["neighbor_idx"][n1]
        n1_valid = pkg["neighbor_valid"][n1].astype(bool)
        second_pool = [
            int(x)
            for x in n1_nbr_idx[n1_valid]
            if int(x) != int(sensor_id)
        ]
        n2 = second_pool[0] if second_pool else n1
        n2_weight = float(adj[n1, n2]) if n2 != n1 else 0.0
        n1_loc = loc_by_index.get(int(n1), {})
        n2_loc = loc_by_index.get(int(n2), {})
        target_to_n1_km = _haversine_km(target_loc.get("lat"), target_loc.get("lon"), n1_loc.get("lat"), n1_loc.get("lon"))
        n1_to_n2_km = _haversine_km(n1_loc.get("lat"), n1_loc.get("lon"), n2_loc.get("lat"), n2_loc.get("lon"))
        n1_valid_count = int(n1_valid.sum())
        uncertainty = max(0.0, min(1.0, 1.0 - (min(n1_valid_count, 6) / 6.0)))
        c_first = n1_weight / max(max_first, 1e-8)
        c_second = min(1.0, n2_weight / max(n1_weight, 1e-8)) if n2_weight > 0 else 0.0
        connectivity = max(0.0, min(1.0, 0.7 * c_first + 0.3 * c_second))
        progress = max(0.0, min(1.0, c_first))
        k_score = 0.4 * (1.0 - uncertainty) + 0.3 * connectivity + 0.3 * progress
        candidates.append({
            "sequence_id": f"seq_raw_{pos + 1}",
            "sensor_chain": [int(sensor_id), n1, int(n2)],
            "external_sensor_chain": [
                target_loc.get("external_sensor_id"),
                n1_loc.get("external_sensor_id"),
                n2_loc.get("external_sensor_id"),
            ],
            "target_to_first_hop_km": target_to_n1_km,
            "first_to_second_hop_km": n1_to_n2_km,
            "first_hop_weight": n1_weight,
            "second_hop_weight": n2_weight,
            "chain_weight_product": float(n1_weight * max(n2_weight, 1e-6)),
            "valid_neighbor_count": n1_valid_count,
            "uncertainty": float(uncertainty),
            "connectivity": float(connectivity),
            "progress": float(progress),
            "k_score": float(k_score),
            "k_score_formula": "0.4*(1-U)+0.3*C+0.3*P, graph-only PEMS adaptation",
            "interpretation": "target -> strongest graph neighbor -> neighbor support",
        })
    candidates.sort(key=lambda row: row["k_score"], reverse=True)
    out = candidates[:max_sequences]
    for rank, row in enumerate(out, start=1):
        row["sequence_id"] = f"seq_{rank}"
        row["rank_by_k_score"] = rank
    return out


def _fallback_a1_decision(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        return {
            "selected_sequence_ids": [],
            "primary_support_sensor": None,
            "evidence_tag": "no_graph_sequence",
        }
    best = max(candidates, key=lambda row: row.get("chain_weight_product", 0.0))
    return {
        "selected_sequence_ids": [best["sequence_id"]],
        "primary_support_sensor": best["sensor_chain"][1],
        "evidence_tag": "graph_similarity_sequence",
    }


def _fallback_a3_guard() -> dict[str, Any]:
    return {
        "guard_policy": "deviation",
        "guard_threshold_mph": 4.0,
        "fallback_policy": "none",
        "evidence_tag": "default_guard",
    }


def _fallback_a4_residual() -> dict[str, Any]:
    return {
        "residual_policy": "none",
        "residual_weight": 0.0,
        "residual_clip_mph": 1.5,
        "evidence_tag": "no_residual",
    }


def _apply_a3_a4(
    *,
    raw_blend: np.ndarray,
    preds_i: dict[str, np.ndarray],
    a0_graph_context: dict[str, Any],
    a3_guard: dict[str, Any],
    a4_residual: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    pred = raw_blend.astype(np.float64).copy()
    anchor_mph = float(a0_graph_context.get("last_step_neighbor_speed_mean_mph", np.mean(pred)))
    raw_mean = float(np.mean(pred))
    triggered = False
    fallback_policy = str(a3_guard.get("fallback_policy") or "none")
    threshold = float(a3_guard.get("guard_threshold_mph", 4.0) or 4.0)
    if str(a3_guard.get("guard_policy") or "none") == "deviation":
        triggered = abs(raw_mean - anchor_mph) > threshold
        if triggered and fallback_policy in preds_i:
            pred = preds_i[fallback_policy].astype(np.float64).copy()

    residual_applied = False
    residual_value = 0.0
    if str(a4_residual.get("residual_policy") or "none") == "graph_anchor_residual":
        weight = float(a4_residual.get("residual_weight", 0.0) or 0.0)
        clip = float(a4_residual.get("residual_clip_mph", 1.5) or 1.5)
        residual_value = float(np.clip(anchor_mph - float(np.mean(pred)), -clip, clip) * weight)
        if abs(residual_value) > 1e-8:
            pred = pred + residual_value
            residual_applied = True

    trace = {
        "raw_blend_mean_mph": raw_mean,
        "graph_anchor_mean_mph": anchor_mph,
        "a3_triggered": bool(triggered),
        "a3_fallback_policy": fallback_policy,
        "a3_guard_threshold_mph": threshold,
        "a4_residual_applied": bool(residual_applied),
        "a4_residual_value_mph": residual_value,
        "final_mean_mph": float(np.mean(pred)),
    }
    return pred.astype(np.float32), trace


def _llm_decision(
    snapshot: dict[str, float],
    model: str,
    *,
    a0_graph_context: dict[str, Any],
    a1_candidates: list[dict[str, Any]],
) -> tuple[dict[str, float], str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    tool_list_text = ", ".join(TOOLS)
    dcrnn_expertise = ""
    if "dcrnn_cstd" in TOOLS:
        dcrnn_expertise = (
            "  - dcrnn_cstd: diffusion convolutional RNN (graph+RNN hybrid).\n"
            "                bidirectional diffusion on the road transition matrix;\n"
            "                fixed-graph variant well-suited to clear directional flow.\n"
        )
    agcrn_expertise = ""
    if "agcrn_cstd" in TOOLS:
        agcrn_expertise = (
            "  - agcrn_cstd: adaptive graph + DCGRU temporal cell (graph+RNN hybrid).\n"
            "                learns node-embedding adjacency from data;\n"
            "                adapts when fixed graph topology is suboptimal.\n"
        )
    fallback_options = ["none", "knn_l1_k6", "pgtft_cstd", "graph_wavenet_cstd"]
    if "dcrnn_cstd" in TOOLS:
        fallback_options.append("dcrnn_cstd")
    if "agcrn_cstd" in TOOLS:
        fallback_options.append("agcrn_cstd")
    context = {
        "dataset": "PEMS/METR benchmark",
        "setup": "shadowed speed forecasting",
        "unit": "mph horizon-mean prediction",
        "tool_snapshot": snapshot,
        "a0_graph_similarity": a0_graph_context,
        "a1_candidate_graph_sequences": a1_candidates,
        "tool_prior": {
            "graph_wavenet_cstd": "strongest RMSE/MAPE tool in this PEMS-BAY Cstd probe",
            "pgtft_cstd": "strongest/near-strongest MAE tool in this PEMS-BAY Cstd probe",
            "knn_l1_k6": "classical neighbor persistence anchor",
            "pgtft_c0": "raw-speed PGTFT backstop",
            "gru_c0": "temporal raw-speed GRU backstop",
            "dcrnn_cstd": "optional expanded-registry DCRNN Cstd expert when present",
            "agcrn_cstd": "optional expanded-registry AGCRN Cstd expert (adaptive graph) when present",
        },
    }
    prompt = (
        "/no_think\n"
        "CONTEXT (read-only, current target-window):\n"
        f"{json.dumps(context, sort_keys=True)}\n\n"
        "PERSONA:\n"
        "You are a senior traffic forecasting analyst specializing in spatiotemporal graph models. "
        "Your job is to choose direct blend weights across seven PEMS-trained forecasting tools "
        "based only on the current target-window tool snapshot.\n\n"
        "A0 GRAPH REFERENCE:\n"
        "PEMS-BAY has no SVI/static road-scene descriptors. Use only the provided "
        "a0_graph_similarity field: Gaussian adjacency top-k weights, valid neighbor count, "
        "and last-step observed neighbor speed statistics. Treat it as supporting spatial evidence, "
        "not as a learned RF/teacher prior.\n\n"
        "A1 GRAPH K-SCORE SEQUENCE SELECTION:\n"
        "Choose one or two candidate graph support sequences from a1_candidate_graph_sequences. "
        "A sequence is target sensor -> first-hop graph neighbor -> second-hop support sensor. "
        "Candidates are ranked by a PEMS graph-only K-score: K=0.4*(1-U)+0.3*C+0.3*P. "
        "Prefer high k_score, but do not let A1 override clear tool_snapshot evidence. "
        "This is the PEMS version of the framework's K-score path/sequence selection; "
        "SVI is unavailable and intentionally not used.\n\n"
        "A3 GUARD POLICY:\n"
        "After A2 blending, decide whether to guard against an implausible deviation from "
        "the A0 graph-neighbor anchor. Use guard_policy='deviation' with a threshold in mph. "
        f"fallback_policy may be one of: {', '.join(fallback_options)}. "
        "Use fallback only for large mismatch; otherwise fallback_policy='none'.\n\n"
        "A4 RESIDUAL POLICY:\n"
        "PEMS has no SVI residual model. Usually use residual_policy='none'. Only use "
        "residual_policy='graph_anchor_residual' for a small clipped correction toward the "
        "A0 graph-neighbor anchor when evidence is stable and the blend is mildly biased.\n\n"
        "TOOL EXPERTISE:\n"
        "  - knn_l1_k6: neighbor persistence anchor; stable when local speed is regular.\n"
        "  - pgtft_c0: raw-speed graph transformer backstop.\n"
        "  - pgtft_cstd: graph transformer with time context; strong average-error tool on PEMS-BAY Cstd.\n"
        "  - gru_c0: temporal raw-speed GRU; conservative momentum backstop.\n"
        "  - graph_wavenet_cstd: diffusion WaveNet with time context; strongest graph propagation tool on PEMS-BAY Cstd.\n"
        f"{dcrnn_expertise}"
        f"{agcrn_expertise}\n"
        "GROUNDING STEPS:\n"
        "  STEP 1: Read the current tool_snapshot values.\n"
        "  STEP 2: Read a0_graph_similarity for graph support strength and neighbor availability.\n"
        "  STEP 3: Select A1 graph K-score support sequence(s) from a1_candidate_graph_sequences.\n"
        "  STEP 4: Identify the strongest Cstd graph/time-context signal: graph_wavenet_cstd or pgtft_cstd.\n"
        "  STEP 5: Assign target-specific A2 weights proportional to expected accuracy, not equal by habit.\n"
        "  STEP 6: Choose A3 guard and A4 residual conservatively.\n\n"
        "SNAPSHOT-PAIRED EXAMPLES (do not copy blindly; match the current snapshot):\n"
        "# Example A — graph_wavenet_cstd and pgtft_cstd strongest, C0/GRU lower:\n"
        'GIVEN {"knn_l1_k6":62.0,"pgtft_c0":61.5,"pgtft_cstd":64.8,"gru_c0":60.9,"graph_wavenet_cstd":65.2,"agcrn_cstd":63.0,"dcrnn_cstd":64.0}\n'
        'OUTPUT {"a1_decision":{"selected_sequence_ids":["seq_1"],"evidence_tag":"strong_graph_sequence"},"a2_decision":{"blend_weights":{"knn_l1_k6":0.05,"pgtft_c0":0.05,"pgtft_cstd":0.30,"gru_c0":0.05,"graph_wavenet_cstd":0.35,"agcrn_cstd":0.08,"dcrnn_cstd":0.12},"forecast_tool":"graph_wavenet_cstd","evidence_tag":"graph_temporal_cstd_dominant"},"a3_guard":{"guard_policy":"deviation","guard_threshold_mph":4.0,"fallback_policy":"none","evidence_tag":"guard_not_needed"},"a4_residual":{"residual_policy":"none","residual_weight":0.0,"residual_clip_mph":1.5,"evidence_tag":"no_residual"},"reasoning":"Strong graph sequence; Cstd graph tools dominate (GWN highest)."}\n'
        "# Example B — pgtft_cstd clearly highest, graph_wavenet close second:\n"
        'GIVEN {"knn_l1_k6":66.0,"pgtft_c0":65.0,"pgtft_cstd":68.3,"gru_c0":64.8,"graph_wavenet_cstd":67.4,"agcrn_cstd":70.5,"dcrnn_cstd":68.0}\n'
        'OUTPUT {"a1_decision":{"selected_sequence_ids":["seq_1"],"evidence_tag":"graph_sequence_support"},"a2_decision":{"blend_weights":{"knn_l1_k6":0.06,"pgtft_c0":0.05,"pgtft_cstd":0.20,"gru_c0":0.05,"graph_wavenet_cstd":0.14,"agcrn_cstd":0.38,"dcrnn_cstd":0.12},"forecast_tool":"agcrn_cstd","evidence_tag":"adaptive_graph_dominant"},"a3_guard":{"guard_policy":"deviation","guard_threshold_mph":4.0,"fallback_policy":"none","evidence_tag":"guard_not_needed"},"a4_residual":{"residual_policy":"none","residual_weight":0.0,"residual_clip_mph":1.5,"evidence_tag":"no_residual"},"reasoning":"AGCRN snapshot value dominates; tilt mass to adaptive-graph expert."}\n'
        "# Example C — all tools nearly tied:\n"
        'GIVEN {"knn_l1_k6":64.0,"pgtft_c0":63.9,"pgtft_cstd":64.2,"gru_c0":63.8,"graph_wavenet_cstd":64.1,"agcrn_cstd":64.0,"dcrnn_cstd":64.1}\n'
        'OUTPUT {"a1_decision":{"selected_sequence_ids":["seq_1"],"evidence_tag":"weak_graph_preference"},"a2_decision":{"blend_weights":{"knn_l1_k6":0.14,"pgtft_c0":0.13,"pgtft_cstd":0.16,"gru_c0":0.13,"graph_wavenet_cstd":0.16,"agcrn_cstd":0.14,"dcrnn_cstd":0.14},"forecast_tool":"pgtft_cstd","evidence_tag":"mixed_evidence"},"a3_guard":{"guard_policy":"deviation","guard_threshold_mph":5.0,"fallback_policy":"none","evidence_tag":"mixed_guard"},"a4_residual":{"residual_policy":"none","residual_weight":0.0,"residual_clip_mph":1.5,"evidence_tag":"no_residual"},"reasoning":"Values within 0.4 mph; near-uniform across all 7 tools."}\n'
        "# Example D — anchor snapshot is far from both Cstd tools:\n"
        'GIVEN {"knn_l1_k6":58.5,"pgtft_c0":59.0,"pgtft_cstd":64.5,"gru_c0":59.2,"graph_wavenet_cstd":64.0,"agcrn_cstd":63.2,"dcrnn_cstd":67.8}\n'
        'OUTPUT {"a1_decision":{"selected_sequence_ids":["seq_1","seq_2"],"evidence_tag":"graph_sequence_support"},"a2_decision":{"blend_weights":{"knn_l1_k6":0.04,"pgtft_c0":0.05,"pgtft_cstd":0.20,"gru_c0":0.04,"graph_wavenet_cstd":0.15,"agcrn_cstd":0.10,"dcrnn_cstd":0.42},"forecast_tool":"dcrnn_cstd","evidence_tag":"diffusion_flow_lock"},"a3_guard":{"guard_policy":"deviation","guard_threshold_mph":3.5,"fallback_policy":"dcrnn_cstd","evidence_tag":"anchor_disagreement_guard"},"a4_residual":{"residual_policy":"none","residual_weight":0.0,"residual_clip_mph":1.5,"evidence_tag":"no_residual"},"reasoning":"DCRNN snapshot value clearly dominates; trust diffusion-flow expert."}\n\n'
        "STRICT OUTPUT RULES:\n"
        "  - DO NOT respond with markdown. DO NOT explain outside JSON.\n"
        "  - Return ONLY one compact JSON object.\n"
        "  - Top-level keys MUST include a1_decision, a2_decision, a3_guard, and a4_residual.\n"
        "  - a1_decision.selected_sequence_ids must use sequence_id values from the context.\n"
        f"  - blend_weights MUST contain exactly these tools: {tool_list_text}.\n"
        "  - Weights must be non-negative and sum to 1.\n"
        "  - Anti-echo rule: do NOT copy any example weights unless the current snapshot truly matches that example.\n"
        "  - Equal weights are allowed only when all snapshot values are nearly identical.\n\n"
        "Now output the a1_decision, a2_decision, a3_guard, and a4_residual for the current target-window."
    )
    # Retry LLM up to 3 times if response is unparseable. Each retry calls
    # call_agent fresh — the underlying LLM may produce different output.
    # On final failure, fall back to uniform weights so the run continues.
    obj: dict[str, Any] | None = None
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = call_agent(prompt, model_name=model, enforce_json=True)
            obj = _extract_json(raw)
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_err = e
            print(f"[pems-plan-c] LLM JSON parse fail attempt {attempt+1}/3: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
    if obj is None:
        print(f"[pems-plan-c] LLM parse failed 3x; falling back to uniform weights. "
              f"last error: {last_err}", flush=True)
        uniform_w = {t: 1.0 / len(TOOLS) for t in TOOLS}
        return (uniform_w, TOOLS[0], "llm_parse_fail",
                _fallback_a1_decision(a1_candidates),
                _fallback_a3_guard(),
                _fallback_a4_residual())
    decision = _unwrap_a2_decision(obj)
    weights = _normalize_weights(decision.get("blend_weights", decision.get("weights", {})))
    forecast_tool = str(decision.get("forecast_tool") or max(weights.items(), key=lambda kv: kv[1])[0])
    evidence_tag = str(decision.get("evidence_tag") or "unknown")
    a1_decision = obj.get("a1_decision") if isinstance(obj.get("a1_decision"), dict) else _fallback_a1_decision(a1_candidates)
    a3_guard = obj.get("a3_guard") if isinstance(obj.get("a3_guard"), dict) else _fallback_a3_guard()
    a4_residual = obj.get("a4_residual") if isinstance(obj.get("a4_residual"), dict) else _fallback_a4_residual()
    return weights, forecast_tool, evidence_tag, a1_decision, a3_guard, a4_residual


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["pems_bay", "metr_la"], default="pems_bay")
    ap.add_argument("--eval-cell", choices=["Cstd"], default="Cstd")
    ap.add_argument("--pgtft-variant", choices=list(PGTFT_VARIANT_FLAGS.keys()), default="b")
    ap.add_argument("--n-sensors", type=int, default=66)
    ap.add_argument("--windows-per-sensor", type=int, default=1)
    ap.add_argument("--sensor-offset", type=int, default=0)
    ap.add_argument("--window-offset", type=int, default=0)
    ap.add_argument(
        "--indices-json",
        type=str,
        default="",
        help="pre-computed indices file (e.g., pems_stratified_1000_indices.json). "
             "If set, overrides --n-sensors/--windows-per-sensor/--*-offset.",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--agent-model", type=str, default="qwen3:14b")
    ap.add_argument("--include-dcrnn-cstd", action="store_true",
                    help="(legacy) keep for backwards compat; core7 default already includes dcrnn_cstd")
    ap.add_argument("--include-agcrn-cstd", action="store_true",
                    help="(legacy) keep for backwards compat; core7 default already includes agcrn_cstd")
    ap.add_argument("--core5-only", action="store_true",
                    help="Disable the core7 additions (AGCRN, DCRNN) and revert to the original 5-tool pool.")
    ap.add_argument("--dry-uniform", action="store_true")
    ap.add_argument("--save-pred-npz", action=argparse.BooleanOptionalAction, default=True,
                    help="Save 24-step y_true, Plan C final predictions, and tool predictions next to output-json.")
    ap.add_argument("--output-json", type=str, default=None)
    args = ap.parse_args()

    global TOOLS
    if args.core5_only:
        TOOLS = list(TOOLS_CORE5)
    else:
        # core7 default = original 5 + agcrn_cstd + dcrnn_cstd
        TOOLS = list(TOOLS_CORE7)

    t0 = time.time()
    print(
        f"[pems-plan-c] dataset={args.dataset} eval_cell={args.eval_cell} "
        f"n_sensors={args.n_sensors} windows_per_sensor={args.windows_per_sensor} "
        f"sensor_offset={args.sensor_offset} window_offset={args.window_offset} "
        f"tools={','.join(TOOLS)}",
        flush=True,
    )
    pkg_cstd = _build_pkg(args.dataset, "Cstd", args.batch_size)
    pkg_c0 = _build_pkg(args.dataset, "C0", args.batch_size)
    adj = load_adj(args.dataset)
    sensor_geo = _load_sensor_geo(args.dataset)

    if args.indices_json:
        with open(args.indices_json, "r", encoding="utf-8") as _f:
            indices = [int(x) for x in json.load(_f)]
        print(f"[pems-plan-c] loaded {len(indices)} indices from {args.indices_json}", flush=True)
    else:
        indices = _indices_by_sensor_windows(
            pkg_cstd["test_loader"].dataset,
            n_sensors=args.n_sensors,
            windows_per_sensor=args.windows_per_sensor,
            sensor_offset=args.sensor_offset,
            window_offset=args.window_offset,
        )
    x_cstd, y_true_t, sids = _batch_from_indices(pkg_cstd["test_loader"].dataset, indices)
    x_c0, _y_true_c0, sids_c0 = _batch_from_indices(pkg_c0["test_loader"].dataset, indices)
    if not np.array_equal(sids, sids_c0):
        raise RuntimeError("C0/Cstd selected sensor order mismatch")
    y_true = y_true_t.numpy()
    print(
        f"[pems-plan-c] selected {len(indices)} samples "
        f"({args.n_sensors} sensors x up to {args.windows_per_sensor} windows)",
        flush=True,
    )

    print("[pems-plan-c] loading PEMS/METR-trained tool models", flush=True)
    pgtft_c0 = _load_pgtft(args.dataset, "C0", args.pgtft_variant, pkg_c0, args.device)
    pgtft_cstd = _load_pgtft(args.dataset, "Cstd", args.pgtft_variant, pkg_cstd, args.device)
    gru_c0 = _load_gru(args.dataset, "C0", pkg_c0, args.device)
    gwn_cstd = _load_gwn(args.dataset, "Cstd", pkg_cstd, args.device)
    dcrnn_cstd = (
        _load_standard_baseline("dcrnn", args.dataset, "Cstd", pkg_cstd, args.device)
        if "dcrnn_cstd" in TOOLS
        else None
    )
    agcrn_cstd = (
        _load_standard_baseline("agcrn", args.dataset, "Cstd", pkg_cstd, args.device)
        if "agcrn_cstd" in TOOLS
        else None
    )

    preds = {
        "knn_l1_k6": _knn_l1_k6_raw(x_cstd, pkg_cstd["mean"], pkg_cstd["std"]),
        "pgtft_c0": _predict_raw(pgtft_c0, x_c0, sids, pkg_c0["mean"], pkg_c0["std"], args.device),
        "pgtft_cstd": _predict_raw(pgtft_cstd, x_cstd, sids, pkg_cstd["mean"], pkg_cstd["std"], args.device),
        "gru_c0": _predict_raw(gru_c0, x_c0, sids, pkg_c0["mean"], pkg_c0["std"], args.device),
        "graph_wavenet_cstd": _predict_raw(gwn_cstd, x_cstd, sids, pkg_cstd["mean"], pkg_cstd["std"], args.device),
    }
    if agcrn_cstd is not None:
        preds["agcrn_cstd"] = _predict_raw(
            agcrn_cstd, x_cstd, sids, pkg_cstd["mean"], pkg_cstd["std"], args.device,
        )
    if dcrnn_cstd is not None:
        preds["dcrnn_cstd"] = _predict_raw(
            dcrnn_cstd,
            x_cstd,
            sids,
            pkg_cstd["mean"],
            pkg_cstd["std"],
            args.device,
        )
    tool_metrics = {tool: _metrics(pred, y_true) for tool, pred in preds.items()}
    print("[pems-plan-c] tool metrics over selected samples:", flush=True)
    for tool, m in tool_metrics.items():
        print(f"  {tool:20s} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} MAPE={m['mape']:.2f}%", flush=True)

    rows = []
    blend_preds = []
    for i, sid in enumerate(sids):
        snapshot = {tool: float(np.mean(preds[tool][i])) for tool in TOOLS}
        a0_graph_context = _a0_graph_similarity_context(
            adj=adj,
            pkg=pkg_cstd,
            sensor_id=int(sid),
            x_cstd_one=x_cstd[i],
            sensor_geo=sensor_geo,
        )
        a1_candidates = _a1_graph_sequence_candidates(
            adj=adj,
            pkg=pkg_cstd,
            sensor_id=int(sid),
            sensor_geo=sensor_geo,
        )
        if args.dry_uniform:
            weights = {tool: 1.0 / len(TOOLS) for tool in TOOLS}
            forecast_tool = "uniform"
            evidence_tag = "dry_uniform"
            a1_decision = _fallback_a1_decision(a1_candidates)
            a3_guard = _fallback_a3_guard()
            a4_residual = _fallback_a4_residual()
        else:
            weights, forecast_tool, evidence_tag, a1_decision, a3_guard, a4_residual = _llm_decision(
                snapshot,
                args.agent_model,
                a0_graph_context=a0_graph_context,
                a1_candidates=a1_candidates,
            )
        blend = np.zeros_like(y_true[i], dtype=np.float64)
        for tool, w in weights.items():
            blend += float(w) * preds[tool][i]
        preds_i = {tool: preds[tool][i] for tool in TOOLS}
        final_pred, guard_residual_trace = _apply_a3_a4(
            raw_blend=blend,
            preds_i=preds_i,
            a0_graph_context=a0_graph_context,
            a3_guard=a3_guard,
            a4_residual=a4_residual,
        )
        blend_preds.append(final_pred)
        rows.append({
            "sensor_id": int(sid),
            "external_sensor_id": sensor_geo.get("loc_by_index", {}).get(int(sid), {}).get("external_sensor_id"),
            "tool_snapshot_mean_mph": snapshot,
            "a0_graph_similarity": a0_graph_context,
            "a1_candidate_graph_sequences": a1_candidates,
            "a1_decision": a1_decision,
            "blend_weights": weights,
            "forecast_tool": forecast_tool,
            "evidence_tag": evidence_tag,
            "a3_guard": a3_guard,
            "a4_residual": a4_residual,
            "guard_residual_trace": guard_residual_trace,
            "target_true_mean_mph": float(np.mean(y_true[i])),
            "blend_pred_mean_mph": float(np.mean(blend)),
            "final_pred_mean_mph": float(np.mean(final_pred)),
        })
        if (i + 1) % 5 == 0 or (i + 1) == len(sids):
            print(f"[pems-plan-c] composed {i + 1}/{len(sids)}", flush=True)

    y_blend = np.stack(blend_preds, axis=0)
    plan_c_metrics = _metrics(y_blend, y_true)
    horizon_metrics = per_horizon_metrics(y_blend, y_true, horizons=(3, 6, 12, 24))
    bucket_mape = _bucket_mape_gt5(y_blend, y_true)
    mean_weights = {tool: float(np.mean([r["blend_weights"][tool] for r in rows])) for tool in TOOLS}

    payload = {
        "protocol": "pems_metr_shadowed_plan_c_direct_sensor_window_sample",
        "dataset": args.dataset,
        "eval_cell": args.eval_cell,
        "n_samples": len(indices),
        "n_sensors_requested": args.n_sensors,
        "windows_per_sensor_requested": args.windows_per_sensor,
        "sensor_offset": args.sensor_offset,
        "window_offset": args.window_offset,
        "sensor_ids": [int(x) for x in sids],
        "external_sensor_ids": [
            sensor_geo.get("loc_by_index", {}).get(int(x), {}).get("external_sensor_id")
            for x in sids
        ],
        "spatial_layer": {
            "source": sensor_geo.get("source"),
            "road_metadata_required": False,
            "svi_required": False,
            "description": "PEMS/METR adapter uses benchmark sensor ids, coordinates, Gaussian adjacency, and observed neighbor speeds.",
        },
        "core_tools": TOOLS,
        "core5": TOOLS,
        "include_dcrnn_cstd": "dcrnn_cstd" in TOOLS,
        "include_agcrn_cstd": "agcrn_cstd" in TOOLS,
        "core_variant": ("core5" if args.core5_only else "core7"),
        "tools_used": list(TOOLS),
        "pgtft_variant": args.pgtft_variant,
        "agent_model": args.agent_model,
        "dry_uniform": bool(args.dry_uniform),
        "tool_metrics_over_selected_samples": tool_metrics,
        "plan_c_direct_metrics_over_selected_samples": plan_c_metrics,
        "plan_c_per_horizon": horizon_metrics,
        "plan_c_bucket_mape_gt5mph": bucket_mape,
        "mean_blend_weights": mean_weights,
        "rows": rows,
        "elapsed_sec": time.time() - t0,
    }
    out = Path(args.output_json) if args.output_json else (
        PROJECT_2 / "results_interp_a" / f"{args.dataset}_shadowed_plan_c_{args.eval_cell}_firstwin.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_pred_npz:
        pred_npz = out.with_name(f"{out.stem}_preds.npz")
        np.savez_compressed(
            pred_npz,
            indices=np.asarray(indices, dtype=np.int64),
            sensor_ids=np.asarray(sids, dtype=np.int64),
            y_true=y_true.astype(np.float32),
            plan_c_final=y_blend.astype(np.float32),
            **{f"tool_{tool}": pred.astype(np.float32) for tool, pred in preds.items()},
        )
        print(f"[pems-plan-c] wrote prediction arrays {pred_npz}", flush=True)
    print(
        f"[pems-plan-c] Plan C RMSE={plan_c_metrics['rmse']:.4f} "
        f"MAE={plan_c_metrics['mae']:.4f} MAPE={plan_c_metrics['mape']:.2f}% "
        f"mean_weights={mean_weights}",
        flush=True,
    )
    print(f"[pems-plan-c] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
