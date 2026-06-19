"""PEMS-BAY / METR-LA dataloader — shadowed prediction setup.

본 mar 77 연구 narrative 와 정렬:
- target sensor 의 self-history (speed 채널) 는 사용 금지 (0 mask)
- 입력: target slot (speed=0) + top-N neighbor slots (Gaussian adj top weight)
- time embeddings (tid/diw) 은 target slot 도 그대로 (시간 = self-history 아님)
- 출력: target sensor 의 미래 24 step (h=120min)

Spatial split (mar 77 비율 24:6:8 그대로):
- TRAIN_V (63.16%): 학습 target pool (PEMS 205 / METR 131)
- VAL_V   (15.79%): 검증 target pool (PEMS 51  / METR 33)
- TEST_V  (21.05%): unseen target pool (PEMS 69  / METR 43)
- neighbor pool 은 split 무관 — 전체 sensor 가 항상 observed neighbor 후보

Train: TRAIN_V_IDS 만 target, leave-one-out (per-sample 1 target sensor mask)
Eval:  random K%-of-N 동시 mask, mask 된 sensor 중 target_pool 에 속하는 것만 평가
       (mask 된 sensor 가 다른 sensor 의 top-N neighbor 면 그 neighbor signal 도 0)
       val:  target_pool = VAL_V_IDS
       test: target_pool = 전체 (mask 안에서 seen/unseen 자동 섞임)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

_DEFAULT_DATA_ROOT = Path(
    r"c:\Users\uaer\OneDrive\1. 왔다갔다\4) obsidian_claude\project_1\input_storage\benchmark_data"
)
_LOCAL_DATA_ROOT = Path(__file__).resolve().parents[1] / "project_1" / "input_storage" / "benchmark_data"
DATA_ROOT = _DEFAULT_DATA_ROOT if _DEFAULT_DATA_ROOT.exists() else _LOCAL_DATA_ROOT

DATASETS = {
    "metr_la": {
        "h5": DATA_ROOT / "metr_la" / "metr-la.h5",
        "adj": DATA_ROOT / "metr_la" / "adj_mx.pkl",
        "num_nodes": 207,
    },
    "pems_bay": {
        "h5": DATA_ROOT / "pems_bay" / "pems-bay.h5",
        "adj": DATA_ROOT / "pems_bay" / "adj_mx_bay.pkl",
        "num_nodes": 325,
    },
}


# ---------------------------------------------------------------------------
# Raw load
# ---------------------------------------------------------------------------
def load_raw(dataset: str) -> pd.DataFrame:
    try:
        df = pd.read_hdf(DATASETS[dataset]["h5"])
    except ImportError as exc:
        if "pytables" not in str(exc).lower() and "tables" not in str(exc).lower():
            raise
        import h5py

        with h5py.File(DATASETS[dataset]["h5"], "r") as f:
            # Files are pandas HDFStore fixed-format frames. PEMS-BAY uses
            # key "speed"; METR-LA commonly uses the default key "df".
            key = "speed" if "speed" in f else "df"
            g = f[key]
            values = g["block0_values"][:]
            columns = g["axis0"][:]
            index = pd.to_datetime(g["axis1"][:])
        df = pd.DataFrame(values, index=index, columns=columns)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df


def load_adj(dataset: str) -> np.ndarray:
    """Load Gaussian thresholded adj_mx (N, N) — DCRNN paper convention."""
    with open(DATASETS[dataset]["adj"], "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if isinstance(item, np.ndarray) and item.ndim == 2:
                return item.astype(np.float32)
        raise ValueError(f"Could not find adj_mx in {obj!r}")
    return np.asarray(obj, dtype=np.float32)


def compute_top_n_neighbors(
    adj: np.ndarray, top_n: int
) -> tuple[np.ndarray, np.ndarray]:
    """For each sensor, pick top-N weighted neighbors (excluding self).

    DCRNN Gaussian adj is sparse: METR-LA 51% / PEMS-BAY 55% of sensors have
    fewer than 8 nonzero neighbors. Padding slots are filled with zero-weight
    indices (any valid sensor id, just to keep tensor shape), and the
    `neighbor_valid` mask marks them so the Dataset can zero out their speed
    channel (PG family padding pattern — identical to target self-mask).

    Returns
    -------
    neighbor_idx   : (N, top_n) int — neighbor indices, sorted by weight desc.
    neighbor_valid : (N, top_n) bool — True iff adj weight > 0 (real neighbor).
    """
    N = adj.shape[0]
    out = np.zeros((N, top_n), dtype=np.int64)
    valid = np.zeros((N, top_n), dtype=bool)
    for i in range(N):
        w = adj[i].copy()
        w[i] = 0.0  # exclude self
        order = np.argsort(-w, kind="stable")
        out[i] = order[:top_n]
        valid[i] = w[out[i]] > 0.0
    return out, valid


# ---------------------------------------------------------------------------
# Channel construction
# ---------------------------------------------------------------------------
def build_channels(
    df: pd.DataFrame, cell: Literal["C0", "Cstd"]
) -> np.ndarray:
    """Return (T, N, C). C=1 for C0, C=3 for Cstd."""
    speed = df.values.astype(np.float32)
    T, N = speed.shape
    if cell == "C0":
        return speed[:, :, None]
    if cell == "Cstd":
        idx = df.index
        sec_in_day = (idx.hour * 3600 + idx.minute * 60 + idx.second).values
        tid = (sec_in_day / 86400.0).astype(np.float32)
        diw = (idx.dayofweek.values / 7.0).astype(np.float32)
        tid_b = np.broadcast_to(tid[:, None], (T, N)).copy()
        diw_b = np.broadcast_to(diw[:, None], (T, N)).copy()
        return np.stack([speed, tid_b, diw_b], axis=-1)
    raise ValueError(f"Unknown cell: {cell}")


def chronological_split(T: int, ratios=(0.7, 0.1, 0.2)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    n_train = int(T * ratios[0])
    n_val = int(T * ratios[1])
    return (
        slice(0, n_train),
        slice(n_train, n_train + n_val),
        slice(n_train + n_val, T),
    )


def sensor_pool_split(
    N: int,
    *,
    train_ratio: float = 24.0 / 38.0,
    val_ratio: float = 6.0 / 38.0,
    seed: int = 2026,
    eligible_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic spatial split into TRAIN / VAL / TEST target pools.

    mar 77 비율 (24:6:8 = 63.16% / 15.79% / 21.05%) 기본값.

    If `eligible_ids` is given, the split only assigns those sensors to one of
    the three pools (sensors outside `eligible_ids` get pool_tag = -1, meaning
    "excluded from any target role" — e.g. sensors with zero valid neighbors
    cannot be predicted in shadowed setup so they never appear as target).

    Returns
    -------
    train_ids : (N_train,) sorted sensor indices for training target pool
    val_ids   : (N_val,)   sorted sensor indices for validation target pool
    test_ids  : (N_test,)  sorted sensor indices for test target pool
    pool_tag  : (N,) int — 0=train, 1=val, 2=test, -1=excluded
    """
    if eligible_ids is None:
        eligible = np.arange(N, dtype=np.int64)
    else:
        eligible = np.asarray(sorted(set(int(s) for s in eligible_ids)), dtype=np.int64)
    M = len(eligible)
    if M < 3:
        raise ValueError(f"Too few eligible sensors for split: {M}")

    n_train = int(round(M * train_ratio))
    n_val = int(round(M * val_ratio))
    n_test = M - n_train - n_val
    if n_test <= 0 or n_train <= 0 or n_val <= 0:
        raise ValueError(
            f"Invalid split for eligible M={M}: train={n_train}, val={n_val}, test={n_test}"
        )

    rng = np.random.default_rng(seed)
    perm = eligible[rng.permutation(M)]
    train_ids = np.sort(perm[:n_train])
    val_ids = np.sort(perm[n_train : n_train + n_val])
    test_ids = np.sort(perm[n_train + n_val :])

    pool_tag = np.full(N, -1, dtype=np.int64)
    pool_tag[train_ids] = 0
    pool_tag[val_ids] = 1
    pool_tag[test_ids] = 2
    return train_ids, val_ids, test_ids, pool_tag


def zscore_fit_speed(data_train: np.ndarray) -> tuple[float, float]:
    speed = data_train[..., 0]
    mean = float(np.nanmean(speed))
    std = float(np.nanstd(speed))
    if std < 1e-6:
        std = 1.0
    return mean, std


def apply_zscore_speed(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    out = data.copy()
    out[..., 0] = (out[..., 0] - mean) / std
    return out


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class ShadowedTrainDataset(Dataset):
    """Leave-one-out training samples on TRAIN_V_IDS target pool.

    Each sample: (target_sensor_i, t_start) — sensor i is the masked target.
    target_pool restricts which sensors are sampled as target (mar 77 패턴:
    TRAIN_V_IDS sensors only). Other sensors (VAL/TEST_V) remain in the neighbor
    pool and are always observed.

    Window: (T_in, 1+top_n, C)  slot 0 = target, slots 1..top_n = neighbors
    target slot speed channel = 0 (masked); time embeddings preserved.
    Output: (T_out,) target sensor speed in normalized units.
    Also returns raw target speed (T_out,) for metric in original units.
    """

    def __init__(
        self,
        data_norm: np.ndarray,
        raw_speed: np.ndarray,
        neighbor_idx: np.ndarray,
        neighbor_valid: np.ndarray,
        input_len: int,
        output_len: int,
        stride: int,
        target_pool: np.ndarray | None = None,
    ):
        self.data = data_norm  # (T, N, C)
        self.raw = raw_speed   # (T, N) original units
        self.nbr = neighbor_idx  # (N, top_n)
        self.nbr_valid = neighbor_valid.astype(bool)  # (N, top_n)
        self.in_len = input_len
        self.out_len = output_len
        self.stride = stride
        T, N, _ = data_norm.shape
        # valid window start positions
        max_start = T - input_len - output_len
        if max_start < 0:
            raise ValueError(f"Insufficient T={T} for in+out={input_len + output_len}")
        self.t_starts = np.arange(0, max_start + 1, stride)
        self.N = N
        self.top_n = neighbor_idx.shape[1]
        # target pool restriction (TRAIN_V_IDS only)
        if target_pool is None:
            self.target_pool = np.arange(N, dtype=np.int64)
        else:
            self.target_pool = np.asarray(target_pool, dtype=np.int64)
        self.n_targets = len(self.target_pool)
        self.length = len(self.t_starts) * self.n_targets

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int):
        t_idx = idx // self.n_targets
        pool_pos = idx % self.n_targets
        sensor_i = int(self.target_pool[pool_pos])
        t_start = int(self.t_starts[t_idx])

        nbr_ids = self.nbr[sensor_i]  # (top_n,)
        nbr_valid_i = self.nbr_valid[sensor_i]  # (top_n,) bool
        # gather: slot 0 = target, slots 1.. = neighbors
        slot_ids = np.concatenate(([sensor_i], nbr_ids))  # (1+top_n,)

        # x_window: (T_in, 1+top_n, C)
        x_window = self.data[t_start : t_start + self.in_len, slot_ids, :].copy()
        # mask target slot speed channel
        x_window[:, 0, 0] = 0.0
        # mask invalid (zero-weight) neighbor slots — PG padding pattern
        for j in range(self.top_n):
            if not nbr_valid_i[j]:
                x_window[:, j + 1, 0] = 0.0

        # target output: (T_out,) normalized speed of sensor_i
        y_norm = self.data[t_start + self.in_len : t_start + self.in_len + self.out_len, sensor_i, 0].copy()
        # target raw: (T_out,) original units
        y_raw = self.raw[t_start + self.in_len : t_start + self.in_len + self.out_len, sensor_i].copy()

        return (
            torch.from_numpy(x_window).float(),
            torch.from_numpy(y_norm).float(),
            torch.from_numpy(y_raw).float(),
            sensor_i,
        )


class ShadowedEvalDataset(Dataset):
    """Random K-of-N masked sensors per window for eval, with target_pool filter.

    Each window (t_start) samples a random subset of size K=int(mask_ratio * N)
    masked sensors (deterministic seed) from the entire sensor pool. Mask
    propagation — if a neighbor sensor is also in the mask set, its speed is 0.

    Only masked sensors that ALSO belong to target_pool are emitted as samples:
    - val:  target_pool = VAL_V_IDS  (model selection on unseen-val)
    - test: target_pool = None (=전체) — sample 별 sensor_id 의 pool_tag 로
            seen/unseen 분리 metric 산출.

    Sample = (target_sensor, t_start, mask_set). Length = sum over windows of
    |mask_set ∩ target_pool|.
    """

    def __init__(
        self,
        data_norm: np.ndarray,
        raw_speed: np.ndarray,
        neighbor_idx: np.ndarray,
        neighbor_valid: np.ndarray,
        input_len: int,
        output_len: int,
        stride: int,
        mask_ratio: float,
        seed: int,
        target_pool: np.ndarray | None = None,
    ):
        self.data = data_norm
        self.raw = raw_speed
        self.nbr = neighbor_idx
        self.nbr_valid = neighbor_valid.astype(bool)
        self.in_len = input_len
        self.out_len = output_len
        T, N, _ = data_norm.shape
        max_start = T - input_len - output_len
        if max_start < 0:
            raise ValueError("Insufficient T for window.")
        self.t_starts = np.arange(0, max_start + 1, stride)
        self.N = N
        self.top_n = neighbor_idx.shape[1]
        self.mask_ratio = mask_ratio
        self.K = max(1, int(round(mask_ratio * N)))

        rng = np.random.default_rng(seed)
        # For each window, pre-sample mask set (sorted for deterministic order).
        self.mask_sets = np.stack(
            [np.sort(rng.choice(N, size=self.K, replace=False))
             for _ in range(len(self.t_starts))],
            axis=0,
        )  # (num_windows, K)
        # Pre-compute frozenset per window for O(1) neighbor-in-mask lookup
        # (avoids rebuilding the set on every __getitem__ call).
        self.mask_sets_lookup = [
            frozenset(int(s) for s in self.mask_sets[wi])
            for wi in range(len(self.t_starts))
        ]

        # target_pool filter — only masked sensors inside the pool become samples
        if target_pool is None:
            target_set: set[int] | None = None
        else:
            target_set = set(int(s) for s in np.asarray(target_pool, dtype=np.int64))

        flat = []
        for wi in range(len(self.t_starts)):
            ms = self.mask_sets[wi]
            for ki in range(self.K):
                if target_set is None or int(ms[ki]) in target_set:
                    flat.append((wi, ki))
        if len(flat) == 0:
            raise ValueError(
                "target_pool ∩ mask_set is empty across all windows; "
                "increase mask_ratio or check target_pool indices."
            )
        self.flat = np.array(flat, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.flat)

    def __getitem__(self, idx: int):
        wi, ki = self.flat[idx]
        t_start = int(self.t_starts[wi])
        sensor_i = int(self.mask_sets[wi, ki])

        nbr_ids = self.nbr[sensor_i]
        nbr_valid_i = self.nbr_valid[sensor_i]  # (top_n,) bool
        slot_ids = np.concatenate(([sensor_i], nbr_ids))

        x_window = self.data[t_start : t_start + self.in_len, slot_ids, :].copy()
        x_window[:, 0, 0] = 0.0  # target speed mask

        # Mask invalid neighbor slots (PG padding pattern) AND propagate eval
        # mask to neighbors that are also in mask_set.
        mask_lookup = self.mask_sets_lookup[wi]
        for slot_j in range(self.top_n):
            if (not nbr_valid_i[slot_j]) or (int(nbr_ids[slot_j]) in mask_lookup):
                x_window[:, slot_j + 1, 0] = 0.0

        y_norm = self.data[t_start + self.in_len : t_start + self.in_len + self.out_len, sensor_i, 0].copy()
        y_raw = self.raw[t_start + self.in_len : t_start + self.in_len + self.out_len, sensor_i].copy()

        return (
            torch.from_numpy(x_window).float(),
            torch.from_numpy(y_norm).float(),
            torch.from_numpy(y_raw).float(),
            sensor_i,
        )


# ---------------------------------------------------------------------------
# One-shot builder
# ---------------------------------------------------------------------------
def build_loaders_shadowed(
    dataset: str,
    cell: Literal["C0", "Cstd"],
    *,
    input_len: int = 12,
    output_len: int = 24,
    top_n: int = 8,
    train_stride: int = 12,
    val_stride: int = 12,
    test_stride: int = 12,
    mask_ratio_eval: float = 0.40,
    eval_seed: int = 2026,
    split_seed: int = 2026,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict:
    """Build train/val/test loaders for shadowed prediction setup with
    spatial generalization split (TRAIN_V / VAL_V / TEST_V).

    Train target pool = TRAIN_V_IDS only (mar 77 pattern).
    Val   target pool = VAL_V_IDS only (model selection).
    Test  target pool = entire sensor set; sensor_id pool_tag enables
                        seen/unseen metric split externally.
    """
    df = load_raw(dataset)
    raw_speed = df.values.astype(np.float32)
    data = build_channels(df, cell)
    T, N, C = data.shape
    tr, vl, te = chronological_split(T)

    train_raw = data[tr]
    mean, std = zscore_fit_speed(train_raw)
    data_norm = apply_zscore_speed(data, mean, std)

    adj = load_adj(dataset)
    nbr_idx, nbr_valid = compute_top_n_neighbors(adj, top_n)  # (N, top_n) x2

    # Per-sensor neighbor-x-neighbor adjacency submatrix for graph models.
    # adj_block[i, j, k] = adj[ nbr_idx[i,j], nbr_idx[i,k] ], with padding
    # rows/cols zeroed via nbr_valid mask.
    adj_block = np.zeros((N, top_n, top_n), dtype=np.float32)
    for i in range(N):
        nbr_i = nbr_idx[i]
        sub = adj[nbr_i, :][:, nbr_i]
        mask = nbr_valid[i].astype(np.float32)
        adj_block[i] = sub * mask[None, :] * mask[:, None]

    # Exclude sensors with zero valid neighbors from any target pool - they
    # have no usable input under the shadowed setup.
    eligible_ids = np.where(nbr_valid.any(axis=1))[0]
    n_excluded = N - len(eligible_ids)

    train_ids, val_ids, test_ids, pool_tag = sensor_pool_split(
        N, seed=split_seed, eligible_ids=eligible_ids,
    )

    train_ds = ShadowedTrainDataset(
        data_norm[tr], raw_speed[tr], nbr_idx, nbr_valid,
        input_len, output_len, train_stride,
        target_pool=train_ids,
    )
    val_ds = ShadowedEvalDataset(
        data_norm[vl], raw_speed[vl], nbr_idx, nbr_valid,
        input_len, output_len, val_stride,
        mask_ratio=mask_ratio_eval, seed=eval_seed + 1,
        target_pool=val_ids,
    )
    # Test target pool = entire eligible set (seen/unseen mixed for metric split).
    test_ds = ShadowedEvalDataset(
        data_norm[te], raw_speed[te], nbr_idx, nbr_valid,
        input_len, output_len, test_stride,
        mask_ratio=mask_ratio_eval, seed=eval_seed + 2,
        target_pool=eligible_ids,
    )

    return {
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val_loader":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test_loader":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "mean": mean,
        "std": std,
        "num_nodes": N,
        "in_channels": C,
        "top_n": top_n,
        "neighbor_idx": nbr_idx,
        "neighbor_valid": nbr_valid,
        "adj_block": adj_block,
        "mask_ratio_eval": mask_ratio_eval,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "eligible_ids": eligible_ids,
        "n_excluded": int(n_excluded),
        "pool_tag": pool_tag,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
    }


def inverse_zscore_speed(x, mean: float, std: float):
    return x * std + mean
