"""Leakage-free M8--M12 utilities.

All functions consume the immutable M0--M2 window manifests.  The operability
target is deliberately a causal *heave-only proxy*: the repository contains no
numeric propeller geometry from which a physical stern trajectory can be made.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from shipmotion.benchmark import (INPUT_STEPS, OUTPUT_STEPS, SEED, _latency,
    _normalize_x, load_windows, split_arrays)
from shipmotion.data.pipeline import MOTION_COLUMNS

FEATURE_SETS = {
    "F0_motion_only": MOTION_COLUMNS,
    "F1_wave_motion": ["wave_m", *MOTION_COLUMNS],
    "F2_wave_speed_motion": ["wave_m", "ship_speed_kn", *MOTION_COLUMNS],
}


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for part in iter(lambda: fh.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def verify_frozen_inputs(root: Path, output_root: Path | None = None) -> dict:
    """Verify M0--M2 inputs against the primary artifact stored in ``output_root``.

    ``output_root`` is configurable so a Kaggle run can restore its prior
    artifacts from a mounted dataset instead of writing into a freshly cloned
    repository each time.
    """
    output_root = output_root or root / "outputs"
    primary = output_root / "primary_benchmark_m37" / "manifest_hashes.json"
    expected = json.loads(primary.read_text())
    base = root / "data" / "splits"; got = {}
    for key in expected:
        dataset, name = key.split("_", 1)
        suffix = ".json" if name.endswith(".json") else ".parquet"
        got[key] = file_hash(base / f"{dataset}_{name}{'' if suffix == '.json' else suffix}")
    bad = {k: {"expected": expected[k], "actual": got[k]} for k in expected if expected[k] != got[k]}
    if bad: raise RuntimeError(f"immutable primary inputs changed: {bad}")
    return {"status": "verified", "hashes": got, "primary_artifact": str(primary)}


def feature_data(root: Path, dataset: str, feature_set: str) -> dict:
    data = load_windows(root, dataset); cols = FEATURE_SETS[feature_set]
    if feature_set == "F2_wave_speed_motion" and data["interp"].ship_speed_kn.isna().all():
        raise ValueError(f"{dataset} has no verified vessel-speed metadata")
    positions = [["wave_m", *MOTION_COLUMNS].index(c) for c in cols if c != "ship_speed_kn"]
    x = data["x"][:, :, positions]
    if "ship_speed_kn" in cols:
        by = []
        for meta in data["metadata"]:
            block = data["interp"][(data["interp"].run_id == meta["run_id"]) & (data["interp"].split == meta["split"])]
            by.append(float(block.ship_speed_kn.iloc[0]))
        speed = np.asarray(by, dtype=np.float32)[:, None, None]
        # physical speed is constant within each confirmed simulation run
        x = np.concatenate([x[:, :, :1], np.repeat(speed, INPUT_STEPS, 1), x[:, :, 1:]], axis=2)
    train_indices, _, _ = split_arrays(data, "train")
    mean = x[train_indices].mean(axis=(0, 1)); scale = x[train_indices].std(axis=(0, 1)); scale[scale == 0] = 1
    data.update(x=x, feature_names=cols, feature_mean=mean.astype("float32"), feature_scale=scale.astype("float32"))
    return data


def motion_metrics(y: np.ndarray, p: np.ndarray, train_std: np.ndarray) -> dict:
    rows = []
    for d, name in enumerate(MOTION_COLUMNS):
        err = p[:, :, d] - y[:, :, d]; ss = np.sum((y[:, :, d] - y[:, :, d].mean()) ** 2)
        rows.append({"dof": name, "rmse": float(np.sqrt(np.mean(err ** 2))), "mae": float(np.mean(abs(err))),
                     "r2": float(1 - np.sum(err ** 2) / ss) if ss else None,
                     "nrmse": float(np.sqrt(np.mean(err ** 2)) / train_std[d])})
    return {"per_dof": rows, "macro_nrmse": float(np.mean([r["nrmse"] for r in rows]))}


def fit_classical(name: str, xtr, ytr, xv, yv, xt, yt, target_std):
    est = Ridge(alpha=1.0) if name == "Ridge" else RandomForestRegressor(
        n_estimators=50, max_depth=15, min_samples_split=5, min_samples_leaf=2, n_jobs=1, random_state=SEED)
    start = time.perf_counter(); est.fit(xtr.reshape(len(xtr), -1), ytr.reshape(len(ytr), -1)); fit = time.perf_counter() - start
    def predict(x): return est.predict(x.reshape(len(x), -1)).reshape(-1, OUTPUT_STEPS, 6)
    p50, p95 = _latency(type("M", (), {"predict": staticmethod(predict)})(), xt[:1])
    return est, predict(xt), {"fit_time_s": fit, "validation": motion_metrics(yv, predict(xv), target_std),
                               "test": motion_metrics(yt, predict(xt), target_std),
                               "cpu_batch1_p50_ms": p50, "cpu_batch1_p95_ms": p95}


def causal_score(values: np.ndarray, depth: float, history: np.ndarray | None = None) -> np.ndarray:
    """Trailing 5-sample half-range / depth; no centred/future samples."""
    z = np.concatenate([history[-4:], values]) if history is not None else values
    out = np.empty(len(values), dtype=np.float32)
    for i in range(len(values)):
        tail = z[max(0, len(z) - len(values) + i - 4):len(z) - len(values) + i + 1]
        out[i] = (tail.max() - tail.min()) / 2.0 / depth
    return out


class MultiTaskTransformer(torch.nn.Module):
    def __init__(self, n_features: int, score: bool = True, flag: bool = True):
        super().__init__(); self.score_enabled, self.flag_enabled = score, flag
        self.proj = torch.nn.Linear(n_features, 128)
        layer = torch.nn.TransformerEncoderLayer(128, 8, 512, 0.1, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = torch.nn.TransformerEncoder(layer, 4)
        pos = torch.arange(INPUT_STEPS).unsqueeze(1); div = torch.exp(torch.arange(0, 128, 2) * (-np.log(10000.) / 128))
        p = torch.zeros(INPUT_STEPS, 128); p[:, 0::2], p[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div); self.register_buffer("pos", p[None])
        self.motion = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(), torch.nn.Linear(256, 300))
        self.score = torch.nn.Linear(128, 50) if score else None; self.flag = torch.nn.Linear(128, 50) if flag else None
    def forward(self, x):
        z = self.encoder(self.proj(x) + self.pos).mean(1)
        return self.motion(z).reshape(-1, 50, 6), (self.score(z) if self.score else None), (self.flag(z) if self.flag else None)


def classification_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    """Classification report for a fixed 0.5 validation-independent threshold."""
    y = np.asarray(y, dtype=int).reshape(-1); prob = np.asarray(prob, dtype=float).reshape(-1)
    classes = sorted(np.unique(y).tolist())
    prevalence = float(y.mean()) if len(y) else 0.0
    if len(classes) < 2:
        return {"status": "N/A", "reason": "test split contains one event class", "classes": classes,
                "positive_support": int(y.sum()), "negative_support": int((1-y).sum()), "prevalence": prevalence}
    pred = (prob >= .5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.; recall = tp / (tp + fn) if tp + fn else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return {"status": "ok", "precision": precision, "recall": recall, "f1": f1,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "pr_auc": float(average_precision_score(y, prob)), "roc_auc": float(roc_auc_score(y, prob)),
            "positive_support": int(y.sum()), "negative_support": int((1-y).sum()), "prevalence": prevalence,
            "confusion_matrix": [[tn, fp], [fn, tp]]}
