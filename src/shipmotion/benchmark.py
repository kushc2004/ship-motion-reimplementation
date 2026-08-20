"""Leakage-free primary motion benchmark for the immutable M0-M2 artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from shipmotion.data.pipeline import MOTION_COLUMNS

FEATURES = ["wave_m", *MOTION_COLUMNS]  # F1: common to both datasets
INPUT_STEPS = 100
OUTPUT_STEPS = 50
STRIDE = 10
SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_windows(root: Path, dataset: str) -> dict:
    base = root / "data" / "splits"
    interp = pd.read_parquet(base / f"{dataset}_interpolated.parquet")
    windows = pd.read_parquet(base / f"{dataset}_windows.parquet")
    scaler = json.loads((base / f"{dataset}_train_scaler.json").read_text())
    x_values, y_values, metadata = [], [], []
    for rec in windows.sort_values(["split", "run_id", "input_start_time_s"]).to_dict("records"):
        block = interp[(interp.run_id == rec["run_id"]) & (interp.split == rec["split"])].sort_values("time_s")
        times = block.time_s.to_numpy()
        i = int(np.searchsorted(times, rec["input_start_time_s"], side="left"))
        j = i + INPUT_STEPS
        k = j + OUTPUT_STEPS
        if k > len(block):
            raise AssertionError(f"window cannot be reconstructed: {rec['window_id']}")
        x_values.append(block.iloc[i:j][FEATURES].to_numpy(dtype=np.float32))
        y_values.append(block.iloc[j:k][MOTION_COLUMNS].to_numpy(dtype=np.float32))
        target_times = block.iloc[j:k].time_s.to_numpy(dtype=np.float32)
        metadata.append({
            "dataset": dataset, "window_id": rec["window_id"], "run_id": rec["run_id"],
            "split": rec["split"], "forecast_origin_time_s": float(block.iloc[j - 1].time_s),
            "target_times": target_times,
        })
    x = np.stack(x_values); y = np.stack(y_values)
    train = interp[interp.split == "train"]
    feature_mean = np.array([scaler["mean"][f] for f in FEATURES], dtype=np.float32)
    feature_scale = np.array([scaler["scale"][f] for f in FEATURES], dtype=np.float32)
    target_mean = train[MOTION_COLUMNS].mean().to_numpy(dtype=np.float32)
    target_scale = train[MOTION_COLUMNS].std(ddof=0).replace(0, 1.0).to_numpy(dtype=np.float32)
    return {"x": x, "y": y, "metadata": metadata, "feature_mean": feature_mean,
            "feature_scale": feature_scale, "target_mean": target_mean,
            "target_scale": target_scale, "interp": interp, "windows": windows}


def split_arrays(data: dict, split: str):
    indices = np.array([i for i, m in enumerate(data["metadata"]) if m["split"] == split], dtype=int)
    return indices, data["x"][indices], data["y"][indices]


def _normalize_x(data: dict, x: np.ndarray) -> np.ndarray:
    return (x - data["feature_mean"][None, None, :]) / data["feature_scale"][None, None, :]


def _denormalize_y(data: dict, y: np.ndarray) -> np.ndarray:
    return y * data["target_scale"][None, None, :] + data["target_mean"][None, None, :]


class Persistence:
    name = "Persistence"
    def fit(self, x, y): return self
    def predict(self, x): return np.repeat(x[:, -1:, 1:], OUTPUT_STEPS, axis=1)


class LinearTrend:
    name = "Linear Trend"
    def __init__(self, trend_steps=20): self.trend_steps = trend_steps
    def fit(self, x, y): return self
    def predict(self, x):
        tail = x[:, -self.trend_steps:, 1:]
        t = np.arange(self.trend_steps, dtype=np.float32)
        future = np.arange(self.trend_steps, self.trend_steps + OUTPUT_STEPS, dtype=np.float32)
        A = np.column_stack([t, np.ones_like(t)])
        rhs = tail.transpose(1, 0, 2).reshape(self.trend_steps, -1)
        coef = np.linalg.lstsq(A, rhs, rcond=None)[0].reshape(2, len(x), 6)
        return (future[None, :, None] * coef[0][:, None, :] +
                coef[1][:, None, :])


class Classical:
    def __init__(self, name, estimator): self.name, self.estimator = name, estimator
    def fit(self, x, y):
        self.estimator.fit(x.reshape(len(x), -1), y.reshape(len(y), -1)); return self
    def predict(self, x):
        return self.estimator.predict(x.reshape(len(x), -1)).reshape(len(x), OUTPUT_STEPS, 6)


class XGBoostMultiDoF:
    """Native multi-output XGBoost on the exact flattened 100 x 7 history."""

    name = "XGBoost"

    def __init__(self):
        # A single native multi-output XGBoost call crashes in the macOS ARM
        # OpenMP runtime on this machine.  The documented sklearn wrapper keeps
        # the identical flattened inputs and trains one supported scalar model
        # per forecast coordinate, serially, avoiding that native failure.
        self.estimator = MultiOutputRegressor(XGBRegressor(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror", random_state=SEED, n_jobs=1,
            verbosity=0, tree_method="hist"), n_jobs=1)

    def fit(self, x, y):
        self.estimator.fit(x.reshape(len(x), -1), y.reshape(len(y), -1))
        return self

    def predict(self, x):
        return self.estimator.predict(x.reshape(len(x), -1)).reshape(len(x), OUTPUT_STEPS, 6)


class ForecastNet(torch.nn.Module):
    def __init__(self, kind: str, n_features: int = 7):
        super().__init__(); self.kind = kind
        self.proj = torch.nn.Linear(n_features, 128)
        if kind == "lstm":
            self.encoder = torch.nn.LSTM(128, 128, num_layers=2, dropout=0.2, batch_first=True)
        else:
            layer = torch.nn.TransformerEncoderLayer(128, 8, dim_feedforward=512, dropout=0.1,
                                                     activation="gelu", batch_first=True, norm_first=True)
            position = torch.arange(INPUT_STEPS).unsqueeze(1)
            div = torch.exp(torch.arange(0, 128, 2) * (-np.log(10000.0) / 128))
            pos = torch.zeros(INPUT_STEPS, 128)
            pos[:, 0::2], pos[:, 1::2] = torch.sin(position * div), torch.cos(position * div)
            self.register_buffer("pos", pos.unsqueeze(0))
            self.encoder = torch.nn.TransformerEncoder(layer, num_layers=3)
        self.head = torch.nn.Sequential(torch.nn.Linear(128, 256), torch.nn.GELU(),
                                         torch.nn.Linear(256, OUTPUT_STEPS * 6))

    def forward(self, x):
        z = self.proj(x)
        if self.kind == "lstm": z, _ = self.encoder(z); z = z[:, -1]
        else: z = self.encoder(z + self.pos).mean(dim=1)
        return self.head(z).reshape(-1, OUTPUT_STEPS, 6)


class Neural:
    def __init__(self, name: str, kind: str, input_mean, input_scale, target_mean, target_scale, out_dir: Path):
        self.name, self.kind = name, kind
        self.input_mean, self.input_scale = input_mean, input_scale
        self.target_mean, self.target_scale = target_mean, target_scale
        self.model = ForecastNet(kind)
        self.out_dir = out_dir
        # Kaggle exposes NVIDIA GPUs through CUDA, while local Apple hardware
        # uses MPS.  Keep CPU as the portable fallback.
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def fit(self, x, y, x_val, y_val):
        torch.manual_seed(SEED); np.random.seed(SEED)
        device = self.device; self.model.to(device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)
                     if self.kind == "transformer" else None)
        loss_fn = torch.nn.MSELoss()
        tx = torch.from_numpy((x - self.input_mean[None, None, :]) / self.input_scale[None, None, :]).to(device)
        ty = torch.from_numpy((y - self.target_mean[None, None, :]) / self.target_scale[None, None, :]).to(device)
        vx = torch.from_numpy((x_val - self.input_mean[None, None, :]) / self.input_scale[None, None, :]).to(device)
        vy = torch.from_numpy((y_val - self.target_mean[None, None, :]) / self.target_scale[None, None, :]).to(device)
        best, best_state, stale, history = float("inf"), None, 0, []
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(tx, ty), batch_size=32, shuffle=True)
        for epoch in range(1, 101):
            self.model.train(); train_loss = 0.0
            for bx, by in loader:
                opt.zero_grad(); loss = loss_fn(self.model(bx), by); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); opt.step(); train_loss += loss.item() * len(bx)
            self.model.eval()
            with torch.no_grad(): val_loss = loss_fn(self.model(vx), vy).item()
            if scheduler is not None: scheduler.step(val_loss)
            history.append({"epoch": epoch, "train_loss": train_loss / len(tx), "validation_loss": val_loss,
                            "learning_rate": opt.param_groups[0]["lr"]})
            if val_loss < best - 1e-7:
                best, best_state, stale = val_loss, {k: v.detach().clone() for k, v in self.model.state_dict().items()}, 0
            else: stale += 1
            if stale >= 10: break
        self.model.load_state_dict(best_state)
        self.history = history; self.epochs = len(history); return self

    def predict(self, x):
        self.model.eval()
        nx = torch.from_numpy((x - self.input_mean[None, None, :]) / self.input_scale[None, None, :])
        with torch.no_grad(): out = self.model(nx.to(self.device)).cpu().numpy()
        return out * self.target_scale[None, None, :] + self.target_mean[None, None, :]


def make_models(data: dict, out_dir: Path):
    return [
        Persistence(), LinearTrend(),
        Classical("Ridge", Ridge(alpha=1.0)),
        Classical("Random Forest", RandomForestRegressor(n_estimators=200, max_depth=15,
            min_samples_split=5, min_samples_leaf=2, bootstrap=True, n_jobs=-1, random_state=SEED)),
        XGBoostMultiDoF(),
        Neural("LSTM", "lstm", data["feature_mean"], data["feature_scale"], data["target_mean"], data["target_scale"], out_dir),
        Neural("Transformer", "transformer", data["feature_mean"], data["feature_scale"], data["target_mean"], data["target_scale"], out_dir),
    ]


def _latency(model, x_one: np.ndarray) -> tuple[float, float]:
    restore_device = None
    if isinstance(model, Neural):
        restore_device = model.device
        model.device = torch.device("cpu")
        model.model.to(model.device)
    for _ in range(20): model.predict(x_one)
    values = []
    for _ in range(500):
        t0 = time.perf_counter(); model.predict(x_one); values.append((time.perf_counter() - t0) * 1000)
    result = float(np.percentile(values, 50)), float(np.percentile(values, 95))
    if restore_device is not None:
        model.device = restore_device
        model.model.to(restore_device)
    return result


def prediction_frame(model_name, dataset, indices, data, pred):
    records = []
    for row, idx in enumerate(indices):
        meta = data["metadata"][idx]
        for h, ts in enumerate(meta["target_times"]):
            for d, dof in enumerate(MOTION_COLUMNS):
                records.append({"dataset": dataset, "run_id": meta["run_id"], "window_id": meta["window_id"],
                    "split": meta["split"], "forecast_origin_time_s": meta["forecast_origin_time_s"],
                    "target_time_s": float(ts), "horizon_step": h + 1, "horizon_s": float((h + 1) / 10),
                    "dof": dof, "y_true": float(data["y"][idx, h, d]), "y_pred": float(pred[row, h, d]),
                    "model": model_name, "seed": SEED})
    return pd.DataFrame(records)


def evaluate(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = predictions[predictions.split == "test"].copy()
    persistence = test[test.model == "Persistence"].set_index(["dataset", "window_id", "horizon_step", "dof"])["y_pred"]
    metrics = []
    for (dataset, model, dof), g in test.groupby(["dataset", "model", "dof"], sort=True):
        err = g.y_pred.to_numpy() - g.y_true.to_numpy(); mse = float(np.mean(err ** 2))
        true = g.y_true.to_numpy(); rmse = float(np.sqrt(mse)); mae = float(np.mean(np.abs(err)))
        denom = float(np.sum((true - true.mean()) ** 2)); r2 = float(1 - np.sum(err ** 2) / denom) if denom else None
        train_std = float(g.attrs.get("train_std", np.nan)) if hasattr(g, "attrs") else np.nan
        pkey = pd.MultiIndex.from_frame(g[["dataset", "window_id", "horizon_step", "dof"]])
        p = persistence.reindex(pkey).to_numpy(); pmse = float(np.mean((p - true) ** 2))
        metrics.append({"dataset": dataset, "model": model, "dof": dof, "rmse": rmse, "mae": mae,
                        "r2": r2, "train_target_std": train_std, "nrmse": np.nan,
                        "persistence_skill": 1 - mse / pmse if pmse else None, "n": len(g)})
    metrics_df = pd.DataFrame(metrics)
    horizon = []
    for (dataset, model, h, dof), g in test.groupby(["dataset", "model", "horizon_step", "dof"], sort=True):
        err = g.y_pred.to_numpy() - g.y_true.to_numpy()
        p = persistence.reindex(pd.MultiIndex.from_frame(g[["dataset", "window_id", "horizon_step", "dof"]])).to_numpy()
        mse, pmse = float(np.mean(err ** 2)), float(np.mean((p - g.y_true.to_numpy()) ** 2))
        horizon.append({"dataset": dataset, "model": model, "horizon_step": h, "horizon_s": h / 10,
                        "dof": dof, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(err))),
                        "persistence_skill": 1 - mse / pmse if pmse else None})
    return metrics_df, pd.DataFrame(horizon)


def run_benchmark(root: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = ["BTP-1", "BTP-2"]
    all_predictions, run_rows = [], []
    immutable = {}
    for dataset in datasets:
        data = load_windows(root, dataset)
        base = root / "data" / "splits"
        for name in ["raw", "summary", "interpolated", "windows", "train_scaler.json", "metadata.json"]:
            immutable[f"{dataset}_{name}"] = sha256_file(base / f"{dataset}_{name if name.endswith('.json') else name + '.parquet'}")
        train_idx, x_train, y_train = split_arrays(data, "train")
        val_idx, x_val, y_val = split_arrays(data, "validation")
        test_idx, x_test, y_test = split_arrays(data, "test")
        nx_train = _normalize_x(data, x_train); nx_val = _normalize_x(data, x_val); nx_test = _normalize_x(data, x_test)
        for model in make_models(data, out_dir):
            start = time.perf_counter()
            if isinstance(model, Neural): model.fit(x_train, y_train, x_val, y_val)
            elif isinstance(model, (Persistence, LinearTrend)): model.fit(x_train, y_train)
            else: model.fit(nx_train, y_train)
            fit_time = time.perf_counter() - start
            pred_by_split = []
            model_inputs = (x_train, x_val, x_test) if isinstance(model, (Persistence, LinearTrend)) else (nx_train, nx_val, nx_test)
            for split, idx, xx in zip(("train", "validation", "test"), (train_idx, val_idx, test_idx), model_inputs):
                pp = model.predict(xx); pred_by_split.append(prediction_frame(model.name, dataset, idx, data, pp))
            pf = pd.concat(pred_by_split, ignore_index=True); all_predictions.append(pf)
            p50, p95 = _latency(model, (x_test if isinstance(model, (Persistence, LinearTrend)) else nx_test)[:1])
            artifact = out_dir / f"{dataset}__{model.name.lower().replace(' ', '_')}__seed{SEED}"
            artifact.mkdir(exist_ok=True)
            if isinstance(model, Neural):
                torch.save(model.model.state_dict(), artifact / "model.pt")
                (artifact / "training_history.json").write_text(json.dumps(model.history, indent=2))
            elif hasattr(model, "estimator"): joblib.dump(model.estimator, artifact / "model.joblib")
            config = {"dataset": dataset, "model": model.name,
                "feature_set": "F1", "features": FEATURES, "input_steps": INPUT_STEPS,
                "output_steps": OUTPUT_STEPS, "stride": STRIDE, "seed": SEED,
                "fit_time_s": fit_time, "cpu_batch1_p50_ms": p50, "cpu_batch1_p95_ms": p95,
                "torch_version": torch.__version__, "xgboost_version": __import__('xgboost').__version__}
            if isinstance(model, XGBoostMultiDoF):
                config["xgboost"] = {"api": "sklearn MultiOutputRegressor(XGBRegressor)", "input_representation": "flattened 100x7 = 700", "target_representation": "flattened 50x6 = 300", "wrapper": "one serial scalar XGBRegressor per target coordinate", "tree_method": "hist", "n_jobs": 1}
            (artifact / "config.json").write_text(json.dumps(config, indent=2))
            run_rows.append({"dataset": dataset, "model": model.name, "feature_set": "F1", "fit_time_s": fit_time,
                "cpu_batch1_p50_ms": p50, "cpu_batch1_p95_ms": p95, "train_windows": len(train_idx),
                "validation_windows": len(val_idx), "test_windows": len(test_idx),
                "epochs": getattr(model, "epochs", None)})
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    metrics, horizon = evaluate(predictions)
    # Attach training-only target standard deviations and calculate NRMSE.
    stds = []
    for dataset in datasets:
        d = load_windows(root, dataset)
        for dof, std in zip(MOTION_COLUMNS, d["target_scale"]): stds.append({"dataset": dataset, "dof": dof, "std": float(std)})
        (out_dir / f"{dataset}_target_scaler.json").write_text(json.dumps({
            "fit_scope": "train split only", "target_columns": MOTION_COLUMNS,
            "mean": {dof: float(mean) for dof, mean in zip(MOTION_COLUMNS, d["target_mean"])},
            "scale": {dof: float(std) for dof, std in zip(MOTION_COLUMNS, d["target_scale"])},
        }, indent=2) + "\n")
    metrics = metrics.merge(pd.DataFrame(stds), on=["dataset", "dof"], how="left")
    metrics["train_target_std"] = metrics.pop("std"); metrics["nrmse"] = metrics.rmse / metrics.train_target_std
    run_df = pd.DataFrame(run_rows); run_df.to_csv(out_dir / "run_artifacts.csv", index=False)
    metrics.to_csv(out_dir / "metrics_by_dof.csv", index=False); horizon.to_csv(out_dir / "metrics_by_horizon.csv", index=False)
    (out_dir / "manifest_hashes.json").write_text(json.dumps(immutable, indent=2) + "\n")
    (out_dir / "environment.json").write_text(json.dumps({"python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__,
        "xgboost": __import__('xgboost').__version__}, indent=2) + "\n")
    return {"predictions": predictions, "metrics": metrics, "horizon": horizon, "runs": run_df}


def write_comparison(result: dict, out_dir: Path):
    metrics = result["metrics"]
    agg = metrics.groupby("model", as_index=False).agg(macro_nrmse=("nrmse", "mean"),
        macro_persistence_skill=("persistence_skill", "mean"))
    runs = result["runs"].groupby("model", as_index=False).agg(fit_time_s=("fit_time_s", "sum"),
        cpu_batch1_p50_ms=("cpu_batch1_p50_ms", "mean"), cpu_batch1_p95_ms=("cpu_batch1_p95_ms", "mean"))
    table = agg.merge(runs, on="model")
    for dof in ["heave_m", "pitch_deg"]:
        vals = metrics[metrics.dof == dof].groupby("model", as_index=False).rmse.mean().rename(columns={"rmse": f"{dof}_rmse"})
        table = table.merge(vals, on="model")
    order = ["Persistence", "Linear Trend", "Ridge", "Random Forest", "XGBoost", "LSTM", "Transformer"]
    table["order"] = table.model.map({v:i for i,v in enumerate(order)}); table = table.sort_values("order").drop(columns="order")
    table.to_csv(out_dir / "primary_comparison.csv", index=False)
    table.to_markdown(out_dir / "primary_comparison.md", index=False, floatfmt=".6f")
    non_transformer = table[table.model != "Transformer"].sort_values("macro_nrmse").iloc[0]
    transformer = table[table.model == "Transformer"].iloc[0]
    comparison = {"strongest_non_transformer": non_transformer.model,
                  "strongest_non_transformer_macro_nrmse": float(non_transformer.macro_nrmse),
                  "transformer_macro_nrmse": float(transformer.macro_nrmse),
                  "transformer_outperforms": bool(transformer.macro_nrmse < non_transformer.macro_nrmse)}
    (out_dir / "transformer_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    return table, comparison
