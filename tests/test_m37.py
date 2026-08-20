import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pandas as pd
import torch

from shipmotion.benchmark import (FEATURES, INPUT_STEPS, OUTPUT_STEPS, ForecastNet,
    LinearTrend, Persistence, evaluate)


def test_primary_contract_and_baseline_shapes():
    assert len(FEATURES) == 7
    x = np.arange(3 * INPUT_STEPS * 7, dtype=np.float32).reshape(3, INPUT_STEPS, 7)
    assert Persistence().predict(x).shape == (3, OUTPUT_STEPS, 6)
    assert LinearTrend().predict(x).shape == (3, OUTPUT_STEPS, 6)
    assert np.allclose(Persistence().predict(x)[:, 0], x[:, -1, 1:])


def test_neural_model_output_shapes():
    x = torch.randn(2, INPUT_STEPS, len(FEATURES))
    for kind in ("lstm", "transformer"):
        assert ForecastNet(kind)(x).shape == (2, OUTPUT_STEPS, 6)


def test_central_evaluator_hand_computed_metrics_and_skill():
    rows = []
    for model, prediction in (("Persistence", 0.0), ("Ridge", 1.0)):
        for h, truth in ((1, 2.0), (2, 4.0)):
            rows.append(dict(dataset="d", model=model, split="test", window_id="w", run_id="r",
                             horizon_step=h, horizon_s=h / 10, dof="surge_m", y_true=truth, y_pred=prediction))
    metric, horizon = evaluate(pd.DataFrame(rows))
    ridge = metric[metric.model == "Ridge"].iloc[0]
    assert np.isclose(ridge.rmse, np.sqrt(5.0))
    assert np.isclose(ridge.mae, 2.0)
    assert np.isclose(ridge.persistence_skill, 0.5)
    assert len(horizon) == 4
