import sys
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to Python path so imports work when running from this folder
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from experiments.run_experiment import run_experiment_from_dict  # noqa
from gnn_sparsify.sparsifiers import SPARSIFIER_REGISTRY, SPARSIFIER_INFO  # noqa

app = FastAPI(title="GNN Sparsify API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExperimentRequest(BaseModel):
    dataset: str = "Cora"
    sparsifier: str = "random"   # "random", "degree", "similarity", "two_stage"
    sparsity: float = 0.5
    epochs: int = 200
    hidden_channels: int = 16
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 0.0005
    seed: int = 42


class OptimizeRequest(BaseModel):
    dataset: str = "Cora"
    sparsifier: str = "random"
    sparsity_values: List[float] = [1.0, 0.9, 0.7, 0.5, 0.3]
    epochs: int = 100
    hidden_channels: int = 16
    dropout: float = 0.5
    lr: float = 0.01
    weight_decay: float = 0.0005
    seed: int = 42
    # Constraints: accuracy drop relative to dense model, and speedup
    max_accuracy_drop: float = 0.02   # 0.02 = max 2% drop
    target_speedup: float = 0.30      # 0.30 = at least 30% faster


@app.get("/")
def root():
    return {"message": "GNN Sparsify API is running"}


@app.get("/sparsifiers")
def list_sparsifiers():
    """
    Return a list of sparsifiers with human friendly metadata for the UI.
    """
    items = []
    for name, cls in SPARSIFIER_REGISTRY.items():
        info = SPARSIFIER_INFO.get(name, {})
        items.append(
            {
                "name": name,
                "label": info.get("label", name),
                "description": info.get("description", ""),
            }
        )
    return items


@app.post("/run")
def run_experiment_api(req: ExperimentRequest):
    config = {
        "seed": req.seed,
        "dataset": {"name": req.dataset},
        "model": {
            "hidden_channels": req.hidden_channels,
            "dropout": req.dropout,
        },
        "training": {
            "epochs": req.epochs,
            "lr": req.lr,
            "weight_decay": req.weight_decay,
        },
        "sparsifier": {
            "name": req.sparsifier,
            "sparsity": req.sparsity,
        },
    }

    metrics = run_experiment_from_dict(config)
    return metrics


@app.post("/optimize")
def optimize_sparsity(req: OptimizeRequest):
    """
    Search over several sparsity values and try to find an 'optimal' point:
    - Accuracy drops by at most `max_accuracy_drop` compared to the full graph.
    - Training time improves by at least `target_speedup`.

    Returns:
      - baseline: metrics at sparsity=1.0 (dense graph)
      - runs: metrics for all sparsity values tested
      - recommended: the best satisfying both constraints (or null)
    """
    # Ensure 1.0 (dense graph) is always evaluated for baseline
    unique_sparsities = sorted(
        {1.0, *[max(0.1, min(1.0, float(s))) for s in req.sparsity_values]}
    )

    runs = []
    baseline = None

    for s in unique_sparsities:
        config = {
            "seed": req.seed,
            "dataset": {"name": req.dataset},
            "model": {
                "hidden_channels": req.hidden_channels,
                "dropout": req.dropout,
            },
            "training": {
                "epochs": req.epochs,
                "lr": req.lr,
                "weight_decay": req.weight_decay,
            },
            "sparsifier": {
                "name": req.sparsifier,
                "sparsity": s,
            },
        }

        metrics = run_experiment_from_dict(config)
        runs.append(metrics)

        if abs(metrics.get("sparsity_config", 0.0) - 1.0) < 1e-6:
            baseline = metrics

    if baseline is None:
        # Should not happen because we force 1.0, but just in case
        return {
            "error": "Baseline at sparsity=1.0 missing",
            "runs": runs,
        }

    baseline_acc = baseline["accuracy"]
    baseline_time = baseline["train_time_sec"]

    best_candidate = None

    for m in runs:
        s_val = m.get("sparsity_config", 0.0)
        # Skip dense baseline in candidate search
        if abs(s_val - 1.0) < 1e-6:
            m["accuracy_drop"] = 0.0
            m["speedup"] = 0.0
            continue

        acc_drop = baseline_acc - m["accuracy"]
        speedup = 1.0 - (m["train_time_sec"] / baseline_time)

        m["accuracy_drop"] = float(acc_drop)
        m["speedup"] = float(speedup)

        if acc_drop <= req.max_accuracy_drop and speedup >= req.target_speedup:
            if best_candidate is None or speedup > best_candidate["speedup"]:
                best_candidate = m

    result = {
        "baseline": baseline,
        "runs": runs,
        "constraints": {
            "max_accuracy_drop": req.max_accuracy_drop,
            "target_speedup": req.target_speedup,
        },
        "recommended": best_candidate,
    }

    return result
