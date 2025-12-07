import time
from typing import Dict

import torch
import torch.nn.functional as F

from gnn_sparsify.datasets import load_dataset
from gnn_sparsify.sparsifiers import SPARSIFIER_REGISTRY
from gnn_sparsify.models import GCN
from gnn_sparsify.evaluation import compute_metrics
from gnn_sparsify.utils import set_seed


def run_experiment(config: Dict) -> Dict:
    """
    Main training pipeline. Used by CLI and FastAPI.

    Config format (example):

    {
      "seed": 42,
      "dataset": {"name": "Cora"},
      "model": {"hidden_channels": 16, "dropout": 0.5},
      "training": {"epochs": 200, "lr": 0.01, "weight_decay": 5e-4},
      "sparsifier": {"name": "random", "sparsity": 0.5}
    }
    """
    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name = config["dataset"]["name"]
    dataset, data = load_dataset(dataset_name)
    data = data.to(device)

    # Build sparsifier
    sparsifier_cfg = config["sparsifier"]
    sparsifier_name = sparsifier_cfg["name"]
    sparsity = float(sparsifier_cfg["sparsity"])

    if sparsifier_name not in SPARSIFIER_REGISTRY:
        raise ValueError(f"Unknown sparsifier: {sparsifier_name}")

    sparsifier_cls = SPARSIFIER_REGISTRY[sparsifier_name]
    sparsifier = sparsifier_cls(sparsity=sparsity)

    start_edges = data.edge_index.size(1)
    data = sparsifier.sparsify(data)
    end_edges = data.edge_index.size(1)
    sparsity_ratio = end_edges / start_edges if start_edges > 0 else 1.0

    # Build model (general: infer from data)
    in_channels = data.x.size(-1)
    out_channels = int(data.y.max().item()) + 1

    hidden_channels = config["model"]["hidden_channels"]
    dropout = config["model"].get("dropout", 0.5)

    model = GCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    epochs = config["training"]["epochs"]

    # More accurate timing, especially on GPU
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time_start = time.time()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time_end = time.time()
    total_time = train_time_end - train_time_start

    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    metrics = compute_metrics(logits, data)

    metrics["train_time_sec"] = float(total_time)
    metrics["sparsity_ratio"] = float(sparsity_ratio)
    metrics["num_edges_before"] = int(start_edges)
    metrics["num_edges_after"] = int(end_edges)
    metrics["dataset"] = dataset_name
    metrics["sparsifier"] = sparsifier_name
    metrics["sparsity_config"] = sparsity

    return metrics
