import argparse
import json
from pathlib import Path

import yaml

from gnn_sparsify.training import run_experiment




def load_config(path: str):
    config_path = Path(path)
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg




def run_experiment_from_dict(config_dict):
    return run_experiment(config_dict)




def main():
    parser = argparse.ArgumentParser(description="Run GNN sparsification experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()


    config = load_config(args.config)
    metrics = run_experiment_from_dict(config)
    print(json.dumps(metrics, indent=2))




if __name__ == "__main__":
    main()

