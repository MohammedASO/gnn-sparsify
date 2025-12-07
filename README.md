# gnn-sparsify

A modular, research friendly toolkit for experimenting with graph sparsification techniques in Graph Neural Networks (GNNs). This project transforms academic work into a clean, reproducible, developer friendly framework that makes it easy to study how sparsity affects accuracy, training time, and memory usage.

## Features

### Core Modules

* Dataset loaders for Cora, Citeseer, PubMed
* Multiple sparsification strategies
* PyTorch Geometric training pipelines
* Evaluation tools for accuracy, F1, runtime, and memory

### Experiment System

* YAML or Hydra configuration
* Reproducible experiment scripts
* Easy comparisons across sparsity levels and models

### Visualization

* Jupyter notebooks for tutorials
* Optional FastAPI + React dashboard to interactively explore results
* Plots for accuracy vs sparsity, time vs sparsity, memory vs sparsity

## Project Structure

```
gnn-sparsify/
│
├── gnn_sparsify/
│   ├── datasets/
│   ├── sparsifiers/
│   ├── models/
│   ├── training/
│   ├── evaluation/
│   └── utils/
│
├── configs/
│   ├── cora.yaml
│   ├── citeseer.yaml
│   └── pubmed.yaml
│
├── experiments/
│   ├── run_experiment.py
│   ├── benchmark.py
│   └── results/
│
├── notebooks/
│   ├── 01_getting_started.ipynb
│   ├── 02_visualizing_sparsity.ipynb
│   └── 03_compare_methods.ipynb
│
├── demo/
│   ├── backend_fastapi/
│   └── frontend_react/
│
└── README.md
```

## Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/gnn-sparsify.git
cd gnn-sparsify
```

### 2. Create environment

```
python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

## Quick Start

### Run an experiment on Cora

```
python experiments/run_experiment.py --config configs/cora.yaml
```

### Choose sparsity level

```
python experiments/run_experiment.py --config configs/cora.yaml sparsity=0.3
```

### Compare all sparsifiers

```
python experiments/benchmark.py --dataset cora
```

## Supported Sparsification Techniques

| Technique               | Description                          |
| ----------------------- | ------------------------------------ |
| Degree-based pruning    | Keeps high-degree edges and nodes    |
| Random sparsification   | Uniform random edge sampling         |
| Spectral sparsification | Laplacian-aware edge sampling        |
| Threshold pruning       | Keeps edges above a weight threshold |
| Custom methods          | Plug in your own class               |

## Metrics Reported

* Node classification accuracy
* Macro and micro F1
* Training time
* Memory usage
* Sparsity ratio
* Edge count reduction

Each experiment automatically logs results under `experiments/results/`.

## Demo (Optional)

A small interactive dashboard allows:

* dataset selection
* choosing sparsification methods
* adjusting sparsity level
* visualizing accuracy and runtime

To run:

```
cd demo/backend_fastapi
uvicorn app:app

cd ../frontend_react
npm install
npm start
```

## Why This Project Matters

This toolkit bridges the gap between research papers and usable ML software. It demonstrates:

* deep understanding of GNNs
* ability to create reproducible ML systems
* research engineering skills
* clean code architecture
* experiment tracking and benchmarking

## License

MIT License.

## Contributing

Pull requests are welcome. Open an issue for bug reports or feature requests.
