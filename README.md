# GNN Sparsify Lab

This repo is my final‑year project at the University of Birmingham.

I am **Mohammed**, an Artificial Intelligence and Computer Science student, and this project turns my dissertation work on **graph sparsification for GNNs** into a reusable, developer‑friendly lab.

The idea is simple:

> Start from a standard GCN on popular benchmark graphs, sparsify the edges in different ways, and see how much structure you can remove before accuracy drops too much.

You can use this repo to:

* Compare multiple sparsification strategies side by side
* Measure the trade‑off between **accuracy, speed and sparsity**
* Run both CLI experiments and a **web dashboard** (FastAPI + React + charts)
* Auto‑search for a sparsity level that gives a good balance between speed and performance

---

## Features

* **Clean Python package**: `gnn_sparsify/` with datasets, sparsifiers, models and training loop
* **Datasets**:

  * Cora, Citeseer, PubMed (Planetoid)
  * ogbn‑arxiv (large OGB benchmark)
* **Sparsification strategies**:

  * `random` – uniform edge sampling baseline
  * `degree` – keep edges between high‑degree nodes
  * `similarity` – cosine similarity on node features
  * `two_stage` – random pre‑sample + degree scoring (aggressive + fast)
* **Metrics**: accuracy, macro/micro F1, edge counts, sparsity ratio, training time
* **Experiment runner**:

  * YAML configs + CLI script
  * FastAPI backend with `/run`, `/optimize`, `/sparsifiers`
* **Web dashboard**:

  * React single‑file frontend (no build step)
  * Controls for dataset, method, sparsity, epochs and hyperparameters
  * History table + **live charts** for sparsity vs accuracy / speedup

---

## Quick start

### 1. Setup

```bash
git clone https://github.com/your-username/gnn-sparsify.git
cd gnn-sparsify

python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

> For PyTorch Geometric and OGB, follow the official install instructions if the basic `pip install` fails.

### 2. Run a CLI experiment

Example: Cora with random sparsification.

```yaml
# configs/cora.yaml
seed: 42

dataset:
  name: "Cora"

model:
  hidden_channels: 16
  dropout: 0.5

training:
  epochs: 200
  lr: 0.01
  weight_decay: 0.0005

sparsifier:
  name: "random"    # "random", "degree", "similarity", "two_stage"
  sparsity: 0.5      # fraction of edges kept
```

Run:

```bash
python -m experiments.run_experiment --config configs/cora.yaml
```

### 3. Launch the web lab

Backend:

```bash
cd demo/backend_fastapi
uvicorn app:app --reload
```

Frontend (in another terminal):

```bash
cd demo/frontend_react
python -m http.server 5500
```

Then open:

```text
http://127.0.0.1:5500/demo/frontend_react/index.html
```

From the UI you can:

* Pick a dataset and sparsifier
* Slide sparsity from full graph to very sparse
* Tune epochs and hyperparameters in the advanced panel
* Run single experiments or **Auto Optimize** (sparsity sweep)
* See history as a table **and** as charts (sparsity vs accuracy / speedup)

---

## Key findings (my FYP results)

Using this toolkit I ran extensive experiments, especially on the large **ogbn‑arxiv** dataset.

### 1. Sparsity vs speed

For ogbn‑arxiv with the **two‑stage** sparsifier (10 epochs) :

| Sparsity (edges kept) | Accuracy (%) | Train time (s) | Speedup vs dense |
| --------------------: | -----------: | -------------: | ---------------: |
|                   1.0 |         24.8 |           4.74 |             0.0% |
|                   0.9 |         24.2 |           4.25 |            10.5% |
|                   0.8 |         23.4 |           3.79 |            20.0% |
|                   0.7 |         22.7 |           3.57 |            24.8% |
|                   0.6 |         22.3 |           3.24 |            31.6% |
|                   0.5 |         21.9 |           2.84 |            40.2% |
|                   0.4 |         20.9 |           2.58 |            45.7% |
|                   0.3 |         21.1 |           2.18 |            53.9% |

**What I learned:**

* Sparsifying ogbn‑arxiv gives **real speed gains**:

  * Around **30% faster** at sparsity ≈ 0.6
  * Around **50% faster** at sparsity ≈ 0.3–0.4
* Accuracy degrades gradually rather than collapsing:

  * Small sparsification (0.9–0.8) gives a mild drop in accuracy for a noticeable speedup
  * More aggressive sparsity (0.6–0.4) roughly keeps 80–90% of the dense accuracy while significantly cutting training time
* Different methods behave differently:

  * `random` tends to be slightly kinder to accuracy
  * `two_stage` is more aggressive on speed, especially on large graphs

In other words, for this setup there is a **useful regime** where we can delete a large fraction of edges and still keep the model reasonably strong while training much faster.

### 2. Simple trade‑off picture (expected trend)

To keep things simple, here is an **expected** trend for how accuracy and speedup behave as you sparsify the graph. The idea is what matters, not the exact numbers.

```text
Sparsity (edges kept)   1.0        0.8        0.6        0.4
----------------------------------------------------------------
Accuracy                ██████     █████      ████       ███
Speedup                 ░          ░░░        ░░░░░      ░░░░░░
```

* As sparsity goes **down** (you keep fewer edges), accuracy drops gradually.
* At the same time, speedup grows, because each training step becomes cheaper.

You can see the **exact** curves for your runs in the web dashboard charts, which are driven directly from your experiment history.

---

## Project structure (short overview)

```text
gnn-sparsify/
├── gnn_sparsify/         # core Python package
│   ├── datasets/         # Cora/Citeseer/PubMed + ogbn-arxiv loaders
│   ├── sparsifiers/      # random, degree, similarity, two_stage
│   ├── models/           # GCN
│   ├── training/         # trainer + pipeline
│   └── evaluation/       # metrics
├── configs/              # YAML configs for experiments
├── experiments/          # CLI entrypoint
├── demo/
│   ├── backend_fastapi/  # FastAPI app
│   └── frontend_react/   # React dashboard with charts
└── notebooks/            # space for analysis notebooks
```

---

## About the project

This repository is the **public version of my final‑year project**:

> *Practical Graph Sparsification for Efficient GNN Training: Accuracy, Time and Memory Trade‑offs*.

It shows that I:

* Designed and ran ML experiments on real graph benchmarks
* Built clean, reusable tooling around research code
* Exposed results through a user‑friendly web interface and visualisations

If you have feedback, ideas for new sparsifiers, or want to use this as a base for your own experiments, feel free to reach out or open an issue.

---

## License

MIT License.
