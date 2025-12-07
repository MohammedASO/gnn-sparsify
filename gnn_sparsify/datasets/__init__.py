from .planetoid import load_planetoid_dataset
from .ogbn_arxiv import load_ogbn_arxiv


__all__ = ["load_planetoid_dataset", "load_dataset"]


def load_dataset(name: str, root: str = "data"):
    """
    Generic dataset loader.

    Supported names:
      - "Cora", "Citeseer", "PubMed"
      - "ogbn-arxiv", "ogbn_arxiv", "arxiv"
    """
    name_lower = name.lower()

    if name_lower in {"cora", "citeseer", "pubmed"}:
        dataset = load_planetoid_dataset(name, root=root)
        data = dataset[0]
        return dataset, data

    if name_lower in {"ogbn-arxiv", "ogbn_arxiv", "arxiv"}:
        dataset, data = load_ogbn_arxiv(root=root)
        return dataset, data

    raise ValueError(f"Unknown dataset: {name}")
