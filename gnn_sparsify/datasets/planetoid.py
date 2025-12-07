import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures




def load_planetoid_dataset(name: str = "Cora", root: str = "data"):
    """
    Load a Planetoid dataset (Cora, Citeseer, PubMed) with normalized features.
    """
    name = name.capitalize()
    dataset = Planetoid(root=root, name=name, transform=NormalizeFeatures())
    return dataset

