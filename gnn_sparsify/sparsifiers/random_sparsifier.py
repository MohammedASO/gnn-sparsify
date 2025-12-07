import torch
from torch_geometric.data import Data

from .base import GraphSparsifier




class RandomSparsifier(GraphSparsifier):
    """
    Uniform random edge sampling.
    """

    def sparsify(self, data: Data) -> Data:
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        keep_edges = int(num_edges * self.sparsity)


        perm = torch.randperm(num_edges)
        keep_idx = perm[:keep_edges]


        new_edge_index = edge_index[:, keep_idx]


        new_data = data.clone()
        new_data.edge_index = new_edge_index
        return new_data

