import torch
from torch_geometric.data import Data

from .base import GraphSparsifier




class DegreeSparsifier(GraphSparsifier):
    """
    Keeps edges that connect higher degree nodes.
    Simple heuristic: score edge by sum of node degrees and keep top-k.
    """

    def sparsify(self, data: Data) -> Data:
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        num_edges = edge_index.size(1)


        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes)


        # score for each edge = deg(u) + deg(v)
        scores = deg[row] + deg[col]
        keep_edges = int(num_edges * self.sparsity)


        keep_idx = torch.topk(scores, k=keep_edges, largest=True).indices
        new_edge_index = edge_index[:, keep_idx]


        new_data = data.clone()
        new_data.edge_index = new_edge_index
        return new_data

