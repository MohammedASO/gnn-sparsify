import torch
from torch_geometric.data import Data

from .base import GraphSparsifier


class TwoStageSparsifier(GraphSparsifier):
    """
    Two-stage sparsification:

    1) Randomly pre-sample edges to a slightly larger intermediate set.
    2) From that subset, keep edges that connect high-degree nodes.

    This tends to keep important 'hub' structure while still being quite cheap.
    """

    def __init__(self, sparsity: float, intermediate_factor: float = 2.0):
        super().__init__(sparsity)
        self.intermediate_factor = max(1.0, float(intermediate_factor))

    def sparsify(self, data: Data) -> Data:
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return data

        # Step 1: random pre-sampling
        target_edges = max(1, int(num_edges * self.sparsity))
        intermediate_edges = min(num_edges, int(target_edges * self.intermediate_factor))

        perm = torch.randperm(num_edges)
        subset_idx = perm[:intermediate_edges]
        subset_edge_index = edge_index[:, subset_idx]

        # Step 2: degree-based scoring within subset
        row, col = subset_edge_index
        num_nodes = data.num_nodes

        deg = torch.bincount(row, minlength=num_nodes)
        scores = deg[row] + deg[col]

        keep_edges = min(intermediate_edges, target_edges)
        keep_edges = max(1, keep_edges)

        keep_idx = torch.topk(scores, k=keep_edges, largest=True).indices
        new_edge_index = subset_edge_index[:, keep_idx]

        new_data = data.clone()
        new_data.edge_index = new_edge_index
        return new_data

