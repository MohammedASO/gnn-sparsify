import torch
from torch_geometric.data import Data

from .base import GraphSparsifier


class SimilaritySparsifier(GraphSparsifier):
    """
    Keeps edges between nodes with similar feature vectors.

    Uses cosine similarity on node features as the edge score.
    """

    def sparsify(self, data: Data) -> Data:
        edge_index = data.edge_index
        num_edges = edge_index.size(1)
        if num_edges == 0:
            return data

        row, col = edge_index
        x = data.x

        x_row = x[row]
        x_col = x[col]

        # Cosine similarity as edge score
        num = (x_row * x_col).sum(dim=1)
        denom = x_row.norm(dim=1) * x_col.norm(dim=1) + 1e-8
        scores = num / denom

        keep_edges = max(1, int(num_edges * self.sparsity))
        keep_edges = min(keep_edges, num_edges)

        keep_idx = torch.topk(scores, k=keep_edges, largest=True).indices
        new_edge_index = edge_index[:, keep_idx]

        new_data = data.clone()
        new_data.edge_index = new_edge_index
        return new_data

