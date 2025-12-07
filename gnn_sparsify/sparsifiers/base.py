from abc import ABC, abstractmethod
from torch_geometric.data import Data




class GraphSparsifier(ABC):
    """
    Base class for graph sparsification.

    Implement sparsify and return a new Data object with fewer edges.
    """

    def __init__(self, sparsity: float):
        """
        sparsity: fraction of edges to keep (0 to 1).
        """
        assert 0 < sparsity <= 1.0
        self.sparsity = sparsity


    @abstractmethod
    def sparsify(self, data: Data) -> Data:
        raise NotImplementedError

