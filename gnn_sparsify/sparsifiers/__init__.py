from .base import GraphSparsifier
from .random_sparsifier import RandomSparsifier
from .degree_sparsifier import DegreeSparsifier
from .similarity_sparsifier import SimilaritySparsifier
from .two_stage_sparsifier import TwoStageSparsifier


SPARSIFIER_REGISTRY = {
    "random": RandomSparsifier,
    "degree": DegreeSparsifier,
    "similarity": SimilaritySparsifier,
    "two_stage": TwoStageSparsifier,
}

# Human friendly metadata for the frontend
SPARSIFIER_INFO = {
    "random": {
        "label": "Random sampling",
        "description": "Keeps a random fraction of edges. Good baseline to compare against structured methods.",
    },
    "degree": {
        "label": "High degree edges",
        "description": "Keeps edges that connect high-degree nodes. Preserves hubs and global structure.",
    },
    "similarity": {
        "label": "Feature similarity",
        "description": "Keeps edges between nodes with similar feature vectors (cosine similarity). Preserves semantic relationships.",
    },
    "two_stage": {
        "label": "Two-stage (random + degree)",
        "description": "First randomly filters edges, then keeps high-degree edges inside that subset. Good compromise between speed and structure.",
    },
}


__all__ = [
    "GraphSparsifier",
    "RandomSparsifier",
    "DegreeSparsifier",
    "SimilaritySparsifier",
    "TwoStageSparsifier",
    "SPARSIFIER_REGISTRY",
    "SPARSIFIER_INFO",
]

