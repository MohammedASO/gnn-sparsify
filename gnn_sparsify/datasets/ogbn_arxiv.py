import torch

# Workaround for PyTorch 2.6+ compatibility with OGB
# OGB uses torch.load internally, and PyTorch 2.6+ defaults to weights_only=True
# We need to patch torch.load to allow OGB's pickle files
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    """Patch torch.load to allow OGB dataset loading with PyTorch 2.6+"""
    # OGB's processed files contain PyG objects that need weights_only=False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

# Only patch if we're on PyTorch 2.6+ (has weights_only parameter)
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.load = _patched_torch_load

from ogb.nodeproppred import PygNodePropPredDataset


def load_ogbn_arxiv(root: str = "data"):
    """
    Load the ogbn-arxiv dataset as a PyG Data object with train/val/test masks.

    Returns:
      dataset: OGB dataset object
      data: PyG Data object with x, y, edge_index, train_mask, val_mask, test_mask
    
    Note: PyTorch 2.6+ compatibility workaround is applied automatically.
    """
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=root)
    data = dataset[0]

    # OGB labels are shape [num_nodes, 1] -> flatten to [num_nodes]
    data.y = data.y.view(-1)

    split_idx = dataset.get_idx_split()
    num_nodes = data.num_nodes

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[split_idx["train"]] = True
    val_mask[split_idx["valid"]] = True
    test_mask[split_idx["test"]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return dataset, data

