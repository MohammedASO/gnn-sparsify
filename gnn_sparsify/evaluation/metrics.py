from typing import Dict

import torch
from sklearn.metrics import f1_score




def compute_metrics(logits: torch.Tensor, data) -> Dict:
    """
    Compute accuracy, macro F1, micro F1 on test set.
    """
    preds = logits.argmax(dim=-1).cpu()
    labels = data.y.cpu()
    test_mask = data.test_mask.cpu()


    test_preds = preds[test_mask]
    test_labels = labels[test_mask]


    correct = (test_preds == test_labels).sum().item()
    total = test_mask.sum().item()
    accuracy = correct / total if total > 0 else 0.0


    macro_f1 = f1_score(test_labels, test_preds, average="macro")
    micro_f1 = f1_score(test_labels, test_preds, average="micro")


    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
    }

