import torch
import numpy as np

def accuracy(logits, labels):
    """
    Calculate classification accuracy: (correct predictions / total)
    logits: (batch, num_classes)
    labels: (batch,)
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return float(correct) / float(total) if total > 0 else 0.0

def accuracy_score(logits, labels):
    """
    Same as the accuracy function, kept for backward compatibility
    """
    return accuracy(logits, labels)

def compute_confusion_matrix(logits, labels, num_classes):
    """
    Calculate confusion matrix, returns num_classes x num_classes
    Rows: true labels, Columns: predicted labels
    """
    preds = torch.argmax(logits, dim=1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
        cm[t, p] += 1
    return cm
