from typing import Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score


def plot_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    title: str = "Confusion Matrix"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a confusion matrix plot.
    
    Args:
        labels: Array of true labels
        predictions: Array of predicted labels
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axes) containing the confusion matrix plot
    """
    fig, ax = plt.subplots()
    cm = confusion_matrix(labels, predictions)
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f"{z}", ha='center', va='center')
    return fig, ax


def plot_roc_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "ROC Curve"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a ROC curve plot with AUC score.
    
    Args:
        labels: Array of true labels
        probabilities: Array of predicted probabilities
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axes) containing the ROC curve plot
    """
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(labels, probabilities)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig, ax

def plot_pr_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "Precision-Recall Curve"
) -> Tuple[plt.Figure, plt.Axes]:
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    auprc = average_precision_score(labels, probabilities)
    fig_pr, ax_pr = plt.subplots()
    ax_pr.step(recall, precision, where="post")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision–Recall Curve (AUPRC = {auprc:.4f})")
    ax_pr.set_xlim([0.0, 1.0])
    ax_pr.set_ylim([0.0, 1.05])
    return fig_pr, ax_pr

def plot_probability_distribution(
    labels: np.ndarray,
    probabilities: np.ndarray,
    title: str = "Probability Distribution"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a histogram of predicted probabilities for positive and negative samples.
    
    Args:
        labels: Array of true labels
        probabilities: Array of predicted probabilities
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axes) containing the probability distribution plot
    """
    fig, ax = plt.subplots()
    pos_probs = probabilities[labels == 1]
    neg_probs = probabilities[labels == 0]
    ax.hist(pos_probs, bins=20, alpha=0.7, label="Positive")
    ax.hist(neg_probs, bins=20, alpha=0.7, label="Negative")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_non_canonical_pair_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    edge_types: np.ndarray,
    pair_types: np.ndarray,
    title: str = "Prediction Accuracy by Non-canonical Pair Type"
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Creates a bar plot showing prediction accuracy for different types of non-canonical pairs.
    Only considers positive pairs (true non-canonical pairs) for accuracy calculation.
    
    Args:
        predictions: Array of predictions
        labels: Array of true labels
        edge_types: Array of edge types
        pair_types: Array of pair types
        title: Title for the plot
        
    Returns:
        Tuple of (figure, axes) containing the accuracy plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    num_pos_edges = len(edge_types)
    
    positive_mask = labels[:num_pos_edges] == 1
    non_canonical_mask = edge_types == 'non-canonical'
    positive_non_canonical_mask = positive_mask & non_canonical_mask
    
    unique_pair_types = np.unique(pair_types[positive_non_canonical_mask])
    
    accuracies = []
    counts = []
    correct_counts = []
    for pair_type in unique_pair_types:
        type_mask = (pair_types == pair_type) & positive_non_canonical_mask
        type_preds = predictions[:num_pos_edges][type_mask.astype(bool)]
        type_labels = labels[:num_pos_edges][type_mask.astype(bool)]
        correct_pred = np.sum(type_preds == type_labels)
        total_count = len(type_labels)
        accuracy = correct_pred / total_count if total_count > 0 else 0
        accuracies.append(accuracy)
        counts.append(total_count)
        correct_counts.append(correct_pred)
    
    bars = ax.bar(unique_pair_types, accuracies)
    ax.set_xlabel("Non-canonical Pair Type")
    ax.set_ylabel("Prediction Accuracy")
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    
    for bar, correct_count, total_count in zip(bars, correct_counts, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., height,
            f'{correct_count}/{total_count}',
            ha='center', 
            va='bottom'
        )
    
    return fig, ax