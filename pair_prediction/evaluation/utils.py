import os
import pickle
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from torch.utils.data import DataLoader

from pair_prediction.visualization.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_probability_distribution,
    plot_pr_curve
)
from pair_prediction.data.dataset import LinkPredictionDataset


def load_dataset(dataset_name: str, root: str = 'data') -> LinkPredictionDataset:
    """
    Load a dataset by name.
    """
    if dataset_name == "validation":
        dataset = LinkPredictionDataset(root=root, mode="validation")
    else:
        dataset = LinkPredictionDataset(root=f"{root}/evaluation/{dataset_name}_clean")
    return dataset

def export_dataset_to_fasta(dataset: LinkPredictionDataset,  output_dir: str, batchsize: int = 1):
    """
    Iterate over the dataset and save each batch into <output_dir> 
    with a `.fasat` extension. 
    """
    os.makedirs(output_dir, exist_ok=True)
    if batchsize:
        dataloader = DataLoader(dataset, batch_size=batchsize, shuffle=False)
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            filename = os.path.join(output_dir, f"batch_{batch_idx}.fasta")
            ids = batch.id
            seqs = batch.seq

            with open(filename, 'w') as f:
                for seq_id, seq in zip(ids, seqs):
                    f.write(f">{seq_id}\n{seq}\n")
    else:
        filename = os.path.join(output_dir, f"batch.fasta")
        with open(filename, 'w') as f:
            for data in dataset:
                seq = data.seq
                id = data.id
                f.write(f">{id}\n{seq}\n")

def collect_and_save_metrics(outputs, output_path):
    preds = torch.cat([x["preds"] for x in outputs], dim=0)
    labels = torch.cat([x["labels"] for x in outputs], dim=0)
    probabilities = torch.cat([x["probabilities"] for x in outputs], dim=0)
    edge_types = np.concatenate([x["edge_types"] for x in outputs])
    pair_types = torch.cat([x["pair_types"] for x in outputs], dim=0)

    preds_np = preds.cpu().numpy().astype(np.float32)
    labels_np = labels.cpu().numpy().astype(np.float32)
    prob_np = probabilities.cpu().numpy().astype(np.float32)
    pair_types = pair_types.cpu().numpy().astype(np.float32)

    # --- Metrics ---
    auprc = average_precision_score(labels_np, prob_np)
    results = {
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
        "auprc": auprc,
    }

    fig_cm, _ = plot_confusion_matrix(labels_np, preds_np)
    fig_roc, _ = plot_roc_curve(labels_np, prob_np)
    fig_hist, _ = plot_probability_distribution(labels_np, prob_np)
    fig_pr, _ =plot_pr_curve(labels_np, prob_np)

    # --- Save ---
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    fig_cm.savefig(f"{output_path}/confusion_matrix.png")
    fig_roc.savefig(f"{output_path}/roc_curve.png")
    fig_hist.savefig(f"{output_path}/probability_distribution.png")
    fig_pr.savefig(f"{output_path}/pr_curve.png")

    with open(f"{output_path}/results.pkl", "wb") as f:
        pickle.dump(outputs, f)

    plt.close(fig_cm)
    plt.close(fig_roc)
    plt.close(fig_hist)
    plt.close(fig_pr)


def rmtree(f: Path):
    if f.is_file():
        f.unlink()
    else:
        for child in f.iterdir():
            rmtree(child)
        f.rmdir()