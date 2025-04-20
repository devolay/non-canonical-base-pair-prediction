import argparse
import numpy as np
import torch
import os
from tqdm import tqdm
from pathlib import Path

from torch_geometric.utils import to_dense_adj
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.model.utils import get_negative_edges
from pair_prediction.visualization.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_probability_distribution,
    plot_non_canonical_pair_accuracy,
)

DATA_DIR = Path("data/")

SPOTRNA_THRESHOLD = 0.335

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SPOT-RNA predictions.")
    parser.add_argument("--predictions-path", type=str, required=True, help="Path to directory with SPOT-RNA predictions.")
    parser.add_argument("--negative-sample-ratio", type=int, default=8, help="Ratio of negative samples to positive samples.")
    parser.add_argument("--output-path", type=str, default="outputs/", help="Path to output directory.")
    return parser.parse_args()

def main(args):
    dataset = LinkPredictionDataset(root=DATA_DIR, validation=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions_path = Path(args.predictions_path)

    outputs = []

    for data in tqdm(dataloader):
        seq_id = data.id[0]
        edge_types = np.concatenate(data.edge_type)
        edge_index = data.edge_index
        # canonical_mask = edge_types == 'canonical'
        non_canonical_mask = edge_types == 'non-canonical'

        bpseq_file = predictions_path / f"{seq_id}.prob"
        if not bpseq_file.exists():
            print(f"File {bpseq_file} does not exist. Skipping.")
            continue

        with open(bpseq_file, "r") as file:
            pred = np.loadtxt(file, dtype=np.float32, delimiter="\t")
            pred = pred + pred.T

        # pos_edge_index = edge_index[:, non_canonical_mask | canonical_mask]
        pos_edge_index = edge_index[:, non_canonical_mask]
        pos_labels = torch.ones(pos_edge_index.size(1), dtype=torch.float32)

        neg_edge_index = get_negative_edges(data, sample_ratio=args.negative_sample_ratio)
        neg_labels = torch.zeros(neg_edge_index.size(1), dtype=torch.float32)

        all_edge_index = torch.concatenate([pos_edge_index, neg_edge_index], axis=1)
        all_labels = torch.concatenate([pos_labels, neg_labels], axis=0)

        all_edge_index = to_dense_adj(all_edge_index, batch=data.batch).squeeze(0)
        all_edge_index_preds = pred[all_edge_index.bool()] > SPOTRNA_THRESHOLD

        outputs.append({
            "preds": torch.from_numpy(all_edge_index_preds),
            "labels": all_labels,
            "probabilities": torch.from_numpy(pred[all_edge_index.bool()]),
            "edge_types": edge_types,
            "pair_types": data.pair_type,
        })
    
    preds = torch.cat([x["preds"] for x in outputs], dim=0)
    labels = torch.cat([x["labels"] for x in outputs], dim=0)
    probabilities = torch.cat([x["probabilities"] for x in outputs], dim=0)
    edge_types = np.concatenate([x["edge_types"] for x in outputs])
    pair_types = torch.cat([x["pair_types"] for x in outputs], dim=0) 
    
    preds_np = preds.cpu().numpy().astype(np.float32)
    labels_np = labels.cpu().numpy().astype(np.float32)
    prob_np = probabilities.cpu().numpy().astype(np.float32)
    pair_types = pair_types.cpu().numpy().astype(np.float32)
    
    fig_cm, _ = plot_confusion_matrix(labels_np, preds_np)
    fig_roc, _ = plot_roc_curve(labels_np, prob_np)
    fig_hist, _ = plot_probability_distribution(labels_np, prob_np)
    # fig_pair_types, _ = plot_non_canonical_pair_accuracy(
    #     preds_np, labels_np, edge_types, pair_types
    # )

    results = {
        "precision": precision_score(labels_np, preds_np, zero_division=0),
        "recall": recall_score(labels_np, preds_np, zero_division=0),
        "f1": f1_score(labels_np, preds_np, zero_division=0),
    }

    os.makedirs(args.output_path, exist_ok=True)
    with open(f"{args.output_path}/results.txt", "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")

    fig_cm.savefig(f"{args.output_path}/confusion_matrix.png")
    fig_roc.savefig(f"{args.output_path}/roc_curve.png")
    fig_hist.savefig(f"{args.output_path}/probability_distribution.png")
    # fig_pair_types.savefig(f"{args.output_path}/non_canonical_pair_accuracy.png")



if __name__ == "__main__":
    args = parse_args()
    main(args)