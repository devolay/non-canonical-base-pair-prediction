from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict, Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from pair_prediction.model.rinalmo_link_predictor_2d import RiNAlmoLinkPredictionModel
from pair_prediction.model.utils import get_negative_edges
from pair_prediction.data.utils import load_dataset


def _compute_prf1(tp: int, fp: int, fn: int):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def _node_counts_by_graph(node_to_graph: torch.Tensor) -> Dict[int, int]:
    return dict(Counter(node_to_graph.tolist()))


def _extract_graph_meta(batch, g: int) -> Dict[str, str]:
    """
    Minimal best-effort metadata extraction.
    """
    fields = ["id", "seq"]
    meta = {k: "Unknown" for k in fields}
    num_graphs = int(batch.num_graphs)
    
    for k in fields:
        if hasattr(batch, k):
            val = getattr(batch, k)
            if isinstance(val, (list, tuple)) and len(val) == num_graphs:
                meta[k] = str(val[g]) if val[g] is not None else "Unknown"
            elif isinstance(val, torch.Tensor) and val.ndim == 1 and len(val) == num_graphs:
                meta[k] = str(val[g].item())
    return meta


@torch.no_grad()
def evaluate_case_study_stats(
    model,
    dataset,
    device: torch.device,
    *,
    batch_size: int = 256,
    threshold: float = 0.5,
) -> List[Dict[str, Any]]:
    """
    Returns a list of dicts per graph with:
      pdb_id, rna_type, family, resolution, organism,
      n_nucleotides, n_canonical, n_noncanonical,
      predicted_noncanonical, TP, TN, FP, FN, precision, recall, f1
    """
    model.to(device)
    model.eval()

    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    results: List[Dict[str, Any]] = []

    for batch in dl:
        batch = batch.to(device)

        # Ground-truth edge info
        edge_type_all = np.concatenate(batch.edge_type)  # array length = E_all (GT edges)
        noncan_mask = (edge_type_all == "non-canonical")

        # Message-passing edges exclude non-canonical (as in your eval)
        mp_edge_index = batch.edge_index[:, ~noncan_mask]

        # RiNALMo tokens + embeddings
        rna_tokens = torch.tensor(model.tokenizer.batch_tokenize(batch.seq), device=device)
        with torch.cuda.amp.autocast():
            node_embeddings = model(batch.features, rna_tokens, mp_edge_index)

        # Positive (GT non-canonical) edges + logits/labels
        pos_edge_index = batch.edge_index[:, noncan_mask]
        pos_logits = model.compute_edge_logits(node_embeddings, pos_edge_index, batch.batch)
        pos_probs = torch.sigmoid(pos_logits)
        pos_preds = (pos_probs > threshold)

        # Negative edges + logits/labels
        neg_edge_index = get_negative_edges(batch, validation=True)
        neg_logits = model.compute_edge_logits(node_embeddings, neg_edge_index, batch.batch)
        neg_probs = torch.sigmoid(neg_logits)
        neg_preds = (neg_probs > threshold)

        # Map edges to graphs (use source node's graph id)
        node_to_graph = batch.batch  # [N]
        pos_graph_ids = node_to_graph[pos_edge_index[0]]  # [E_pos]
        neg_graph_ids = node_to_graph[neg_edge_index[0]]  # [E_neg]

        # Canonical / non-canonical counts per graph from GT edges
        gt_graph_ids = node_to_graph[batch.edge_index[0]].cpu().tolist()
        n_can_per_g = defaultdict(int)
        n_non_per_g = defaultdict(int)
        for et, g in zip(edge_type_all, gt_graph_ids):
            if et == "canonical":
                n_can_per_g[g] += 1
            elif et == "non-canonical":
                n_non_per_g[g] += 1

        # Nucleotides per graph
        n_nodes_per_g = _node_counts_by_graph(node_to_graph)

        # Confusion counts per graph
        tp_per_g = defaultdict(int); fn_per_g = defaultdict(int)
        tn_per_g = defaultdict(int); fp_per_g = defaultdict(int)

        for g, pred in zip(pos_graph_ids.tolist(), pos_preds.tolist()):
            if pred: tp_per_g[g] += 1
            else:    fn_per_g[g] += 1

        for g, pred in zip(neg_graph_ids.tolist(), neg_preds.tolist()):
            if pred: fp_per_g[g] += 1
            else:    tn_per_g[g] += 1

        # Predicted non-canonical per graph (predicted positives across pos+neg)
        pred_non_per_g = defaultdict(int)
        for g, pred in zip(pos_graph_ids.tolist(), pos_preds.tolist()):
            if pred: pred_non_per_g[g] += 1
        for g, pred in zip(neg_graph_ids.tolist(), neg_preds.tolist()):
            if pred: pred_non_per_g[g] += 1

        # Emit records
        graphs_in_batch = set(n_nodes_per_g.keys())
        for g in sorted(graphs_in_batch):
            tp = tp_per_g[g]; tn = tn_per_g[g]; fp = fp_per_g[g]; fn = fn_per_g[g]
            prec, rec, f1 = _compute_prf1(tp, fp, fn)
            meta = _extract_graph_meta(batch, g)

            results.append({
                "id": meta["id"],
                "seq": meta["seq"],
                "n_nucleotides": int(n_nodes_per_g.get(g, 0)),
                "n_canonical": int(n_can_per_g.get(g, 0)),
                "n_noncanonical": int(n_non_per_g.get(g, 0)),
                "predicted_noncanonical": int(pred_non_per_g.get(g, 0)),
                "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
                "precision": float(prec), "recall": float(rec), "f1": float(f1),
            })

    return results


def _write_json(path: str, data: Any):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _write_csv(path: str, rows: List[Dict[str, Any]]):
    import csv
    fieldnames = list(rows[0].keys()) if rows else [
        "pdb_id","rna_type","family","resolution","organism",
        "n_nucleotides","n_canonical","n_noncanonical","predicted_noncanonical",
        "TP","TN","FP","FN","precision","recall","f1"
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Per-graph case-study stats (Top-K best / Bottom-K worst by F1).")
    p.add_argument("--batch-size", type=int, default=1, help="Eval batch size.")
    p.add_argument("--model-path", type=str, default="/workspace/non-canonical-base-pair-prediction/models/model.ckpt",)
    p.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for positive class.")
    p.add_argument("--out-csv", type=str, default=None, help="Path to write ALL results as CSV.")
    p.add_argument("--out-json", type=str, default=None, help="Path to write ALL results as JSON.")
    p.add_argument("--k", type=int, default=5, help="K for top/bottom selection.")
    p.add_argument("--print", dest="do_print", action="store_true",
                   help="Print compact Top-K and Bottom-K tables to stdout.")
    p.add_argument("remainder", nargs=argparse.REMAINDER,
                   help="Optional '-- key=value ...' forwarded to loader as kwargs.")
    args = p.parse_args(argv)

    # Parse forwarded kwargs after "--"
    loader_kwargs: Dict[str, Any] = {}
    if args.remainder:
        toks = [t for t in args.remainder if t != "--"]
        key = None
        for t in toks:
            if t.startswith("--"):
                key = t.lstrip("-").replace("-", "_")
                loader_kwargs[key] = True  # bare flag
            else:
                if key is None:
                    raise SystemExit(f"Unexpected token without key: {t}")
                loader_kwargs[key] = t
                key = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiNAlmoLinkPredictionModel(
        in_channels=1280,
        gnn_channels=[1280, 512, 256, 128],
        cnn_head_embed_dim=64,
        cnn_head_num_blocks=3
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    checkpoint['state_dict'] = {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(checkpoint['state_dict'])

    dataset = load_dataset("ts_combined")

    stats = evaluate_case_study_stats(
        model, dataset, device,
        batch_size=args.batch_size,
        threshold=args.threshold
    )

    by_f1_desc = sorted(stats, key=lambda r: r["f1"], reverse=True)
    by_f1_asc  = list(reversed(by_f1_desc))
    top_k = by_f1_desc[: args.k] if args.k > 0 else []
    bottom_k = by_f1_asc[: args.k] if args.k > 0 else []

    if args.out_json:
        _write_json(args.out_json, stats)
        base, *ext = args.out_json.rsplit(".", 1)
        ext = ext[0] if ext else "json"
        _write_json(f"{base}_top{args.k}.{ext}", top_k)
        _write_json(f"{base}_bottom{args.k}.{ext}", bottom_k)

    if args.out_csv:
        _write_csv(args.out_csv, stats)
        base, *ext = args.out_csv.rsplit(".", 1)
        ext = ext[0] if ext else "csv"
        _write_csv(f"{base}_top{args.k}.{ext}", top_k)
        _write_csv(f"{base}_bottom{args.k}.{ext}", bottom_k)

    if args.do_print:
        cols = ["pdb_id","n_nucleotides","n_canonical","n_noncanonical",
                "predicted_noncanonical","TP","TN","FP","FN","precision","recall","f1"]
        def fmt(x): return f"{x:.3f}" if isinstance(x, float) else str(x)

        def print_block(title: str, rows: List[Dict[str, Any]]):
            print(f"\n=== {title} ===")
            print("\t".join(cols))
            for r in rows:
                print("\t".join(fmt(r.get(c, "")) for c in cols))

        print_block(f"Top-{args.k} by F1", top_k)
        print_block(f"Bottom-{args.k} by F1", bottom_k)

    return 0


if __name__ == "__main__":
    sys.exit(main())
