from __future__ import annotations
import argparse, pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import Batch

from pair_prediction.constants import BASE_DIR
from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.model.rinalmo_link_predictor_2d import RiNAlmoLinkPredictionModel
from pair_prediction.model.utils import enumerate_negative_candidates


def load_model(ckpt_path: pathlib.Path, device: torch.device) -> RiNAlmoLinkPredictionModel:
    model = RiNAlmoLinkPredictionModel(
        in_channels=1280,
        gnn_channels=[1280, 512, 256, 128],
        cnn_head_embed_dim=64,
        cnn_head_num_blocks=3,
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt["state_dict"] = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


def plot_single_graph(
    batch,                      # PyG Batch with exactly one graph
    model,
    device,
    temperature: float,
) -> None:
    edge_mask = np.concatenate(batch.edge_type) == "non-canonical"
    msg_edge_index = batch.edge_index[:, ~edge_mask]

    rna_tokens = torch.tensor(model.tokenizer.batch_tokenize(batch.seq), device=device)
    with torch.cuda.amp.autocast():
        node_emb = model(
            batch.features.to(device),
            rna_tokens,
            msg_edge_index.to(device),
        )

    neg_edge_index, _ = enumerate_negative_candidates(batch)
    logits = model.compute_edge_logits(node_emb, neg_edge_index, batch.batch)
    losses = F.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits), reduction="none"
    ).detach().cpu().numpy()

    row, col = neg_edge_index.detach().cpu().numpy()
    L = batch.num_nodes
    heat = np.full((L, L), np.nan, dtype=np.float32)
    heat[row, col] = losses

    probs = torch.softmax(torch.tensor(losses) / temperature, dim=0).numpy()

    seq_str = batch.seq[0] if isinstance(batch.seq, (list, tuple)) else batch.seq
    bases   = list(seq_str)
    assert len(bases) == L, "sequence length ≠ num_nodes"

    step = 1 if L <= 50 else max(1, L // 50)

    fig_h, ax_h = plt.subplots(figsize=(8, 8), dpi=350)

    im = ax_h.imshow(heat, cmap="viridis", origin="upper")
    ax_h.set_title("Negative-edge BCE loss heat-map", pad=12)
    ax_h.set_xlabel("Node j  (sequence order)")
    ax_h.set_ylabel("Node i  (sequence order)")

    ax_h.set_xticks(np.arange(0, L, step))
    ax_h.set_yticks(np.arange(0, L, step))
    ax_h.set_xticklabels(bases[::step], rotation=90, fontsize=6)
    ax_h.set_yticklabels(bases[::step], fontsize=6)

    cbar = plt.colorbar(im, ax=ax_h, fraction=0.045, pad=0.04)
    cbar.set_label("BCE loss")

    plt.tight_layout()
    fig_h.savefig(f"{batch.id}_heatmap.png", dpi=350)
    plt.close(fig_h)


    fig_d, ax1 = plt.subplots(figsize=(10, 5), dpi=350)

    ax1.hist(losses, bins=80, density=True, alpha=0.65, label="Loss density")
    ax1.set_xlabel("Per-edge BCE loss")
    ax1.set_ylabel("Density")
    ax1.set_title("Loss distribution vs. sampling probability")
    ax1.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    ax2 = ax1.twinx()
    idx = np.argsort(losses)
    ax2.plot(losses[idx], probs[idx], color="tab:red", label=f"Sampling probability (T={temperature})")
    ax2.set_ylabel("Sampling probability")
    ax2.set_ylim(0, probs.max() * 1.1)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="upper left")

    plt.tight_layout()
    fig_d.savefig(f"{batch.id}_loss_hist.png", dpi=350)
    plt.close(fig_d)

    print(f"Saved: {batch.id}_heatmap.png  and  {batch.id}_loss_hist.png")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ds = LinkPredictionDataset(pathlib.Path(args.data_root), mode="validation")
    g_idx = args.graph_idx
    if g_idx < 0 or g_idx >= len(ds):
        raise ValueError(f"--graph-idx must be in [0, {len(ds)-1}]")
    data = ds[g_idx].to(device)
    batch = Batch.from_data_list([data])

    model = load_model(pathlib.Path(args.checkpoint), device)
    plot_single_graph(batch, model, device, args.temperature)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Plot losses for one validation graph.")
    p.add_argument("--data-root", type=str, default=str(BASE_DIR / "data"))
    p.add_argument("--checkpoint", type=str, required=True, help="Path to model .ckpt")
    p.add_argument("--graph-idx", type=int, default=0, help="Which validation graph to plot")
    p.add_argument("--temperature", type=float, default=1.0, help="Soft-max temperature")
    return p


if __name__ == "__main__":
    main(build_parser().parse_args())
