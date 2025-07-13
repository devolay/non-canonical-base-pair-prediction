import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, List, Tuple
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from pair_prediction.model.rinalmo_link_predictor_2d import RiNAlmoLinkPredictionModel

BASE2IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
PAIR2IDX = {
    'AA': 0, 'AC': 1, 'AG': 2, 'AU': 3,
    'CA': 4, 'CC': 5, 'CG': 6, 'CU': 7,
    'GA': 8, 'GC': 9, 'GG': 10, 'GU': 11,
    'UA': 12, 'UC': 13, 'UG': 14, 'UU': 15
}
CANONICAL_IDXS = [3, 6, 9, 12, 11, 14] 

def create_pair_matrix(seq: str, device=None):
    """Generate a pair matrix for a given sequence."""
    idx = torch.tensor([BASE2IDX[b] for b in seq], device=device)
    return 4 * idx[:, None] + idx[None, :]


def enumerate_negative_candidates(batched_data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Enumerate *all* negative-edge candidates in the mini-batch.

    Returns
    -------
    edge_index_all : Tensor[2, M]
        Concatenated negative edge indices for the whole batch.
    graph_ids      : Tensor[M]
        graph_ids[j] == g  ⇒  edge_index_all[:, j] belongs to graph g.
    """
    ptr = batched_data.ptr          # [G+1]
    num_graphs = ptr.numel() - 1
    device = batched_data.batch.device
    pos_adj = to_dense_adj(batched_data.edge_index, batch=batched_data.batch).squeeze(0).bool()

    neg_edges:   List[torch.Tensor] = []
    graph_ids:   List[torch.Tensor] = []

    for g in range(num_graphs):
        start, end = ptr[g].item(), ptr[g + 1].item()
        seq  = batched_data.seq[g]

        # positive mask → square matrix (pad to max-len)
        pos_mask  = pos_adj[g]
        pad_size  = pos_mask.size(-1) - len(seq)
        pad  = nn.ZeroPad2d((0, pad_size, 0, pad_size))

        # canonical / self-loop removal
        pair_matrix = create_pair_matrix(seq, device=device)                 # [L,L]
        canonical_table = torch.zeros(16, device=device, dtype=torch.bool)
        canonical_table[CANONICAL_IDXS] = True
        candidate = ~canonical_table[pair_matrix]
        candidate.fill_diagonal_(False)
        candidate = pad(candidate)
        candidate &= ~pos_mask

        row, col = torch.nonzero(candidate, as_tuple=True)
        idx = torch.stack([row + start, col + start], dim=0)           # [2,⋯]

        neg_edges.append(idx)
        graph_ids.append(torch.full((idx.size(1),), g, device=device, dtype=torch.long))

    edge_index_all = torch.cat(neg_edges, dim=1)   # [2, M]
    graph_ids      = torch.cat(graph_ids, dim=0)   # [M]
    return edge_index_all, graph_ids


def sample_negative_edges(
    candidate_edge_index: torch.Tensor,
    edge_types: List[str],
    sample_ratio: float,
    device: torch.device,
    edge_losses: Optional[torch.Tensor] = None,
    mode: str = "uniform",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Down-sample negatives for **one** graph.  If `edge_losses` is given the
    `"weighted"` strategy favours hard negatives (prob ∝ exp(loss/T)).
    """
    N = candidate_edge_index.size(1)
    pos_noncanon = sum(et == "non-canonical" for et in edge_types)
    keep = int(sample_ratio * pos_noncanon)

    if keep == 0 or keep >= N:
        return candidate_edge_index

    if mode == "weighted" and edge_losses is not None:
        probs = torch.softmax(edge_losses / temperature, dim=0)
        idx   = torch.multinomial(probs, keep, replacement=False)
    else:
        idx = torch.randperm(N, device=device)[:keep]

    return candidate_edge_index[:, idx]


def get_negative_edges(
    batched_data: Data,
    sample_ratio: Optional[float] = None,
    validation: bool = False,
    hard_negative_sampling: bool = False,
    model: Optional["RiNAlmoLinkPredictionModel"] = None,
    node_embeddings: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Return a (possibly sampled) negative-edge tensor for the mini-batch.
    """
    # 1. full pool  ────────────────────────────────────────────────────────────
    edge_idx_all, graph_ids = enumerate_negative_candidates(batched_data)

    # If no sampling requested (ratio=None) or running validation → done
    if sample_ratio is None or validation:
        return edge_idx_all

    # 2. score once for *all* negatives  ───────────────────────────────────────
    edge_losses_all = None
    if hard_negative_sampling:
        if model is None or node_embeddings is None:
            raise ValueError("hard_negative_sampling=True requires `model` and ""`node_embeddings`.")
        with torch.no_grad():  # logits used only for sampling
            logits = model.compute_edge_logits(node_embeddings, edge_idx_all, batched_data.batch)
        edge_losses_all = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="none"
        )

    # 3. per-graph sampling  ──────────────────────────────────────────────────
    sampled: List[torch.Tensor] = []
    num_graphs = batched_data.ptr.numel() - 1
    device = batched_data.batch.device

    for g in range(num_graphs):
        mask         = graph_ids == g
        cand_edges   = edge_idx_all[:, mask]
        losses_g     = edge_losses_all[mask] if edge_losses_all is not None else None
        mode         = "weighted" if losses_g is not None else "uniform"

        neg_idx_g = sample_negative_edges(
            cand_edges,
            batched_data.edge_type[g],
            sample_ratio,
            device,
            edge_losses=losses_g,
            mode=mode,
            temperature=1.0,
        )
        sampled.append(neg_idx_g)

    return torch.cat(sampled, dim=1)
