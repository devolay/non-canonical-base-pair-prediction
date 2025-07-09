import torch
import torch.nn as nn

from typing import Optional
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse

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

def get_negative_edges(batched_data: Data, sample_ratio: Optional[int] = None, validation: bool = False) -> torch.Tensor:
    """
    Given a batched Data object (from PyG's DataLoader) that contains:
      - batched_data.ptr: a tensor of shape [num_graphs+1] indicating node boundaries.
      - batched_data.edge_index: Tensor of shape [2, total_edges] holding the positive edges.
      - batched_data.edge_type: a list (length = num_graphs) where each element is a list of edge types
            corresponding to the edges in that graph (aligned with batched_data.edge_index).
      
    For each graph, this function enumerates all possible candidate edges (excluding self-loops)
    and removes those that are either positive or involve a canonical relationship (i.e. have edge type "canonical"),
    returning a tensor of negative edge candidates.
    
    Returns:
        neg_edge_index: Tensor of shape [2, total_neg_edges] containing negative edge candidates.
    """
    neg_edges_list = []
    ptr = batched_data.ptr
    num_graphs = ptr.size(0) - 1

    pos_edge_masks = to_dense_adj(batched_data.edge_index, batch=batched_data.batch).squeeze(0).bool()

    for i in range(num_graphs):
        start = ptr[i].item()
        end = ptr[i+1].item()
        device = batched_data.batch.device
        sequence = batched_data.seq[i]

        pos_edge_mask = pos_edge_masks[i,:]
        pad_value = pos_edge_mask.size(-1) - len(sequence)
        pad = nn.ZeroPad2d((0, pad_value, 0, pad_value))
        
        all_pairings_matrix = create_pair_matrix(sequence, device=device)
        canonical_lookup_table = torch.zeros(16, dtype=torch.bool, device=device)
        canonical_lookup_table[CANONICAL_IDXS] = True
        candidate_mask = ~canonical_lookup_table[all_pairings_matrix]
        candidate_mask.fill_diagonal_(False)
        candidate_mask = pad(candidate_mask)
        candidate_mask &= ~pos_edge_mask

        row, col = torch.nonzero(candidate_mask, as_tuple=True)
        candidate_edge_index = torch.stack([row + start, col + start], dim=0) 
        
        if sample_ratio is not None and not validation:
            edge_types = batched_data.edge_type[i]
            num_neg_edges = candidate_edge_index.size(1)
            non_canonical_pairs_num = sum([1 for et in edge_types if et == "non-canonical"]) 
            num_samples = int(sample_ratio * non_canonical_pairs_num)
            if num_samples < num_neg_edges:
                indices = torch.randperm(num_neg_edges, device=device)[:num_samples]
                candidate_edge_index = candidate_edge_index[:, indices]
        neg_edges_list.append(candidate_edge_index)

    neg_edge_index = torch.cat(neg_edges_list, dim=1)
    return neg_edge_index
