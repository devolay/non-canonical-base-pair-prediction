import torch

from typing import Optional
from torch_geometric.data import Data

def get_negative_edges(batched_data: Data, sample_ratio: Optional[int] = None, consider_multiplets: bool = False, validation: bool = False) -> torch.Tensor:
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
    pos_edge_index = batched_data.edge_index
    num_graphs = ptr.size(0) - 1

    for i in range(num_graphs):
        # Determine node boundaries for the current graph.
        start = ptr[i].item()
        end = ptr[i+1].item()
        device = batched_data.batch.device
        nodes = torch.arange(start, end, device=device)

        # Generate all possible candidate edges for this graph (excluding self-loops).
        u, v = torch.meshgrid(nodes, nodes, indexing='ij')
        u = u.reshape(-1)
        v = v.reshape(-1)
        candidate_edges = torch.stack([u, v], dim=0)
        candidate_edges = candidate_edges[:, candidate_edges[0] != candidate_edges[1]]
        candidate_edges = {tuple(edge) for edge in candidate_edges.t().tolist()}
        
        # Extract the positive edges for the current graph (phosphodiester or canonical).
        mask_pos = (pos_edge_index[0] >= start) & (pos_edge_index[0] < end)
        pos_edges_i = pos_edge_index[:, mask_pos]
        pos_list = [tuple(edge) for edge in pos_edges_i.t().tolist()]
        pos_set = set(pos_list)
        
        # Get the edge types for these positive edges.
        edge_types_i = batched_data.edge_type[i]
        
        # Remove any candidate edge that is already positive.
        neg_set = candidate_edges - pos_set

        # Identify all potential multiplet edges.
        if consider_multiplets:
            canonical_nodes = set()
            for edge, et in zip(pos_list, edge_types_i):
                if et == "canonical":
                    canonical_nodes.update(edge)  # add both nodes of the edge
            
            multiplet_edges = {
                edge for edge in candidate_edges 
                if (edge[0] in canonical_nodes or edge[1] in canonical_nodes) and edge not in pos_set
            }
            neg_set += multiplet_edges
        
        neg_edges = torch.tensor(list(neg_set), dtype=torch.long, device=batched_data.batch.device).t().contiguous()
        
        if sample_ratio is not None and not validation:
            non_canonical_pairs = sum([1 for et in edge_types_i if et == "non-canonical"]) 
            num_neg_edges = len(neg_set)
            num_samples = int(sample_ratio * non_canonical_pairs)
            if num_samples < num_neg_edges:
                indices = torch.randperm(num_neg_edges, device=device)[:num_samples]
                neg_edges = neg_edges[:, indices]
        neg_edges_list.append(neg_edges)

    neg_edge_index = torch.cat(neg_edges_list, dim=1)
    return neg_edge_index
