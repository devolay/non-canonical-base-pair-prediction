import torch

import torch


def prepare_val_negative_edges(batched_data):
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
        end = ptr[i + 1].item()
        device = batched_data.batch.device
        nodes = torch.arange(start, end, device=device)

        # Generate all possible candidate edges for this graph (excluding self-loops).
        u, v = torch.meshgrid(nodes, nodes, indexing="ij")
        u = u.reshape(-1)
        v = v.reshape(-1)
        candidate_edges = torch.stack([u, v], dim=0)
        candidate_edges = candidate_edges[:, candidate_edges[0] != candidate_edges[1]]

        # Extract the positive edges for the current graph (phosphodiester or canonical).
        mask_pos = (pos_edge_index[0] >= start) & (pos_edge_index[0] < end)
        pos_edges_i = pos_edge_index[:, mask_pos]
        pos_list = [tuple(edge) for edge in pos_edges_i.t().tolist()]
        pos_set = set(pos_list)

        # Get the edge types for these positive edges.
        edge_types_i = batched_data.edge_type[i]

        # Identify all nodes that are in a canonical relationship.
        canonical_nodes = set()
        for edge, et in zip(pos_list, edge_types_i):
            if et == "canonical":
                canonical_nodes.update(edge)  # add both nodes of the edge

        # From the candidate set, remove any edge that contains a node in canonical_nodes.
        candidate_list = [tuple(edge) for edge in candidate_edges.t().tolist()]
        filtered_candidates = {
            edge
            for edge in candidate_list
            if edge[0] not in canonical_nodes and edge[1] not in canonical_nodes
        }

        # Remove any candidate edge that is already positive.
        neg_set = filtered_candidates - pos_set
        neg_edges = (
            torch.tensor(list(neg_set), dtype=torch.long, device=batched_data.batch.device)
            .t()
            .contiguous()
        )
        neg_edges_list.append(neg_edges)
        neg_edge_index = torch.cat(neg_edges_list, dim=1)

    return neg_edge_index
