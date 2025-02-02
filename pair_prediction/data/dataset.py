import torch
import random
import networkx as nx
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torch_geometric.utils import from_networkx

from pair_prediction.data.processing import create_rna_graph
from pair_prediction.data.read import read_idx_file, read_matrix_file


class LinkPredictionDataset(Dataset):
    def __init__(self, idx_dir: Path, matrix_dir: Path):
        self.idx_dir = idx_dir
        self.matrix_dir = matrix_dir
        self.file_names = [idx_file.stem for idx_file in self.idx_dir.glob("*.idx")]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]
        idx_file = self.idx_dir / f"{filename}.idx"
        amt_file = self.matrix_dir / f"{filename}.amt"

        seq, details = read_idx_file(idx_file)
        amt_matrix = read_matrix_file(amt_file)

        graph = create_rna_graph(seq, amt_matrix)

        noncanonical_edges = []
        for u, v, data in list(graph.edges(data=True)):
            if data.get("edge_type") == "non-canonical":
                noncanonical_edges.append((u, v))
                graph.remove_edge(u, v) # Mask to avoid message passing through non-canonical edges

        pyg_graph = from_networkx(graph)

        pos_edge_indices = torch.tensor(noncanonical_edges, dtype=torch.float32)
        neg_edge_indices = self.sample_negative_edges(pyg_graph, noncanonical_edges)

        return pyg_graph, pos_edge_indices, neg_edge_indices

    def sample_negative_edges(self, graph, pos_edges):
        """
        Samples negative edges for link prediction based on non-canonical edges.
        """
        nodes = list(range(graph.num_nodes))
        existing_edges = set()
        for u, v in pos_edges:
            existing_edges.add((u, v))
            existing_edges.add((v, u))

        neg_edges = []
        while len(neg_edges) < len(pos_edges):
            u, v = random.sample(nodes, 2)
            if (u, v) not in existing_edges and (v, u) not in existing_edges:
                neg_edges.append((u, v))
        neg_edge_indices = torch.tensor(neg_edges, dtype=torch.float32)
        return neg_edge_indices
