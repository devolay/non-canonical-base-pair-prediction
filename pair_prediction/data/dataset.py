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
        pyg_graph = from_networkx(graph)

        pos_edge_indices = torch.tensor(list(pyg_graph.edge_index.t().numpy()), dtype=torch.long)
        neg_edge_indices = self.sample_negative_edges(pyg_graph)

        return pyg_graph, pos_edge_indices, neg_edge_indices

    def sample_negative_edges(self, graph):
        """
        Samples negative edges for link prediction.
        """
        nodes = list(range(graph.num_nodes))
        existing_edges = set(tuple(edge) for edge in graph.edge_index.t().numpy())
        neg_edges = []
        while len(neg_edges) < len(existing_edges):
            u, v = random.sample(nodes, 2)
            if (u, v) not in existing_edges:
                neg_edges.append((u, v))

        neg_edge_indices = torch.tensor(neg_edges, dtype=torch.long)
        return neg_edge_indices
