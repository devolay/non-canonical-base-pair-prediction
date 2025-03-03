import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.nn import GCNConv, GATv2Conv, GAT


class LinkPredictorModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        gnn_channels: list = [64, 64],
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels (int): Number of input node features.
            gnn_channels (list of int): Hidden channels for the GNN encoder.
                E.g., [64, 64] creates two GAT layers:
                - First layer: in_channels -> 64.
                - Second layer: 64 -> 64.
            cnn_channels (list of int): Hidden channels for the CNN encoder.
                E.g., [64, 64] creates two residual blocks:
                - First block: in_channels -> 64.
                - Second block: 64 -> 64.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GAT(in_channels, gnn_channels[0]))
        for i in range(1, len(gnn_channels)):
            self.convs.append(GAT(gnn_channels[i-1], gnn_channels[i]))

        predictor_in_dim = 2 * gnn_channels[-1]
        self.link_predictor = nn.Sequential(
            nn.Linear(predictor_in_dim, gnn_channels[-1]),
            nn.ReLU(),
            nn.Linear(gnn_channels[-1], 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute node embeddings given a graph.
        """
        node_embeddings = x
        for conv in self.convs:
            node_embeddings = conv(node_embeddings, edge_index)
            node_embeddings = F.relu(node_embeddings)
            node_embeddings = F.dropout(node_embeddings, p=self.dropout, training=self.training)
        return node_embeddings

    def compute_edge_logits(
        self, node_embeddings: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logits for edges using node embeddings and an edge index tensor.
        """
        indices = edge_index.int().T
        src = indices[:, 0]
        dst = indices[:, 1]
        h_src = node_embeddings[src].squeeze(1)
        h_dst = node_embeddings[dst].squeeze(1)
        edge_features = torch.cat([h_src, h_dst], dim=1)
        logits = self.link_predictor(edge_features).squeeze(-1)
        return logits
