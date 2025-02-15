import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.nn import GCNConv


class LinkPredictorModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels (int): Number of input node features.
            hidden_channels (int): Hidden layer size.
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.link_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute node embeddings given a graph.
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def compute_edge_logits(self, node_embeddings: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
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
