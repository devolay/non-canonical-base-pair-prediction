import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

from pair_prediction.model.residual_block import ResidualBlock1d


class LinkPredictorGlobalModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        gnn_channels: list = [64, 64],
        cnn_channels: list = [64, 64],
        dropout: float = 0.0,
    ):
        """
        Args:
            in_channels (int): Number of input node features.
            gnn_channels (list of int): Hidden channel sizes for the GNN encoder layers.
                For example, [64, 64] creates a GNN with two layers:
                - First layer: input in_channels, output 64.
                - Second layer: input 64, output 64.
            cnn_channels (list of int): Hidden channel sizes for the CNN encoder layers.
                For example, [64, 64] creates a CNN with two residual blocks:
                - First block: input in_channels, output 64.
                - Second block: input 64, output 64.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.dropout = dropout

        self.gnn_convs = nn.ModuleList()
        self.gnn_convs.append(GCNConv(in_channels, gnn_channels[0]))
        for i in range(1, len(gnn_channels)):
            self.gnn_convs.append(GCNConv(gnn_channels[i-1], gnn_channels[i]))

        cnn_blocks = []
        cnn_blocks.append(ResidualBlock1d(in_channels, cnn_channels[0]))
        for i in range(1, len(cnn_channels)):
            cnn_blocks.append(ResidualBlock1d(cnn_channels[i-1], cnn_channels[i]))
        self.cnn_encoder = nn.Sequential(*cnn_blocks)

        predictor_in_dim = 2 * gnn_channels[-1] + cnn_channels[-1]
        self.link_predictor = nn.Sequential(
            nn.Linear(predictor_in_dim, gnn_channels[-1]),
            nn.ReLU(),
            nn.Linear(gnn_channels[-1], 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """
        Forward pass:
          - Computes node embeddings with the GNN encoder.
          - Computes a global graph representation with the CNN encoder.
          
        Args:
            x (torch.Tensor): Node feature matrix of shape [total_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity information.
            batch (torch.Tensor): Batch vector that assigns each node to a graph.
            
        Returns:
            node_embeddings (torch.Tensor): Output from GNN encoder.
            global_reps (torch.Tensor): Global representations from CNN encoder.
        """
        node_embeddings = x
        for conv in self.gnn_convs:
            node_embeddings = conv(node_embeddings, edge_index)
            node_embeddings = F.relu(node_embeddings)
            node_embeddings = F.dropout(node_embeddings, p=self.dropout, training=self.training)
        
        # CNN encoder: create a dense batch for the global graph representation.
        # x_dense shape: [B, N, in_channels] where N is the maximum number of nodes per graph.
        x_dense, mask = to_dense_batch(x, batch)
        # Transpose to [B, in_channels, N] for Conv1d input.
        x_dense = x_dense.transpose(1, 2)
        features = self.cnn_encoder(x_dense)
        # Aggregate over the node dimension (e.g., using mean pooling).
        global_reps = features.mean(dim=2)  # shape: [B, cnn_channels[-1]]
        
        return node_embeddings, global_reps

    def compute_edge_logits(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        global_reps: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute logits for edges by concatenating local node embeddings (src & dst)
        with the corresponding global representation.

        Args:
            node_embeddings (torch.Tensor): Node embeddings [total_nodes, hidden_channels].
            edge_index (torch.Tensor): Tensor of edge indices.
            global_reps (torch.Tensor): Global representations [batch_size, hidden_channels].
            batch (torch.Tensor): Batch vector assigning each node to a graph.

        Returns:
            logits (torch.Tensor): Logits for each edge.
        """
        # Transpose edge_index to shape [num_edges, 2]
        indices = edge_index.int().T
        src = indices[:, 0]
        dst = indices[:, 1]

        h_src = node_embeddings[src]
        h_dst = node_embeddings[dst]

        edge_batch = batch[src]
        global_rep_edges = global_reps[edge_batch]

        edge_features = torch.cat([h_src, h_dst, global_rep_edges], dim=1)
        logits = self.link_predictor(edge_features).squeeze(-1)
        return logits
