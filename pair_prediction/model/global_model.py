import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch

class LinkPredictorGlobalModel(nn.Module):
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

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels , kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
        )

        self.link_predictor = nn.Sequential(
            nn.Linear(3 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor):
        """
        Compute node embeddings using GCN layers and global representations using the CNN encoder.
        
        Args:
            x (torch.Tensor): Node feature matrix of shape [total_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity.
            batch (torch.Tensor): Batch vector assigning each node to a graph.
            
        Returns:
            node_embeddings (torch.Tensor): Local node embeddings.
            global_reps (torch.Tensor): Global representation for each graph in the batch.
            batch (torch.Tensor): The original batch vector (passed along for edge-level processing).
        """
        node_embeddings = x
        for conv in self.convs:
            node_embeddings = conv(node_embeddings, edge_index)
            node_embeddings = F.relu(node_embeddings)
            node_embeddings = F.dropout(node_embeddings, p=self.dropout, training=self.training)

        x_dense, mask = to_dense_batch(x, batch)  
        x_dense = x_dense.transpose(1, 2)
        global_reps = self.cnn_encoder(x_dense).squeeze(-1)

        return node_embeddings, global_reps

    def compute_edge_logits(
        self, 
        node_embeddings: torch.Tensor, 
        edge_index: torch.Tensor, 
        global_reps: torch.Tensor,
        batch: torch.Tensor
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