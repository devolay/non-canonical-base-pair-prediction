import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.nn import GCNConv
from pytorch_lightning.loggers import NeptuneLogger

class LitLinkPredictor(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3
    ):
        """
        Args:
            in_channels (int): Number of input node features.
            hidden_channels (int): Hidden layer size.
            num_layers (int): Number of GCN layers.
            dropout (float): Dropout probability.
            lr (float): Learning rate.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.link_predictor = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, data):
        """
        Compute node embeddings given a graph.
        """
        x, edge_index = data.features, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def compute_edge_logits(self, node_embeddings, edge_index):
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

    def training_step(self, batch, batch_idx):
        """
        Training step expects a tuple: (pyg_graph, pos_edge_indices, neg_edge_indices)
        """
        data, pos_edge_index, neg_edge_index = batch
        node_embeddings = self.forward(data)
        
        # Compute logits for positive and negative edges.
        pos_logits = self.compute_edge_logits(node_embeddings, pos_edge_index)
        neg_logits = self.compute_edge_logits(node_embeddings, neg_edge_index)
        
        # Create binary labels.
        pos_labels = torch.ones(pos_logits.size(0), device=self.device, dtype=torch.float32)
        neg_labels = torch.zeros(neg_logits.size(0), device=self.device, dtype=torch.float32)
        
        # Concatenate logits and labels.
        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step computing loss and accuracy.
        """
        data, pos_edge_index, neg_edge_index = batch
        node_embeddings = self.forward(data)
        pos_logits = self.compute_edge_logits(node_embeddings, pos_edge_index)
        neg_logits = self.compute_edge_logits(node_embeddings, neg_edge_index)
        
        pos_labels = torch.ones(pos_logits.size(0), device=self.device)
        neg_labels = torch.zeros(neg_logits.size(0), device=self.device)
        
        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        preds = (torch.sigmoid(all_logits) > 0.5).float()
        acc = (preds == all_labels).float().mean()
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
