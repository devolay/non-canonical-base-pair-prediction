import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl

from torch_geometric.utils import batched_negative_sampling

from pair_prediction.model.model import LinkPredictorModel

class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
    ):
        """
        LightningModule that wraps the LinkPredictorModel.

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
        self.model = LinkPredictorModel(in_channels, hidden_channels, num_layers, dropout)

    def forward(self, data):
        """
        Forward pass using the underlying LinkPredictorModel.
        """
        return self.model(data)
    
    def _step(self, batch, batch_idx):
        """
        Shared step for training and validation.
        """
        edge_mask = np.concatenate(batch.edge_type)
        edge_mask = edge_mask == 'non-canonical'
        
        pos_edge_index = batch.edge_index[:, edge_mask]
        neg_edge_index = batched_negative_sampling(
            batch.edge_index, batch=batch.batch, num_neg_samples=pos_edge_index.size(1)
        )
        message_passing_edge_index = batch.edge_index[:, ~edge_mask]

        node_embeddings = self.model(batch.features, message_passing_edge_index)
        pos_logits = self.model.compute_edge_logits(node_embeddings, pos_edge_index)
        neg_logits = self.model.compute_edge_logits(node_embeddings, neg_edge_index)

        pos_labels = torch.ones(pos_logits.size(0), device=self.device, dtype=torch.float32)
        neg_labels = torch.zeros(neg_logits.size(0), device=self.device, dtype=torch.float32)

        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        return all_logits, all_labels

    def training_step(self, batch, batch_idx):
        """
        Training step computing loss for link prediction.
        """
        all_logits, all_labels = self._step(batch, batch_idx)
        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step computing loss and accuracy.
        """
        all_logits, all_labels = self._step(batch, batch_idx)
        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        preds = (torch.sigmoid(all_logits) > 0.5).float()
        acc = (preds == all_labels).float().mean()
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
