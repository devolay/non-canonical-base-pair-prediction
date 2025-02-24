import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from torch_geometric.utils import batched_negative_sampling
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score

from pair_prediction.model.model import LinkPredictorModel
from pair_prediction.model.global_model import LinkPredictorGlobalModel
from pair_prediction.model.utils import prepare_val_negative_edges

class LitWrapper(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        lr: float = 1e-3,
        model_type: str = "local",
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
        self.model_type = model_type

        if model_type == "local":
            self.model = LinkPredictorModel(in_channels, hidden_channels, num_layers, dropout)
        elif model_type == "global":
            self.model = LinkPredictorGlobalModel(in_channels, hidden_channels, num_layers, dropout)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.val_outputs = []

    def forward(self, data):
        """
        Forward pass using the underlying LinkPredictorModel.
        """
        return self.model(data)    

    def training_step(self, batch, batch_idx):
        """
        Training step computing loss for link prediction.
        """
        edge_mask = np.concatenate(batch.edge_type)
        edge_mask = edge_mask == 'non-canonical'

        message_passing_edge_index = batch.edge_index[:, ~edge_mask]

        if self.model_type == "global":
            node_embeddings, global_representation = self.model(batch.features, message_passing_edge_index, batch.batch)
        else:
            node_embeddings = self.model(batch.features, message_passing_edge_index)

        pos_edge_index = batch.edge_index[:, edge_mask]

        if self.model_type == "global":
            pos_logits = self.model.compute_edge_logits(node_embeddings, pos_edge_index, global_representation, batch.batch)
        else:
            pos_logits = self.model.compute_edge_logits(node_embeddings, pos_edge_index)

        pos_labels = torch.ones(pos_logits.size(0), device=self.device, dtype=torch.float32)

        num_neg_samples = pos_edge_index.size(1) // (len(batch.ptr) - 1)
        neg_edge_index = batched_negative_sampling(
            batch.edge_index, batch=batch.batch, num_neg_samples=num_neg_samples
        )
        if self.model_type == "global":
            neg_logits = self.model.compute_edge_logits(node_embeddings, neg_edge_index, global_representation, batch.batch)
        else:
            neg_logits = self.model.compute_edge_logits(node_embeddings, neg_edge_index)

        neg_labels = torch.zeros(neg_logits.size(0), device=self.device, dtype=torch.float32)

        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)

        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step computing loss, accuracy, and collects predictions,
        labels, and probabilities for logging.
        """
        edge_mask = np.concatenate(batch.edge_type)
        edge_mask = edge_mask == 'non-canonical'
        
        message_passing_edge_index = batch.edge_index[:, ~edge_mask]
        if self.model_type == "global":
            node_embeddings, global_representation = self.model(batch.features, message_passing_edge_index, batch.batch)
        else:
            node_embeddings = self.model(batch.features, message_passing_edge_index)

        pos_edge_index = batch.edge_index[:, edge_mask]
        if self.model_type == "global":
            pos_logits = self.model.compute_edge_logits(node_embeddings, pos_edge_index, global_representation, batch.batch)
        else:
            pos_logits = self.model.compute_edge_logits(node_embeddings, pos_edge_index)
        pos_labels = torch.ones(pos_logits.size(0), device=self.device, dtype=torch.float32)

        neg_edge_index = prepare_val_negative_edges(batch)
        if self.model_type == "global":
            neg_logits = self.model.compute_edge_logits(node_embeddings, neg_edge_index, global_representation, batch.batch)
        else:
            neg_logits = self.model.compute_edge_logits(node_embeddings, neg_edge_index)
        neg_labels = torch.zeros(neg_logits.size(0), device=self.device, dtype=torch.float32)
        
        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)

        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        probabilities = torch.sigmoid(all_logits)
        preds = (probabilities > 0.5).long()
        labels = all_labels.long()

        self.val_outputs.append({
            "preds": preds.detach(),
            "labels": labels.detach(),
            "probabilities": probabilities.detach()
        })
        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.val_outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in self.val_outputs], dim=0)
        all_probabilities = torch.cat([x["probabilities"] for x in self.val_outputs], dim=0)

        self._log_validation(all_preds, all_labels, all_probabilities)
        self.val_outputs.clear()
        
    def _log_validation(self, preds, labels, probabilities):
        """
        Logs the confusion matrix, ROC curve (with AUC), and probability distribution plots.
        Uses Neptune logger's log_image method.
        """
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        prob_np = probabilities.cpu().numpy()
        
        # Confusion Matrix.
        fig_cm, ax_cm = plt.subplots()
        cm = confusion_matrix(labels_np, preds_np)
        cax = ax_cm.matshow(cm, cmap='Blues')
        fig_cm.colorbar(cax)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        ax_cm.set_title('Confusion Matrix')
        for (i, j), z in np.ndenumerate(cm):
            ax_cm.text(j, i, f"{z}", ha='center', va='center')
        
        # ROC Curve and AUC.
        fpr, tpr, _ = roc_curve(labels_np, prob_np)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.2f})")
        ax_roc.plot([0, 1], [0, 1], "k--")
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend(loc="lower right")
        
        # Probability Distribution.
        fig_hist, ax_hist = plt.subplots()
        pos_probs = prob_np[labels_np == 1]
        neg_probs = prob_np[labels_np == 0]
        ax_hist.hist(pos_probs, bins=20, alpha=0.7, label="Positive")
        ax_hist.hist(neg_probs, bins=20, alpha=0.7, label="Negative")
        ax_hist.set_xlabel("Predicted Probability")
        ax_hist.set_ylabel("Count")
        ax_hist.set_yscale("log")
        ax_hist.set_title("Probability Distribution")
        ax_hist.legend()
        
        self.logger.experiment["val_confusion_matrix"].append(fig_cm)
        self.logger.experiment["val_roc_curve"].append(fig_roc)
        self.logger.experiment["val_probs_distribution"].append(fig_hist)
        
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall = recall_score(labels_np, preds_np, zero_division=0)
        f1 = f1_score(labels_np, preds_np, zero_division=0)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
        
        plt.close(fig_cm)
        plt.close(fig_roc)
        plt.close(fig_hist)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
