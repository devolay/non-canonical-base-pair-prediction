import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from pair_prediction.model.model import LinkPredictorModel
from pair_prediction.model.rinalmo_link_predictor import RiNAlmoLinkPredictionModel
from pair_prediction.model.global_model import LinkPredictorGlobalModel
from pair_prediction.model.utils import get_negative_edges
from pair_prediction.config import ModelConfig
from pair_prediction.visualization.metrics import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_probability_distribution,
    plot_non_canonical_pair_accuracy,
)
from pair_prediction.constants import BASE_DIR

from rinalmo.data.alphabet import Alphabet


class LitWrapper(pl.LightningModule):
    def __init__(self, config = ModelConfig):
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
        self.save_hyperparameters(config.__dict__)
        self.lr = config.lr
        self.min_lr = config.min_lr

        self.model_type = config.model_type
        self.negative_sample_ratio = config.negative_sample_ratio
        self.hard_negative_sampling = config.hard_negative_sampling
        self.hard_negative_sampling_temperature = config.hard_negative_sampling_temperature

        self.use_scheduler = config.use_scheduler
        self.scheduler_patience = config.scheduler_patience

        if self.model_type == "local":
            self.model = LinkPredictorModel(
                in_channels=config.in_channels,
                hidden_channels=config.gnn_channels[0],
                num_layers=len(config.gnn_channels),
                dropout=config.dropout
            )
        elif self.model_type == "global":
            if config.cnn_channels is None:
                raise ValueError("cnn_channels must be provided for global models.")
            self.model = LinkPredictorGlobalModel(
                in_channels=config.in_channels,
                gnn_channels=config.gnn_channels,
                cnn_channels=config.cnn_channels,
                dropout=config.dropout
            )
        elif self.model_type == "rinalmo":
            self.model = RiNAlmoLinkPredictionModel(
                in_channels=config.in_channels,
                gnn_channels=config.gnn_channels,
                cnn_head_embed_dim=config.cnn_head_embed_dim,
                cnn_head_num_blocks=config.cnn_head_num_blocks,
                kernel_size=config.kernel_size,
                dropout=config.dropout
            )
            self.model._load_pretrained_lm_weights(
                BASE_DIR / "models" / "rinalmo" / "rinalmo_giga_pretrained.pt",
                freeze_lm=config.freeze_embeddings
            )
            self.tokenizer = Alphabet()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        self.val_outputs = []

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)     

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def _common_step(self, batch, validation: bool = False):
        if self.model_type == "global":
            all_logits, all_labels = self._global_model_step(batch, validation)
        elif self.model_type == "rinalmo":
            all_logits, all_labels = self._rinalmo_step(batch, validation)
        else:
            all_logits, all_labels = self._local_model_step(batch, validation)
        return all_logits, all_labels

    def _global_model_step(self, batch, validation: bool = False):
        edge_mask = np.concatenate(batch.edge_type)
        edge_mask = edge_mask == 'non-canonical'
        message_passing_edge_index = batch.edge_index[:, ~edge_mask]
        node_embeddings, global_representation = self.model(batch.features, message_passing_edge_index, batch.batch)

        pos_edge_index = batch.edge_index[:, edge_mask]
        pos_logits = self.model.compute_edge_logits(node_embeddings, pos_edge_index, global_representation, batch.batch)
        pos_labels = torch.ones(pos_logits.size(0), device=self.device, dtype=torch.float32)

        neg_edge_index = get_negative_edges(
            batch, validation=validation, sample_ratio=self.negative_sample_ratio, 
            hard_negative_sampling=self.hard_negative_sampling, model=self.model, node_embeddings=node_embeddings,
            hard_negative_sampling_temperature=self.hard_negative_sampling_temperature
        )
        neg_logits = self.model.compute_edge_logits(node_embeddings, neg_edge_index, global_representation, batch.batch)
        neg_labels = torch.zeros(neg_logits.size(0), device=self.device, dtype=torch.float32)

        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        return all_logits, all_labels
    
    def _rinalmo_step(self, batch, validation: bool = False):
        edge_mask = np.concatenate(batch.edge_type)
        edge_mask = edge_mask == 'non-canonical'
        message_passing_edge_index = batch.edge_index[:, ~edge_mask]
        rna_tokens = torch.tensor(self.tokenizer.batch_tokenize(batch.seq), device=self.device)
        with torch.cuda.amp.autocast():
            node_embeddings = self.model(batch.features, rna_tokens, message_passing_edge_index)

        pos_edge_index = batch.edge_index[:, edge_mask]
        pos_logits = self.model.compute_edge_logits(node_embeddings, pos_edge_index, batch.batch)
        pos_labels = torch.ones(pos_logits.size(0), device=self.device, dtype=pos_logits.dtype)

        neg_edge_index = get_negative_edges(
            batch, validation=validation, sample_ratio=self.negative_sample_ratio, 
            hard_negative_sampling=self.hard_negative_sampling, model=self.model, node_embeddings=node_embeddings,
            hard_negative_sampling_temperature=self.hard_negative_sampling_temperature
        )
        neg_logits = self.model.compute_edge_logits(node_embeddings, neg_edge_index, batch.batch)
        neg_labels = torch.zeros(neg_logits.size(0), device=self.device, dtype=pos_logits.dtype)

        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        return all_logits, all_labels
    
    def _local_model_step(self, batch, validation: bool = False):
        edge_mask = np.concatenate(batch.edge_type)
        edge_mask = edge_mask == 'non-canonical'
        message_passing_edge_index = batch.edge_index[:, ~edge_mask]
        node_embeddings = self.model(batch.features, message_passing_edge_index)

        pos_edge_index = batch.edge_index[:, edge_mask]
        pos_logits = self.model.compute_edge_logits(node_embeddings, pos_edge_index)
        pos_labels = torch.ones(pos_logits.size(0), device=self.device, dtype=torch.float32)

        neg_edge_index = get_negative_edges(batch, validation=validation, sample_ratio=self.negative_sample_ratio)
        neg_logits = self.model.compute_edge_logits(node_embeddings, neg_edge_index)
        neg_labels = torch.zeros(neg_logits.size(0), device=self.device, dtype=torch.float32)

        all_logits = torch.cat([pos_logits, neg_logits], dim=0)
        all_labels = torch.cat([pos_labels, neg_labels], dim=0)
        return all_logits, all_labels

    def training_step(self, batch, batch_idx):
        """
        Training step computing loss for link prediction.
        """
        all_logits, all_labels = self._common_step(batch)
        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step computing loss, accuracy, and collects predictions,
        labels, and probabilities for logging.
        """
        all_logits, all_labels = self._common_step(batch, validation=True)

        loss = F.binary_cross_entropy_with_logits(all_logits, all_labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        
        probabilities = torch.sigmoid(all_logits)
        preds = (probabilities > 0.5)

        edge_types = np.concatenate(batch.edge_type)
        pair_types = batch.pair_type
    
        self.val_outputs.append({ 
            "preds": preds.detach(),
            "labels": all_labels.detach(),
            "probabilities": probabilities.detach(),
            "edge_types": edge_types,
            "pair_types": pair_types.detach(),
            "loss": loss.detach(),
        })
        return loss    

    def on_validation_epoch_end(self):
        all_preds = torch.cat([x["preds"] for x in self.val_outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in self.val_outputs], dim=0)
        all_probabilities = torch.cat([x["probabilities"] for x in self.val_outputs], dim=0)
        all_edge_types = np.concatenate([x["edge_types"] for x in self.val_outputs])
        all_pair_types = torch.cat([x["pair_types"] for x in self.val_outputs], dim=0) 
        
        self._log_validation(
            all_preds, all_labels, all_probabilities, 
            all_edge_types, all_pair_types,
        )
        
        self.val_outputs.clear()
        
    def _log_validation(self, preds, labels, probabilities, edge_types, pair_types):
        """
        Logs the confusion matrix, ROC curve (with AUC), and probability distribution plots.
        Uses Neptune logger's log_image method.
        """
        preds_np = preds.cpu().numpy().astype(np.float32)
        labels_np = labels.cpu().numpy().astype(np.float32)
        prob_np = probabilities.cpu().numpy().astype(np.float32)
        pair_types = pair_types.cpu().numpy().astype(np.float32)
        
        fig_cm, _ = plot_confusion_matrix(labels_np, preds_np)
        fig_roc, _ = plot_roc_curve(labels_np, prob_np)
        fig_hist, _ = plot_probability_distribution(labels_np, prob_np)
        fig_pair_types, _ = plot_non_canonical_pair_accuracy(
            preds_np, labels_np, edge_types, pair_types
        )
        
        self.logger.experiment["val_confusion_matrix"].append(fig_cm)
        self.logger.experiment["val_roc_curve"].append(fig_roc)
        self.logger.experiment["val_probs_distribution"].append(fig_hist)
        self.logger.experiment["val_pair_type_accuracy"].append(fig_pair_types)
        
        plt.close(fig_cm)
        plt.close(fig_roc)
        plt.close(fig_hist)
        plt.close(fig_pair_types)
        
        # Calculate and log metrics
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall = recall_score(labels_np, preds_np, zero_division=0)
        f1 = f1_score(labels_np, preds_np, zero_division=0)
        self.log("val_precision", precision, prog_bar=True)
        self.log("val_recall", recall, prog_bar=True)
        self.log("val_f1", f1, prog_bar=True)
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if not self.use_scheduler:
            return self.optimizer
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            min_lr=self.min_lr,
            patience=self.scheduler_patience,
        )

        return {
            "optimizer": self.optimizer, 
            "lr_scheduler": self.scheduler, 
            "monitor": "val_loss"
        }
    