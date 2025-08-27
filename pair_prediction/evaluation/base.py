import torch
import numpy as np
from tqdm import tqdm
from typing import Any, List, Dict

from torch_geometric.loader import DataLoader

from pair_prediction.data.dataset import LinkPredictionDataset
from pair_prediction.model.rinalmo_link_predictor import RiNAlmoLinkPredictionModel
from pair_prediction.model.utils import get_negative_edges


def base_eval(
    model: RiNAlmoLinkPredictionModel,
    dataset: LinkPredictionDataset,
    device: torch.device,
    **kwargs: Any
) -> List[Dict[str, Any]]:
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=kwargs.get("batch_size", 256), shuffle=False)

    outputs = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating predictions"):
            batch = batch.to(device)

            # Mask for non-canonical edges
            edge_mask = np.concatenate(batch.edge_type) == 'non-canonical'
            message_passing_edge_index = batch.edge_index[:, ~edge_mask]

            # Prepare tokens for RiNALMo (as in _rinalmo_step)
            rna_tokens = torch.tensor(model.tokenizer.batch_tokenize(batch.seq), device=device)
            with torch.cuda.amp.autocast():
                node_embeddings = model(batch.features, rna_tokens, message_passing_edge_index)

            # Positive (non-canonical) edges
            pos_edge_index = batch.edge_index[:, edge_mask]
            pos_logits = model.compute_edge_logits(node_embeddings, pos_edge_index, batch.batch)
            pos_labels = torch.ones(pos_logits.size(0), device=device, dtype=torch.float32)

            # Negative edges
            neg_edge_index = get_negative_edges(batch, validation=True)
            neg_logits = model.compute_edge_logits(node_embeddings, neg_edge_index, batch.batch)
            neg_labels = torch.zeros(neg_logits.size(0), device=device, dtype=torch.float32)

            # Concatenate logits and labels
            all_logits = torch.cat([pos_logits, neg_logits], dim=0)
            all_labels = torch.cat([pos_labels, neg_labels], dim=0)
            probabilities = torch.sigmoid(all_logits)

            threshold = kwargs.get("threshold", 0.5)
            predictions = (probabilities > threshold)

            outputs.append({
                "data": batch.cpu(),
                "preds": predictions.cpu(),
                "labels": all_labels.cpu(),
                "probabilities": probabilities.cpu(),
                "edge_types": np.concatenate(batch.edge_type),
                "pair_types": batch.pair_type.cpu(),
            })
    
    return outputs
